"""Hermes identity adapter: physical rows are witnesses, never message identities."""
import hashlib
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

SCHEMA_VERSION = "1"


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def fingerprint(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def decoded(value, fallback=None):
    if value is None:
        return fallback
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def decode_content(value):
    from hermes_state import SessionDB
    return SessionDB._decode_content(value)


def uid(value):
    return value if isinstance(value, str) and value else None


def eligible(row):
    return bool(row.get("active", 1) or row.get("compacted", 0))


def logical_message(row):
    value = uid(row.get("message_uid"))
    return "uid:" + value if value else "legacy-row:" + str(row["id"])


def absorbed(row):
    value = decoded(row.get("absorbed_message_uids"), [])
    return [item for item in value if uid(item)] if isinstance(value, list) else []


def call_occurrences(row):
    calls = decoded(row.get("tool_calls"), [])
    mapping = decoded(row.get("tool_call_uids"), {})
    if not isinstance(calls, list) or not isinstance(mapping, dict):
        raise ValueError(f"Invalid tool identity on Hermes row {row['id']}")
    offsets = {}
    for position, call in enumerate(calls):
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ValueError(f"Invalid tool call on Hermes row {row['id']}")
        from agent.message_sanitization import coalesce_tool_call_id, tool_call_id_variants
        provider_id = coalesce_tool_call_id(call)
        occurrence = mapping.get(provider_id)
        if occurrence is None:
            witnesses = [mapping[key] for key in sorted(tool_call_id_variants(call)) if key in mapping]
            if witnesses and any(value != witnesses[0] for value in witnesses[1:]):
                raise ValueError(f"Conflicting tool occurrence witnesses on Hermes row {row['id']}")
            occurrence = witnesses[0] if witnesses else None
        if isinstance(occurrence, list):
            offset = offsets.get(provider_id, 0)
            if offset >= len(occurrence):
                raise ValueError(f"Incomplete tool occurrence witness on Hermes row {row['id']}")
            occurrence = occurrence[offset]
            offsets[provider_id] = offset + 1
        occurrence = uid(occurrence)
        yield position, provider_id, occurrence, call["function"]


def components(row):
    """Yield (part, logical identity, version, kind, visible text, original payload)."""
    from .store import encoded, visible
    role = row["role"]
    content = decode_content(row.get("content"))
    logical = logical_message(row)
    if role in ("user", "assistant") and visible(content):
        kind = "user" if role == "user" else "talk"
        yield "message", logical, fingerprint([kind, content]), kind, visible(content), content
    for position, provider_id, occurrence, function in call_occurrences(row):
        identity = "call:" + occurrence if occurrence else f"legacy-call:{logical}:{position}"
        yield f"call:{position}", identity, fingerprint(function), "tool", encoded(function), function
    if role == "tool":
        occurrence = uid(row.get("tool_call_uid"))
        identity = "result:" + occurrence if occurrence else "result-" + logical
        kind = "work" if row.get("tool_name") == "delegate_task" else "echo"
        yield "result", identity, fingerprint([kind, content]), kind, visible(content), content


def initialize(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS identities (
            logical TEXT NOT NULL, version TEXT NOT NULL, event INTEGER NOT NULL,
            PRIMARY KEY(logical,version));
        CREATE TABLE IF NOT EXISTS row_events (
            row_id INTEGER NOT NULL, part TEXT NOT NULL, logical TEXT NOT NULL,
            version TEXT NOT NULL, event INTEGER NOT NULL, PRIMARY KEY(row_id,part));
        CREATE TABLE IF NOT EXISTS active_events (event INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS active_units (id INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS tool_occurrences (
            uid TEXT PRIMARY KEY, provider_id TEXT NOT NULL, row_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS event_order (event INTEGER PRIMARY KEY, ordinal INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS unit_order (id INTEGER PRIMARY KEY, ordinal INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS event_sequence ON event_order(ordinal);
        CREATE INDEX IF NOT EXISTS unit_sequence ON unit_order(ordinal);
        CREATE INDEX IF NOT EXISTS tool_provider ON tool_occurrences(provider_id,row_id DESC,ordinal DESC);
        CREATE TABLE IF NOT EXISTS hook_bindings (key TEXT PRIMARY KEY, uid TEXT);
        CREATE TABLE IF NOT EXISTS raw_heads (
            source TEXT PRIMARY KEY, original_source TEXT NOT NULL, event INTEGER);
    """)
    with db:
        db.execute("INSERT OR IGNORE INTO meta VALUES ('identity_schema',?)", (SCHEMA_VERSION,))


def connect_snapshot(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def current_rows(conn, session, before_row=None):
    """Stream one current visible version per uid, retaining physical-row witnesses.

    The first pass reads only identity/state columns, not large tool payloads. Both
    passes execute inside the caller's read transaction, including same-row edits.
    """
    columns = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
    fields = [name for name in ("id", "message_uid", "active", "compacted", "absorbed_message_uids") if name in columns]
    latest, first, removed = {}, {}, {}
    for record in conn.execute(f"SELECT {','.join(fields)} FROM messages WHERE session_id=? ORDER BY id", (session,)):
        row = dict(record)
        logical = logical_message(row)
        first.setdefault(logical, row["id"])
        latest[logical] = row
        for value in [row.get("message_uid"), *absorbed(row)]:
            if uid(value):
                removed[value] = not eligible(row)
    candidates = {logical:row for logical,row in latest.items()
                  if eligible(row) and not removed.get(row.get("message_uid"),False)}
    absorbed_uids = {value for row in candidates.values() for value in absorbed(row)}
    selected = {row["id"]: logical for logical, row in candidates.items()
                if (before_row is None or row["id"] < before_row)
                and not (logical.startswith("uid:") and logical[4:] in absorbed_uids)}
    for row_id in sorted(selected, key=lambda value: first[selected[value]]):
        row = conn.execute("SELECT * FROM messages WHERE id=? AND session_id=?", (row_id,session)).fetchone()
        yield dict(row), True
    for record in conn.execute("SELECT * FROM messages WHERE session_id=? ORDER BY id", (session,)):
        row = dict(record)
        if row["id"] not in selected:
            yield row, False


def sync(store, home, session, before_row=None):
    path = Path(home).resolve() / "state.db"
    if not path.exists():
        return
    with store.lock:
        identity = (path.stat().st_dev, path.stat().st_ino)
        cached = store._hermes_readers.get(str(path))
        if cached is None or cached[0] != identity:
            if cached is not None:
                cached[1].close()
            cached = (identity, connect_snapshot(path))
            store._hermes_readers[str(path)] = cached
        conn = cached[1]
        revision = conn.execute("PRAGMA data_version").fetchone()[0]
        key = (str(path), identity, session, revision, before_row)
        if store._sync_key == key:
            return
        old_total = store.total
        old_leaf_total = store.leaf_total
        try:
            conn.execute("BEGIN")
            store._syncing = True
            with store.db:
                active = {}
                for row, selected in current_rows(conn, session, before_row):
                    if before_row is not None and row["id"] >= before_row:
                        continue
                    if not eligible(row):
                        store.db.execute("DELETE FROM row_events WHERE row_id=?", (row["id"],))
                        continue
                    # Copies which are not the selected version still receive an address
                    # when that exact version already exists, but cannot revive it.
                    parts = list(components(row))
                    previous = {r["part"]: r for r in store.db.execute("SELECT * FROM row_events WHERE row_id=?", (row["id"],))}
                    for removed_part in previous.keys() - {part[0] for part in parts}:
                        store.db.execute("DELETE FROM row_events WHERE row_id=? AND part=?", (row["id"],removed_part))
                    for part, logical, version, kind, text, payload in parts:
                        match = store.db.execute("SELECT event FROM identities WHERE logical=? AND version=?", (logical, version)).fetchone()
                        prior = previous.get(part)
                        if match is None and prior is not None and prior["version"] == version and ("legacy-row:" in prior["logical"] or prior["logical"].startswith("legacy-call:")):
                            # A uid backfilled onto this same durable row is a witness,
                            # unlike matching the text of an unrelated physical row.
                            match = (prior["event"],)
                            store.db.execute("INSERT INTO identities VALUES (?,?,?)", (logical, version, match[0]))
                        if match is None and not selected:
                            store.db.execute("DELETE FROM row_events WHERE row_id=? AND part=?", (row["id"],part))
                            continue
                        if match is None:
                            raw_source = "tool-result:" + logical[7:] if logical.startswith("result:") else f"row:{row['id']}"
                            raw = store.db.execute("""SELECT r.payload,h.event FROM raw_heads h JOIN raw_activity r
                                ON r.source=h.original_source WHERE h.source=?""", (raw_source,)).fetchone()
                            if raw is not None and raw[1] is None:
                                payload = json.loads(raw[0])
                                if logical.startswith("result:"):
                                    from .store import visible
                                    text = visible(payload)
                            event = store.append(logical + ":" + version, kind, text, payload, row.get("timestamp"))
                            if raw is not None and raw[1] is None:
                                store.db.execute("UPDATE raw_heads SET event=? WHERE source=?", (event,raw_source))
                            store.db.execute("INSERT INTO identities VALUES (?,?,?)", (logical, version, event))
                        else:
                            event = match[0]
                        if prior is None or (prior["logical"],prior["version"],prior["event"]) != (logical,version,event):
                            store.db.execute("INSERT OR REPLACE INTO row_events VALUES (?,?,?,?,?)", (row["id"],part,logical,version,event))
                        if selected:
                            if logical not in active or row["id"] >= active[logical][0]:
                                active[logical] = (row["id"],event)
                    if selected:
                        for position, provider_id, occurrence, function in call_occurrences(row):
                            if occurrence:
                                store.db.execute("""INSERT INTO tool_occurrences VALUES (?,?,?,?) ON CONFLICT(uid)
                                    DO UPDATE SET provider_id=excluded.provider_id,row_id=excluded.row_id,ordinal=excluded.ordinal
                                    WHERE tool_occurrences.provider_id<>excluded.provider_id OR tool_occurrences.row_id<>excluded.row_id
                                    OR tool_occurrences.ordinal<>excluded.ordinal""", (occurrence,provider_id,row["id"],position))
                store.reconcile_active([event for _,event in active.values()])
            conn.rollback()
        except BaseException:
            store.db.rollback()
            conn.rollback()
            store.total = old_total
            store.leaf_total = old_leaf_total
            raise
        finally:
            store._syncing = False
        store._sync_key = key


def observe_tool(store, phase, provider_id, *, tool_name=None, args=None, result=None,
                 tool_call_uid=None, turn_id=None, api_request_id=None):
    """Journal hooks without inventing provider-id events beside durable occurrences."""
    key = dumps([turn_id or "", api_request_id or "", provider_id])
    with store.lock, store.db:
        occurrence = uid(tool_call_uid)
        if phase == "pre":
            if occurrence is None:
                row = store.db.execute("""SELECT t.uid FROM tool_occurrences t WHERE t.provider_id=?
                    AND EXISTS (SELECT 1 FROM identities i JOIN active_events a ON a.event=i.event WHERE i.logical='call:'||t.uid)
                    AND NOT EXISTS (SELECT 1 FROM identities i JOIN active_events a ON a.event=i.event WHERE i.logical='result:'||t.uid)
                    ORDER BY t.row_id DESC,t.ordinal DESC LIMIT 1""", (provider_id,)).fetchone()
                occurrence = row[0] if row else None
            store.db.execute("INSERT OR REPLACE INTO hook_bindings VALUES (?,?)", (key, occurrence))
        elif occurrence is None:
            row = store.db.execute("SELECT uid FROM hook_bindings WHERE key=?", (key,)).fetchone()
            occurrence = row[0] if row else None
        payload = {"tool_call_id": provider_id, "tool_call_uid": occurrence,
                   "tool_name": tool_name, "args": args} if phase == "pre" else result
        # A hook without a durable witness remains a raw journal record. It must
        # never be guessed onto an unrelated call just because the id/text repeats.
        source = ("tool-call:" if phase == "pre" else "tool-result:") + occurrence if occurrence else "unbound-hook:" + phase + ":" + fingerprint(key) + ":" + uuid4().hex
        store.record_raw(source, payload)
        return occurrence
