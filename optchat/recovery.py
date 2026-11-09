"""Offline, non-destructive recovery of row-addressed OptChat archives.

``recover_archive`` accepts a read-only old chat.db (or its archive directory),
a consistent Hermes database snapshot, a NEW archive root and destination
session. It never rewrites either source. Ambiguous tool occurrences fail closed
unless the caller explicitly preserves those originals in the old archive.
"""
import fcntl
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
import hashlib
import json
import os
import tempfile
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .identity import call_occurrences, components, connect_snapshot, current_rows, decoded, dumps, fingerprint
from .store import Store, encoded, initialize_audit, leaf_text, visible

RECOVERY_VERSION = "5"


class RecoveryAmbiguityError(RuntimeError):
    def __init__(self, ambiguities):
        self.ambiguities = ambiguities
        super().__init__("Ambiguous legacy tool occurrence identities; originals have not been changed: " + dumps(ambiguities))


def _table(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _manifest(report):
    result = dict(report)
    result["event_map"] = list(report["event_map"].items())
    result["leaf_map"] = list(report["leaf_map"].items())
    result["node_map"] = [[list(old), list(new)] for old, new in report["node_map"].items()]
    result["unmapped_reasons"] = list(report["unmapped_reasons"].items())
    return encoded(result)


def _report(text):
    result = json.loads(text)
    for name in ("event_map", "leaf_map", "unmapped_reasons"):
        result[name] = {int(key): value for key, value in result[name]}
    result["node_map"] = {tuple(old): tuple(new) for old, new in result["node_map"]}
    return result


def _text_matches(event, component, raw_admission=None):
    if event["kind"] != component[3]:
        return False
    if event["kind"] == "tool":
        return decoded(event["text"]) == decoded(component[4])
    if event["text"] != component[4]:
        return False
    if event["kind"] in ("user","talk","echo","work"):
        original = decoded(event["payload"])
        current = component[5]
        if isinstance(original,(dict,list)) or isinstance(current,(dict,list)):
            # Attachment placeholders are not a content-version witness. Even an
            # old raw admission is source-only if its structured version differs.
            return original == current
        if event["kind"] in ("user","talk"):
            return original == current or (raw_admission is not None and original == raw_admission)
    return True


def _plan(source, hermes, source_session, preserve_ambiguous):
    """Resolve identities first. No destination exists when ambiguity is reported."""
    canonical, rows, providers = {}, {}, {}
    snapshot = hashlib.sha256()
    for row, selected in current_rows(hermes, source_session):
        # SQLite rows contain metadata BLOBs (for example display_identity), not
        # just JSON values. Tag those bytes for the snapshot hash, not the log.
        snapshot.update(dumps({key:{"sqlite_blob_hex":value.hex()} if isinstance(value,bytes) else value
                               for key,value in row.items()}).encode("utf-8"))
        parts = list(components(row))
        rows[row["id"]] = (row, parts)
        if selected:
            for component in parts:
                identity = component[1]
                previous = canonical.get(identity)
                if previous is None or row["id"] >= previous[0]["id"]:
                    canonical[identity] = (row, component)
        # Nonselected and withdrawn rows still witness provider-id reuse. Dropping
        # them here would make a surviving, unrelated occurrence look unique.
        part_identities = {part[0]: part[1] for part in parts}
        for position, provider, occurrence, function in call_occurrences(row):
            providers.setdefault(("tool", provider), set()).add(part_identities[f"call:{position}"])
        if row["role"] == "tool":
            for component in parts:
                providers.setdefault((component[3], row.get("tool_call_id")), set()).add(component[1])
    # The insertion order is the first logical occurrence, not the latest copied row.
    keys = {(component[1], component[2]): index for index, (row, component) in enumerate(canonical.values())}
    old_identity = {}
    if _table(source, "identities"):
        for row in source.execute("SELECT * FROM identities"):
            old_identity.setdefault(row["event"],[]).append((row["logical"],row["version"]))
    mapping, reasons, originals, ambiguous = {}, {}, {}, {}
    for record in source.execute("SELECT * FROM events ORDER BY id"):
        event = dict(record)
        event_id, origin = event["id"], event["source"]
        candidate = None
        reason = "untracked-source"
        if event_id in old_identity:
            matches = [identity for identity in old_identity[event_id] if identity in keys]
            targets = {keys[identity] for identity in matches}
            if len(targets) > 1:
                raise RecoveryAmbiguityError({event_id: {"source": origin, "candidates": sorted(targets)}})
            candidate = matches[0] if matches else None
            reason = "noncanonical-version"
        elif origin.startswith("row:"):
            try:
                row_id = int(origin[4:])
            except ValueError:
                row_id = None
            witness = rows.get(row_id)
            if witness is None:
                reason = "physical-row-absent-from-snapshot"
            else:
                row, parts = witness
                raw = source.execute("SELECT payload FROM raw_activity WHERE source=?", (origin,)).fetchone() if _table(source,"raw_activity") else None
                admission = decoded(raw[0]) if raw else None
                matches = [part for part in parts if _text_matches(event,part,admission)]
                if len(matches) == 1:
                    candidate = (matches[0][1], matches[0][2])
                    reason = "withdrawn-or-superseded-version"
                else:
                    reason = "historical-row-version-not-in-snapshot"
                    if event["kind"] in ("user","talk","echo","work") and any(part[3] == event["kind"] and part[4] == event["text"] for part in parts):
                        reason = "raw-admission-or-structured-payload-not-version-verifiable"
        elif origin.startswith(("call:", "result:")):
            provider = origin.split(":", 1)[1]
            candidates = providers.get((event["kind"], provider), set())
            # Ambiguity is determined from every occurrence witness, before
            # excluding versions which cannot enter the canonical projection.
            if len(candidates) > 1:
                ambiguous[event_id] = {"source": origin, "candidates": sorted(candidates)}
                reason = "ambiguous-tool-occurrence-preserved-in-source"
            elif candidates:
                logical = next(iter(candidates))
                current = canonical.get(logical)
                if current is None:
                    reason = "withdrawn-or-superseded-tool-occurrence"
                elif _text_matches(event, current[1]):
                    candidate = (current[1][1], current[1][2])
                else:
                    reason = "tool-content-version-not-exact"
        target = keys.get(candidate)
        mapping[event_id] = target
        if target is None:
            reasons[event_id] = reason
        else:
            originals.setdefault(target, event)
    if ambiguous and not preserve_ambiguous:
        raise RecoveryAmbiguityError(ambiguous)
    return list(canonical.values()), rows, mapping, reasons, originals, snapshot.hexdigest(), ambiguous


def _reuse(source, store, event_map, report):
    boundary = store.settings.summary_boundary
    unit_map = {}
    if boundary == "event":
        for old, new in event_map.items():
            if new is not None and leaf_text(source, "event", old) == store.leaf_source(new):
                unit_map[old] = new
    elif _table(source, "leaves"):
        lookup = {tuple(row)[1:]: row[0] for row in store.db.execute("SELECT id,event_start,event_stop,start_offset,stop_offset,source FROM leaves")}
        for leaf in source.execute("SELECT * FROM leaves ORDER BY id"):
            mapped = [event_map.get(event) for event in range(leaf["event_start"],leaf["event_stop"])]
            if not mapped or mapped[0] is None or mapped != list(range(mapped[0],mapped[0]+len(mapped))):
                continue
            span = (mapped[0],mapped[-1]+1,leaf["start_offset"],leaf["stop_offset"],leaf["source"])
            if span in lookup:
                unit_map[leaf["id"]] = lookup[span]
        report["leaf_map"] = unit_map
    for node in source.execute("SELECT * FROM nodes ORDER BY n,start"):
        start, n = node["start"], node["n"]
        if not store.can_copy(node["text"]):
            continue
        mapped = [unit_map.get(unit) for unit in range(start,start+n)]
        if not mapped or mapped[0] is None or mapped[0] % n or mapped != list(range(mapped[0],mapped[0]+n)):
            continue
        target = mapped[0]
        if n > 1 and not store.valid_span(target,n):
            continue
        if n == 1:
            # Projection/capping is part of a leaf's source contract.
            from .chunks import capped
            original = leaf_text(source,boundary,start)
            if boundary == "event" and source.execute("SELECT kind FROM events WHERE id=?", (start,)).fetchone()[0] in ("tool", "echo"):
                original = capped(original,store.settings.tool_chars)
            if original != store.summary_source(target):
                continue
        else:
            children = [source.execute("SELECT text FROM nodes WHERE start=? AND n=?", (child,n//2)).fetchone()
                        for child in (start,start+n//2)]
            if any(child is None or store.node(target+offset,n//2) != child[0]
                   for offset, child in zip((0,n//2),children)):
                continue
        existing = store.db.execute("SELECT text,metrics FROM nodes WHERE start=? AND n=?", (target,n)).fetchone()
        if existing is not None and tuple(existing) != (node["text"],node["metrics"]):
            continue
        store.db.execute("INSERT OR IGNORE INTO nodes VALUES (?,?,?,?)", (target,n,node["text"],node["metrics"]))
        store.db.execute("DELETE FROM jobs WHERE start=? AND n=?", (target,n))
        report["node_map"][(start,n)] = (target,n)
    if store.settings.compression_policy == "eager":
        store._queue_eager_jobs()


def recover_archive(source_archive, hermes_db, destination_root, session, *,
                    settings=None, budget=128000, source_session=None,
                    preserve_ambiguous=False):
    """Publish a NEW canonical archive; return a durable, idempotent address map.

    source_session identifies the source Hermes session and must agree with the
    old archive's metadata. session is the destination session (it may differ).
    Ambiguous/untracked originals are never deleted or guessed: the report names
    every source-only event and retains the old archive path and SHA-256.
    A pre-existing destination is accepted only for exactly the same inputs.
    Use the source profile's projection/summary settings, particularly tool_chars
    and turn_source_bytes, so reused nodes retain their original source contract.
    Keep source_archive available after recovery: unmapped originals and their
    attachments intentionally remain there, not in the active canonical view.
    """
    settings = settings or Settings()
    source_path = Path(source_archive).resolve()
    if source_path.is_dir():
        source_path /= "chat.db"
    hermes_path = Path(hermes_db).resolve()
    destination_root = Path(destination_root).resolve()
    destination = destination_root / hashlib.sha256(session.encode()).hexdigest()
    if source_path == destination / "chat.db" or hermes_path == destination / "chat.db":
        raise ValueError("Recovery requires a NEW destination, never a source database")
    source = hermes = source_guard = None
    try:
        source_lock = source_path.parent / "writer.lock"
        if source_lock.exists():
            source_guard = source_lock.open("rb")
            try:
                fcntl.flock(source_guard,fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Source archive has a live writer; stop it and take an offline snapshot") from exc
        source = connect_snapshot(source_path)
        hermes = connect_snapshot(hermes_path)
        source.execute("BEGIN")
        hermes.execute("BEGIN")
        meta = dict(source.execute("SELECT key,value FROM meta"))
        old_session = meta.get("session")
        source_session = source_session or old_session or session
        if old_session and source_session != old_session:
            raise ValueError("source_session disagrees with the old archive session")
        if meta.get("summary_boundary", "event") != settings.summary_boundary:
            raise ValueError("Recovery settings must preserve the source summary_boundary")
        canonical, rows, event_map, reasons, originals, hermes_hash, ambiguities = _plan(source,hermes,source_session,preserve_ambiguous)
        if not rows and source.execute("SELECT 1 FROM events LIMIT 1").fetchone():
            raise ValueError("The Hermes snapshot has no rows for source_session; refusing empty recovery")
        with source_path.open("rb") as original_file:
            source_hash = hashlib.file_digest(original_file,"sha256").hexdigest()
        signature = fingerprint([RECOVERY_VERSION,str(source_path),source_hash,hermes_hash,source_session,session,asdict(settings),budget,preserve_ambiguous])
        destination_root.mkdir(parents=True,exist_ok=True,mode=0o700)
        with (destination_root / (".recover-" + destination.name + ".lock")).open("a+b") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            if destination.exists():
                existing = connect_snapshot(destination / "chat.db")
                try:
                    values = dict(existing.execute("SELECT key,value FROM meta WHERE key IN ('recovery_signature','recovery_report')"))
                finally:
                    existing.close()
                if values.get("recovery_signature") != signature:
                    raise FileExistsError("Destination already exists with different recovery inputs; choose a NEW destination root/session")
                return _report(values["recovery_report"])
            report = dict(recovery_version=RECOVERY_VERSION,source_archive=str(source_path),source_sha256=source_hash,
                          hermes_db=str(hermes_path),hermes_fingerprint=hermes_hash,
                          source_session=source_session,destination_session=session,destination=str(destination),
                          event_map=event_map,leaf_map={},node_map={},preserved_unmapped=sorted(reasons),
                          unmapped_reasons=reasons,ambiguities=ambiguities,
                          source_events=len(event_map),canonical_events=len(canonical),
                          source_nodes=source.execute("SELECT count(*) FROM nodes").fetchone()[0],
                          source_raw_records=source.execute("SELECT count(*) FROM raw_activity").fetchone()[0] if _table(source,"raw_activity") else 0,
                          source_image_records=source.execute("SELECT count(*) FROM images").fetchone()[0] if _table(source,"images") else 0)
            with tempfile.TemporaryDirectory(prefix=".recover-build-",dir=destination_root) as scratch:
                store = Store(scratch,session,budget,settings)
                try:
                    store._syncing = True
                    with store.db:
                        if _table(source,"raw_activity"):
                            # These records belong to the previous generation.
                            # Exact mapped evidence is already bound to its event;
                            # none may become an unconsumed admission for a later edit.
                            store.db.executemany("INSERT INTO raw_activity VALUES (?,?)",
                                (("previous-raw:" + row["source"], row["payload"])
                                 for row in source.execute("SELECT source,payload FROM raw_activity")))
                        identities = {}
                        for index, (row, part) in enumerate(canonical):
                            name, logical, version, kind, text, payload = part
                            original = originals.get(index)
                            if original is not None:
                                payload = decoded(original["payload"])
                                if kind in ("tool","echo","work"):
                                    text = original["text"]
                                    # A projected/pruned result is not its exact original.
                                    # Changing text also invalidates incompatible cached spans.
                                    if kind != "tool" and payload is not None:
                                        text = visible(payload)
                            event = store.append(logical+":"+version,kind,text,payload,row.get("timestamp"),historical=True)
                            store.db.execute("INSERT INTO identities VALUES (?,?,?)", (logical,version,event))
                            identities[(logical,version)] = event
                        for row, parts in rows.values():
                            if not (row.get("active",1) or row.get("compacted",0)):
                                continue
                            for name, logical, version, kind, text, payload in parts:
                                event = identities.get((logical,version))
                                if event is not None:
                                    store.db.execute("INSERT INTO row_events VALUES (?,?,?,?,?)", (row["id"],name,logical,version,event))
                        if _table(source,"images"):
                            for image in source.execute("SELECT * FROM images"):
                                if image["data"] is None and image["error"] is None:
                                    continue
                                event = event_map.get(image["event"])
                                if event is not None and originals[event]["id"] == image["event"]:
                                    store.db.execute("INSERT OR REPLACE INTO images VALUES (?,?,?,?,?)", (event,image["ordinal"],image["mime"],image["data"],image["error"]))
                    store._syncing = False
                    store.finish_turn()
                    with store.db:
                        _reuse(source,store,event_map,report)
                        report["reused_nodes"] = len(set(report["node_map"].values()))
                        report["reused_parent_nodes"] = len({node for node in report["node_map"].values() if node[1] > 1})
                        report["pending_jobs"] = store.db.execute("SELECT count(*) FROM jobs").fetchone()[0]
                        store.db.execute("INSERT INTO meta VALUES ('recovery_signature',?)", (signature,))
                        store.db.execute("INSERT INTO meta VALUES ('recovery_report',?)", (_manifest(report),))
                    store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    built = store.root
                finally:
                    store.close()
                os.rename(built,destination)
                directory = os.open(destination_root,os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            return _report(_manifest(report))
    finally:
        if hermes is not None:
            hermes.close()
        if source is not None:
            source.close()
        if source_guard is not None:
            source_guard.close()


def _preserve_audit(source, target, working_root, published_root, source_path, maximum):
    """Seal a consistent audit snapshot and retain old children for exact zoom."""
    audit_file = Path(working_root) / "audit-source.db"
    with closing(sqlite3.connect(audit_file)) as audit:
        source.backup(audit)
        audit.execute("PRAGMA journal_mode=DELETE")
    audit_file.chmod(0o400)
    with audit_file.open("rb") as original:
        checksum = hashlib.file_digest(original,"sha256").hexdigest()
    published = str(Path(published_root) / audit_file.name)
    initialize_audit(target)
    with target:
        target.executemany("INSERT OR IGNORE INTO audit_nodes VALUES (?,?,?,?,?)",
            ((start,n,text,metrics,published) for start,n,text,metrics in source.execute(
                "SELECT start,n,text,metrics FROM nodes WHERE length(CAST(text AS BLOB))>?",
                (maximum,))))
    return dict(source_archive=published,source_origin=str(source_path),source_sha256=checksum)


def _build_migration_jobs(worker, jobs, progress=None):
    """Bound independent compactions by the configured concurrency, not serial retries."""
    model_jobs = []
    for job in jobs:
        if worker.store.can_copy(job[2]):
            worker.build(*job)
        else:
            model_jobs.append(job)
    if not model_jobs:
        return
    lanes = min(worker.settings.summary_concurrency,len(model_jobs))
    groups = [model_jobs[index::lanes] for index in range(lanes)]
    def build(group):
        if worker.settings.batching == "auto" and hasattr(worker.summarize,"summarize_many"):
            worker.build_many(group)
        else:
            for job in group:
                worker.build(*job)
    if lanes == 1:
        build(groups[0])
        return
    with ThreadPoolExecutor(max_workers=lanes,thread_name_prefix="optchat-migration") as pool:
        futures = [pool.submit(copy_context().run,build,group) for group in groups]
        for future in as_completed(futures):
            future.result()
            if progress:
                progress(f"OptChat migration committed {worker.completed_nodes} new summaries")


def migrate_summary_limits(source_archive, destination_root, *, settings, summarize,
                           budget=128000, progress=None):
    """Create a strict-byte cache generation without replaying the original log.

    All originals, identities, attachments and the persisted view are copied.
    Conforming cached nodes are imported unchanged. Only nonconforming nodes in
    that view are recompressed, using their previous summaries as migration
    sources, not pretending to reread the entire conversation. Preserved audit
    children remain available through zoom; active descendants are regenerated
    from originals only when the current projection needs them.

    The source must be offline. A standalone read-only snapshot is retained
    inside the new generation, so reopening the source cannot change its audit.
    Unrecovered provider errors and fallback-only summaries prevent publication.
    Normal batch-to-singleton transport recovery remains supported.
    Callers publish the returned generation separately, after verification.
    """
    from .batch import SummaryJob
    from .worker import Worker

    if settings.chunk_limit != "bytes" or settings.compression_policy != "eager":
        raise ValueError("Summary-limit migration requires eager, strict-byte settings")
    source_path = Path(source_archive).resolve()
    if source_path.is_dir():
        source_path /= "chat.db"
    destination_root = Path(destination_root).resolve()
    guard = source = store = worker = None
    try:
        lock_path = source_path.parent / "writer.lock"
        if lock_path.exists():
            guard = lock_path.open("rb")
            try:
                fcntl.flock(guard, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Source archive has a live writer; take an offline snapshot") from exc
        source = connect_snapshot(source_path)
        source.execute("BEGIN")
        metadata = dict(source.execute("SELECT key,value FROM meta"))
        session = metadata["session"]
        if metadata.get("identity_schema") != "1":
            raise ValueError("Recover legacy message identities before migrating summary limits")
        if metadata.get("summary_boundary", "event") != settings.summary_boundary:
            raise ValueError("Migration must preserve the summary boundary")
        destination = destination_root / hashlib.sha256(session.encode()).hexdigest()
        if destination == source_path.parent:
            raise ValueError("Migration requires a NEW destination generation")
        destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (destination_root / (".migrate-" + destination.name + ".lock")).open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if destination.exists():
                raise FileExistsError("Destination already exists; choose a NEW generation")
            with tempfile.TemporaryDirectory(prefix=".migrate-build-", dir=destination_root) as scratch:
                copied = Path(scratch) / destination.name
                copied.mkdir(mode=0o700)
                copied_db = copied / "chat.db"
                with closing(sqlite3.connect(copied_db)) as target:
                    source.backup(target)
                    audit = _preserve_audit(source,target,copied,destination,source_path,
                                            settings.summary_limit.maximum)
                    # Prune only the unpublished active cache. Its immutable
                    # historical children remain in the separate audit table.
                    with target:
                        target.execute("DROP TRIGGER nodes_no_delete")
                        target.execute("DELETE FROM nodes WHERE length(CAST(text AS BLOB))>?",
                                       (settings.summary_limit.maximum,))
                        target.execute("""CREATE TRIGGER nodes_no_delete BEFORE DELETE ON nodes
                            BEGIN SELECT RAISE(ABORT,'summaries are immutable'); END""")
                        target.execute("DELETE FROM jobs")
                        target.execute("DELETE FROM summary_view")
                        target.execute("""DELETE FROM meta WHERE key IN
                            ('main_cache_head','summary_cache_head','summary_epoch','view_target',
                             'view_byte_target','summary_view_target','summary_view_byte_target')""")
                        target.execute("INSERT OR REPLACE INTO meta VALUES ('jobs_initialized','1')")
                store = Store(scratch, session, budget, settings)
                worker = Worker(store, settings, summarize, start=False)
                parts = store.parts()
                jobs = []
                seeds = []
                for start,n in parts:
                    if store.node(start,n) is not None:
                        continue
                    old = source.execute("SELECT text FROM nodes WHERE start=? AND n=?",
                                         (start,n)).fetchone()
                    if old is None and n != 1:
                        raise ValueError("The persisted view contains an unbuilt parent; settle it before migration")
                    text = old[0] if old else store.summary_source(start)
                    action = ("Recompress this preserved historical summary for a new cache generation"
                              if old else "Compress this historical message")
                    worker.snapshots[(start,n)] = SummaryJob(
                        f"{start}+{n}", "", text, settings.summary_limit.output_bytes,
                        action, "bytes", settings.summary_limit.target,
                    )
                    jobs.append((start,n,text,None))
                    seeds.append({"start": start, "n": n, "source": "previous_summary" if old else "original",
                                  "source_bytes": len(text.encode("utf-8"))})
                _build_migration_jobs(worker,jobs,progress)
                for start,n,_,_ in jobs:
                    row = store.db.execute("SELECT text,metrics FROM nodes WHERE start=? AND n=?",
                                           (start,n)).fetchone()
                    if row is None or not store.can_copy(row[0]) or "fallback" in json.loads(row[1]):
                        raise RuntimeError("Migration could not produce a valid summary; no generation published")
                    if progress:
                        progress(f"OptChat migration committed {start}+{n}")
                store.prepare(store.count(),next_message=True)
                view = store.render()
                if len(view.encode("utf-8")) > store.byte_budget:
                    raise RuntimeError("Migrated cover still exceeds the byte budget; no generation published")
                report = dict(
                    migration_version=2, **audit, destination=str(destination),
                    session=session, node_bytes=settings.summary_limit.target,
                    view_bytes=len(view.encode("utf-8")), byte_budget=store.byte_budget,
                    rebuilt_frontier_nodes=len(jobs), migration_sources=seeds,
                    source_events=source.execute("SELECT count(*) FROM events").fetchone()[0],
                    preserved_originals=store.db.execute("SELECT count(*) FROM events").fetchone()[0],
                    imported_nodes=store.db.execute("SELECT count(*) FROM nodes").fetchone()[0]-len(jobs),
                )
                with store.db:
                    store.db.execute("INSERT OR REPLACE INTO meta VALUES ('limit_migration_report',?)",
                                     (encoded(report),))
                store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                worker.close()
                worker = None
                os.rename(copied, destination)
                directory = os.open(destination_root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                return report
    finally:
        if worker is not None:
            worker.close()
        if store is not None:
            store.close()
        if source is not None:
            source.close()
        if guard is not None:
            guard.close()
