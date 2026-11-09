"""Append-only originals and nodes; transactional, monotonically coarsening view."""
import fcntl
from contextlib import nullcontext
import hashlib
import heapq
import json
import sqlite3
import threading
import time
from pathlib import Path

from .limits import MAX_HISTORY_BYTES, measure


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def size(text):
    return len(text.encode("utf-8"))


def visible(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        return "\n".join(p.get("text", "") if p.get("type") in ("text", "input_text")
                         else f"[attachment: {p.get('type', 'unknown')}]" for p in content if isinstance(p, dict))
    return encoded(content)


def leaf_text(db, boundary, start, raw=False):
    """Read projected or exact visible leaf text, including from a read-only export."""
    if boundary == "event":
        event = db.execute("SELECT kind,text FROM events WHERE id=?",(start,)).fetchone()
        if event is None:
            raise ValueError("No such message")
        return event["kind"]+": "+event["text"]
    leaf = db.execute("SELECT * FROM leaves WHERE id=?",(start,)).fetchone()
    if leaf is None:
        raise ValueError("No such summary leaf")
    if not raw:
        return leaf["source"]
    out = []
    for event in db.execute("SELECT id,kind,text FROM events WHERE id>=? AND id<? ORDER BY id",(leaf["event_start"],leaf["event_stop"])):
        a = leaf["start_offset"] if event["id"] == leaf["event_start"] else 0
        b = leaf["stop_offset"] if event["id"] == leaf["event_stop"]-1 else len(event["text"])
        out.append(event["kind"]+": "+event["text"][a:b])
    return "\n".join(out)


VIEW_ACK = "(The view continues in the next message.)"


def merge_priority(end, start, n):
    # Age from the inclusive last message of the sibling pair. Subtracting
    # 2*n-1 is essential: subtracting 2*n just reproduces the old ordering.
    return (end-(start+2*n-1))/(4*n)


class MergeFrontier:
    def __init__(self, parts, end, eligible):
        self.parts, self.end, self.eligible = list(parts), end, eligible
        self.following = list(range(1,len(parts)+1))
        self.preceding = list(range(-1,len(parts)-1))
        self.pairs = []
        for j in range(len(parts)):
            self.offer(j)

    def offer(self, j):
        if j < 0 or j >= len(self.parts) or self.parts[j] is None:
            return
        k = self.following[j]
        start,n = self.parts[j]
        if k >= len(self.parts) or self.parts[k] != (start+n,n) or start % (2*n) or not self.eligible(start,n):
            return
        heapq.heappush(self.pairs,(-merge_priority(self.end,start,n),start,j,k,n))

    def take(self):
        while self.pairs:
            _,start,j,k,n = heapq.heappop(self.pairs)
            if self.parts[j] == (start,n) and self.following[j] == k and self.parts[k] == (start+n,n):
                return j,k,start,n
        return None

    def merge(self, j, k, parent):
        self.parts[j],self.parts[k] = parent,None
        self.following[j] = self.following[k]
        if self.following[k] < len(self.parts):
            self.preceding[self.following[k]] = j
        self.offer(self.preceding[j])
        self.offer(j)


def sticky_head(head, text, split):
    """A line-aligned head of text to end a message at, so caches keyed on message or marker
    boundaries can reuse it next turn. The head stays while text still starts with it (merges
    mostly change the end) and moves when a merge reaches above it or text has grown far past it."""
    if not split:
        return None
    if head and text.startswith(head) and len(text)*split/2 <= len(head) < len(text):
        return head
    cut = text.rfind("\n",0,int(len(text)*split))+1
    return text[:cut] if cut > len("<chat>\n") else None


def initialize_audit(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS audit_nodes (
            start INTEGER, n INTEGER, text TEXT NOT NULL, metrics TEXT NOT NULL,
            source_archive TEXT NOT NULL, PRIMARY KEY(start,n));
        CREATE TRIGGER IF NOT EXISTS audit_nodes_no_update BEFORE UPDATE ON audit_nodes
            BEGIN SELECT RAISE(ABORT,'audit summaries are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS audit_nodes_no_delete BEFORE DELETE ON audit_nodes
            BEGIN SELECT RAISE(ABORT,'audit summaries are immutable'); END;
    """)


class Store:
    def __init__(self, root, session, budget, settings=None):
        from .config import Settings
        self.settings = settings or Settings()
        self.root = Path(root) / hashlib.sha256(session.encode()).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lockfile = open(self.root / "writer.lock", "a+b")
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lockfile.close()
            raise RuntimeError("OptChat archive already has a writer; resume it in the owning Hermes process")
        self.lock = threading.RLock()
        self.db = None
        self.closed = False
        self._hermes_readers = {}
        self._sync_key = None
        self._syncing = False
        try:
            self.db = sqlite3.connect(self.root / "chat.db", check_same_thread=False)
            self.db.row_factory = sqlite3.Row
            self.initialize(session,budget)
        except BaseException:
            self.close()
            raise

    def _check_cached_limits(self, settings):
        if (settings.chunk_limit == "bytes"
                and self.db.execute("SELECT 1 FROM sqlite_master WHERE name='nodes'").fetchone()
                and self.db.execute(
                    "SELECT 1 FROM nodes WHERE length(CAST(text AS BLOB))>? LIMIT 1",
                    (settings.summary_limit.maximum,),
                ).fetchone()):
            raise RuntimeError(
                "Cached summaries exceed the selected UTF-8 byte limit; use "
                "optchat.recovery.migrate_summary_limits with a NEW destination "
                "generation before resuming. The existing archive is unchanged."
            )

    def initialize(self, session, budget):
        # Legacy nonempty archives use event leaves. Check before changing their schema.
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='meta'").fetchone():
            row = self.db.execute("SELECT value FROM meta WHERE key='summary_boundary'").fetchone()
            boundary = row[0] if row else "event"
            populated = self.db.execute("SELECT 1 FROM events LIMIT 1").fetchone()
            if populated and boundary != self.settings.summary_boundary:
                raise RuntimeError(f"This archive uses summary_boundary={boundary}; start a new Hermes session for {self.settings.summary_boundary}")
            identity = self.db.execute("SELECT value FROM meta WHERE key='identity_schema'").fetchone()
            if populated and identity is None:
                raise RuntimeError("Legacy OptChat archive requires offline recovery: use optchat.recovery.recover_archive with a consistent Hermes database and a NEW destination root; the original archive is unchanged")
            if identity is not None and identity[0] != "1":
                raise RuntimeError("Unsupported OptChat identity schema; recover into a new archive")
        self._check_cached_limits(self.settings)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, source TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
                text TEXT NOT NULL, payload TEXT NOT NULL, date REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS nodes (
                start INTEGER, n INTEGER, text TEXT NOT NULL, metrics TEXT NOT NULL,
                PRIMARY KEY(start,n));
            CREATE TABLE IF NOT EXISTS view (start INTEGER PRIMARY KEY, n INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS summary_view (start INTEGER PRIMARY KEY, n INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, start INTEGER, n INTEGER, date REAL, metrics TEXT);
            CREATE TABLE IF NOT EXISTS raw_activity (source TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS images (
                event INTEGER, ordinal INTEGER, mime TEXT, data BLOB, error TEXT,
                PRIMARY KEY(event,ordinal));
            CREATE TABLE IF NOT EXISTS jobs (start INTEGER, n INTEGER, PRIMARY KEY(start,n));
            CREATE TABLE IF NOT EXISTS leaves (
                id INTEGER PRIMARY KEY, event_start INTEGER NOT NULL, event_stop INTEGER NOT NULL,
                start_offset INTEGER NOT NULL, stop_offset INTEGER NOT NULL, source TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS leaves_events ON leaves(event_stop,id);
            CREATE INDEX IF NOT EXISTS jobs_leaf ON jobs(n,start);
            CREATE INDEX IF NOT EXISTS jobs_end ON jobs(start+n,n);
            CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT,'originals are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT,'originals are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS nodes_no_update BEFORE UPDATE ON nodes
                BEGIN SELECT RAISE(ABORT,'summaries are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS nodes_no_delete BEFORE DELETE ON nodes
                BEGIN SELECT RAISE(ABORT,'summaries are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS raw_no_update BEFORE UPDATE ON raw_activity
                BEGIN SELECT RAISE(ABORT,'activity is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS raw_no_delete BEFORE DELETE ON raw_activity
                BEGIN SELECT RAISE(ABORT,'activity is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS leaves_no_update BEFORE UPDATE ON leaves
                BEGIN SELECT RAISE(ABORT,'leaf spans are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS leaves_no_delete BEFORE DELETE ON leaves
                BEGIN SELECT RAISE(ABORT,'leaf spans are immutable'); END;
        """)
        initialize_audit(self.db)
        from .identity import initialize as initialize_identity
        initialize_identity(self.db)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO event_order SELECT event,event FROM active_events")
            self.db.execute("INSERT OR IGNORE INTO unit_order SELECT id,id FROM active_units")
        row = self.db.execute("SELECT value FROM meta WHERE key='logical_reorder'").fetchone()
        self.logical_reorder = bool(row and row[0] == "1")
        self.budget = budget
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('session', ?)", (session,))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('summary_boundary', ?)",(self.settings.summary_boundary,))
        # One writer holds the lock, so the count is kept in memory instead of scanning per append.
        self.total = self.db.execute("SELECT coalesce(max(id)+1,0) FROM events").fetchone()[0]
        self.leaf_total = self.db.execute("SELECT coalesce(max(id)+1,0) FROM leaves").fetchone()[0]
        self.history_end = None
        self.plan_key = None
        self.configured = False
        self.configure(self.settings)

    def _queue_eager_jobs(self):
        """Restore missing active leaves and ready compatible parents after recovery."""
        self.db.execute("""INSERT OR IGNORE INTO jobs SELECT u.id,1 FROM active_units u
            LEFT JOIN nodes n ON n.start=u.id AND n.n=1 WHERE n.start IS NULL""")
        candidates = self.db.execute("""SELECT a.start,a.n*2 FROM nodes a
            JOIN nodes b ON b.start=a.start+a.n AND b.n=a.n
            LEFT JOIN nodes p ON p.start=a.start AND p.n=a.n*2
            WHERE a.start % (a.n*2)=0 AND p.start IS NULL""").fetchall()
        self.db.executemany("INSERT OR IGNORE INTO jobs VALUES (?,?)",
            ((start,n) for start,n in candidates if self.valid_span(start,n)))

    def configure(self, settings):
        with self.lock, self.db:
            if self.configured and settings == self.settings:
                return
            boundary = self.db.execute("SELECT value FROM meta WHERE key='summary_boundary'").fetchone()[0]
            if boundary != settings.summary_boundary:
                raise RuntimeError(f"This archive uses summary_boundary={boundary}; start a new Hermes session for {settings.summary_boundary}")
            self._check_cached_limits(settings)
            row = self.db.execute("SELECT value FROM meta WHERE key='compression_policy'").fetchone()
            policy = row[0] if row else "eager"
            self.settings = settings
            unit = self.db.execute("SELECT value FROM meta WHERE key='view_unit'").fetchone()
            if (unit[0] if unit else "bytes") != settings.chunk_limit:
                self.db.execute("DELETE FROM meta WHERE key IN ('view_target','summary_view_target','view_byte_target','summary_view_byte_target')")
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('view_unit',?)",(settings.chunk_limit,))
            if policy != settings.compression_policy:
                # Jobs are derived state. Retain immutable nodes, discard speculative work.
                self.db.execute("DELETE FROM jobs")
            initialized = self.db.execute("SELECT 1 FROM meta WHERE key='jobs_initialized'").fetchone()
            if settings.compression_policy == "eager" and (not initialized or policy != settings.compression_policy):
                self._queue_eager_jobs()
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('compression_policy',?)",(settings.compression_policy,))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('batching',?)",(settings.batching,))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('chunk_limit',?)",(settings.chunk_limit,))
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('jobs_initialized','1')")
            self.plan_key = None
            self.configured = True

    def close(self):
        with self.lock:
            if not self.closed:
                for _, connection in self._hermes_readers.values():
                    connection.close()
                self._hermes_readers.clear()
                if self.db is not None:
                    self.db.close()
                self.lockfile.close()
                self.closed = True

    def count(self):
        return self.total

    def unit_count(self):
        return self.total if self.settings.summary_boundary == "event" else self.leaf_total

    def unit_end(self, event_end):
        if self.settings.summary_boundary == "event":
            return event_end
        with self.lock:
            if self.db.execute("SELECT 1 FROM leaves l JOIN active_units u ON u.id=l.id WHERE l.event_start<? AND l.event_stop>?",(event_end,event_end)).fetchone():
                raise RuntimeError("A sealed turn crosses the current input; archive boundary is inconsistent")
            return self.db.execute("SELECT coalesce(max(l.id)+1,0) FROM leaves l JOIN active_units u ON u.id=l.id WHERE l.event_stop<=?",(event_end,)).fetchone()[0]

    def seal(self, end):
        """Seal prior turns, including a recovered interrupted turn, never the live suffix."""
        if self.settings.summary_boundary == "event":
            return
        from .chunks import turn_chunks
        with self.lock:
            with self.db:
                covered = {row[0] for row in self.db.execute("""
                    SELECT DISTINCT e.event FROM active_events e JOIN leaves l
                    ON e.event>=l.event_start AND e.event<l.event_stop
                    JOIN active_units u ON u.id=l.id""")}
                active = self.db.execute("SELECT value FROM meta WHERE key='active_start'").fetchone()
                active = int(active[0]) if active else end
                events = [dict(r) for r in self.db.execute(
                    "SELECT e.* FROM events e JOIN active_events a ON a.event=e.id JOIN event_order o ON o.event=e.id WHERE e.id<? ORDER BY o.ordinal", (end,))
                    if r["id"] not in covered]
                groups, group = [], []
                for event in events:
                    # A withdrawn version is a real gap, never part of a new leaf.
                    if group and (event["id"] != group[-1]["id"]+1 or event["id"] == active or event["kind"] == "user" and event["id"] < active):
                        groups.append(group)
                        group = []
                    group.append(event)
                if group:
                    groups.append(group)
                leaf_id = self.leaf_total
                for group in groups:
                    for span in turn_chunks(group,self.settings.turn_source_bytes,self.settings.tool_chars):
                        self.db.execute("INSERT INTO leaves VALUES (?,?,?,?,?,?)",(leaf_id,*span))
                        self.db.execute("INSERT INTO active_units VALUES (?)",(leaf_id,))
                        self.db.execute("INSERT INTO unit_order VALUES (?,?)",(leaf_id,leaf_id))
                        self.db.execute("INSERT INTO view VALUES (?,1)",(leaf_id,))
                        if self.settings.compression_policy == "eager":
                            self.db.execute("INSERT INTO jobs VALUES (?,1)",(leaf_id,))
                        leaf_id += 1
            self.leaf_total = leaf_id
            with self.db:
                self.order_units()
            self.message_boundary()

    def start_turn(self, event_start):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('active_start',?)",(str(event_start),))

    def finish_turn(self):
        self.seal(self.total)
        with self.lock, self.db:
            self.db.execute("DELETE FROM meta WHERE key='active_start'")

    def leaf_source(self, start, raw=False):
        with self.lock:
            return leaf_text(self.db,self.settings.summary_boundary,start,raw)

    def summary_source(self, start):
        source = self.leaf_source(start)
        if self.settings.summary_boundary == "event" and self.event(start)["kind"] in ("tool","echo"):
            from .chunks import capped
            source = capped(source,self.settings.tool_chars)
        return source

    def address(self, start, n):
        return ("leaf:" if self.settings.summary_boundary == "user_turn" else "")+f"{start}+{n}|"

    def append(self, source, kind, text, payload=None, date=None, *, historical=False):
        with self.lock:
            with nullcontext() if self._syncing else self.db:
                old = self.db.execute("SELECT id FROM events WHERE source=?", (source,)).fetchone()
                if old:
                    return old[0]
                i = self.total
                self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",
                                (i, source, kind, text, encoded(payload), time.time() if date is None else date))
                self.db.execute("INSERT INTO active_events VALUES (?)", (i,))
                ordinal = self.db.execute("SELECT coalesce(max(ordinal)+1,0) FROM event_order").fetchone()[0]
                self.db.execute("INSERT INTO event_order VALUES (?,?)",(i,ordinal))
                if kind in ("user","talk","echo","work"):
                    from .images import capture
                    capture(self.db,i,payload,historical=historical)
                if self.settings.summary_boundary == "event":
                    self.db.execute("INSERT INTO active_units VALUES (?)", (i,))
                    self.db.execute("INSERT INTO unit_order VALUES (?,?)",(i,ordinal))
                    self.db.execute("INSERT INTO view VALUES (?,1)", (i,))
                    if self.settings.compression_policy == "eager":
                        self.db.execute("INSERT INTO jobs VALUES (?,1)",(i,))
            self.total = i+1
            if self.settings.summary_boundary == "event" and not self._syncing:
                self.message_boundary()
            return i

    def message_boundary(self):
        if self.settings.compression_policy == "eager" and self.db.execute(
                "SELECT 1 FROM meta WHERE key IN ('view_target','view_byte_target') LIMIT 1").fetchone():
            with self.db:
                self.fit(next_message=True)

    def event(self, i):
        with self.lock:
            row = self.db.execute("SELECT * FROM events WHERE id=?", (i,)).fetchone()
            if row is None:
                raise ValueError("No such message")
            return dict(row)

    def record_raw(self, source, payload):
        with self.lock, nullcontext() if self._syncing else self.db:
            from .identity import fingerprint
            value = encoded(payload)
            old = self.db.execute("SELECT payload FROM raw_activity WHERE source=?", (source,)).fetchone()
            original_source = source if old is None or old[0] == value else source+":version:"+fingerprint(payload)
            self.db.execute("INSERT OR IGNORE INTO raw_activity VALUES (?,?)",(original_source,value))
            self.db.execute("""INSERT INTO raw_heads VALUES (?,?,NULL) ON CONFLICT(source)
                DO UPDATE SET original_source=excluded.original_source,event=NULL
                WHERE raw_heads.original_source<>excluded.original_source""", (source,original_source))

    def event_for_row(self, row_id):
        """Canonical event for this physical Hermes row, or None when withdrawn."""
        with self.lock:
            row = self.db.execute("SELECT event FROM row_events WHERE row_id=? ORDER BY CASE part WHEN 'message' THEN 0 WHEN 'result' THEN 1 ELSE 2 END,part LIMIT 1", (row_id,)).fetchone()
            return row[0] if row else None

    def observe_tool(self, phase, provider_id, **kwargs):
        from .identity import observe_tool
        return observe_tool(self, phase, provider_id, **kwargs)

    @property
    def copy_limit(self):
        return min(MAX_HISTORY_BYTES,max(1,self.byte_budget//2-128))

    def can_copy(self, text):
        return self.settings.summary_limit.can_copy(text,max_bytes=self.copy_limit)

    def valid_span(self, start, n):
        count, low, high = self.db.execute(
            "SELECT count(*),min(ordinal-id),max(ordinal-id) FROM unit_order WHERE id>=? AND id<?",
            (start,start+n)).fetchone()
        return count == n and low == high

    def order_units(self):
        if self.settings.summary_boundary == "event":
            ordered = [r[0] for r in self.db.execute("SELECT o.event FROM event_order o JOIN active_units u ON u.id=o.event ORDER BY o.ordinal")]
        else:
            ordered = [r[0] for r in self.db.execute("""SELECT l.id FROM leaves l
                JOIN active_units u ON u.id=l.id JOIN event_order o ON o.event=l.event_start
                ORDER BY o.ordinal,l.start_offset,l.id""")]
        self.db.execute("DELETE FROM unit_order")
        self.db.executemany("INSERT INTO unit_order VALUES (?,?)",((unit,rank) for rank,unit in enumerate(ordered)))
        self.logical_reorder = ordered != sorted(ordered)
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('logical_reorder',?)",("1" if self.logical_reorder else "0",))

    def reconcile_active(self, events):
        """Project current versions in source order; originals/addresses never change."""
        ordered = list(dict.fromkeys(sorted(events) if isinstance(events,(set,frozenset)) else events))
        events = set(ordered)
        old_order = [row[0] for row in self.db.execute("SELECT event FROM event_order ORDER BY ordinal")]
        previous_parts = self.parts()
        old = {row[0] for row in self.db.execute("SELECT event FROM active_events")}
        removed = old-events
        self.db.executemany("DELETE FROM active_events WHERE event=?",((event,) for event in removed))
        self.db.executemany("INSERT INTO active_events VALUES (?)",((event,) for event in events-old))
        self.db.execute("DELETE FROM event_order")
        self.db.executemany("INSERT INTO event_order VALUES (?,?)",((event,rank) for rank,event in enumerate(ordered)))
        if self.settings.summary_boundary == "event":
            units = events
        else:
            units = {row[0] for row in self.db.execute("""SELECT l.id FROM leaves l
                WHERE (SELECT count(*) FROM active_events a WHERE a.event>=l.event_start AND a.event<l.event_stop)=l.event_stop-l.event_start
                AND (SELECT min(ordinal-event)=max(ordinal-event) FROM event_order o WHERE o.event>=l.event_start AND o.event<l.event_stop)
                AND EXISTS (SELECT 1 FROM active_units u WHERE u.id=l.id)""")}
        previous = {row[0] for row in self.db.execute("SELECT id FROM active_units")}
        if units == previous and not removed and ordered == old_order:
            return
        self.db.execute("DELETE FROM active_units")
        self.db.executemany("INSERT INTO active_units VALUES (?)",((unit,) for unit in units))
        self.order_units()
        parts, covered = [], set()
        def retain(start, n):
            members = [unit for unit in range(start,start+n) if unit in units]
            if len(members) == n and self.valid_span(start,n):
                parts.append((start,n))
                covered.update(members)
            elif members and n > 1:
                retain(start,n//2)
                retain(start+n//2,n//2)
        for start,n in previous_parts:
            retain(start,n)
        parts.extend((unit,1) for unit in units-covered)
        self.db.execute("DELETE FROM view")
        self.db.executemany("INSERT INTO view VALUES (?,?)",parts)
        self.db.execute("DELETE FROM summary_view")
        self.db.execute("DELETE FROM meta WHERE key IN ('main_cache_head','summary_epoch','view_target','view_byte_target','summary_view_target','summary_view_byte_target')")
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('view_epoch',?)",(str(self.view_epoch()+1),))
        for start,n in self.db.execute("SELECT start,n FROM jobs").fetchall():
            if not self.valid_span(start,n):
                self.db.execute("DELETE FROM jobs WHERE start=? AND n=?",(start,n))
        if self.settings.compression_policy == "eager":
            for unit in units-previous:
                if self.node(unit,1) is None:
                    self.require(unit,1)
                    continue
                n = 2
                while n <= self.unit_count():
                    start = unit//n*n
                    if not self.valid_span(start,n):
                        break
                    if self.node(start,n) is None:
                        if self.node(start,n//2) is not None and self.node(start+n//2,n//2) is not None:
                            self.db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?)",(start,n))
                        break
                    n *= 2
        self.plan_key = None

    def node(self, start, n):
        row = self.db.execute("SELECT text FROM nodes WHERE start=? AND n=?", (start, n)).fetchone()
        return row[0] if row else None

    def parts(self):
        return [tuple(r) for r in self.db.execute("SELECT v.start,v.n FROM view v JOIN unit_order o ON o.id=v.start ORDER BY o.ordinal")]

    def lines(self, end=None, ids=True, raw=None, *, summary=False, ready_prefix=False):
        with self.lock:
            raw = self.settings.compression_policy == "on_demand" if raw is None else raw
            out = []
            # Bulk-read complete view parts. Historical boundaries and missing
            # ready-prefix parents descend only as far as their available cover.
            table = "summary_view" if summary else "view"
            rows = self.db.execute(f"""SELECT v.start,v.n,c.text FROM {table} v
                LEFT JOIN nodes c ON c.start=v.start AND c.n=v.n
                JOIN unit_order o ON o.id=v.start
                WHERE ? IS NULL OR v.start<? ORDER BY o.ordinal""",(end,end)).fetchall()
            def add(start, n, text=None, fetched=False):
                if end is not None and start >= end:
                    return True
                if end is not None and start+n > end:
                    if n > 1:
                        return add(start,n//2) and add(start+n//2,n//2)
                    return True
                if not fetched:
                    text = self.node(start, n)
                if text is None and n == 1 and raw and not ready_prefix:
                    text = self.leaf_source(start,raw=True)
                if text is None:
                    if ready_prefix:
                        return n > 1 and add(start,n//2) and add(start+n//2,n//2)
                    raise RuntimeError("Summary not ready")
                out.append((self.address(start,n) if ids else "") + " ".join(text.splitlines()))
                return True
            for start, n, text in rows:
                if not add(start,n,text,fetched=True):
                    break
            return out

    def render(self, end=None):
        return "<chat>\n" + "\n".join(self.lines(end)) + "\n</chat>"

    @property
    def view_budget(self):
        return self.settings.history_limit(self.budget)

    @property
    def byte_budget(self):
        ceiling = min(self.budget, MAX_HISTORY_BYTES)
        return min(self.view_budget, ceiling) if self.settings.chunk_limit == "bytes" else ceiling

    def framing(self, unit=None):
        return 1 if (unit or self.settings.chunk_limit) == "sentences" else 15

    def line_cost(self, text, unit=None):
        unit = unit or self.settings.chunk_limit
        return measure(text,unit)+(0 if unit == "sentences" else 1)

    def view_cost(self, end=None):
        return self.framing()+sum(self.line_cost(line) for line in self.lines(end))

    def fit(self, end=None, limit=None, *, next_message=False):
        return self._fit("view",end,limit,self.view_budget,self.byte_budget,next_message=next_message)

    def _fit(self, table, end, limit, budget, byte_budget=None, *, next_message=False):
        byte_budget = self.byte_budget if byte_budget is None else byte_budget
        rows = self.db.execute(f"""SELECT v.start,v.n,c.text,p.text FROM {table} v
            LEFT JOIN nodes c ON c.start=v.start AND c.n=v.n
            LEFT JOIN nodes p ON p.start=v.start AND p.n=v.n*2
            JOIN unit_order o ON o.id=v.start
            WHERE ? IS NULL OR v.start+v.n<=? ORDER BY o.ordinal""",(end,end)).fetchall()
        parts = [(start,n) for start,n,_,_ in rows]
        nodes = {}
        for start,n,current,parent in rows:
            nodes[(start,n)] = current
            nodes[(start,n*2)] = parent
        def node(start, n):
            key = (start,n)
            # A merge may expose a higher ancestor not present in the initial
            # join. Cache its lookup, including absence, only for this fit call.
            if key not in nodes:
                nodes[key] = self.node(start,n)
            return nodes[key]
        def cost(part):
            start, n = part
            text = node(start, n)
            if text is None and n == 1 and self.settings.compression_policy == "on_demand":
                text = self.leaf_source(start,raw=True)
            line = self.address(start,n) + (" ".join(text.splitlines()) if text is not None else "(waiting)")
            return self.line_cost(line),size(line)+1
        costs = {part:cost(part) for part in parts}
        total = self.framing()+sum(value[0] for value in costs.values())
        byte_total = 15+sum(value[1] for value in costs.values())
        before,byte_before = total,byte_total
        T = self.unit_count() if end is None else end
        explicit = limit is not None
        if limit is None:
            limit = self.compression_target(total,budget,table+"_target",minimum=self.framing())
        byte_limit = self.compression_target(byte_total,byte_budget,table+"_byte_target") if self.settings.chunk_limit != "bytes" else limit
        if not explicit and not next_message and total <= budget and byte_total <= byte_budget:
            return total
        frontier = MergeFrontier(parts,T,lambda start,n:node(start,2*n) is not None and self.valid_span(start,2*n))
        changed = False
        while total > limit or byte_total > byte_limit:
            pair = frontier.take()
            if pair is None:
                break
            j,k,start,n = pair
            up = (start,2*n)
            parent,left,right = cost(up),costs[(start,n)],costs[(start+n,n)]
            total += parent[0]-left[0]-right[0]
            byte_total += parent[1]-left[1]-right[1]
            costs[up] = parent
            frontier.merge(j,k,up)
            changed = True
        # A background completion must save a whole batch (or a full budget
        # during a large import). Message boundaries may publish partial progress.
        publish = explicit or next_message or (
            (before <= budget or total <= max(budget//2,before-budget)) and
            (byte_before <= byte_budget or byte_total <= max(byte_budget//2,byte_before-byte_budget)))
        if changed and not publish:
            return before
        if changed:
            original = dict(parts)
            merged = dict(part for part in frontier.parts if part is not None)
            removed = [(start,) for start,n in parts if merged.get(start) != n]
            replacements = [(start,n) for start,n in merged.items() if original.get(start) != n]
            self.db.executemany(f"DELETE FROM {table} WHERE start=?",removed)
            self.db.executemany(f"INSERT INTO {table} VALUES (?,?)",replacements)
        if total <= limit:
            self.db.execute("DELETE FROM meta WHERE key=?",(table+"_target",))
        if byte_total <= byte_limit:
            self.db.execute("DELETE FROM meta WHERE key=?",(table+"_byte_target",))
        if changed and table == "view":
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('view_epoch',?)",(str(self.view_epoch()+1),))
        return total

    def view_epoch(self):
        row = self.db.execute("SELECT value FROM meta WHERE key='view_epoch'").fetchone()
        return int(row[0]) if row else 0

    def cache_head(self, key, text, split):
        """Persist the chosen message boundary as well as the view it splits."""
        with self.lock, self.db:
            row = self.db.execute("SELECT value FROM meta WHERE key=?",(key,)).fetchone()
            saved = json.loads(row[0]) if row else {}
            head = sticky_head(saved.get("head") if saved.get("split") == split else None,text,split)
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",(key,encoded({"head":head,"split":split})))
            return head

    def compression_target(self, total, budget, key, trigger=1, minimum=15):
        row = self.db.execute("SELECT value FROM meta WHERE key=?",(key,)).fetchone()
        if row or total > budget*trigger:
            target = min(int(row[0]),max(minimum,budget//2)) if row else max(minimum,budget//2)
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",(key,str(target)))
            return target
        return budget

    def ordered_summary_lines(self, end, raw):
        """A bounded logical prefix when an edited version has a later log ID."""
        row = self.db.execute("SELECT ordinal FROM unit_order WHERE id=?",(end,)).fetchone()
        if row is None:
            return []
        boundary = row[0]
        budget = max(2*self.framing(),min(self.settings.history_limit(32000),self.view_budget//4))
        byte_budget = max(30,self.byte_budget//4)
        kept, total, byte_total = [], self.framing(), 15
        def visit(start,n):
            nonlocal total,byte_total
            row = self.db.execute("SELECT ordinal FROM unit_order WHERE id=?",(start,)).fetchone()
            if row is None or row[0] >= boundary:
                return True
            text = self.node(start,n)
            if row[0]+n > boundary or text is None:
                if n > 1:
                    return visit(start,n//2) and visit(start+n//2,n//2)
                if not raw:
                    return False
                text = self.leaf_source(start,raw=True)
            line = " ".join(text.splitlines())
            cost, byte_cost = self.line_cost(line),size(line)+1
            if total+cost > budget or byte_total+byte_cost > byte_budget:
                return False
            kept.append(line)
            total += cost
            byte_total += byte_cost
            return True
        for start,n in self.parts():
            if not visit(start,n):
                break
        return kept

    def summary_lines(self, end, raw=False):
        """Smaller context in the selected unit, resynchronized with main-view merges."""
        with self.lock, self.db:
            if self.logical_reorder:
                return self.ordered_summary_lines(end,raw)
            epoch = self.view_epoch()
            row = self.db.execute("SELECT value FROM meta WHERE key='summary_epoch'").fetchone()
            if row is None or int(row[0]) != epoch:
                self.db.execute("DELETE FROM summary_view")
                self.db.execute("INSERT INTO summary_view SELECT * FROM view")
                self.db.execute("DELETE FROM meta WHERE key IN ('summary_view_target','summary_view_byte_target')")
                self.db.execute("INSERT OR REPLACE INTO meta VALUES ('summary_epoch',?)",(str(epoch),))
                # A new main batch establishes a fresh smaller summarizer view.
                limit = max(self.framing(),min(self.settings.history_limit(32000),self.view_budget//4)//2)
                self.db.execute("INSERT OR REPLACE INTO meta VALUES ('summary_view_target',?)",(str(limit),))
            else:
                stop = self.db.execute("SELECT coalesce(max(start+n),0) FROM summary_view").fetchone()[0]
                self.db.execute("INSERT INTO summary_view SELECT * FROM view WHERE start>=?",(stop,))
                limit = None
            budget = max(2*self.framing(),min(self.settings.history_limit(32000),self.view_budget//4))
            byte_budget = min(MAX_HISTORY_BYTES//4,max(30,self.byte_budget//4))
            self._fit("summary_view",end,limit,budget,byte_budget)
            lines = self.lines(end,ids=False,raw=raw,summary=True,ready_prefix=not raw)
            # A partial historical boundary can descend into finer children.
            # Keep a complete prefix of lines, never half a summary or a gap.
            kept,total,byte_total = [],self.framing(),15
            for line in lines:
                total += self.line_cost(line)
                byte_total += size(line)+1
                if total > budget or byte_total > byte_budget:
                    break
                kept.append(line)
            return kept

    def require(self, start, n):
        """Queue only a requested node's missing dependency subtree; caller holds lock."""
        if self.node(start,n) is not None:
            return
        if not self.valid_span(start,n):
            raise RuntimeError("Cannot summarize a withdrawn archive span")
        if self.db.execute("SELECT 1 FROM jobs WHERE start=? AND n=?",(start,n)).fetchone():
            return
        if n > 1:
            self.require(start,n//2)
            self.require(start+n//2,n//2)
        self.db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?)",(start,n))

    def prepare(self, end, *, next_message=False):
        """Predict the smallest older prefix of work needed for compression headroom."""
        with self.lock, self.db:
            self.history_end = end
            if self.settings.compression_policy == "eager":
                total = self.fit(next_message=next_message)
                # A persisted/coarsened view need not retain every descendant
                # cache. A later edit or historical boundary can expose one.
                missing = self.db.execute("""SELECT v.start,v.n FROM view v
                    LEFT JOIN nodes n ON n.start=v.start AND n.n=v.n
                    WHERE n.start IS NULL""").fetchall()
                for start,n in missing:
                    self.require(start,n)
                # Imported siblings may both exist without a completion event to
                # schedule their parent. Only restore work on the selected cover.
                parents = self.db.execute("""SELECT a.start,a.n*2 FROM view a
                    JOIN view b ON b.start=a.start+a.n AND b.n=a.n
                    JOIN nodes l ON l.start=a.start AND l.n=a.n
                    JOIN nodes r ON r.start=b.start AND r.n=b.n
                    LEFT JOIN nodes p ON p.start=a.start AND p.n=a.n*2
                    WHERE a.start % (a.n*2)=0 AND p.start IS NULL""").fetchall()
                self.db.executemany("INSERT OR IGNORE INTO jobs VALUES (?,?)",
                    ((start,n) for start,n in parents if self.valid_span(start,n)))
                return total
            # Demand mode retains its raw-history headroom policy. Sawtooth
            # batches apply to the summary-only eager view.
            target = max(self.framing(),int(self.view_budget*self.settings.compression_trigger))
            self.fit(end,target)
            unit = self.settings.chunk_limit
            total = self.view_cost(end)
            if size(self.render(end))+1 > self.byte_budget:
                unit = "bytes"
                target = max(15,int(self.byte_budget*self.settings.compression_trigger))
                total = size(self.render(end))+1
            if total <= target:
                return self.view_cost(end)
            key = (end,self.view_budget,self.byte_budget,unit,self.settings.compression_trigger)
            if self.plan_key == key and self.db.execute("SELECT 1 FROM jobs LIMIT 1").fetchone():
                return self.view_cost(end)
            self.plan_key = key
            parts = [p for p in self.parts() if p[0]+p[1] <= end]
            costs = {}
            pending = set(tuple(r) for r in self.db.execute("SELECT start,n FROM jobs"))
            limit = self.settings.summary_limit
            def address_cost(part):
                return measure(self.address(*part),unit)+(0 if unit == "sentences" else 1)
            def text_cost(part,text):
                return self.line_cost(self.address(*part)+" ".join(text.splitlines()),unit)
            reservation = limit.target if unit == self.settings.chunk_limit else limit.output_bytes
            def estimate(part):
                return reservation+(0 if unit == "sentences" else address_cost(part))
            for start,n in parts:
                text = self.node(start,n)
                if text is None:
                    text = self.leaf_source(start,raw=True)
                costs[(start,n)] = text_cost((start,n),text)
            # Pay for older leaves before merging. Account for queued work so a
            # notification does not keep scheduling unnecessary newer summaries.
            for part in parts:
                if total <= target:
                    break
                if part[1] != 1 or self.node(*part) is not None:
                    continue
                source = self.summary_source(part[0])
                source_cost = text_cost(part,source)
                projected = source_cost if self.can_copy(source) else estimate(part)
                if projected < costs[part] or part in pending:
                    self.require(*part)
                    total += projected-costs[part]
                    costs[part] = projected
            # Plan binary carries only as far as needed. Actual replacements use
            # committed summaries and publish at the next native turn boundary.
            def projected_merge(start,n):
                parent = (start,2*n)
                child_text = sum(costs[p]-(0 if unit == "sentences" else address_cost(p)) for p in ((start,n),(start+n,n)))
                if unit != "sentences":
                    child_text += 1
                parent_cost = min(reservation,child_text)+(0 if unit == "sentences" else address_cost(parent))
                return costs[(start,n)]+costs[(start+n,n)]-parent_cost,parent_cost
            # Zero-saving carries can expose a larger, compressible sentence range.
            frontier = MergeFrontier(parts,end,lambda start,n:self.valid_span(start,2*n) and projected_merge(start,n)[0] >= 0)
            while total > target:
                pair = frontier.take()
                if pair is None:
                    break
                j,k,start,n = pair
                parent = (start,2*n)
                saving,parent_cost = projected_merge(start,n)
                self.require(*parent)
                total -= saving
                costs[parent] = parent_cost
                frontier.merge(j,k,parent)
            if (self.view_cost(end) > self.view_budget or size(self.render(end))+1 > self.byte_budget) and not self.db.execute("SELECT 1 FROM jobs LIMIT 1").fetchone():
                for start,n in self.parts():
                    if n == 1 and start < end and self.node(start,1) is None and not self.can_copy(self.summary_source(start)):
                        self.require(start,1)
                        break
            return self.view_cost(end)

    def save_node(self, start, n, text, metrics):
        if not self.can_copy(text):
            raise ValueError("Summary exceeds its configured length or framed-view byte allowance")
        with self.lock, self.db:
            inserted = self.db.execute("INSERT OR IGNORE INTO nodes VALUES (?,?,?,?)", (start,n,text,encoded(metrics))).rowcount == 1
            self.db.execute("DELETE FROM jobs WHERE start=? AND n=?",(start,n))
            parent = start-start%(2*n)
            if self.settings.compression_policy == "eager" and self.valid_span(parent,2*n) and self.node(parent,n) is not None and self.node(parent+n,n) is not None and self.node(parent,2*n) is None:
                self.db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?)",(parent,2*n))
            if self.settings.compression_policy == "eager":
                self.fit()
            elif self.history_end is not None:
                self.prepare(self.history_end)
            return inserted

    def record_attempt(self, start, n, metrics):
        with self.lock, self.db:
            self.db.execute("INSERT INTO attempts(start,n,date,metrics) VALUES (?,?,?,?)",
                            (start,n,time.time(),encoded(metrics)))

    def ready_job(self, kind=None, skip=()):
        with self.lock:
            if kind != "parent":
                for (first,) in self.db.execute("SELECT start FROM jobs WHERE n=1 ORDER BY start"):
                    if (first,1) not in skip and self.valid_span(first,1):
                        event = self.event(first) if self.settings.summary_boundary == "event" else None
                        return first, 1, self.leaf_source(first), event["kind"] if event else "turn"
            if kind == "leaf":
                return None
            for start, n, a, b in self.db.execute("""
                SELECT j.start,j.n,a.text,b.text FROM jobs j
                JOIN nodes a ON a.start=j.start AND a.n=j.n/2
                JOIN nodes b ON b.start=j.start+j.n/2 AND b.n=j.n/2
                WHERE j.n>1 AND j.start+j.n<=?
                ORDER BY j.start+j.n,j.n
            """, (self.unit_count(),)):
                if (start,n) not in skip and self.valid_span(start,n):
                    return start, n, a+"\n"+b, None
            return None

    def ready_jobs(self, kind=None, skip=()):
        with self.lock:
            out = []
            if kind != "parent":
                for (start,) in self.db.execute("SELECT start FROM jobs WHERE n=1 ORDER BY start"):
                    if (start,1) not in skip and self.valid_span(start,1):
                        kind_name = self.event(start)["kind"] if self.settings.summary_boundary == "event" else "turn"
                        out.append((start,1,self.leaf_source(start),kind_name))
            if kind != "leaf":
                end = self.unit_count()
                for start,n,a,b in self.db.execute("""SELECT j.start,j.n,a.text,b.text FROM jobs j
                    JOIN nodes a ON a.start=j.start AND a.n=j.n/2
                    JOIN nodes b ON b.start=j.start+j.n/2 AND b.n=j.n/2
                    WHERE j.n>1 AND j.start+j.n<=? ORDER BY j.start+j.n,j.n""",(end,)):
                    if (start,n) not in skip and self.valid_span(start,n):
                        out.append((start,n,a+"\n"+b,None))
            return out

    def parents(self, i):
        n = 2
        while (i+1) % n == 0:
            start = i+1-n
            with self.lock:
                if self.node(start,n) is None:
                    a, b = self.node(start,n//2), self.node(start+n//2,n//2)
                    if a is None or b is None:
                        return
                    yield start, n, a + "\n" + b
            n *= 2

    def audit_node(self, start, n):
        """Preserved historical text for zoom only, never an active-view node."""
        with self.lock:
            if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='audit_nodes'").fetchone():
                return None
            row = self.db.execute("SELECT text,source_archive FROM audit_nodes WHERE start=? AND n=?",
                                  (start,n)).fetchone()
            return dict(row) if row is not None else None

    def _zoom_summary(self, start, n):
        text = self.node(start,n)
        result = {"id":start,"n":n,"text":text}
        if text is None:
            audit = self.audit_node(start,n)
            if audit is None:
                raise ValueError("Children not summarized yet")
            result.update(text=audit["text"],historical=True,source_archive=audit["source_archive"])
        return result

    def zoom(self, i, n, offset=0, limit=30000):
        with self.lock:
            if self.settings.summary_boundary == "user_turn" and n != 1:
                raise ValueError("Original events use n=1; open summary ranges with unit=leaf")
            if type(i) is not int or type(n) is not int or i < 0 or n < 1 or n & (n-1) or i % n or i+n > self.total:
                raise ValueError("Require aligned id+n, n a power of two, within the log")
            if n == 1:
                event = self.event(i)
                text = event["text"]
                if type(offset) is not int or offset < 0 or offset > len(text):
                    raise ValueError("Invalid character offset")
                stop = min(len(text), offset+limit)
                return {"id": i, "kind": event["kind"], "text": text[offset:stop],
                        "offset": offset, "next_offset": stop if stop < len(text) else None,
                        "total_characters": len(text)}
            children = []
            for start in (i,i+n//2):
                children.append(self._zoom_summary(start,n//2))
            return {"children":children}

    def zoom_leaf(self, i, n, offset=0, limit=128):
        with self.lock:
            if type(i) is not int or type(n) is not int or i < 0 or n < 1 or n & (n-1) or i % n or i+n > self.unit_count():
                raise ValueError("Require an aligned leaf id+n within the summary tree")
            if self.settings.summary_boundary == "event":
                return self.zoom(i,n,offset)
            if n > 1:
                children = []
                for start in (i,i+n//2):
                    child = self._zoom_summary(start,n//2)
                    children.append(dict(child,unit="leaf"))
                return {"children":children}
            leaf = self.db.execute("SELECT * FROM leaves WHERE id=?",(i,)).fetchone()
            count = leaf["event_stop"]-leaf["event_start"]
            if type(offset) is not int or offset < 0 or offset > count:
                raise ValueError("Invalid event-reference offset")
            events = []
            for event in self.db.execute("SELECT id,kind,length(text) AS length FROM events WHERE id>=? AND id<? ORDER BY id LIMIT ?",(leaf["event_start"]+offset,leaf["event_stop"],limit)):
                events.append({"id":event["id"],"kind":event["kind"],"unit":"event",
                               "offset":leaf["start_offset"] if event["id"] == leaf["event_start"] else 0,
                               "end_offset":leaf["stop_offset"] if event["id"] == leaf["event_stop"]-1 else event["length"]})
            stop = offset+len(events)
            result = {"id":i,"n":1,"unit":"leaf","text":self.node(i,1),"events":events,
                      "offset":offset,"next_offset":stop if stop < count else None,"total_events":count}
            if result["text"] is None:
                audit = self.audit_node(i,1)
                if audit is not None:
                    result.update(text=audit["text"],historical=True,source_archive=audit["source_archive"])
            return result

    def search(self, query, after=-1, limit=20):
        # Literal substring search remains exact and works without optional SQLite FTS.
        with self.lock:
            return [dict(r) for r in self.db.execute(
                "SELECT e.id,e.kind,substr(e.text,max(1,instr(lower(e.text),lower(?))-80),240) AS excerpt "
                "FROM events e JOIN active_events a ON a.event=e.id "
                "WHERE e.id>? AND instr(lower(e.text),lower(?))>0 ORDER BY e.id LIMIT ?", (query,after,query,limit))]

    def sync_hermes(self, home, session, before_row=None):
        from .identity import sync
        return sync(self, home, session, before_row)
