"""Append-only originals and nodes; transactional, monotonically coarsening view."""
import fcntl
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


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
        try:
            self.db = sqlite3.connect(self.root / "chat.db", check_same_thread=False)
            self.db.row_factory = sqlite3.Row
            self.initialize(session,budget)
        except BaseException:
            self.close()
            raise

    def initialize(self, session, budget):
        # Legacy nonempty archives use event leaves. Check before changing their schema.
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='meta'").fetchone():
            row = self.db.execute("SELECT value FROM meta WHERE key='summary_boundary'").fetchone()
            boundary = row[0] if row else "event"
            populated = self.db.execute("SELECT 1 FROM events LIMIT 1").fetchone()
            if populated and boundary != self.settings.summary_boundary:
                raise RuntimeError(f"This archive uses summary_boundary={boundary}; start a new Hermes session for {self.settings.summary_boundary}")
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
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, start INTEGER, n INTEGER, date REAL, metrics TEXT);
            CREATE TABLE IF NOT EXISTS raw_activity (source TEXT PRIMARY KEY, payload TEXT NOT NULL);
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

    def configure(self, settings):
        with self.lock, self.db:
            if self.configured and settings == self.settings:
                return
            boundary = self.db.execute("SELECT value FROM meta WHERE key='summary_boundary'").fetchone()[0]
            if boundary != settings.summary_boundary:
                raise RuntimeError(f"This archive uses summary_boundary={boundary}; start a new Hermes session for {settings.summary_boundary}")
            row = self.db.execute("SELECT value FROM meta WHERE key='compression_policy'").fetchone()
            policy = row[0] if row else "eager"
            self.settings = settings
            if policy != settings.compression_policy:
                # Jobs are derived state. Retain immutable nodes, discard speculative work.
                self.db.execute("DELETE FROM jobs")
            initialized = self.db.execute("SELECT 1 FROM meta WHERE key='jobs_initialized'").fetchone()
            if settings.compression_policy == "eager" and (not initialized or policy != settings.compression_policy):
                table = "events" if boundary == "event" else "leaves"
                self.db.execute(f"INSERT OR IGNORE INTO jobs SELECT e.id,1 FROM {table} e LEFT JOIN nodes n ON n.start=e.id AND n.n=1 WHERE n.start IS NULL")
                self.db.execute("""INSERT OR IGNORE INTO jobs SELECT a.start,a.n*2 FROM nodes a
                    JOIN nodes b ON b.start=a.start+a.n AND b.n=a.n
                    LEFT JOIN nodes p ON p.start=a.start AND p.n=a.n*2
                    WHERE a.start % (a.n*2)=0 AND p.start IS NULL""")
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('compression_policy',?)",(settings.compression_policy,))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('batching',?)",(settings.batching,))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('chunk_limit',?)",(settings.chunk_limit,))
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('jobs_initialized','1')")
            self.plan_key = None
            self.configured = True

    def close(self):
        with self.lock:
            if not self.closed:
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
            if self.db.execute("SELECT 1 FROM leaves WHERE event_start<? AND event_stop>?",(event_end,event_end)).fetchone():
                raise RuntimeError("A sealed turn crosses the current input; archive boundary is inconsistent")
            return self.db.execute("SELECT coalesce(max(id)+1,0) FROM leaves WHERE event_stop<=?",(event_end,)).fetchone()[0]

    def seal(self, end):
        """Seal prior turns, including a recovered interrupted turn, never the live suffix."""
        if self.settings.summary_boundary == "event":
            return
        from .chunks import turn_chunks
        with self.lock:
            with self.db:
                start = self.db.execute("SELECT coalesce(max(event_stop),0) FROM leaves").fetchone()[0]
                if end <= start:
                    return
                active = self.db.execute("SELECT value FROM meta WHERE key='active_start'").fetchone()
                active = int(active[0]) if active else end
                events = [dict(r) for r in self.db.execute("SELECT * FROM events WHERE id>=? AND id<? ORDER BY id",(start,end))]
                groups, group = [], []
                for event in events:
                    # Imported history has no native turn journal. User rows are its best
                    # available boundaries; admitted live turns also include steering rows.
                    if group and (event["id"] == active or event["kind"] == "user" and event["id"] < active):
                        groups.append(group)
                        group = []
                    group.append(event)
                if group:
                    groups.append(group)
                leaf_id = self.leaf_total
                for group in groups:
                    for span in turn_chunks(group,self.settings.turn_source_bytes,self.settings.tool_chars):
                        self.db.execute("INSERT INTO leaves VALUES (?,?,?,?,?,?)",(leaf_id,*span))
                        self.db.execute("INSERT INTO view VALUES (?,1)",(leaf_id,))
                        if self.settings.compression_policy == "eager":
                            self.db.execute("INSERT INTO jobs VALUES (?,1)",(leaf_id,))
                        leaf_id += 1
            self.leaf_total = leaf_id

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

    def append(self, source, kind, text, payload=None, date=None):
        with self.lock:
            with self.db:
                old = self.db.execute("SELECT id FROM events WHERE source=?", (source,)).fetchone()
                if old:
                    return old[0]
                i = self.total
                self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",
                                (i, source, kind, text, encoded(payload), time.time() if date is None else date))
                if self.settings.summary_boundary == "event":
                    self.db.execute("INSERT INTO view VALUES (?,1)", (i,))
                    if self.settings.compression_policy == "eager":
                        self.db.execute("INSERT INTO jobs VALUES (?,1)",(i,))
            self.total = i+1
            return i

    def event(self, i):
        with self.lock:
            row = self.db.execute("SELECT * FROM events WHERE id=?", (i,)).fetchone()
            if row is None:
                raise ValueError("No such message")
            return dict(row)

    def record_raw(self, source, payload):
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO raw_activity VALUES (?,?)",(source,encoded(payload)))

    def node(self, start, n):
        row = self.db.execute("SELECT text FROM nodes WHERE start=? AND n=?", (start, n)).fetchone()
        return row[0] if row else None

    def parts(self):
        return [tuple(r) for r in self.db.execute("SELECT start,n FROM view ORDER BY start")]

    def lines(self, end=None, ids=True, raw=None):
        with self.lock:
            raw = self.settings.compression_policy == "on_demand" if raw is None else raw
            out = []
            # Bulk-read complete view parts. Only a part crossing the requested
            # prefix boundary needs point lookups while descending its children.
            rows = self.db.execute("""SELECT v.start,v.n,c.text FROM view v
                LEFT JOIN nodes c ON c.start=v.start AND c.n=v.n
                WHERE ? IS NULL OR v.start<? ORDER BY v.start""",(end,end)).fetchall()
            def add(start, n, text=None, fetched=False):
                if end is not None and start >= end:
                    return
                if end is not None and start+n > end:
                    if n > 1:
                        add(start,n//2)
                        add(start+n//2,n//2)
                    return
                if not fetched:
                    text = self.node(start, n)
                if text is None and n == 1 and raw:
                    text = self.leaf_source(start,raw=True)
                if text is None:
                    raise RuntimeError("Summary not ready")
                out.append((self.address(start,n) if ids else "") + " ".join(text.splitlines()))
            for start, n, text in rows:
                add(start,n,text,fetched=True)
            return out

    def render(self, end=None):
        return "<chat>\n" + "\n".join(self.lines(end)) + "\n</chat>"

    def fit(self, end=None, limit=None):
        # Count rendered UTF-8 bytes, including addresses and framing, rather than targets.
        rows = self.db.execute("""SELECT v.start,v.n,c.text,p.text FROM view v
            LEFT JOIN nodes c ON c.start=v.start AND c.n=v.n
            LEFT JOIN nodes p ON p.start=v.start AND p.n=v.n*2
            WHERE ? IS NULL OR v.start+v.n<=? ORDER BY v.start""",(end,end)).fetchall()
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
            return size(self.address(start,n) + (" ".join(text.splitlines()) if text is not None else "(waiting)")) + 1
        total = 15 + sum(map(cost, parts))
        T = self.unit_count() if end is None else end
        limit = self.budget if limit is None else limit
        while total > limit:
            best = None
            for j, (start, n) in enumerate(parts[:-1]):
                if start % (2*n) or parts[j+1] != (start+n, n):
                    continue
                if node(start, 2*n) is None:
                    continue
                due = (T-start) / (4*n)
                if best is None or due > best[0]:
                    best = (due, j, start, n)
            if best is None:
                break
            _, j, start, n = best
            total += cost((start,2*n)) - cost(parts[j]) - cost(parts[j+1])
            self.db.execute("DELETE FROM view WHERE start IN (?,?)", (start,start+n))
            self.db.execute("INSERT INTO view VALUES (?,?)", (start,2*n))
            parts[j:j+2] = [(start,2*n)]
        return total

    def require(self, start, n):
        """Queue only a requested node's missing dependency subtree; caller holds lock."""
        if self.node(start,n) is not None:
            return
        if self.db.execute("SELECT 1 FROM jobs WHERE start=? AND n=?",(start,n)).fetchone():
            return
        if n > 1:
            self.require(start,n//2)
            self.require(start+n//2,n//2)
        self.db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?)",(start,n))

    def prepare(self, end):
        """Predict the smallest older prefix of work needed for compression headroom."""
        with self.lock, self.db:
            self.history_end = end
            if self.settings.compression_policy == "eager":
                return self.fit()
            target = max(15,int(self.budget*self.settings.compression_trigger))
            self.fit(end,target)
            total = size(self.render(end))+1
            if total <= target:
                return total
            key = (end,self.budget,self.settings.compression_trigger)
            if self.plan_key == key and self.db.execute("SELECT 1 FROM jobs LIMIT 1").fetchone():
                return total
            self.plan_key = key
            parts = [p for p in self.parts() if p[0]+p[1] <= end]
            costs = {}
            pending = set(tuple(r) for r in self.db.execute("SELECT start,n FROM jobs"))
            limit = self.settings.summary_limit
            def estimate(part):
                text_bytes = limit.target if limit.unit == "bytes" else limit.output_bytes
                return size(self.address(*part))+text_bytes+1
            for start,n in parts:
                text = self.node(start,n)
                if text is None:
                    text = self.leaf_source(start,raw=True)
                costs[(start,n)] = size(self.address(start,n)+" ".join(text.splitlines()))+1
            # Pay for older leaves before merging. Account for queued work so a
            # notification does not keep scheduling unnecessary newer summaries.
            for part in parts:
                if total <= target:
                    break
                if part[1] != 1 or self.node(*part) is not None:
                    continue
                source = self.summary_source(part[0])
                source_cost = size(self.address(*part)+" ".join(source.splitlines()))+1
                projected = source_cost if limit.can_copy(source,self.settings.summary_bytes) else estimate(part)
                if projected < costs[part] or part in pending:
                    self.require(*part)
                    total += projected-costs[part]
                    costs[part] = projected
            # Plan binary carries only as far as needed. Actual replacements use
            # committed summaries and publish at the next native turn boundary.
            while total > target:
                best = None
                for j,(start,n) in enumerate(parts[:-1]):
                    if start % (2*n) or parts[j+1] != (start+n,n):
                        continue
                    parent = (start,2*n)
                    child_text = sum(costs[p]-size(self.address(*p))-1 for p in ((start,n),(start+n,n)))+1
                    text_bytes = limit.target if limit.unit == "bytes" else limit.output_bytes
                    parent_cost = size(self.address(*parent))+min(text_bytes,child_text)+1
                    saving = costs[(start,n)]+costs[(start+n,n)]-parent_cost
                    if saving <= 0:
                        continue
                    score = (end-start)/(4*n)
                    if best is None or score > best[0]:
                        best = score,j,parent,saving,parent_cost
                if best is None:
                    break
                _,j,parent,saving,parent_cost = best
                self.require(*parent)
                total -= saving
                costs[parent] = parent_cost
                parts[j:j+2] = [parent]
            # A length-unit reservation can exceed a small history budget even
            # though the model could return a shorter valid summary. When the
            # raw view cannot fit, try an unbuilt paid leaf before declaring the
            # binary forest impossible. Actual committed bytes remain decisive.
            if size(self.render(end))+1 > self.budget and not self.db.execute("SELECT 1 FROM jobs LIMIT 1").fetchone():
                for start,n in self.parts():
                    if n == 1 and start < end and self.node(start,1) is None and not limit.can_copy(self.summary_source(start),self.settings.summary_bytes):
                        self.require(start,1)
                        break
            return size(self.render(end))+1

    def save_node(self, start, n, text, metrics):
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO nodes VALUES (?,?,?,?)", (start,n,text,encoded(metrics)))
            self.db.execute("DELETE FROM jobs WHERE start=? AND n=?",(start,n))
            parent = start-start%(2*n)
            if self.settings.compression_policy == "eager" and self.node(parent,n) is not None and self.node(parent+n,n) is not None and self.node(parent,2*n) is None:
                self.db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?)",(parent,2*n))
            if self.settings.compression_policy == "eager":
                self.fit()
            elif self.history_end is not None:
                self.prepare(self.history_end)

    def record_attempt(self, start, n, metrics):
        with self.lock, self.db:
            self.db.execute("INSERT INTO attempts(start,n,date,metrics) VALUES (?,?,?,?)",
                            (start,n,time.time(),encoded(metrics)))

    def ready_job(self, kind=None, skip=()):
        # Serial leaves have fully summarized prior context. Binary carries build parents
        # before the next leaf; no speculative parents and no missing context. Parents in
        # skip are already claimed by another merge lane.
        with self.lock:
            pending = self.db.execute("SELECT start FROM jobs WHERE n=1 ORDER BY start LIMIT 1").fetchone()
            first = pending[0] if pending else self.unit_count()
            if kind != "parent" and pending and (first,1) not in skip:
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
            """, (self.unit_count() if self.settings.compression_policy == "on_demand" or self.settings.summary_boundary == "user_turn" else first,)):
                if (start, n) not in skip:
                    return start, n, a+"\n"+b, None
            return None

    def ready_jobs(self, kind=None, skip=()):
        with self.lock:
            out = []
            if kind != "parent":
                if self.settings.summary_boundary == "event":
                    job = self.ready_job("leaf",skip)
                    if job:
                        out.append(job)
                else:
                    for row in self.db.execute("SELECT start FROM jobs WHERE n=1 ORDER BY start"):
                        if (row[0],1) not in skip:
                            out.append((row[0],1,self.leaf_source(row[0]),"turn"))
            if kind != "leaf":
                pending = self.db.execute("SELECT start FROM jobs WHERE n=1 ORDER BY start LIMIT 1").fetchone()
                end = pending[0] if pending and self.settings.summary_boundary == "event" and self.settings.compression_policy == "eager" else self.unit_count()
                for start,n,a,b in self.db.execute("""SELECT j.start,j.n,a.text,b.text FROM jobs j
                    JOIN nodes a ON a.start=j.start AND a.n=j.n/2
                    JOIN nodes b ON b.start=j.start+j.n/2 AND b.n=j.n/2
                    WHERE j.n>1 AND j.start+j.n<=? ORDER BY j.start+j.n,j.n""",(end,)):
                    if (start,n) not in skip:
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
                text = self.node(start,n//2)
                if text is None:
                    raise ValueError("Children not summarized yet")
                children.append({"id":start,"n":n//2,"text":text})
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
                    text = self.node(start,n//2)
                    if text is None:
                        raise ValueError("Children not summarized yet")
                    children.append({"id":start,"n":n//2,"unit":"leaf","text":text})
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
            return {"id":i,"n":1,"unit":"leaf","text":self.node(i,1),"events":events,
                    "offset":offset,"next_offset":stop if stop < count else None,"total_events":count}

    def search(self, query, after=-1, limit=20):
        # Literal substring search remains exact and works without optional SQLite FTS.
        with self.lock:
            return [dict(r) for r in self.db.execute(
                "SELECT id,kind,substr(text,max(1,instr(lower(text),lower(?))-80),240) AS excerpt FROM events "
                "WHERE id>? AND instr(lower(text),lower(?))>0 ORDER BY id LIMIT ?", (query,after,query,limit))]

    def sync_hermes(self, home, session, before_row=None):
        dbpath = Path(home) / "state.db"
        if not dbpath.exists():
            return
        with self.lock:
            row = self.db.execute("SELECT value FROM meta WHERE key='cursor'").fetchone()
            cursor = int(row[0]) if row else 0
        from hermes_state import SessionDB
        conn = sqlite3.connect(dbpath.as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM messages WHERE session_id=? AND id>? ORDER BY id", (session,cursor))
            for r in rows:
                r = dict(r)
                if before_row is not None and r["id"] >= before_row:
                    break
                role = r["role"]
                content = SessionDB._decode_content(r["content"])
                # Only visible conversation: native encrypted reasoning remains Hermes-owned.
                if role in ("user","assistant") and visible(content):
                    with self.lock:
                        raw = self.db.execute("SELECT payload FROM raw_activity WHERE source=?",(f"row:{r['id']}",)).fetchone()
                    self.append(f"row:{r['id']}", "user" if role == "user" else "talk",
                                visible(content), json.loads(raw[0]) if raw else content, r["timestamp"])
                calls = json.loads(r["tool_calls"]) if r.get("tool_calls") else []
                for call in calls:
                    self.append("call:"+call["id"], "tool", encoded(call["function"]), call["function"],r["timestamp"])
                if role == "tool":
                    self.append("result:"+r["tool_call_id"], "work" if r.get("tool_name")=="delegate_task" else "echo", visible(content), content,r["timestamp"])
                with self.lock, self.db:
                    self.db.execute("INSERT OR REPLACE INTO meta VALUES ('cursor',?)", (str(r["id"]),))
        finally:
            conn.close()
