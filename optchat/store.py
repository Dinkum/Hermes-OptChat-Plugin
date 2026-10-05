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
    def __init__(self, root, session, budget):
        self.root = Path(root) / hashlib.sha256(session.encode()).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lockfile = open(self.root / "writer.lock", "a+b")
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lockfile.close()
            raise RuntimeError("OptChat archive already has a writer; resume it in the owning Hermes process")
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "chat.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
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
        """)
        self.budget = budget
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('session', ?)", (session,))
            if not self.db.execute("SELECT 1 FROM meta WHERE key='jobs_initialized'").fetchone():
                self.db.execute("INSERT OR IGNORE INTO jobs SELECT e.id,1 FROM events e LEFT JOIN nodes n ON n.start=e.id AND n.n=1 WHERE n.start IS NULL")
                self.db.execute("""INSERT OR IGNORE INTO jobs SELECT a.start,a.n*2 FROM nodes a
                    JOIN nodes b ON b.start=a.start+a.n AND b.n=a.n
                    LEFT JOIN nodes p ON p.start=a.start AND p.n=a.n*2
                    WHERE a.start % (a.n*2)=0 AND p.start IS NULL""")
                self.db.execute("INSERT INTO meta VALUES ('jobs_initialized','1')")
        # One writer holds the lock, so the count is kept in memory instead of scanning per append.
        self.total = self.db.execute("SELECT coalesce(max(id)+1,0) FROM events").fetchone()[0]
        self.closed = False

    def close(self):
        with self.lock:
            if not self.closed:
                self.db.close()
                self.lockfile.close()
                self.closed = True

    def count(self):
        return self.total

    def append(self, source, kind, text, payload=None, date=None):
        with self.lock:
            with self.db:
                old = self.db.execute("SELECT id FROM events WHERE source=?", (source,)).fetchone()
                if old:
                    return old[0]
                i = self.total
                self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",
                                (i, source, kind, text, encoded(payload), time.time() if date is None else date))
                self.db.execute("INSERT INTO view VALUES (?,1)", (i,))
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

    def lines(self, end=None, ids=True):
        with self.lock:
            out = []
            def add(start, n):
                if end is not None and start >= end:
                    return
                if end is not None and start+n > end:
                    if n > 1:
                        add(start,n//2)
                        add(start+n//2,n//2)
                    return
                text = self.node(start, n)
                if text is None:
                    raise RuntimeError("Summary not ready")
                out.append((f"{start}+{n}|" if ids else "") + " ".join(text.splitlines()))
            for start, n in self.parts():
                add(start,n)
            return out

    def render(self, end=None):
        return "<chat>\n" + "\n".join(self.lines(end)) + "\n</chat>"

    def fit(self):
        # Count rendered UTF-8 bytes, including addresses and framing, rather than targets.
        parts = self.parts()
        def cost(part):
            start, n = part
            text = self.node(start, n)
            return size(f"{start}+{n}|" + (" ".join(text.splitlines()) if text is not None else "(waiting)")) + 1
        total = 15 + sum(map(cost, parts))
        T = self.count()
        while total > self.budget:
            best = None
            for j, (start, n) in enumerate(parts[:-1]):
                if start % (2*n) or parts[j+1] != (start+n, n):
                    continue
                if self.node(start, 2*n) is None:
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

    def save_node(self, start, n, text, metrics):
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO nodes VALUES (?,?,?,?)", (start,n,text,encoded(metrics)))
            self.db.execute("DELETE FROM jobs WHERE start=? AND n=?",(start,n))
            parent = start-start%(2*n)
            if self.node(parent,n) is not None and self.node(parent+n,n) is not None and self.node(parent,2*n) is None:
                self.db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?)",(parent,2*n))
            self.fit()

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
            first = pending[0] if pending else self.total
            if kind != "parent" and first < self.total:
                event = self.event(first)
                return first, 1, event["kind"] + ": " + event["text"], event["kind"]
            if kind == "leaf":
                return None
            for start, n, a, b in self.db.execute("""
                SELECT j.start,j.n,a.text,b.text FROM jobs j
                JOIN nodes a ON a.start=j.start AND a.n=j.n/2
                JOIN nodes b ON b.start=j.start+j.n/2 AND b.n=j.n/2
                WHERE j.n>1 AND j.start+j.n<=?
                ORDER BY j.start+j.n,j.n
            """, (first,)):
                if (start, n) not in skip:
                    return start, n, a+"\n"+b, None
            return None

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
