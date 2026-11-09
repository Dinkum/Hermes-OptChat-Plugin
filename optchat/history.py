"""Explicit audit retrieval from a preserved pre-recovery generation."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading

from .config import Settings
from .store import Store, encoded


class PreviousArchive(Store):
    """Read-only adapter reusing the existing exact zoom/pagination implementation."""
    def __init__(self, current):
        with current.lock:
            row = current.db.execute("SELECT value FROM meta WHERE key='recovery_report'").fetchone()
        if row is None:
            raise ValueError("No previous archive is recorded for this session")
        path = Path(json.loads(row[0])["source_archive"]).resolve()
        self.db = sqlite3.connect(path.as_uri()+"?mode=ro",uri=True)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA query_only=ON")
        self.lock = threading.RLock()
        boundary = self.db.execute("SELECT value FROM meta WHERE key='summary_boundary'").fetchone()
        self.settings = Settings(summary_boundary=boundary[0] if boundary else "event")
        self.total = self.db.execute("SELECT coalesce(max(id)+1,0) FROM events").fetchone()[0]
        self.leaf_total = (self.db.execute("SELECT coalesce(max(id)+1,0) FROM leaves").fetchone()[0]
                           if self.settings.summary_boundary == "user_turn" else 0)

    def close(self):
        self.db.close()

    def search(self, query, after=-1, limit=20):
        # Audit search deliberately includes source-only/withdrawn originals.
        return [dict(row) for row in self.db.execute(
            "SELECT id,kind,substr(text,max(1,instr(lower(text),lower(?))-80),240) AS excerpt FROM events "
            "WHERE id>? AND instr(lower(text),lower(?))>0 ORDER BY id LIMIT ?",
            (query,after,query,limit))]


def previous_call(current, name, args):
    try:
        with closing(PreviousArchive(current)) as archive:
            blocks = []
            if name == "optchat_zoom":
                unit = args.get("unit","event")
                if unit not in ("event","leaf"):
                    raise ValueError("unit must be event or leaf")
                result = (archive.zoom_leaf(args["id"],args["n"],args.get("offset",0)) if unit == "leaf"
                          else archive.zoom(args["id"],args["n"],args.get("offset",0)))
                original = args["n"] == 1 and (unit == "event" or archive.settings.summary_boundary == "event")
                if original and (args.get("offset",0) == 0 or "image_offset" in args):
                    from .images import page
                    blocks, metadata = page(archive,args["id"],args.get("image_offset",0),read_only=True)
                    if metadata["total_images"]:
                        result.update(metadata)
            elif name == "optchat_search":
                query = args["query"]
                if not isinstance(query,str) or not query:
                    raise ValueError("A nonempty literal query is required")
                result = {"matches":archive.search(query,args.get("after",-1))}
            elif name == "optchat_date":
                result = {"id":args["id"],"date":datetime.fromtimestamp(
                    archive.event(args["id"])["date"],timezone.utc).isoformat()}
            else:
                raise ValueError("Unknown OptChat tool")
            result = dict(result,archive="previous")
            if blocks:
                text = encoded(result)
                return {"_multimodal":True,"content":[{"type":"text","text":text}]+blocks,"text_summary":text}
            return result
    except sqlite3.Error as exc:
        raise RuntimeError(f"Previous archive is unavailable: {exc}") from exc
