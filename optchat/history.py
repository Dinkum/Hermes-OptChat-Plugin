"""Explicit audit retrieval through preserved migration/recovery generations."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading

from .config import Settings
from .store import Store, encoded


class PreviousArchive(Store):
    """Read-only adapter retaining Store's exact zoom and preserved-child fallback."""
    def __init__(self, current, generation=1):
        if type(generation) is not int or generation < 1:
            raise ValueError("generation must be a positive integer")
        with current.lock:
            reports = dict(current.db.execute("""SELECT key,value FROM meta
                WHERE key IN ('limit_migration_report','recovery_report')"""))
        visited = {Path(current.root / "chat.db").resolve()}
        for depth in range(generation):
            report = reports.get("limit_migration_report") or reports.get("recovery_report")
            if report is None:
                raise ValueError("No previous archive is recorded at this generation")
            path = Path(json.loads(report)["source_archive"]).resolve()
            if path in visited:
                raise ValueError("Previous archive chain contains a cycle")
            visited.add(path)
            connection = sqlite3.connect(path.as_uri()+"?mode=ro",uri=True)
            if depth == generation-1:
                self.db = connection
                break
            try:
                reports = dict(connection.execute("""SELECT key,value FROM meta
                    WHERE key IN ('limit_migration_report','recovery_report')"""))
            finally:
                connection.close()
        self.root = path.parent
        self.source_archive = str(path)
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
        with closing(PreviousArchive(current,args.get("generation",1))) as archive:
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
            result = dict(result,archive="previous",source_archive=archive.source_archive)
            if "generation" in args:
                result["generation"] = args["generation"]
            if blocks:
                text = encoded(result)
                return {"_multimodal":True,"content":[{"type":"text","text":text}]+blocks,"text_summary":text}
            return result
    except sqlite3.Error as exc:
        raise RuntimeError(f"Previous archive is unavailable: {exc}") from exc
