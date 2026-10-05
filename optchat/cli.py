"""Profile setup, status and portable archive browsing through native Hermes CLI registration."""
import html
import json
import sqlite3
from pathlib import Path

from .store import leaf_text


def setup(parser):
    parser.add_argument("action",choices=["setup","status","export"])
    parser.add_argument("--session",help="Hermes session ID (status lists all if omitted)")
    parser.add_argument("--output",type=Path,help="HTML output file for export")


def handle(args):
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    if args.action == "setup":
        from .configure import prepare_configuration, write_configuration
        config, cfg = prepare_configuration(home)
        write_configuration(config,cfg)
        print(f"Configured OptChat in {home}. Configuration backup: {home/'config.before-optchat.yaml'}.")
        return
    root = home/"optchat"
    results = []
    for path in sorted(root.glob("*/chat.db")):
        conn = sqlite3.connect(path.as_uri()+"?mode=ro",uri=True)
        conn.row_factory = sqlite3.Row
        try:
            session = conn.execute("SELECT value FROM meta WHERE key='session'").fetchone()[0]
            if args.session and args.session != session:
                continue
            events = conn.execute("SELECT count(*) FROM events").fetchone()[0]
            nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
            meta = dict(conn.execute("SELECT key,value FROM meta"))
            boundary,policy = meta.get("summary_boundary","event"),meta.get("compression_policy","eager")
            parts = conn.execute("SELECT v.start,v.n,n.text FROM view v LEFT JOIN nodes n ON n.start=v.start AND n.n=v.n ORDER BY v.start").fetchall()
            attempts = [json.loads(r[0]) for r in conn.execute("SELECT metrics FROM attempts")]
            results.append({"session":session,"events":events,"nodes":nodes,"view_parts":len(parts),
                            "summary_boundary":boundary,"compression_policy":policy,"batching":meta.get("batching","off"),
                            "chunk_limit":meta.get("chunk_limit","bytes"),
                            "raw_view_parts":sum(p["text"] is None and p["n"] == 1 and policy == "on_demand" for p in parts),
                            "pending_view_parts":sum(p["text"] is None and (policy != "on_demand" or p["n"] > 1) for p in parts),
                            "summary_calls":sum(a.get("usage_accounted",True) for a in attempts),
                            "summary_seconds":sum(a.get("seconds",0) for a in attempts),
                            "summary_cost_usd":sum(a["usage"]["cost_usd"] for a in attempts) if all(a.get("usage",{}).get("cost_usd") is not None for a in attempts) else None,
                            "archive":str(path)})
            if args.action == "export":
                if not args.session or not args.output:
                    raise ValueError("Export requires --session and --output")
                def escaped(text):
                    return html.escape(str(text))
                with args.output.open("w",encoding="utf-8") as out:
                    out.write('<!doctype html><meta charset="utf-8"><title>OptChat archive</title><style>body{max-width:1000px;margin:2em auto;padding:1em;font:16px system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:1em 0}</style>')
                    out.write("<h1>"+escaped(session)+"</h1><h2>Overview</h2>")
                    for p in parts:
                        text = p["text"]
                        if text is None and policy == "on_demand" and p["n"] == 1:
                            text = leaf_text(conn,boundary,p["start"],raw=True)
                        anchor = "leaf" if boundary == "user_turn" else "m"
                        out.write(f'<p><a href="#{anchor}{p["start"]}">{p["start"]}+{p["n"]}</a> '+escaped(text if text is not None else "summary pending")+"</p>")
                    out.write("<h2>Originals</h2>")
                    for r in conn.execute("SELECT * FROM events ORDER BY id"):
                        out.write(f'<details id="m{r["id"]}"><summary>{r["id"]} · '+escaped(r["kind"])+" · "+escaped(r["date"])+"</summary><pre>"+escaped(r["text"])+"</pre><details><summary>Original payload</summary><pre>"+escaped(r["payload"])+"</pre></details></details>")
                    if boundary == "user_turn":
                        out.write("<h2>Turn chunks</h2>")
                        for r in conn.execute("SELECT * FROM leaves ORDER BY id"):
                            out.write(f'<details id="leaf{r["id"]}"><summary>Leaf {r["id"]}</summary><pre>'+escaped(r["source"])+"</pre><p>Originals: ")
                            for event_id in range(r["event_start"],r["event_stop"]):
                                out.write(f'<a href="#m{event_id}">{event_id}</a> ')
                            out.write(f'</p><p>First offset: {r["start_offset"]}; last offset: {r["stop_offset"]}.</p></details>')
                    out.write("<h2>Summary tree</h2>")
                    for r in conn.execute("SELECT start,n,text FROM nodes ORDER BY n DESC,start"):
                        out.write(f'<details><summary>{r["start"]}+{r["n"]}</summary><pre>'+escaped(r["text"])+"</pre></details>")
                    out.write("<h2>Admission journal</h2>")
                    for r in conn.execute("SELECT source,payload FROM raw_activity ORDER BY rowid"):
                        out.write("<details><summary>"+escaped(r["source"])+"</summary><pre>"+escaped(r["payload"])+"</pre></details>")
        finally:
            conn.close()
    if args.session and not results:
        raise ValueError("No OptChat archive for that session")
    print(json.dumps(results,indent=2))
