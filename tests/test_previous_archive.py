"""Recovered agents can explicitly retrieve source-only originals without reviving them."""
import hashlib
import json
import sqlite3
from types import SimpleNamespace

from optchat.config import Settings
from optchat.engine import OptChatEngine
from optchat.store import Store


def test_previous_generation_is_explicit_exact_paginated_and_read_only(tmp_path):
    root = tmp_path
    previous = root / "previous.db"
    with sqlite3.connect(previous) as db:
        db.executescript("CREATE TABLE events(id INTEGER PRIMARY KEY,kind TEXT,text TEXT,payload TEXT,date REAL);"
                         "CREATE TABLE nodes(start INTEGER,n INTEGER,text TEXT);"
                         "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);")
        db.execute("INSERT INTO meta VALUES ('summary_boundary','event')")
        db.execute("INSERT INTO events VALUES (0,'user',?,'null',1)", ("old-original:" + "x"*40000,))
    before = hashlib.sha256(previous.read_bytes()).hexdigest()
    store = Store(root / "current", "recovered", 128000, Settings())
    try:
        store.append("fixture-current", "user", "Current instruction", "Current instruction")
        with store.db:
            store.db.execute("INSERT INTO meta VALUES ('recovery_report',?)", (json.dumps({"source_archive":str(previous)}),))
        engine = OptChatEngine(settings=Settings())
        engine.worker = SimpleNamespace(store=store)
        current = json.loads(engine.handle_tool_call("optchat_zoom", {"id":0,"n":1}))
        assert current["text"] == "Current instruction"
        old = json.loads(engine.handle_tool_call("optchat_zoom", {"id":0,"n":1,"archive":"previous"}))
        assert old["text"] == ("old-original:" + "x"*40000)[:30000]
        assert old["next_offset"] == 30000
        assert old["archive"] == "previous"
        tail = json.loads(engine.handle_tool_call("optchat_zoom", {"id":0,"n":1,"offset":30000,"archive":"previous"}))
        assert tail["text"] == ("old-original:" + "x"*40000)[30000:]
        assert tail["next_offset"] is None
        assert hashlib.sha256(previous.read_bytes()).hexdigest() == before
    finally:
        store.close()

def test_previous_search_never_reintroduces_withdrawn_text_into_current_memory(tmp_path):
    root = tmp_path
    previous = root / "previous.db"
    with sqlite3.connect(previous) as db:
        db.executescript("CREATE TABLE events(id INTEGER PRIMARY KEY,kind TEXT,text TEXT,payload TEXT,date REAL);"
                         "CREATE TABLE nodes(start INTEGER,n INTEGER,text TEXT);"
                         "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);")
        db.execute("INSERT INTO meta VALUES ('summary_boundary','event')")
        db.execute("INSERT INTO events VALUES (0,'user','unique-retired','null',1)")
    store = Store(root / "current", "recovered", 128000, Settings())
    try:
        store.append("fixture-current", "user", "unique-current", "unique-current")
        with store.db:
            store.db.execute("INSERT INTO meta VALUES ('recovery_report',?)", (json.dumps({"source_archive":str(previous)}),))
        engine = OptChatEngine(settings=Settings())
        engine.worker = SimpleNamespace(store=store)
        assert json.loads(engine.handle_tool_call("optchat_search", {"query":"unique-retired"}))["matches"] == []
        matches = json.loads(engine.handle_tool_call("optchat_search", {"query":"unique-retired","archive":"previous"}))
        assert matches == {"archive":"previous","matches":[{"id":0,"kind":"user","excerpt":"unique-retired"}]}
        assert "unique-retired" not in store.event(0)["text"]
    finally:
        store.close()


def test_previous_zoom_paginates_preserved_images_without_refetch_or_writes(tmp_path, monkeypatch):
    import base64
    from io import BytesIO
    from PIL import Image
    root = tmp_path
    old = Store(root/"previous","session",128000,Settings())
    urls = []
    for colour in range(6):
        image = BytesIO()
        Image.new("RGB",(2,2),(colour,23,41)).save(image,format="PNG")
        urls.append("data:image/png;base64,"+base64.b64encode(image.getvalue()).decode("ascii"))
    old.append("photos","user","Six photos",[{"type":"image_url","image_url":{"url":url}} for url in urls])
    old.append("uncached","user","Uncached photo",[{"type":"image_url","image_url":{"url":"https://example.invalid/old.png"}}])
    previous = old.root/"chat.db"
    old.close()
    digest = hashlib.sha256(previous.read_bytes()).hexdigest()
    def forbidden_refetch(source):
        raise AssertionError("Previous evidence must never be resolved from a mutable reference")
    monkeypatch.setattr("optchat.images.resolve",forbidden_refetch)
    current = Store(root/"current","session",128000,Settings())
    try:
        with current.db:
            current.db.execute("INSERT INTO meta VALUES ('recovery_report',?)",(json.dumps({"source_archive":str(previous)}),))
        engine = OptChatEngine(settings=Settings())
        engine.worker = SimpleNamespace(store=current)
        first = engine.handle_tool_call("optchat_zoom",{"id":0,"n":1,"archive":"previous"})
        assert isinstance(first,dict), first
        assert first["_multimodal"] is True
        metadata = json.loads(first["content"][0]["text"])
        assert (metadata["archive"],metadata["total_images"],metadata["next_image_offset"]) == ("previous",6,4)
        second = engine.handle_tool_call("optchat_zoom",{"id":0,"n":1,"archive":"previous","image_offset":4})
        assert json.loads(second["content"][0]["text"])["next_image_offset"] is None
        assert [block["image_url"]["url"] for block in first["content"][1:]+second["content"][1:]] == urls
        missing = json.loads(engine.handle_tool_call("optchat_zoom",{"id":1,"n":1,"archive":"previous"}))
        assert missing["images"] == [{"index":0,"error":"Preserved image data is unavailable in the previous archive"}]
        assert hashlib.sha256(previous.read_bytes()).hexdigest() == digest
    finally:
        current.close()
