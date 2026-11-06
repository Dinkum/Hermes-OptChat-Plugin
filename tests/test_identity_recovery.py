"""Public archive regressions using isolated temporary Hermes databases."""
import base64
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from optchat.config import Settings
from optchat.store import Store

SESSION = "identity-regression"
SETTINGS = Settings(compression_policy="on_demand", chunk_limit="bytes")
PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2ioAAAAASUVORK5CYII="
PNG_BYTES = base64.b64decode(PNG_BASE64)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path
    home = root / "hermes"
    home.mkdir()
    conn = sqlite3.connect(home / "state.db")
    conn.executescript("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
            timestamp REAL, tool_calls TEXT, tool_call_id TEXT, tool_name TEXT,
            active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0,
            message_uid TEXT, absorbed_message_uids TEXT,
            tool_call_uids TEXT, tool_call_uid TEXT);
    """)
    try:
        yield root, home, conn
    finally:
        conn.close()


def message(conn, row, uid, text, role="user", **fields):
    from hermes_state import SessionDB
    content = SessionDB._encode_content(text)
    data = dict(id=row, session_id=SESSION, role=role, content=content,
                timestamp=float(row), message_uid=uid, **fields)
    conn.execute(f"INSERT INTO messages ({','.join(data)}) VALUES ({','.join('?' for _ in data)})", tuple(data.values()))
    conn.commit()


def store_at(root, settings=SETTINGS):
    return Store(root / "archives", SESSION, 128000, settings)


def test_copied_uid_version_is_one_event_across_sync_and_restart(workspace):
    root, home, conn = workspace
    message(conn, 1, "first", "Original")
    store = store_at(root)
    store.record_raw("row:1", {"original": "raw admission"})
    store.sync_hermes(home, SESSION)
    first = store.event_for_row(1)
    assert store.event(first)["payload"] == '{"original":"raw admission"}'
    store.close()
    conn.execute("UPDATE messages SET active=0, compacted=1 WHERE id=1")
    conn.commit()
    message(conn, 9, "first", "Original")
    store = store_at(root)
    try:
        store.sync_hermes(home, SESSION)
        assert store.count() == 1
        assert store.event_for_row(9) == first == store.event_for_row(1)
        assert store.zoom(first, 1)["text"] == "Original"
    finally:
        store.close()


def test_identical_text_with_distinct_uids_is_not_deduplicated(workspace):
    root, home, conn = workspace
    message(conn, 1, "one", "Again")
    message(conn, 2, "two", "Again")
    store = store_at(root)
    try:
        store.sync_hermes(home, SESSION)
        assert store.count() == 2
        assert store.event_for_row(1) != store.event_for_row(2)
        assert store.render().count("Again") == 2
    finally:
        store.close()


def test_same_row_edit_creates_version_and_withdraws_old_projection(workspace):
    root, home, conn = workspace
    message(conn, 1, "editable", "Before")
    store = store_at(root)
    try:
        store.sync_hermes(home, SESSION)
        old = store.event_for_row(1)
        conn.execute("UPDATE messages SET content=? WHERE id=1", ("After",))
        conn.commit()
        store.sync_hermes(home, SESSION)
        assert store.count() == 2
        assert store.event_for_row(1) != old
        assert "After" in store.render() and "Before" not in store.render()
        assert store.zoom(old, 1)["text"] == "Before"
        store.sync_hermes(home, SESSION)
        assert store.count() == 2
    finally:
        store.close()


def test_rewind_never_imports_or_resurrects_withdrawn_rows(workspace):
    root, home, conn = workspace
    message(conn, 1, "kept", "Keep", active=0, compacted=1)
    message(conn, 2, "withdrawn-before-import", "Never import", active=0)
    message(conn, 3, "rewound", "Take back")
    store = store_at(root)
    try:
        store.sync_hermes(home, SESSION)
        assert store.count() == 2
        old = store.event_for_row(3)
        conn.execute("UPDATE messages SET active=0, compacted=0 WHERE id=3")
        conn.commit()
        store.sync_hermes(home, SESSION)
        assert "Take back" not in store.render()
        assert "Never import" not in store.render()
        assert "Keep" in store.render()
        assert store.event_for_row(3) is None
        assert store.zoom(old, 1)["text"] == "Take back"
    finally:
        store.close()


def test_latest_active_uid_version_and_absorbed_merge_replace_constituents(workspace):
    root, home, conn = workspace
    message(conn, 1, "a", "Alpha")
    message(conn, 2, "b", "Beta")
    store = store_at(root)
    try:
        store.sync_hermes(home, SESSION)
        message(conn, 3, "a", "Alpha\n\nBeta", absorbed_message_uids=json.dumps(["b"]))
        conn.execute("UPDATE messages SET active=0, compacted=0 WHERE id=2")
        conn.commit()
        store.sync_hermes(home, SESSION)
        assert store.count() == 3
        assert store.render().count("Alpha") == 1
        assert store.render().count("Beta") == 1
        assert store.event_for_row(3) != store.event_for_row(1)
        assert store.event_for_row(2) is None
        assert store.zoom(0, 1)["text"] == "Alpha"
    finally:
        store.close()


def test_missing_uid_uses_row_identity_without_text_dedup_and_adopts_backfill(workspace):
    root, home, conn = workspace
    message(conn, 1, None, "Legacy")
    message(conn, 2, None, "Legacy")
    store = store_at(root)
    try:
        store.sync_hermes(home, SESSION)
        assert store.count() == 2
        original = store.event_for_row(1)
        conn.execute("UPDATE messages SET message_uid='backfilled' WHERE id=1")
        conn.commit()
        store.sync_hermes(home, SESSION)
        assert store.count() == 2
        assert store.event_for_row(1) == original
        message(conn, 3, "backfilled", "Legacy")
        store.sync_hermes(home, SESSION)
        assert store.count() == 2
        assert store.event_for_row(3) == original
    finally:
        store.close()


def calls(uid):
    return dict(tool_calls=json.dumps([{"id": "reused", "function": {"name": "read", "arguments": "{}"}}]),
                tool_call_uids=json.dumps({"reused": uid}))


def test_tool_occurrences_hooks_and_database_share_identity_and_keep_raw(workspace, monkeypatch):
    import optchat
    root, home, conn = workspace
    message(conn, 1, "assistant-one", "", role="assistant", **calls("occurrence-one"))
    store = store_at(root)
    engine = SimpleNamespace(home=home, session=SESSION,
                             worker=SimpleNamespace(store=store, notify=lambda: None))
    monkeypatch.setattr(optchat, "current", lambda session: engine)
    try:
        optchat.pre_tool(session_id=SESSION, tool_call_id="reused", tool_name="read", args={}, api_request_id="request-one")
        optchat.post_tool(session_id=SESSION, tool_call_id="reused", tool_name="read", result={"raw": "first full result"}, api_request_id="request-one")
        message(conn, 2, "result-one", "first projected result", role="tool", tool_call_id="reused", tool_name="read", tool_call_uid="occurrence-one")
        store.sync_hermes(home, SESSION)
        first = store.event_for_row(2)
        assert json.loads(store.event(first)["payload"]) == {"raw": "first full result"}
        message(conn, 3, "assistant-one", "", role="assistant", **calls("occurrence-one"))
        store.sync_hermes(home, SESSION)
        assert store.count() == 2
        message(conn, 4, "assistant-two", "", role="assistant", **calls("occurrence-two"))
        optchat.pre_tool(session_id=SESSION, tool_call_id="reused", tool_name="read", args={}, api_request_id="request-two")
        optchat.post_tool(session_id=SESSION, tool_call_id="reused", tool_name="read", result={"raw": "second full result"}, api_request_id="request-two")
        message(conn, 5, "result-two", "second projected result", role="tool", tool_call_id="reused", tool_name="read", tool_call_uid="occurrence-two")
        store.sync_hermes(home, SESSION)
        assert store.count() == 4
        assert json.loads(store.event(store.event_for_row(5))["payload"]) == {"raw": "second full result"}
        assert store.zoom(first, 1)["text"] != store.zoom(store.event_for_row(5), 1)["text"]
    finally:
        store.close()


def legacy_archive(root, events, nodes=()):
    path = root / "old-chat.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE events(id INTEGER PRIMARY KEY,source TEXT UNIQUE,kind TEXT,text TEXT,payload TEXT,date REAL);
        CREATE TABLE nodes(start INTEGER,n INTEGER,text TEXT,metrics TEXT,PRIMARY KEY(start,n));
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE raw_activity(source TEXT PRIMARY KEY,payload TEXT);
        CREATE TABLE leaves(id INTEGER PRIMARY KEY,event_start INTEGER,event_stop INTEGER,start_offset INTEGER,stop_offset INTEGER,source TEXT);
        CREATE TABLE images(event INTEGER,ordinal INTEGER,mime TEXT,data BLOB,error TEXT,PRIMARY KEY(event,ordinal));
    """)
    conn.executemany("INSERT INTO meta VALUES (?,?)", [("session", SESSION), ("summary_boundary", "event")])
    conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", events)
    conn.executemany("INSERT INTO nodes VALUES (?,?,?,?)", nodes)
    conn.commit()
    conn.close()
    return path


def test_offline_recovery_maps_copies_preserves_originals_and_reuses_exact_subtrees(workspace):
    from optchat.recovery import recover_archive
    root, home, conn = workspace
    message(conn, 1, "a", "Alpha")
    message(conn, 2, "b", "Beta", role="assistant")
    message(conn, 3, "a", "Alpha")
    message(conn, 4, "withdrawn", "Withdrawn", active=0)
    source = legacy_archive(root, [
        (0, "row:1", "user", "Alpha", '"Alpha"', 1.0),
        (1, "row:2", "talk", "Beta", '"Beta"', 2.0),
        (2, "row:3", "user", "Alpha", '"Alpha"', 3.0),
        (3, "row:4", "user", "Withdrawn", '"Withdrawn"', 4.0),
        (4, "row:999", "user", "Untracked", '"Untracked"', 5.0),
    ], [(0, 1, "Summary A", '{"original_metrics":1}'),
        (1, 1, "Summary B", '{"original_metrics":2}'),
        (0, 2, "Summary AB", '{"original_metrics":3}'),
        (2, 1, "Summary A", '{"original_metrics":4}')])
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = root / "recovered"
    report = recover_archive(source, home / "state.db", destination, SESSION, settings=SETTINGS)
    assert report["event_map"] == {0: 0, 1: 1, 2: 0, 3: None, 4: None}
    assert set(report["preserved_unmapped"]) == {3, 4}
    assert report["node_map"][(0, 2)] == (0, 2)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    recovered = Store(destination, SESSION, 128000, SETTINGS)
    try:
        assert recovered.count() == 2
        assert recovered.zoom(0, 1)["text"] == "Alpha"
        assert json.loads(recovered.event(0)["payload"]) == "Alpha"
        assert recovered.node(0, 2) == "Summary AB"
        recovered.sync_hermes(home,SESSION)
        assert recovered.count() == 2
        assert recovered.node(0,2) == "Summary AB"
        assert "Untracked" not in recovered.render()
    finally:
        recovered.close()
    repeated = recover_archive(source, home / "state.db", destination, SESSION, settings=SETTINGS)
    assert repeated == report


def test_recovery_fails_on_ambiguous_provider_id_without_occurrence_witness(workspace):
    from optchat.recovery import RecoveryAmbiguityError, recover_archive
    root, home, conn = workspace
    message(conn, 1, "a", "", role="assistant", **calls("one"))
    message(conn, 2, "b", "", role="assistant", **calls("two"))
    source = legacy_archive(root, [(0, "call:reused", "tool", '{"name":"read","arguments":"{}"}', '{"name":"read","arguments":"{}"}', 1.0)])
    with pytest.raises(RecoveryAmbiguityError, match="ambiguous|occurrence"):
        recover_archive(source, home / "state.db", root / "recovered", SESSION, settings=SETTINGS)
    assert not (root / "recovered" / hashlib.sha256(SESSION.encode()).hexdigest() / "chat.db").exists()


def test_legacy_archive_requires_explicit_offline_recovery(workspace):
    root, home, conn = workspace
    path = legacy_archive(root, [(0, "row:1", "user", "Old", '"Old"', 1.0)])
    archive = root / "archives" / hashlib.sha256(SESSION.encode()).hexdigest()
    archive.mkdir(parents=True)
    (archive / "chat.db").write_bytes(path.read_bytes())
    with pytest.raises(RuntimeError, match="recover_archive|offline recovery"):
        store_at(root)


@pytest.mark.parametrize("boundary", ["event", "user_turn"])
def test_rewrites_remove_summaries_that_contain_withdrawn_versions(workspace, boundary):
    root, home, conn = workspace
    settings = Settings(summary_boundary=boundary, compression_policy="on_demand", chunk_limit="bytes")
    message(conn, 1, "a", "Retired content")
    message(conn, 2, "b", "Keep", role="assistant")
    store = store_at(root, settings)
    try:
        store.sync_hermes(home, SESSION)
        store.finish_turn()
        store.save_node(0, 1, "Summary of retired content", {})
        conn.execute("UPDATE messages SET content=? WHERE id=1", ("Replacement",))
        conn.commit()
        store.sync_hermes(home, SESSION)
        store.finish_turn()
        assert "retired" not in store.render().lower()
        assert "Replacement" in store.render()
        assert "Keep" in store.render()
        assert store.zoom(0, 1)["text"] == "Retired content"
    finally:
        store.close()


def test_host_content_sentinel_and_literal_json_are_not_conflated(workspace):
    root, home, conn = workspace
    message(conn, 1, "literal", '{"literal":"JSON text, not structured content"}')
    message(conn, 2, "parts", [{"type":"text","text":"Multimodal text"}])
    message(conn, 3, "reserved", '\x00json:{"literal":"reserved prefix in plaintext"}')
    store = store_at(root)
    try:
        store.sync_hermes(home, SESSION)
        assert store.zoom(store.event_for_row(1),1)["text"] == '{"literal":"JSON text, not structured content"}'
        assert store.zoom(store.event_for_row(2),1)["text"] == "Multimodal text"
        assert json.loads(store.event(store.event_for_row(2))["payload"]) == [{"type":"text","text":"Multimodal text"}]
        assert store.zoom(store.event_for_row(3),1)["text"] == '\x00json:{"literal":"reserved prefix in plaintext"}'
    finally:
        store.close()


def test_explicit_ambiguity_preservation_keeps_source_only_records(workspace):
    from optchat.recovery import recover_archive
    root, home, conn = workspace
    message(conn,1,"a","",role="assistant",**calls("one"))
    message(conn,2,"b","",role="assistant",**calls("two"))
    source = legacy_archive(root, [(0,"call:reused","tool",'{"name":"read","arguments":"{}"}','{"name":"read","arguments":"{}"}',1.0)])
    report = recover_archive(source,home/"state.db",root/"recovered",SESSION,settings=SETTINGS,preserve_ambiguous=True)
    assert report["event_map"] == {0:None}
    assert report["preserved_unmapped"] == [0]
    assert "ambiguous" in report["unmapped_reasons"][0]
    recovered = Store(root/"recovered",SESSION,128000,SETTINGS)
    try:
        assert recovered.count() == 2
        assert recovered.node(0,1) is None
        assert recovered.node(1,1) is None
    finally:
        recovered.close()


def test_recovery_preserves_user_turn_summary_leaf_and_zoom_references(workspace):
    from optchat.recovery import recover_archive
    root, home, conn = workspace
    settings = Settings(summary_boundary="user_turn",compression_policy="on_demand",chunk_limit="bytes")
    message(conn,1,"a","Question")
    message(conn,2,"b","Answer",role="assistant")
    source = Store(root/"old",SESSION,128000,settings)
    source.append("row:1","user","Question","Question",1.0)
    source.append("row:2","talk","Answer","Answer",2.0)
    source.finish_turn()
    source.save_node(0,1,"Preserved turn summary",{"genuine":True})
    source_path = source.root/"chat.db"
    source.close()
    report = recover_archive(source_path,home/"state.db",root/"recovered",SESSION,settings=settings)
    assert report["leaf_map"] == {0:0}
    assert report["node_map"] == {(0,1):(0,1)}
    recovered = Store(root/"recovered",SESSION,128000,settings)
    try:
        leaf = recovered.zoom_leaf(0,1)
        assert leaf["text"] == "Preserved turn summary"
        assert [event["id"] for event in leaf["events"]] == [0,1]
        assert recovered.zoom(0,1)["text"] == "Question"
        assert recovered.zoom(1,1)["text"] == "Answer"
    finally:
        recovered.close()


def test_folded_tool_uid_list_keeps_reused_provider_id_occurrences(workspace):
    root, home, conn = workspace
    message(conn,1,"folded","",role="assistant",
            tool_calls=json.dumps([{"id":"reused","function":{"name":"read","arguments":"{}"}},
                                   {"id":"reused","function":{"name":"read","arguments":"{}"}}]),
            tool_call_uids=json.dumps({"reused":["one","two"]}))
    message(conn,2,"result-one","One",role="tool",tool_call_id="reused",tool_call_uid="one")
    message(conn,3,"result-two","Two",role="tool",tool_call_id="reused",tool_call_uid="two")
    store = store_at(root)
    try:
        store.sync_hermes(home,SESSION)
        assert store.count() == 4
        assert store.render().count('tool:') == 2
        assert store.zoom(store.event_for_row(2),1)["text"] == "One"
        assert store.zoom(store.event_for_row(3),1)["text"] == "Two"
    finally:
        store.close()


def test_raw_admission_cannot_leak_into_later_same_row_version(workspace):
    root, home, conn = workspace
    message(conn,1,"a","Before")
    store = store_at(root)
    try:
        store.record_raw("row:1",{"original":"Before"})
        store.sync_hermes(home,SESSION)
        original = store.event_for_row(1)
        conn.execute("UPDATE messages SET content='After' WHERE id=1")
        conn.commit()
        store.sync_hermes(home,SESSION)
        assert json.loads(store.event(store.event_for_row(1))["payload"]) == "After"
        assert json.loads(store.event(original)["payload"]) == {"original":"Before"}
        store.record_raw("row:1",{"original":"Final"})
        conn.execute("UPDATE messages SET content='Final' WHERE id=1")
        conn.commit()
        store.sync_hermes(home,SESSION)
        assert json.loads(store.event(store.event_for_row(1))["payload"]) == {"original":"Final"}
        assert json.loads(store.event(original)["payload"]) == {"original":"Before"}
    finally:
        store.close()


def test_recovery_restores_unambiguous_raw_tool_result_and_exact_summaries(workspace):
    from optchat.recovery import recover_archive
    root, home, conn = workspace
    message(conn,1,"a","",role="assistant",**calls("one"))
    message(conn,2,"b","Pruned",role="tool",tool_call_id="reused",tool_call_uid="one")
    source = legacy_archive(root, [
        (0,"call:reused","tool",'{"name":"read","arguments":"{}"}','{"name":"read","arguments":"{}"}',1.0),
        (1,"result:reused","echo","Pruned",'"Full original result"',2.0),
    ],[(0,1,"Original call summary",'{"genuine":1}'),(1,1,"Original result summary",'{"genuine":2}'),(0,2,"Original parent summary",'{"genuine":3}')])
    report = recover_archive(source,home/"state.db",root/"recovered",SESSION,settings=SETTINGS)
    assert report["event_map"] == {0:0,1:1}
    assert report["node_map"][(0,2)] == (0,2)
    recovered = Store(root/"recovered",SESSION,128000,SETTINGS)
    try:
        assert recovered.zoom(1,1)["text"] == "Pruned"
        assert json.loads(recovered.event(1)["payload"]) == "Full original result"
        assert recovered.node(0,2) == "Original parent summary"
        assert recovered.zoom(0,2)["children"][1]["text"] == "Original result summary"
    finally:
        recovered.close()


@pytest.mark.parametrize("changed_attachment",[False,True])
def test_recovery_requires_full_structured_version_not_attachment_placeholder(workspace,changed_attachment):
    from optchat.recovery import recover_archive
    root, home, conn = workspace
    before = [{"type":"text","text":"Photo"},{"type":"image_url","image_url":{"url":"https://example.invalid/old.png"}}]
    current = [{"type":"text","text":"Photo"},{"type":"image_url","image_url":{"url":"https://example.invalid/new.png"}}] if changed_attachment else before
    message(conn,1,"a",current)
    source = legacy_archive(root,[(0,"row:1","user","Photo\n[attachment: image_url]",json.dumps(before),1.0)],
                            [(0,1,"Old photo summary",'{"genuine":1}')])
    original = sqlite3.connect(source)
    original.execute("INSERT INTO raw_activity VALUES (?,?)",("row:1",json.dumps(before)))
    original.commit()
    original.close()
    report = recover_archive(source,home/"state.db",root/"recovered",SESSION,settings=SETTINGS)
    recovered = Store(root/"recovered",SESSION,128000,SETTINGS)
    try:
        assert json.loads(recovered.event(0)["payload"]) == current
        if changed_attachment:
            assert report["event_map"] == {0:None}
            assert report["preserved_unmapped"] == [0]
            assert recovered.node(0,1) is None
            assert "version" in report["unmapped_reasons"][0]
        else:
            assert report["event_map"] == {0:0}
            assert recovered.node(0,1) == "Old photo summary"
    finally:
        recovered.close()


def test_tool_same_row_edit_does_not_resurrect_already_claimed_raw_result(workspace,monkeypatch):
    import optchat
    root, home, conn = workspace
    message(conn,1,"a","",role="assistant",**calls("one"))
    store = store_at(root)
    engine = SimpleNamespace(home=home,session=SESSION,worker=SimpleNamespace(store=store,notify=lambda:None))
    monkeypatch.setattr(optchat,"current",lambda session:engine)
    try:
        optchat.pre_tool(session_id=SESSION,tool_call_id="reused",tool_name="read",args={},api_request_id="one")
        optchat.post_tool(session_id=SESSION,tool_call_id="reused",tool_name="read",result="Full original result",api_request_id="one")
        message(conn,2,"b","Projected result",role="tool",tool_call_id="reused",tool_call_uid="one")
        store.sync_hermes(home,SESSION)
        old = store.event_for_row(2)
        assert store.zoom(old,1)["text"] == "Full original result"
        conn.execute("UPDATE messages SET content='Edited result' WHERE id=2")
        conn.commit()
        store.sync_hermes(home,SESSION)
        current = store.event_for_row(2)
        assert current != old
        assert store.zoom(current,1)["text"] == "Edited result"
        assert json.loads(store.event(current)["payload"]) == "Edited result"
        assert store.zoom(old,1)["text"] == "Full original result"
        assert "Full original result" not in store.render()
    finally:
        store.close()


def test_offline_recovery_refuses_source_with_live_writer(workspace):
    from optchat.recovery import recover_archive
    root, home, conn = workspace
    message(conn,1,"a","Question")
    source = Store(root/"old",SESSION,128000,SETTINGS)
    source.append("row:1","user","Question","Question")
    try:
        with pytest.raises(RuntimeError,match="live writer"):
            recover_archive(source.root/"chat.db",home/"state.db",root/"recovered",SESSION,settings=SETTINGS)
    finally:
        source.close()
    assert not (root/"recovered"/hashlib.sha256(SESSION.encode()).hexdigest()/"chat.db").exists()


def test_recovery_does_not_alias_different_tool_result_version_by_provider_id(workspace):
    from optchat.recovery import recover_archive
    root, home, conn = workspace
    message(conn,1,"a","",role="assistant",**calls("one"))
    message(conn,2,"b","Edited result",role="tool",tool_call_id="reused",tool_call_uid="one")
    source = legacy_archive(root,[(0,"result:reused","echo","Original result",'"Original result"',1.0)],
                            [(0,1,"Original summary",'{"genuine":1}')])
    report = recover_archive(source,home/"state.db",root/"recovered",SESSION,settings=SETTINGS)
    assert report["event_map"] == {0:None}
    assert report["preserved_unmapped"] == [0]
    recovered = Store(root/"recovered",SESSION,128000,SETTINGS)
    try:
        assert recovered.zoom(recovered.event_for_row(2),1)["text"] == "Edited result"
        assert "Original result" not in recovered.render()
        assert recovered.node(1,1) is None
    finally:
        recovered.close()


@pytest.mark.parametrize("historical_state", ["withdrawn", "nonselected"])
def test_recovery_counts_historical_provider_occurrences_before_matching(workspace, historical_state):
    from optchat.recovery import RecoveryAmbiguityError, recover_archive
    root, home, conn = workspace
    historical = {"active": 0} if historical_state == "withdrawn" else {}
    assistant_uid = "same-assistant" if historical_state == "nonselected" else "old-assistant"
    result_uid = "same-result" if historical_state == "nonselected" else "old-result"
    message(conn, 1, assistant_uid, "", role="assistant", **calls("old"), **historical)
    message(conn, 2, result_uid, "Identical result", role="tool",
            tool_call_id="reused", tool_call_uid="old", **historical)
    message(conn, 3, assistant_uid if historical_state == "nonselected" else "new-assistant",
            "", role="assistant", **calls("new"))
    message(conn, 4, result_uid if historical_state == "nonselected" else "new-result",
            "Identical result", role="tool", tool_call_id="reused", tool_call_uid="new")
    source = legacy_archive(root, [
        (0, "call:reused", "tool", '{"name":"read","arguments":"{}"}',
         '{"name":"read","arguments":"{}"}', 1.0),
        (1, "result:reused", "echo", "Identical result", '"Withdrawn raw result"', 2.0),
    ], [(1, 1, "Withdrawn result summary", '{"genuine":1}')])
    original_bytes = source.read_bytes()
    with pytest.raises(RecoveryAmbiguityError) as failure:
        recover_archive(source, home / "state.db", root / "rejected", SESSION, settings=SETTINGS)
    assert set(failure.value.ambiguities) == {0, 1}
    assert not (root / "rejected").exists()
    report = recover_archive(source, home / "state.db", root / "recovered", SESSION,
                             settings=SETTINGS, preserve_ambiguous=True)
    assert report["event_map"] == {0: None, 1: None}
    assert report["preserved_unmapped"] == [0, 1]
    assert report["source_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    recovered = Store(root / "recovered", SESSION, 128000, SETTINGS)
    try:
        recovered.sync_hermes(home, SESSION)
        assert json.loads(recovered.event(recovered.event_for_row(4))["payload"]) == "Identical result"
        assert "Withdrawn" not in recovered.render()
        assert recovered.node(1, 1) is None
    finally:
        recovered.close()
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("kind,tool_name", [("echo", "read"), ("work", "delegate_task")])
@pytest.mark.parametrize("origin", ["row:2", "result:reused"])
@pytest.mark.parametrize("changed_attachment", [False, True])
def test_recovery_checks_structured_tool_result_versions(workspace, kind, tool_name, origin, changed_attachment):
    from optchat.recovery import recover_archive
    root, home, conn = workspace
    before = [{"type": "text", "text": "Photo"},
              {"type": "image_url", "image_url": {"url": "https://example.invalid/old.png"}}]
    current = ([{"type": "text", "text": "Photo"},
                {"type": "image_url", "image_url": {"url": "https://example.invalid/new.png"}}]
               if changed_attachment else before)
    message(conn, 1, "assistant", "", role="assistant",
            tool_calls=json.dumps([{"id": "reused", "function": {"name": tool_name, "arguments": "{}"}}]),
            tool_call_uids=json.dumps({"reused": "one"}))
    message(conn, 2, "result", current, role="tool", tool_call_id="reused",
            tool_call_uid="one", tool_name=tool_name)
    source = legacy_archive(root, [(0, origin, kind, "Photo\n[attachment: image_url]", json.dumps(before), 2.0)],
                            [(0, 1, "Old photo summary", '{"genuine":1}')])
    image = PNG_BYTES
    with sqlite3.connect(source) as original:
        original.execute("INSERT INTO raw_activity VALUES (?,?)", (origin, json.dumps(before)))
        original.execute("INSERT INTO images VALUES (0,0,'image/png',?,NULL)", (image,))
    original_bytes = source.read_bytes()
    report = recover_archive(source, home / "state.db", root / "recovered", SESSION, settings=SETTINGS)
    recovered = Store(root / "recovered", SESSION, 128000, SETTINGS)
    try:
        recovered.sync_hermes(home, SESSION)
        event = recovered.event_for_row(2)
        assert json.loads(recovered.event(event)["payload"]) == current
        if changed_attachment:
            assert report["event_map"] == {0: None}
            assert recovered.node(event, 1) is None
            assert all(row[0] != image for row in recovered.db.execute("SELECT data FROM images"))
        else:
            assert report["event_map"] == {0: event}
            assert recovered.node(event, 1) == "Old photo summary"
            assert recovered.db.execute("SELECT data FROM images WHERE event=? AND ordinal=0", (event,)).fetchone()[0] == image
    finally:
        recovered.close()
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("initial_mapping", ["source-only", "mapped"])
def test_recovery_raw_admissions_cannot_revive_after_row_loses_its_projection(workspace, initial_mapping):
    from optchat.recovery import recover_archive
    root, home, conn = workspace
    old_payload = [{"type": "text", "text": "Withdrawn photo"},
                   {"type": "image_url", "image_url": {"url": "https://example.invalid/old.png"}}]
    message(conn, 1, "editable", old_payload if initial_mapping == "mapped" else "")
    raw_payload = json.dumps(old_payload)
    source = legacy_archive(root, [
        (0, "row:1", "user", "Withdrawn photo\n[attachment: image_url]", raw_payload, 1.0),
    ])
    with sqlite3.connect(source) as original:
        original.execute("INSERT INTO raw_activity VALUES ('row:1',?)", (raw_payload,))
        original.execute("INSERT INTO images VALUES (0,0,'image/png',?,NULL)", (PNG_BYTES,))
    original_bytes = source.read_bytes()
    report = recover_archive(source, home / "state.db", root / "recovered", SESSION, settings=SETTINGS)
    assert report["event_map"] == {0: 0 if initial_mapping == "mapped" else None}
    recovered = Store(root / "recovered", SESSION, 128000, SETTINGS)
    try:
        recovered.sync_hermes(home, SESSION)
        if initial_mapping == "mapped":
            conn.execute("UPDATE messages SET content='' WHERE id=1")
            conn.commit()
            recovered.sync_hermes(home, SESSION)
            assert recovered.event_for_row(1) is None
        conn.execute("UPDATE messages SET content='Fresh edit' WHERE id=1")
        conn.commit()
        recovered.sync_hermes(home, SESSION)
        event = recovered.event_for_row(1)
        assert recovered.zoom(event, 1)["text"] == "Fresh edit"
        assert json.loads(recovered.event(event)["payload"]) == "Fresh edit"
        assert recovered.db.execute("SELECT 1 FROM images WHERE event=?", (event,)).fetchone() is None
        assert recovered.db.execute("SELECT payload FROM raw_activity WHERE source='previous-raw:row:1'").fetchone()[0] == raw_payload
        assert "Withdrawn photo" not in recovered.render()
        if initial_mapping == "mapped":
            assert json.loads(recovered.event(0)["payload"]) == old_payload
            assert recovered.db.execute("SELECT data FROM images WHERE event=0 AND ordinal=0").fetchone()[0] == PNG_BYTES
    finally:
        recovered.close()
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("boundary,unit", [("event", "event"), ("event", "leaf"), ("user_turn", "event")])
def test_previous_zoom_pages_preserved_images_read_only(workspace, monkeypatch, boundary, unit):
    from optchat.engine import OptChatEngine
    from optchat.history import previous_call
    from optchat.recovery import recover_archive
    import optchat.images
    root, home, conn = workspace
    settings = Settings(summary_boundary=boundary, compression_policy="on_demand", chunk_limit="bytes")
    message(conn, 1, "current", "Current text")
    payload = [{"type": "text", "text": "Old photo"}] + [
        {"type": "image_url", "image_url": {"url": f"https://example.invalid/{index}.png"}}
        for index in range(6)]
    text = "Old photo" + "\n[attachment: image_url]" * 6
    source = legacy_archive(root, [(0, "row:999", "user", text, json.dumps(payload), 1.0)])
    with sqlite3.connect(source) as original:
        original.execute("UPDATE meta SET value=? WHERE key='summary_boundary'", (boundary,))
        original.executemany("INSERT INTO images VALUES (0,?,'image/png',?,NULL)",
                             [(index, PNG_BYTES) for index in range(5)])
    original_bytes = source.read_bytes()
    report = recover_archive(source, home / "state.db", root / "recovered", SESSION, settings=settings)
    assert report["event_map"] == {0: None}
    attempted_resolutions = []
    def forbidden_resolution(reference):
        attempted_resolutions.append(reference)
        raise AssertionError("Previous archive must not resolve image references")
    monkeypatch.setattr(optchat.images, "resolve", forbidden_resolution)
    recovered = Store(root / "recovered", SESSION, 128000, settings)
    try:
        engine = SimpleNamespace(worker=SimpleNamespace(store=recovered), settings=settings)
        args = {"id": 0, "n": 1, "unit": unit, "archive": "previous"}
        first = previous_call(recovered, "optchat_zoom", args)
        assert set(first) == {"_multimodal", "content", "text_summary"}
        assert first["_multimodal"] is True
        assert first["content"][0] == {"type": "text", "text": first["text_summary"]}
        metadata = json.loads(first["text_summary"])
        assert metadata["archive"] == "previous"
        assert metadata["text"] == text
        assert metadata["image_offset"] == 0
        assert metadata["next_image_offset"] == 4
        assert metadata["total_images"] == 6
        assert [item["index"] for item in metadata["images"]] == [0, 1, 2, 3]
        block = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_BASE64}"}}
        assert first["content"][1:] == [block] * 4
        assert OptChatEngine.handle_tool_call(engine, "optchat_zoom", args) == first
        second = previous_call(recovered, "optchat_zoom", dict(args, offset=1, image_offset=4))
        metadata = json.loads(second["text_summary"])
        assert metadata["image_offset"] == 4
        assert metadata["next_image_offset"] is None
        assert metadata["text"] == text[1:]
        assert second["content"][1:] == [block]
        assert metadata["images"][0] == {"index": 4, "attached": True, "mime": "image/png", "bytes": len(PNG_BYTES)}
        assert metadata["images"][1]["index"] == 5
        assert "unavailable" in metadata["images"][1]["error"].lower()
        text_only = previous_call(recovered, "optchat_zoom", dict(args, offset=1))
        assert text_only["text"] == text[1:]
        assert "images" not in text_only and "_multimodal" not in text_only
        exhausted = previous_call(recovered, "optchat_zoom", dict(args, image_offset=6))
        assert exhausted["images"] == [] and exhausted["next_image_offset"] is None
        for offset in (-1, True, 7):
            with pytest.raises(ValueError, match="offset"):
                previous_call(recovered, "optchat_zoom", dict(args, image_offset=offset))
        matches = previous_call(recovered, "optchat_search", {"query": "Old photo"})
        assert matches["archive"] == "previous" and matches["matches"][0]["id"] == 0
        assert recovered.zoom(0, 1)["text"] == "Current text"
        assert attempted_resolutions == []
    finally:
        recovered.close()
    assert source.read_bytes() == original_bytes
    assert hashlib.sha256(original_bytes).hexdigest() == report["source_sha256"]


def test_previous_zoom_without_an_image_table_reports_unavailable(workspace, monkeypatch):
    from optchat.history import previous_call
    from optchat.recovery import recover_archive
    import optchat.images
    root, home, conn = workspace
    message(conn, 1, "current", "Current text")
    payload = [{"type": "image_url", "image_url": {"url": "/missing/old-image.png"}}]
    source = legacy_archive(root, [(0, "row:999", "user", "[attachment: image_url]", json.dumps(payload), 1.0)])
    with sqlite3.connect(source) as original:
        original.execute("DROP TABLE images")
    original_bytes = source.read_bytes()
    recover_archive(source, home / "state.db", root / "recovered", SESSION, settings=SETTINGS)
    attempted_resolutions = []
    def forbidden_resolution(reference):
        attempted_resolutions.append(reference)
        raise AssertionError("Previous archive must not resolve image references")
    monkeypatch.setattr(optchat.images, "resolve", forbidden_resolution)
    recovered = Store(root / "recovered", SESSION, 128000, SETTINGS)
    try:
        result = previous_call(recovered, "optchat_zoom", {"id": 0, "n": 1})
        assert result["archive"] == "previous"
        assert result["total_images"] == 1
        assert "unavailable" in result["images"][0]["error"].lower()
        assert "_multimodal" not in result
        assert attempted_resolutions == []
    finally:
        recovered.close()
    assert source.read_bytes() == original_bytes


def test_recovery_algorithm_version_prevents_stale_idempotent_reuse(workspace, monkeypatch):
    import optchat.recovery as recovery
    root, home, conn = workspace
    message(conn, 1, "current", "Exact text")
    source = legacy_archive(root, [(0, "row:1", "user", "Exact text", '"Exact text"', 1.0)])
    original_bytes = source.read_bytes()
    destination = root / "recovered"
    report = recovery.recover_archive(source, home / "state.db", destination, SESSION, settings=SETTINGS)
    assert report["recovery_version"] == recovery.RECOVERY_VERSION
    monkeypatch.setattr(recovery, "RECOVERY_VERSION", recovery.RECOVERY_VERSION + "-next")
    with pytest.raises(FileExistsError, match="different recovery inputs"):
        recovery.recover_archive(source, home / "state.db", destination, SESSION, settings=SETTINGS)
    assert source.read_bytes() == original_bytes


def test_recovery_never_backdates_mutable_image_file_bytes(workspace):
    from PIL import Image
    from optchat.engine import OptChatEngine
    from optchat.recovery import recover_archive
    from optchat.store import visible
    root,home,conn = workspace
    image = root/"mutable.png"
    Image.new("RGB",(2,2),(255,0,0)).save(image)
    payload = [{"type":"image_url","image_url":{"url":str(image)}}]
    message(conn,1,"photo",payload)
    old = legacy_archive(root,[(0,"row:1","user",visible(payload),json.dumps(payload),1.0)])
    Image.new("RGB",(2,2),(0,0,255)).save(image)
    recover_archive(old,home/"state.db",root/"recovered",SESSION,settings=SETTINGS)
    store = Store(root/"recovered",SESSION,128000,SETTINGS)
    try:
        engine = OptChatEngine(settings=SETTINGS)
        engine.worker = SimpleNamespace(store=store)
        response = engine.handle_tool_call("optchat_zoom",{"id":0,"n":1})
        assert isinstance(response,str), "Today's image must not be attached as historical evidence"
        result = json.loads(response)
        assert "error" in result["images"][0]
        assert not result["images"][0].get("attached",False)
    finally:
        store.close()


def test_recovery_rebuilds_an_oversized_immutable_sentence_node(workspace):
    from optchat.engine import Usage
    from optchat.recovery import recover_archive
    from optchat.worker import Worker
    root,home,conn = workspace
    original = "x"*130000
    message(conn,1,"large",original)
    old = legacy_archive(root,[(0,"row:1","user",original,json.dumps(original),1.0)],
                         [(0,1,original,"{}")])
    settings = Settings(summary_concurrency=1,summary_tries=1,settle_seconds=2)
    recover_archive(old,home/"state.db",root/"recovered",SESSION,settings=settings)
    store = Store(root/"recovered",SESSION,128000,settings)
    def summarize(messages):
        return SimpleNamespace(text="user: Large original retained; zoom for its exact content.",
                               model="fixture",provider="none",usage=Usage())
    worker = Worker(store,settings,summarize)
    try:
        view = worker.settle(1)
        assert len(view.encode()) <= store.byte_budget
        assert "Large original retained" in view
        pages,offset = [],0
        while True:
            page = store.zoom(0,1,offset)
            pages.append(page["text"])
            offset = page["next_offset"]
            if offset is None:
                break
        assert "".join(pages) == original
    finally:
        worker.close()
