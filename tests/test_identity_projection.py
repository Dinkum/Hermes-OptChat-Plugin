"""Logical projection boundaries that differ from immutable archive addresses."""
import json
import os
import sqlite3

import pytest
from optchat.config import Settings
from test_identity_recovery import SESSION, message, store_at, workspace


def test_newest_rewind_is_a_tombstone_for_compacted_copies(workspace):
    root,home,db = workspace
    message(db,1,"same","Withdrawn",active=0,compacted=1)
    store = store_at(root)
    try:
        store.sync_hermes(home,SESSION)
        original = store.event_for_row(1)
        message(db,2,"same","Withdrawn",active=0,compacted=0)
        store.sync_hermes(home,SESSION)
        assert "Withdrawn" not in store.render()
        assert store.search("Withdrawn") == []
        assert store.zoom(original,1)["text"] == "Withdrawn"
    finally:
        store.close()


def test_removed_merged_carrier_tombstones_absorbed_copies(workspace):
    root,home,db = workspace
    message(db,1,"absorbed","Withdrawn constituent",active=0,compacted=1)
    message(db,2,"carrier","Merged",active=0,compacted=0,
            absorbed_message_uids=json.dumps(["absorbed"]))
    store = store_at(root)
    try:
        store.sync_hermes(home,SESSION)
        assert store.render() == "<chat>\n\n</chat>"
    finally:
        store.close()


@pytest.mark.parametrize("boundary",["event","user_turn"])
def test_edit_preserves_source_order_and_original_addresses(workspace,boundary):
    root,home,db = workspace
    message(db,1,"question","Old question")
    message(db,2,"answer","Unchanged assistant",role="assistant")
    settings = Settings(summary_boundary=boundary,compression_policy="on_demand",chunk_limit="bytes")
    store = store_at(root,settings)
    try:
        store.sync_hermes(home,SESSION)
        store.seal(store.count())
        original = store.event_for_row(1)
        db.execute("UPDATE messages SET content='Corrected question' WHERE id=1")
        db.commit()
        store.sync_hermes(home,SESSION)
        store.seal(store.count())
        view = store.render()
        assert view.index("Corrected question") < view.index("Unchanged assistant")
        assert "Old question" not in view
        assert store.search("Old question") == []
        assert store.zoom(original,1)["text"] == "Old question"
        assistant_unit = (store.event_for_row(2) if boundary == "event" else
            store.db.execute("SELECT id FROM leaves WHERE event_start=1 ORDER BY id DESC LIMIT 1").fetchone()[0])
        context = "\n".join(store.summary_lines(assistant_unit,raw=True))
        assert "Corrected question" in context
        assert "Unchanged assistant" not in context
    finally:
        store.close()
    store = store_at(root,settings)
    try:
        store.sync_hermes(home,SESSION)
        assert store.render().index("Corrected question") < store.render().index("Unchanged assistant")
    finally:
        store.close()


def test_prefix_withdrawal_requeues_unfinished_eager_summary(workspace):
    root,home,db = workspace
    message(db,1,"question","Keep this instruction")
    store = store_at(root,Settings(compression_policy="eager",chunk_limit="bytes"))
    try:
        store.sync_hermes(home,SESSION)
        store.sync_hermes(home,SESSION,before_row=1)
        assert store.ready_job() is None
        store.sync_hermes(home,SESSION)
        start,n,text,kind = store.ready_job()
        store.save_node(start,n,text,{"free":True})
        assert "Keep this instruction" in store.render()
    finally:
        store.close()


def test_replacing_state_database_invalidates_connection_local_revision(workspace):
    root,home,db = workspace
    message(db,1,"question","Before restore")
    store = store_at(root)
    try:
        store.sync_hermes(home,SESSION)
        replacement = home/"replacement.db"
        with sqlite3.connect(replacement) as restored:
            db.backup(restored)
            restored.execute("UPDATE messages SET content='After restore' WHERE id=1")
        os.replace(replacement,home/"state.db")
        store.sync_hermes(home,SESSION)
        assert "After restore" in store.render()
        assert "Before restore" not in store.render()
        assert store.zoom(0,1)["text"] == "Before restore"
    finally:
        store.close()


def test_effective_bridge_call_id_binds_durable_occurrence(workspace):
    root,home,db = workspace
    call = {"id":"fc_item","call_id":"call_pair","function":{"name":"read","arguments":"{}"}}
    message(db,1,"assistant","",role="assistant",tool_calls=json.dumps([call]),
            tool_call_uids=json.dumps({"call_pair":"occurrence"}))
    store = store_at(root)
    try:
        store.sync_hermes(home,SESSION)
        assert store.observe_tool("pre","call_pair",tool_name="read",args={}) == "occurrence"
        assert store.count() == 1
        assert store.zoom(0,1)["kind"] == "tool"
    finally:
        store.close()


def test_reactivated_cached_leaves_restore_missing_parent_work(workspace):
    root,home,db = workspace
    message(db,1,"question","Instruction")
    message(db,2,"reply","Acknowledged",role="assistant")
    store = store_at(root,Settings(compression_policy="eager",chunk_limit="bytes"))
    try:
        store.sync_hermes(home,SESSION)
        store.save_node(0,1,"user: Instruction",{"free":True})
        store.save_node(1,1,"talk: Acknowledged",{"free":True})
        store.sync_hermes(home,SESSION,before_row=2)
        store.sync_hermes(home,SESSION)
        start,n,source,_ = store.ready_job("parent")
        store.save_node(start,n,source,{"free":True})
        with store.db:
            store.fit(limit=1,next_message=True)
        assert "0+2|" in store.render()
        assert [child["text"] for child in store.zoom(0,2)["children"]] == [
            "user: Instruction","talk: Acknowledged"]
    finally:
        store.close()


def test_cached_summary_range_remains_auditable_after_an_edit(workspace):
    from optchat.engine import OptChatEngine
    from optchat.worker import Worker
    root,home,db = workspace
    message(db,1,"question","Old question")
    message(db,2,"answer","Unchanged assistant",role="assistant")
    store = store_at(root)
    store.sync_hermes(home,SESSION)
    def forbidden_provider(messages):
        raise AssertionError("Cached exact audit must not call a provider")
    worker = Worker(store,store.settings,forbidden_provider)
    try:
        worker.ensure(0,2)
        db.execute("UPDATE messages SET content='Corrected question' WHERE id=1")
        db.commit()
        store.sync_hermes(home,SESSION)
        engine = OptChatEngine(settings=store.settings)
        engine.worker = worker
        result = json.loads(engine.handle_tool_call("optchat_zoom",{"id":0,"n":2}))
        assert [child["text"] for child in result["children"]] == [
            "user: Old question","talk: Unchanged assistant"]
        assert store.render().index("Corrected question") < store.render().index("Unchanged assistant")
        assert store.search("Old question") == []
    finally:
        worker.close()
