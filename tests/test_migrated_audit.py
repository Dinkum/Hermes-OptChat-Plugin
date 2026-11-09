"""Historical tree navigation stays separate from the active bounded projection."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from optchat.config import Settings
from optchat.engine import Usage
from optchat.recovery import migrate_summary_limits
from optchat.store import Store
from test_identity_recovery import SESSION, engine_at, message, tool_result, workspace


def strict_settings(boundary="event"):
    return Settings(summary_boundary=boundary, turn_node_bytes=512, summary_tries=1,
                    summary_concurrency=1, merge_lanes=1, settle_seconds=3,
                    summary_timeout=1, retry_seconds=0.01)


def coarsened_archive(workspace, settings):
    root, home, database = workspace
    originals = [f"Keep-current-{i}" for i in range(8)]
    originals[3] = "Superseded-private-3"
    originals[7] = "Withdrawn-private-7"
    for i, text in enumerate(originals):
        message(database, i + 1, f"message-{i}", text)
    old = Store(root / "old", SESSION, 128000, replace(settings, chunk_limit="sentences"))
    summaries = {}
    try:
        old.sync_hermes(home, SESSION)
        old.finish_turn()
        assert old.unit_count() == 8
        for i in range(8):
            summaries[i, 1] = originals[i] + " legacy summary " + "é" * 900
            old.save_node(i, 1, summaries[i, 1], {})
            for start, n, _ in old.parents(i):
                summaries[start, n] = " / ".join(originals[start:start + n]) + " legacy summary " + "é" * 900
                old.save_node(start, n, summaries[start, n], {})
        with old.lock, old.db:
            old.fit(limit=1, next_message=True)
        assert old.parts() == [(0, 8)]
        source = old.root / "chat.db"
    finally:
        old.close()
    return source, originals, summaries


def migration_summary(calls):
    def summarize(messages):
        calls.append(messages)
        return SimpleNamespace(text="Current bounded memory; zoom for original instructions.",
                               model="fixture", provider="none", usage=Usage())
    return summarize


def assert_audit_children(result, start, n, summaries, source_archive, unit):
    expected = []
    for child in (start, start + n // 2):
        item = {"id": child, "n": n // 2, "text": summaries[child, n // 2],
                "historical": True, "source_archive": source_archive}
        if unit == "leaf":
            item["unit"] = "leaf"
        expected.append(item)
    assert result["children"] == expected


@pytest.mark.parametrize("boundary,unit", [("event", "event"), ("user_turn", "leaf")])
def test_migrated_zoom_preserves_pruned_children_after_withdrawal_and_edit(workspace, engine_at, boundary, unit):
    _, home, database = workspace
    settings = strict_settings(boundary)
    source, originals, summaries = coarsened_archive(workspace, settings)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    migration_calls = []
    report = migrate_summary_limits(source, home / "optchat", settings=settings,
                                    summarize=migration_summary(migration_calls))
    assert len(migration_calls) == 1, "Only the coarsened frontier needs recompression"
    audit_source = Path(report["source_archive"])
    assert audit_source.is_file() and audit_source != source
    audit_hash = hashlib.sha256(audit_source.read_bytes()).hexdigest()
    retrieval_calls = []

    def no_regeneration(messages):
        retrieval_calls.append(messages)
        raise AssertionError("Preserved audit children must not be regenerated")

    engine = engine_at(settings, summarizer=no_regeneration)
    assert engine.worker is not None, engine.problem
    store = engine.worker.store
    assert store.parts() == [(0, 8)]
    assert store.node(0, 4) is None and store.node(4, 4) is None
    for start, n in ((0, 8), (4, 4), (6, 2)):
        result = tool_result(engine, "optchat_zoom", {"id": start, "n": n, "unit": unit})
        assert_audit_children(result, start, n, summaries, str(audit_source), unit)
    assert retrieval_calls == []
    assert store.node(0, 4) is None and store.node(4, 4) is None

    database.execute("UPDATE messages SET content=? WHERE id=4", ("Edited-current-only",))
    database.execute("UPDATE messages SET active=0,compacted=0 WHERE id=8")
    database.commit()
    message(database, 9, "fresh-current", "Fresh current question")
    current = {"role": "user", "content": "Fresh current question", "_row_id": 9}
    engine.begin_turn("after-edit", [current], user_message=current["content"])
    selected = engine.select_context([current], conversation_messages=[current], incoming_message=current)
    selected_text = json.dumps(selected, ensure_ascii=False)
    assert "Edited-current-only" in selected_text
    for retired in (originals[3], originals[7]):
        assert retired not in selected_text
        assert tool_result(engine, "optchat_search", {"query": retired})["matches"] == []
    assert len(engine.view.encode()) <= store.byte_budget
    engine.on_turn_complete([current])
    current_view = engine.worker.settle(store.count())
    assert len(current_view.encode()) <= store.byte_budget
    assert all(len(store.node(start, n).encode()) <= 512 for start, n in store.parts())

    historical = tool_result(engine, "optchat_zoom", {"id": 0, "n": 8, "unit": unit})
    assert_audit_children(historical, 0, 8, summaries, str(audit_source), unit)
    if unit == "leaf":
        leaf = tool_result(engine, "optchat_zoom", {"id": 7, "n": 1, "unit": "leaf"})
        assert leaf["text"] == summaries[7, 1]
        assert leaf["historical"] is True and leaf["source_archive"] == str(audit_source)
        assert [event["id"] for event in leaf["events"]] == [7]
    for event in (3, 7):
        assert tool_result(engine, "optchat_zoom", {"id": event, "n": 1})["text"] == originals[event]
    edited = store.event_for_row(4)
    assert edited != 3
    assert tool_result(engine, "optchat_zoom", {"id": edited, "n": 1})["text"] == "Edited-current-only"
    assert retrieval_calls == []
    assert store.node(0, 4) is None and store.node(4, 4) is None
    assert store.render() == current_view
    for retired in (originals[3], originals[7]):
        assert retired not in store.render()
        assert tool_result(engine, "optchat_search", {"query": retired})["matches"] == []
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert hashlib.sha256(audit_source.read_bytes()).hexdigest() == audit_hash


@pytest.mark.parametrize("boundary,unit", [("event", "event"), ("user_turn", "leaf")])
def test_previous_migrated_generation_zooms_its_pruned_audit_children(workspace, engine_at, boundary, unit):
    root, home, _ = workspace
    settings = strict_settings(boundary)
    source, originals, summaries = coarsened_archive(workspace, settings)
    first_calls = []
    first = migrate_summary_limits(source, root / "intermediate", settings=settings,
                                   summarize=migration_summary(first_calls))
    assert len(first_calls) == 1
    intermediate = Path(first["destination"]) / "chat.db"
    second_calls = []
    second = migrate_summary_limits(intermediate, home / "optchat", settings=settings,
                                    summarize=migration_summary(second_calls))
    assert second_calls == [], "An already strict cover must not be recompressed"
    snapshots = [Path(first["source_archive"]), Path(second["source_archive"])]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshots}
    engine = engine_at(settings)
    assert engine.worker is not None, engine.problem
    assert engine.worker.store.node(0, 4) is None

    for generation in (1, 2):
        args = {"id": 0, "n": 8, "unit": unit, "archive": "previous", "generation": generation}
        result = tool_result(engine, "optchat_zoom", args)
        assert result["archive"] == "previous" and result["generation"] == generation
        assert result["source_archive"] == str(snapshots[-generation])
        assert [(child["id"], child["n"], child["text"]) for child in result["children"]] == [
            (0, 4, summaries[0, 4]), (4, 4, summaries[4, 4]),
        ]
        if generation == 1:
            assert_audit_children(result, 0, 8, summaries, first["source_archive"], unit)
        original = tool_result(engine, "optchat_zoom", {
            "id": 7, "n": 1, "archive": "previous", "generation": generation,
        })
        assert original["text"] == originals[7]
    assert engine.worker.store.node(0, 4) is None
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == checksum for path, checksum in before.items())


@pytest.mark.parametrize("provider_fails", [False, True])
def test_missing_child_without_audit_still_builds_or_fails_closed(workspace, engine_at, provider_fails):
    _, _, database = workspace
    original = "Uncached-original-" + "é" * 900
    message(database, 1, "first", original)
    message(database, 2, "second", original)
    settings = replace(strict_settings(), compression_policy="on_demand",
                       settle_seconds=0.3 if provider_fails else 3, retry_seconds=1)
    calls = []

    def summarize(messages):
        calls.append(messages)
        if provider_fails:
            raise ConnectionError("fixture provider unavailable")
        return SimpleNamespace(text="Generated current retrieval child.", model="fixture",
                               provider="none", usage=Usage())

    engine = engine_at(settings, summarizer=summarize)
    assert engine.worker is not None, engine.problem
    result = tool_result(engine, "optchat_zoom", {"id": 0, "n": 2})
    assert calls
    if provider_fails:
        assert "Retrieval summaries did not settle" in result["error"]
        assert "fixture provider unavailable" in result["error"]
        assert "children" not in result and "_multimodal" not in result
    else:
        assert result == {"children": [
            {"id": 0, "n": 1, "text": "Generated current retrieval child."},
            {"id": 1, "n": 1, "text": "Generated current retrieval child."},
        ]}
    assert tool_result(engine, "optchat_zoom", {"id": 0, "n": 1})["text"] == original
