"""Resumed coarse covers remain useful without replaying pruned descendants."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from optchat.config import Settings
from optchat.engine import Usage
from optchat.recovery import migrate_summary_limits
from optchat.store import Store
from optchat.worker import Worker


def test_migrated_coldresume_uses_retained_cover_for_the_next_summary(tmp_path):
    settings = Settings(chunk_limit="sentences", summary_tries=1, cache_split=0)
    old = Store(tmp_path / "old", "resume", 128000, settings)
    oldest = "user: Keep the oldest deployment constraint."
    historical = "user: Deploy blue, never green; future-only instruction seven applies to the complete cover."
    try:
        for i in range(8):
            old.append(f"message:{i}", "user", f"RAW_ONLY_ORIGINAL_{i} " * 100)
            text = {0: oldest, 2: "user: Do not skip the missing earlier instruction.",
                    7: "user: FUTURE_LEAF_SEVEN_PRIVATE"}.get(i, "é" * 900)
            old.save_node(i, 1, text, {})
            for start, n, _ in old.parents(i):
                old.save_node(start, n, "é" * 900, {})
        with old.lock, old.db:
            old.fit(limit=1, next_message=True)
        assert old.parts() == [(0, 8)]
        source = old.root / "chat.db"
    finally:
        old.close()

    def migrate(messages):
        return SimpleNamespace(text=historical, model="fixture", provider="none", usage=Usage())

    strict = replace(settings, chunk_limit="bytes")
    migrate_summary_limits(source, tmp_path / "new", settings=strict, summarize=migrate)
    # Reopening the published generation, rather than retaining migration state,
    # is what previously reset the completed-leaf cursor in front of a valid cover.
    store = Store(tmp_path / "new", "resume", 128000, strict)
    calls = []

    def summarize(messages):
        calls.append(messages)
        return SimpleNamespace(text="user: Apply the deployment constraint to the new request.",
                               model="fixture", provider="none", usage=Usage())

    worker = Worker(store, strict, summarize, start=False)
    try:
        assert store.parts() == [(0, 8)]
        assert store.node(0, 1) == oldest
        assert store.node(1, 1) is None
        assert store.node(0, 4) is None
        assert store.render() == f"<chat>\n0+8|{historical}\n</chat>"
        # A boundary inside the retained parent must descend missing parents,
        # stop at the first absent leaf, and not resume at a later cached sibling.
        assert store.summary_lines(0) == []
        assert store.summary_lines(6) == [oldest]
        with pytest.raises(RuntimeError, match="Summary not ready"):
            store.render(6)
        assert store.summary_lines(8) == [historical]

        store.append("message:8", "user", "NEW_REQUEST_REQUIRING_CONTEXT " * 100)
        store.prepare(store.count())
        with pytest.raises(RuntimeError, match="Summary not ready"):
            store.render()
        job = store.ready_job("leaf")
        assert job is not None and job[:2] == (8, 1)
        worker.build(*job)
        assert calls, "The new oversized message must reach the real Worker model seam"
        prompt = "\n".join(message["content"] for message in calls[-1])
        context = prompt.split("<chat>\n", 1)[1].split("</chat>", 1)[0]
        assert context.strip() == historical
        assert "RAW_ONLY_ORIGINAL" not in context
        assert "NEW_REQUEST_REQUIRING_CONTEXT" not in context
        assert store.node(1, 1) is None, "Coldresume must not replay hidden historical leaves"
        assert "8+1|user: Apply the deployment constraint" in store.render()
        assert len(store.render().encode("utf-8")) <= store.byte_budget
    finally:
        worker.close()


def test_migrated_selected_siblings_supply_missing_parent_work_under_byte_pressure(tmp_path):
    settings = Settings(chunk_limit="sentences", summary_tries=1, cache_split=0)
    old = Store(tmp_path / "old", "siblings", 128000, settings)
    try:
        for i in range(8):
            old.append(f"message:{i}", "user", f"Historical original {i}: " + "é" * 900)
            old.save_node(i, 1, "é" * 900, {})
            for start, n, _ in old.parents(i):
                if n == 2:
                    old.save_node(start, n, f"user: Historical pair {start}-{start+1}: ".ljust(500, "x"), {})
        with old.lock, old.db:
            old.fit(limit=1, next_message=True)
        assert old.parts() == [(0, 2), (2, 2), (4, 2), (6, 2)]
        source = old.root / "chat.db"
    finally:
        old.close()

    def no_recompression(messages):
        raise AssertionError("The selected pair summaries already conform to the byte limit")

    strict = replace(settings, chunk_limit="bytes")
    migrate_summary_limits(source, tmp_path / "new", settings=strict, budget=2048,
                           summarize=no_recompression)
    store = Store(tmp_path / "new", "siblings", 2048, strict)
    calls = []
    current = "user: Handle the current request. " + "n" * 180
    merged = "user: Retain the combined historical constraints zero through three."
    responses = iter([current, merged])

    def summarize(messages):
        calls.append(messages)
        return SimpleNamespace(text=next(responses), model="fixture", provider="none", usage=Usage())

    worker = Worker(store, strict, summarize, start=False)
    try:
        store.prepare(store.count())
        assert [(job[0], job[1]) for job in store.ready_jobs("parent")] == [(0, 4), (4, 4)]
        assert store.ready_job("leaf") is None, "Selected cached pairs must not replay their pruned leaves"
        assert store.node(0, 1) is None

        store.append("message:8", "user", "NEW_REQUEST_OVER_BYTE_PRESSURE " * 100)
        store.prepare(store.count())
        leaf = store.ready_job("leaf")
        assert leaf is not None and leaf[:2] == (8, 1)
        worker.build(*leaf)
        assert len(store.render().encode("utf-8")) > store.byte_budget
        parent = store.ready_job("parent")
        assert parent is not None and parent[:2] == (0, 4)
        worker.build(*parent)
        merge_prompt = "\n".join(message["content"] for message in calls[-1])
        assert "Merge these two lines" in merge_prompt
        assert "user: Historical pair 0-1:" in merge_prompt
        assert "user: Historical pair 2-3:" in merge_prompt

        store.prepare(store.count(), next_message=True)
        view = store.render()
        assert store.parts() == [(0, 4), (4, 2), (6, 2), (8, 1)]
        assert f"0+4|{merged}" in view
        assert f"8+1|{current}" in view
        assert len(view.encode("utf-8")) <= store.byte_budget
        assert store.ready_job("leaf") is None
        assert store.node(0, 1) is None
    finally:
        worker.close()
