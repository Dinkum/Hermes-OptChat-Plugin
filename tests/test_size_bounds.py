"""Strict UTF-8 summary bounds through real worker/archive interfaces."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from optchat.config import Settings
from optchat.store import Store
from optchat.worker import Worker
from test_identity_recovery import engine_at, workspace


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@pytest.fixture
def summarize_history(tmp_path, workspace, engine_at):
    workers = []

    def prepare(original, responses, *, target=512):
        settings = Settings(
            chunk_limit="bytes", node_bytes=target, batching="off",
            summary_concurrency=1, merge_lanes=1, settle_seconds=2,
            summary_timeout=1, retry_seconds=0.01, summary_tries=2,
        )
        store = Store(tmp_path / f"summary-{len(workers)}", "size-bounds", 128000, settings)
        store.append("message:0", "user", original)
        replies = iter(responses)

        def summarize(messages):
            return SimpleNamespace(
                text=next(replies, responses[-1]), model="fixture", provider="none", usage=Usage(),
            )

        worker = Worker(store, settings, summarize)
        workers.append(worker)
        worker.ensure(0, 1)
        return store

    try:
        yield prepare
    finally:
        for worker in reversed(workers):
            worker.close()


@pytest.mark.parametrize("original", ["x" * 507, "é" * 252 + "界"])
def test_original_above_512_utf8_bytes_is_not_copied_as_a_summary(summarize_history, original):
    # The visible "user: " prefix adds six bytes, making either source 513 bytes.
    summary = "user: Preserve the original instruction; zoom 0+1 for exact text."
    store = summarize_history(original, [summary])
    assert store.node(0, 1) == summary
    assert store.zoom(0, 1)["text"] == original


@pytest.mark.parametrize("target,candidate", [
    (256, "x" * 257),
    (512, "x" * 513),
    (512, "é" * 255 + "界"),
    (512, "😀" * 128 + "x"),
    (2048, "x" * 2049),
])
def test_generated_summary_above_its_byte_target_requires_rewriting(summarize_history, target, candidate):
    original = "Preserve this historical instruction. " * 100
    summary = "user: Preserve the historical instruction; zoom 0+1 for exact text."
    store = summarize_history(original, [candidate, summary], target=target)
    assert store.node(0, 1) == summary
    assert store.zoom(0, 1)["text"] == original


def test_multibyte_original_at_exactly_512_bytes_is_preserved(summarize_history):
    # Six prefix bytes + 251 two-byte characters + one four-byte character = 512.
    original = "é" * 251 + "😀"
    store = summarize_history(original, ["Unexpected replacement summary."])
    assert store.node(0, 1) == "user: " + original
    assert store.zoom(0, 1)["text"] == original


def test_unusable_multibyte_summary_keeps_a_retrievable_bounded_view(summarize_history):
    original = "a" + "😀" * 500
    store = summarize_history(original, ["x" * 513])
    summary = store.node(0, 1)
    assert len(summary.encode("utf-8")) <= 512
    assert "\ufffd" not in summary
    assert len(store.render().encode("utf-8")) <= 128000
    assert store.zoom(0, 1)["text"] == original


