"""Consumer-visible readiness cancellation/progress contracts; no provider calls."""
from dataclasses import dataclass
import threading
from types import SimpleNamespace
import unittest

import pytest

from optchat.config import Settings
from optchat.store import Store
from optchat.worker import Worker
from agent.context_engine import ContextEngineError
from test_identity_recovery import engine_at, message, workspace

def original_text(store, event):
    parts, offset = [], 0
    while True:
        page = store.zoom(event,1,offset)
        parts.append(page["text"])
        offset = page["next_offset"]
        if offset is None:
            return "".join(parts)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

class ReadinessTests(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def worker_workspace(self, tmp_path, workspace, engine_at):
        self.worker_root = tmp_path
        self.worker_count = 0
        self.hermes_workspace = workspace
        self.engine_at = engine_at

    def make_worker(self, source_count, summarize, *, settings=None, byte_budget=128000, source=None):
        root = self.worker_root / f"worker-{self.worker_count}"
        self.worker_count += 1
        settings = settings or Settings(chunk_limit="bytes", batching="off", summary_concurrency=1,
                                        merge_lanes=1, settle_seconds=2, summary_timeout=1,
                                        retry_seconds=0.01, summary_tries=1)
        store = Store(root, "readiness", byte_budget, settings)
        for i in range(source_count):
            store.append(f"message:{i}", "user", source or f"Historical instruction {i}. " * 80)
        worker = Worker(store, settings, summarize)
        return store, worker

    def make_selection(self, summarize, *, settle_seconds=10):
        _, _, database = self.hermes_workspace
        original = "Historical instruction. " * 500
        message(database, 1, "history", original)
        message(database, 2, "current", "current task")
        settings = Settings(
            compression_policy="on_demand", chunk_limit="bytes", view_bytes=4096,
            summary_concurrency=1, merge_lanes=1, settle_seconds=settle_seconds,
            summary_timeout=10, retry_seconds=0.01, summary_tries=1,
        )
        engine = self.engine_at(settings, summarize)
        history = {"role": "user", "content": original, "_row_id": 1}
        current = {"role": "user", "content": "current task", "_row_id": 2}
        conversation = [history, current]
        engine.begin_turn("readiness-turn", conversation, user_message="current task")
        return engine, conversation, current, original

    def test_cancelled_readiness_unwinds_before_slow_summary_finishes(self):
        started, release = threading.Event(), threading.Event()
        waiting, cancelled, finished = threading.Event(), threading.Event(), threading.Event()
        outcome = []

        def summarize(messages):
            started.set()
            release.wait(10)
            return SimpleNamespace(text="user: Retain the historical instruction.",
                                   model="fixture", provider="none", usage=Usage())

        engine, conversation, current, original = self.make_selection(summarize)

        def cancellation():
            value = cancelled.is_set()
            if started.is_set() and not value:
                waiting.set()
            return value

        def select():
            try:
                engine.select_context(
                    conversation, conversation_messages=conversation,
                    incoming_message=current, cancelled=cancellation,
                )
            except BaseException as exc:
                outcome.append(exc)
            finally:
                finished.set()

        caller = threading.Thread(target=select)
        caller.start()
        try:
            self.assertTrue(started.wait(2), "Summary never began")
            self.assertTrue(waiting.wait(2), "Selection never observed the running summary")
            cancelled.set()
            self.assertTrue(finished.wait(2), "Cancellation waited for provider completion")
            self.assertEqual(len(outcome), 1)
            self.assertIsInstance(outcome[0], KeyboardInterrupt)
            self.assertFalse(release.is_set())
            self.assertEqual(engine.worker.store.zoom(0, 1)["text"], original)
        finally:
            release.set()
            caller.join(10)

    def test_progress_reports_committed_work_not_wait_ticks(self):
        started, release = threading.Event(), threading.Event()

        def summarize(messages):
            started.set()
            release.wait(10)
            return SimpleNamespace(text="user: Retain the historical instruction.",
                                   model="fixture", provider="none", usage=Usage())

        store, worker = self.make_worker(1, summarize)
        reports, idle_reports = [], []
        checks = 0

        def unblock_after_idle():
            nonlocal checks
            checks += 1
            if checks == 3:
                idle_reports.extend(reports)
                release.set()
            return False

        try:
            self.assertTrue(started.wait(2))
            view = worker.settle(1, progress=reports.append, cancelled=unblock_after_idle)
            self.assertIn("Retain the historical instruction.", view)
            self.assertEqual(idle_reports, [])
            self.assertEqual(len(reports), 1)
        finally:
            release.set()
            worker.close()

    def test_readiness_timeout_is_fail_closed_and_keeps_original(self):
        release = threading.Event()

        def summarize(messages):
            release.wait(10)
            return SimpleNamespace(text="user: Retain the historical instruction.",
                                   model="fixture", provider="none", usage=Usage())

        engine, conversation, current, original = self.make_selection(summarize, settle_seconds=0.05)
        try:
            with self.assertRaises(ContextEngineError):
                engine.select_context(
                    conversation, conversation_messages=conversation, incoming_message=current,
                )
            self.assertEqual(engine.worker.store.zoom(0, 1)["text"], original)
        finally:
            release.set()

    def test_already_committed_view_and_retrieval_do_not_report_activity(self):
        started, release = threading.Event(), threading.Event()
        def summarize(messages):
            started.set()
            release.wait(3)
            return SimpleNamespace(text="user: A committed summary.",
                                   model="fixture", provider="none", usage=Usage())
        store, worker = self.make_worker(1, summarize)
        try:
            self.assertTrue(started.wait(2))
            release.set()
            self.assertIn("A committed summary.", worker.settle(1))
            reports = []
            self.assertIn("A committed summary.", worker.settle(1, progress=reports.append))
            worker.ensure(0, 1, progress=reports.append)
            self.assertEqual(reports, [])
        finally:
            release.set()
            worker.close()



    def test_sentence_copy_requires_framed_view_byte_allowance(self):
        for policy in ("eager", "on_demand"):
            for byte_budget in (128000, 4096):
                with self.subTest(policy=policy, byte_budget=byte_budget):
                    calls = []
                    def summarize(messages):
                        calls.append(messages)
                        return SimpleNamespace(text="user: A very long original instruction; zoom 0+1 for exact text.",
                                               model="fixture", provider="none", usage=Usage())
                    settings = Settings(compression_policy=policy, chunk_limit="sentences",
                                        summary_concurrency=1, merge_lanes=1, settle_seconds=2,
                                        summary_timeout=1, retry_seconds=0.01, summary_tries=1)
                    original = "x" * 130000
                    store, worker = self.make_worker(1, summarize, settings=settings,
                                                    byte_budget=byte_budget, source=original)
                    try:
                        view = worker.settle(1)
                        self.assertEqual(len(calls), 1, "A one-sentence oversized original still needs compression")
                        self.assertLessEqual(len(store.node(0, 1).encode()), store.copy_limit)
                        self.assertLessEqual(len(view.encode()), store.byte_budget)
                        self.assertEqual(original_text(store,0), original)
                    finally:
                        worker.close()

    def test_empty_summary_fallback_keeps_exact_zoom_and_fits_view(self):
        for byte_budget in (128000, 4096):
            for unit in ("bytes", "characters", "sentences"):
                with self.subTest(byte_budget=byte_budget, unit=unit):
                    def summarize(messages):
                        return SimpleNamespace(text="", model="fixture", provider="none", usage=Usage())
                    settings = Settings(
                        chunk_limit=unit, summary_concurrency=1, merge_lanes=1, settle_seconds=2,
                        summary_timeout=1, retry_seconds=0.01, summary_tries=1,
                    )
                    original = "é" * 130000
                    store, worker = self.make_worker(
                        1, summarize, settings=settings, byte_budget=byte_budget, source=original,
                    )
                    try:
                        view = worker.settle(1)
                        self.assertIn("0+1", store.node(0, 1))
                        self.assertLessEqual(len(view.encode()), byte_budget)
                        self.assertEqual(original_text(store, 0), original)
                    finally:
                        worker.close()

    def test_oversized_generated_summary_uses_only_configured_attempts(self):
        calls = []
        def summarize(messages):
            calls.append(messages)
            return SimpleNamespace(text="z" * 130000, model="fixture", provider="none", usage=Usage())
        settings = Settings(summary_concurrency=1, merge_lanes=1, settle_seconds=2,
                            summary_timeout=1, retry_seconds=0.01, summary_tries=2)
        store, worker = self.make_worker(1, summarize, settings=settings,
                                        byte_budget=4096, source="x" * 130000)
        try:
            view = worker.settle(1)
            self.assertEqual(len(calls), settings.summary_tries)
            self.assertIn("0+1", store.node(0, 1))
            self.assertTrue(store.can_copy(store.node(0, 1)))
            self.assertLessEqual(len(view.encode()), store.byte_budget)
        finally:
            worker.close()


