"""Consumer-visible readiness cancellation/progress contracts; no provider calls."""
from dataclasses import dataclass
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pytest

from optchat.config import Settings
from optchat.limits import SummaryLimit
from optchat.store import Store
from optchat.worker import Worker

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
    def workspace(self, tmp_path):
        self.worker_root = tmp_path
        self.worker_count = 0

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

    def test_cancelled_readiness_unwinds_before_slow_summary_finishes(self):
        started, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        def summarize(messages):
            started.set()
            release.wait(3)
            return SimpleNamespace(text="user: Retain the historical instruction.",
                                   model="fixture", provider="none", usage=Usage())
        store, worker = self.make_worker(1, summarize)
        try:
            self.assertTrue(started.wait(1), "Summary never began")
            timer = threading.Timer(0.03, cancelled.set)
            timer.start()
            began = time.monotonic()
            with self.assertRaises(KeyboardInterrupt):
                worker.settle(1, cancelled=cancelled.is_set)
            self.assertLess(time.monotonic() - began, 0.6)
            self.assertFalse(release.is_set(), "Cancellation waited for provider completion")
            self.assertEqual(store.zoom(0, 1)["text"], "Historical instruction 0. " * 80)
            timer.join()
        finally:
            release.set()
            worker.close()

    def test_progress_reports_committed_work_not_wait_ticks(self):
        started, release = threading.Event(), threading.Event()
        def summarize(messages):
            started.set()
            release.wait(3)
            return SimpleNamespace(text="user: Retain the historical instruction.",
                                   model="fixture", provider="none", usage=Usage())
        store, worker = self.make_worker(1, summarize)
        reports, idle_reports = [], []
        try:
            self.assertTrue(started.wait(1))
            def unlock():
                time.sleep(0.55)
                idle_reports.extend(reports)
                release.set()
            releaser = threading.Thread(target=unlock)
            releaser.start()
            view = worker.settle(1, progress=reports.append)
            releaser.join()
            self.assertIn("0+1|user: Retain the historical instruction.", view)
            self.assertEqual(idle_reports, [], "Idle waits must not refresh the watchdog")
            self.assertEqual(len(reports), 1, reports)
            self.assertIn("1 summaries committed", reports[0])
        finally:
            release.set()
            worker.close()

    def test_readiness_timeout_keeps_actionable_backlog_and_original(self):
        started, release = threading.Event(), threading.Event()
        def summarize(messages):
            started.set()
            release.wait(3)
            return SimpleNamespace(text="user: Retain the historical instruction.",
                                   model="fixture", provider="none", usage=Usage())
        store, worker = self.make_worker(1, summarize)
        try:
            self.assertTrue(started.wait(1))
            # Short fixture timeout; production keeps its configured deadline.
            from dataclasses import replace
            worker.settings = replace(worker.settings, settle_seconds=0.05)
            with self.assertRaisesRegex(RuntimeError, r"pending|backlog"):
                worker.settle(1)
            self.assertEqual(store.zoom(0, 1)["text"], "Historical instruction 0. " * 80)
        finally:
            release.set()
            worker.close()

    def test_already_committed_view_and_retrieval_do_not_report_activity(self):
        started, release = threading.Event(), threading.Event()
        def summarize(messages):
            started.set()
            release.wait(3)
            return SimpleNamespace(text="user: A committed summary.",
                                   model="fixture", provider="none", usage=Usage())
        store, worker = self.make_worker(1, summarize)
        try:
            self.assertTrue(started.wait(1))
            release.set()
            with worker.changed:
                self.assertTrue(worker.changed.wait_for(lambda: worker.completed_nodes == 1, timeout=1))
            reports = []
            self.assertIn("0+1|user: A committed summary.", worker.settle(1, progress=reports.append))
            worker.ensure(0, 1, progress=reports.append)
            self.assertEqual(reports, [])
        finally:
            release.set()
            worker.close()

    def test_commit_during_readiness_is_reported_before_return(self):
        started, release = threading.Event(), threading.Event()
        def summarize(messages):
            started.set()
            release.wait(3)
            return SimpleNamespace(text="user: A committed summary.",
                                   model="fixture", provider="none", usage=Usage())
        store, worker = self.make_worker(1, summarize)
        try:
            self.assertTrue(started.wait(1))
            render = store.render
            def commit_then_render(end):
                worker.save_node(0, 1, "user: A committed summary.", {"fixture": True})
                return render(end)
            reports = []
            with patch.object(store, "render", side_effect=commit_then_render):
                self.assertIn("0+1|user: A committed summary.", worker.settle(1, progress=reports.append))
            self.assertEqual(len(reports), 1, reports)
            self.assertIn("1 summaries committed", reports[0])
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
                    settings = Settings(compression_policy=policy, summary_concurrency=1, merge_lanes=1,
                                        settle_seconds=2, summary_timeout=1, retry_seconds=0.01, summary_tries=1)
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
            with self.subTest(byte_budget=byte_budget):
                def summarize(messages):
                    return SimpleNamespace(text="", model="fixture", provider="none", usage=Usage())
                settings = Settings(summary_concurrency=1, merge_lanes=1, settle_seconds=2,
                                    summary_timeout=1, retry_seconds=0.01, summary_tries=1)
                original = "é" * 130000
                store, worker = self.make_worker(1, summarize, settings=settings,
                                                byte_budget=byte_budget, source=original)
                try:
                    view = worker.settle(1)
                    node = store.node(0, 1)
                    self.assertIn("0+1", node)
                    self.assertTrue(store.can_copy(node))
                    self.assertLessEqual(len(node.encode()), store.copy_limit)
                    self.assertLessEqual(len(view.encode()), store.byte_budget)
                    self.assertEqual(original_text(store,0), original)
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

    def test_summary_limits_preserve_unit_target_and_whole_reference(self):
        for unit, target in (("bytes", 512), ("characters", 500), ("sentences", 5)):
            with self.subTest(unit=unit):
                limit = SummaryLimit(unit, target)
                self.assertFalse(limit.can_copy("x" * 130000, max_bytes=1024))
                fallback = limit.fallback("é" * 130000, "123456+1024", max_bytes=1024)
                self.assertIn("123456+1024", fallback)
                self.assertTrue(limit.can_copy(fallback, max_bytes=1024))
                with self.assertRaises(ValueError):
                    limit.fallback("source", "123456+1024", max_bytes=8)

