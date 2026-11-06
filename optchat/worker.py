import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

from . import batch
from .chunks import capped
from .store import VIEW_ACK, encoded, size

logger = logging.getLogger(__name__)
COMPACT = Path(__file__).with_name("compact.txt").read_text()
TURN_COMPACT = Path(__file__).with_name("compact-turn.txt").read_text()


class Worker:
    def __init__(self, store, settings, summarize, *, on_close=None):
        self.store, self.settings, self.summarize = store, settings, summarize
        self.store.configure(settings)
        self.on_close = on_close
        self.changed = threading.Condition()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.errors = {}
        self.busy = set()
        self.snapshots = {}
        self.slots = threading.BoundedSemaphore(settings.summary_concurrency)
        self.context_head = None
        self.claim = threading.Lock()
        self.waiters = 0
        self.completed_nodes = 0
        from agent.memory_provider import spawn_context_thread
        leaves = [(f"leaf-{k}","leaf") for k in range(settings.summary_concurrency)]
        if settings.batching == "auto":
            lanes = [(f"batch-{k}",None) for k in range(settings.summary_concurrency)]
        else:
            lanes = leaves+[(f"parent-{k}","parent") for k in range(settings.merge_lanes)]
        self.running = len(lanes)
        self.threads = [spawn_context_thread(lambda lane=lane,kind=kind:self.run(lane,kind),name="optchat-"+lane)
                        for lane,kind in lanes]
        for thread in self.threads:
            thread.start()

    @property
    def error(self):
        return "; ".join(dict.fromkeys(self.errors.copy().values())) or None

    def notify(self):
        if self.store.settings.compression_policy == "on_demand" and self.store.history_end is not None:
            self.store.prepare(self.store.history_end)
        self.wake.set()

    def save_node(self, start, n, text, metrics):
        if not self.store.can_copy(text):
            raise RuntimeError("Summary exceeds its configured length or framed-view byte allowance")
        if self.store.save_node(start,n,text,metrics):
            with self.changed:
                self.completed_nodes += 1
                self.changed.notify_all()

    def check_cancelled(self, cancelled):
        if cancelled is not None and cancelled():
            raise KeyboardInterrupt("OptChat context preparation cancelled")
        if self.stop.is_set():
            raise RuntimeError("OptChat is shutting down")

    def report_progress(self, progress, previous, *, ready=False):
        with self.changed:
            revision = self.completed_nodes
        if progress is not None and revision > previous:
            with self.store.lock:
                pending = self.store.db.execute("SELECT count(*) FROM jobs").fetchone()[0]
            state = "ready" if ready else "preparing"
            progress(f"OptChat history {state}: {revision} summaries committed; {pending} pending nodes")
        return revision

    @property
    def prompt(self):
        prompt = TURN_COMPACT if self.settings.summary_boundary == "user_turn" else COMPACT
        return self.settings.summary_limit.prompt(prompt)

    def source(self, source, kind):
        if kind in ("tool","echo"):
            source = capped(source,self.settings.tool_chars)
        return source

    def description(self, start, n, source, kind=None, prepared=False):
        with self.store.lock:
            context = "\n".join(self.store.summary_lines(start))
        limit = self.settings.summary_limit
        return batch.SummaryJob(f"{start}+{n}",context,source if prepared else self.source(source,kind),limit.output_bytes,
                                "Compress this completed turn chunk" if kind == "turn" else "Compress this message" if n == 1 else "Merge these two lines",
                                limit.unit,limit.target)

    def build(self, start, n, source, kind=None):
        source = self.source(source,kind)
        limit = self.settings.summary_limit
        if self.store.can_copy(source):
            self.save_node(start,n,source,{"free":True})
            return
        job = getattr(self,"snapshots",{}).get((start,n)) or self.description(start,n,source,kind,prepared=True)
        context = job.context
        step = limit.instruction(job.action,source)
        chat = f"<chat>\n{context}\n"
        head = self.context_head = self.store.cache_head("summary_cache_head",chat,self.settings.cache_split)
        if head:
            # Same boundary as the main view: calls share the context's head as a cacheable message.
            messages = [{"role":"system","content":self.prompt},{"role":"user","content":head},
                        {"role":"assistant","content":VIEW_ACK},
                        {"role":"user","content":f"{chat[len(head):]}</chat>\n\n{step}"}]
        else:
            messages = [{"role":"system","content":self.prompt},
                        {"role":"user","content":f"{chat}</chat>\n\n{step}"}]
        base = messages
        tries = []
        for _ in range(self.settings.summary_tries):
            # Finish an in-flight call, but leave the job queued instead of starting
            # another request during shutdown. Originals and attempts stay durable.
            if self.stop.is_set():
                return
            began = time.monotonic()
            result = self.summarize(messages)
            text = result.text.strip()
            metrics = {"seconds":time.monotonic()-began,"bytes":size(text),
                       "limit_unit":limit.unit,"target_length":limit.target,"length":limit.measure(text),
                       "model":result.model,"provider":result.provider,"usage":asdict(result.usage)}
            self.store.record_attempt(start,n,metrics)
            if not text:
                if tries:
                    break
                fallback = limit.fallback(source,f"{start}+{n}",max_bytes=self.store.copy_limit)
                self.save_node(start,n,fallback,{"fallback":"empty_response","tries":1})
                return
            tries.append(text)
            if self.store.can_copy(text):
                break
            feedback = limit.feedback(text)
            if size(text) > self.store.copy_limit:
                feedback += f" It is {size(text)} UTF-8 bytes; the view allows at most {self.store.copy_limit} bytes per summary."
            messages = base+[{"role":"user","content":feedback}]
        accepted = [text for text in tries if self.store.can_copy(text)]
        if accepted:
            self.save_node(start,n,min(accepted,key=lambda text:(limit.measure(text),size(text))),{"tries":len(tries)})
        else:
            fallback = limit.fallback(source,f"{start}+{n}",max_bytes=self.store.copy_limit)
            self.save_node(start,n,fallback,{"fallback":"oversized_response","tries":len(tries)})

    def take_jobs(self, kind):
        """Claim a resource-bounded batch from one committed snapshot."""
        if self.settings.batching == "off" or not hasattr(self.summarize,"summarize_many"):
            job = self.store.ready_job(kind,self.busy)
            return [job] if job else []
        candidates = self.store.ready_jobs(kind,self.busy)
        if not candidates:
            return []
        chosen, descriptions = [], []
        for job in sorted(candidates,key=lambda job:(job[0]+job[1],job[1])):
            # Do cheap work at its chronological position, not ahead of every
            # older model-backed job: large imports otherwise starve transport.
            if self.store.can_copy(self.source(job[2],job[3])):
                if not chosen:
                    return [job]
                continue
            description = self.description(*job)
            packed = descriptions+[description]
            if chosen and (size(encoded(batch.messages(packed,self.prompt))) > self.settings.batch_input_bytes
                           or sum(j.target_bytes+size(j.id)+64 for j in packed) > self.settings.batch_output_bytes):
                break
            chosen.append(job)
            descriptions.append(description)
            # A single huge job runs as a singleton; batching never truncates it.
            if len(chosen) == 1 and size(encoded(batch.messages(packed,self.prompt))) > self.settings.batch_input_bytes:
                break
        return chosen

    def build_many(self, jobs):
        if len(jobs) == 1:
            self.build(*jobs[0])
            return
        with self.store.lock:
            descriptions = [getattr(self,"snapshots",{}).get(job[:2]) or self.description(*job) for job in jobs]
        if (size(encoded(batch.messages(descriptions,self.prompt))) > self.settings.batch_input_bytes
            or sum(j.target_bytes+size(j.id)+64 for j in descriptions) > self.settings.batch_output_bytes):
            middle = len(jobs)//2
            self.build_many(jobs[:middle])
            if not self.stop.is_set():
                self.build_many(jobs[middle:])
            return
        began = time.monotonic()
        try:
            result = self.summarize.summarize_many(descriptions,self.prompt)
        except Exception:
            # An endpoint that cannot serve batches still has the singleton path.
            for job in jobs:
                if self.stop.is_set():
                    break
                self.build(*job)
            return
        texts = batch.results(result.text,{job.id for job in descriptions})
        usage = asdict(result.usage)
        batch_id = f"{descriptions[0].id}:{time.time_ns()}"
        for index,(start,n,_,_) in enumerate(jobs):
            text = texts.get(f"{start}+{n}","")
            # Account for the request once, even when it supplies several nodes.
            accounted = usage if index == 0 else {key:0 if key != "cost_status" else usage[key] for key in usage}
            self.store.record_attempt(start,n,{"seconds":time.monotonic()-began if index == 0 else 0,
                                      "bytes":size(text),"model":result.model,"provider":result.provider,
                                      "limit_unit":self.settings.chunk_limit,"target_length":self.settings.summary_limit.target,
                                      "length":self.settings.summary_limit.measure(text),
                                      "usage":accounted,"batch_id":batch_id,"batch_items":len(jobs),
                                      "usage_accounted":index == 0})
        unresolved = []
        for job in jobs:
            text = texts.get(f"{job[0]}+{job[1]}")
            if text and self.store.can_copy(text):
                self.save_node(job[0],job[1],text,{"batch_id":batch_id})
            else:
                unresolved.append(job)
        # Commit valid siblings first; repair only missing, duplicate or oversized items.
        for job in unresolved:
            if self.stop.is_set():
                break
            self.build(*job)

    def run(self, lane, kind):
        try:
            self.run_jobs(lane,kind)
        finally:
            with self.claim:
                self.running -= 1
                finished = self.running == 0 and self.stop.is_set()
            if finished:
                self.finish_close()

    def run_jobs(self, lane, kind):
        while not self.stop.is_set():
            jobs = []
            try:
                if self.settings.batching == "auto" and kind != "leaf" and not self.waiters:
                    # Bounded collection time is skipped when a request needs the work.
                    self.stop.wait(self.settings.batch_wait_seconds)
                    if self.stop.is_set():
                        break
                if not self.slots.acquire(timeout=0.1):
                    continue
                try:
                    with self.claim, self.store.lock:
                        jobs = self.take_jobs(kind)
                        self.busy.update(job[:2] for job in jobs)
                        for job in jobs:
                            self.snapshots[job[:2]] = self.description(*job)
                    if jobs:
                        self.build_many(jobs)
                        self.errors.pop(lane,None)
                finally:
                    self.slots.release()
                if jobs:
                    with self.claim:
                        self.busy.difference_update(job[:2] for job in jobs)
                        for job in jobs:
                            self.snapshots.pop(job[:2],None)
                    self.wake.set()
                    with self.changed:
                        self.changed.notify_all()
                    continue
            except Exception as exc:
                if self.errors.get(lane) != str(exc):
                    logger.error("OptChat summary failed; originals retained: %s", exc)
                self.errors[lane] = str(exc)
                with self.changed:
                    self.changed.notify_all()
                # The failed node stays claimed while it waits, so other lanes don't retry it early.
                self.stop.wait(self.settings.retry_seconds)
                if jobs:
                    with self.claim:
                        self.busy.difference_update(job[:2] for job in jobs)
                        for job in jobs:
                            self.snapshots.pop(job[:2],None)
                continue
            self.wake.wait(1)
            self.wake.clear()

    def settle(self, end, *, progress=None, cancelled=None):
        with self.changed:
            observed = self.completed_nodes
        self.check_cancelled(cancelled)
        self.store.seal(end)
        end = self.store.unit_end(end)
        self.store.prepare(end,next_message=True)
        return self.wait_for_view(end,progress=progress,cancelled=cancelled,_observed=observed)

    def wait_for_view(self, end, *, progress=None, cancelled=None, _observed=None):
        if _observed is None:
            with self.changed:
                _observed = self.completed_nodes
        with self.claim:
            self.waiters += 1
        try:
            return self.await_view(end,_observed,progress=progress,cancelled=cancelled)
        finally:
            with self.claim:
                self.waiters -= 1

    def await_view(self, end, observed, *, progress=None, cancelled=None):
        self.notify()
        deadline = time.monotonic()+self.settings.settle_seconds
        reported = observed
        fitted = None
        while True:
            self.check_cancelled(cancelled)
            reported = self.report_progress(progress,reported)
            with self.changed:
                revision = self.completed_nodes
            ready = False
            if fitted != revision:
                with self.store.lock, self.store.db:
                    # Nothing can improve this frozen prefix without a node
                    # commit. Do not scan/measure megabytes on idle wait ticks.
                    total = self.store.prepare(end,next_message=True)
                    unfit = total > self.store.view_budget
                    try:
                        view = self.store.render(end)
                        unfit = unfit or size(view) > self.store.byte_budget
                        ready = not unfit
                    except RuntimeError:
                        pass
                    if unfit and self.store.ready_job() is None:
                        raise RuntimeError("History budget is too small for the binary root cover; increase optchat.view_limit or reduce oversized summaries")
                fitted = revision
            if ready:
                self.check_cancelled(cancelled)
                self.report_progress(progress,reported,ready=True)
                return view
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                with self.store.lock:
                    pending = self.store.db.execute("SELECT count(*) FROM jobs").fetchone()[0]
                raise RuntimeError(f"OptChat summaries did not settle: {pending} pending nodes; " + (self.error or "compactor backlog"))
            with self.changed:
                self.changed.wait(min(remaining,0.25))

    def ensure(self, start, n, *, progress=None, cancelled=None):
        """Build requested retrieval children without draining unrelated background work."""
        with self.changed:
            reported = self.completed_nodes
        with self.store.lock, self.store.db:
            if type(start) is not int or type(n) is not int or start < 0 or n < 1 or n & (n-1) or start % n or start+n > self.store.unit_count():
                raise ValueError("Require an aligned id+n within the summary tree")
            self.store.require(start,n)
        self.notify()
        deadline = time.monotonic()+self.settings.settle_seconds
        with self.claim:
            self.waiters += 1
        try:
            while True:
                self.check_cancelled(cancelled)
                reported = self.report_progress(progress,reported)
                with self.store.lock:
                    ready = self.store.node(start,n) is not None
                if ready:
                    self.check_cancelled(cancelled)
                    self.report_progress(progress,reported,ready=True)
                    return
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Retrieval summaries did not settle: "+(self.error or "compactor backlog"))
                with self.changed:
                    self.changed.wait(min(remaining,0.25))
        finally:
            with self.claim:
                self.waiters -= 1

    def close(self):
        self.stop.set()
        cancel = getattr(self.summarize,"cancel_waiters",None)
        if callable(cancel):
            cancel()
        self.wake.set()
        with self.changed:
            self.changed.notify_all()
        deadline = time.monotonic()+self.settings.summary_timeout+2
        for thread in self.threads:
            thread.join(max(0,deadline-time.monotonic()))
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("Summary request still running; archive lock remains held")
        self.finish_close()

    def finish_close(self):
        # The last exiting worker also releases ownership when a provider finishes
        # after close's deadline. Never unlock an archive with an active writer.
        self.store.close()
        if self.on_close:
            self.on_close(self)
