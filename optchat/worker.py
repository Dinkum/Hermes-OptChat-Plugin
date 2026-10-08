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
        self.context_head = None
        self.claim = threading.Lock()
        self.waiters = 0
        from agent.memory_provider import spawn_context_thread
        # Leaves depend on preceding leaves, so one lane builds them in order. Merges of
        # finished parts are independent and run in parallel lanes beside it.
        if settings.batching == "auto":
            lanes = [("leaf","leaf"),("batch","parent")] if settings.summary_boundary == "event" else [("batch",None)]
        else:
            lanes = [("leaf","leaf")]+[(f"parent-{k}","parent") for k in range(settings.merge_lanes)]
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
            context = "\n".join(self.store.summary_lines(start if n == 1 else start+n,
                               raw=self.settings.compression_policy == "on_demand" or self.settings.summary_boundary == "user_turn"))
        limit = self.settings.summary_limit
        return batch.SummaryJob(f"{start}+{n}",context,source if prepared else self.source(source,kind),limit.output_bytes,
                                "Compress this completed turn chunk" if kind == "turn" else "Compress this message" if n == 1 else "Merge these two lines",
                                limit.unit,limit.target)

    def build(self, start, n, source, kind=None):
        source = self.source(source,kind)
        limit = self.settings.summary_limit
        if limit.can_copy(source,self.settings.summary_bytes):
            self.store.save_node(start,n,source,{"free":True})
            return
        job = self.description(start,n,source,kind,prepared=True)
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
                raise RuntimeError("Empty summary; original remains durable")
            tries.append(text)
            if limit.accepts(text):
                break
            messages.extend([{"role":"assistant","content":text},
                             {"role":"user","content":limit.feedback(text)}])
        self.store.save_node(start,n,min(tries,key=lambda text:(limit.measure(text),size(text))),{"tries":len(tries)})

    def take_jobs(self, kind):
        """Claim a resource-bounded batch from one committed snapshot."""
        if self.settings.batching == "off" or kind == "leaf" or not hasattr(self.summarize,"summarize_many"):
            job = self.store.ready_job(kind,self.busy)
            return [job] if job else []
        candidates = self.store.ready_jobs(kind,self.busy)
        if not candidates:
            return []
        # Free nodes need no transport and can expose new parent jobs immediately.
        for job in candidates:
            if self.settings.summary_limit.can_copy(self.source(job[2],job[3]),self.settings.summary_bytes):
                return [job]
        chosen, descriptions = [], []
        for job in sorted(candidates,key=lambda job:(job[0]+job[1],job[1])):
            description = self.description(*job)
            packed = descriptions+[description]
            if chosen and (size(encoded(batch.messages(packed,self.prompt))) > self.settings.batch_input_bytes
                           or sum(j.target_bytes+size(j.id)+64 for j in packed) > self.settings.batch_output_bytes):
                continue
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
            descriptions = [self.description(*job) for job in jobs]
        if (size(encoded(batch.messages(descriptions,self.prompt))) > self.settings.batch_input_bytes
            or sum(j.target_bytes+size(j.id)+64 for j in descriptions) > self.settings.batch_output_bytes):
            # Other lanes may have committed context since these jobs were claimed.
            # Recheck the actual envelope, splitting without losing job identities.
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
            if text and self.settings.summary_limit.accepts(text):
                self.store.save_node(job[0],job[1],text,{"batch_id":batch_id})
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
                with self.claim:
                    jobs = self.take_jobs(kind)
                    self.busy.update(job[:2] for job in jobs)
                if jobs:
                    self.build_many(jobs)
                    self.errors.pop(lane,None)
                    with self.claim:
                        self.busy.difference_update(job[:2] for job in jobs)
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
                continue
            self.wake.wait(1)
            self.wake.clear()

    def settle(self, end):
        self.store.seal(end)
        end = self.store.unit_end(end)
        self.store.prepare(end)
        return self.wait_for_view(end)

    def wait_for_view(self, end):
        with self.claim:
            self.waiters += 1
        try:
            return self.await_view(end)
        finally:
            with self.claim:
                self.waiters -= 1

    def await_view(self, end):
        self.notify()
        deadline = time.monotonic()+self.settings.settle_seconds
        while True:
            if self.stop.is_set():
                raise RuntimeError("OptChat is shutting down")
            with self.store.lock, self.store.db:
                total = self.store.prepare(end)
                if total > self.store.budget and self.store.ready_job() is None:
                    raise RuntimeError("History budget is too small for the binary root cover; increase optchat.view_bytes")
                try:
                    view = self.store.render(end)
                    if total <= self.store.budget and size(view) <= self.store.budget:
                        return view
                except RuntimeError:
                    pass
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise RuntimeError("OptChat summaries did not settle: " + (self.error or "compactor backlog"))
            with self.changed:
                self.changed.wait(min(remaining,0.25))

    def ensure(self, start, n):
        """Build requested retrieval children without draining unrelated background work."""
        with self.store.lock, self.store.db:
            if type(start) is not int or type(n) is not int or start < 0 or n < 1 or n & (n-1) or start % n or start+n > self.store.unit_count():
                raise ValueError("Require an aligned id+n within the summary tree")
            self.store.require(start,n)
        self.notify()
        deadline = time.monotonic()+self.settings.settle_seconds
        with self.claim:
            self.waiters += 1
        try:
            while not self.stop.is_set():
                with self.store.lock:
                    if self.store.node(start,n) is not None:
                        return
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Retrieval summaries did not settle: "+(self.error or "compactor backlog"))
                with self.changed:
                    self.changed.wait(min(remaining,0.25))
            raise RuntimeError("OptChat is shutting down")
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
