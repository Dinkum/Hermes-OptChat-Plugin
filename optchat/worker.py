import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .store import VIEW_ACK, size, sticky_head

logger = logging.getLogger(__name__)
COMPACT = Path(__file__).with_name("compact.txt").read_text()


def capped(text, limit):
    # Tool I/O reaches the summarizer as head and tail, as the design caps it when logged.
    # The archive keeps the whole original for zoom.
    if len(text) <= limit:
        return text
    half = limit//2
    return text[:half]+f"\n[... {len(text)-2*half} characters cut ...]\n"+text[-half:]


class Worker:
    def __init__(self, store, settings, summarize):
        self.store, self.settings, self.summarize = store, settings, summarize
        self.changed = threading.Condition()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.errors = {}
        self.busy = set()
        self.context_head = None
        self.claim = threading.Lock()
        from agent.memory_provider import spawn_context_thread
        # Leaves depend on preceding leaves, so one lane builds them in order. Merges of
        # finished parts are independent and run in parallel lanes beside it.
        lanes = [("leaf","leaf")]+[(f"parent-{k}","parent") for k in range(settings.merge_lanes)]
        self.threads = [spawn_context_thread(lambda lane=lane,kind=kind:self.run(lane,kind),name="optchat-"+lane)
                        for lane,kind in lanes]
        for thread in self.threads:
            thread.start()

    @property
    def error(self):
        return "; ".join(dict.fromkeys(self.errors.copy().values())) or None

    def notify(self):
        self.wake.set()

    def build(self, start, n, source, kind=None):
        if kind in ("tool","echo"):
            source = capped(source,self.settings.tool_chars)
        if size(source) <= self.settings.node_bytes:
            self.store.save_node(start,n,source,{"free":True})
            return
        with self.store.lock:
            context = "\n".join(self.store.lines(start if n == 1 else start+n, ids=False))
        scale = "user: Keep the deployment private; approval is required before publishing. echo: The authentication test passes, but the browser check has not run. talk: The database stores originals and summaries separately. user: Prefer exact wording and retain project names, decisions, amounts and reasons. echo: Retrieved the source document successfully. "
        scale = (scale * (self.settings.node_bytes // len(scale) + 1))[:self.settings.node_bytes]
        step = f"For scale, this line is exactly {self.settings.node_bytes} bytes:\n{scale}\n\n"
        step += ("Compress this message" if n == 1 else "Merge these two lines")
        step += f" into one line, in at most {self.settings.node_bytes} bytes:\n{source}"
        chat = f"<chat>\n{context}\n"
        head = self.context_head = sticky_head(self.context_head,chat,self.settings.cache_split)
        if head:
            # Same boundary as the main view: calls share the context's head as a cacheable message.
            messages = [{"role":"system","content":COMPACT},{"role":"user","content":head},
                        {"role":"assistant","content":VIEW_ACK},
                        {"role":"user","content":f"{chat[len(head):]}</chat>\n\n{step}"}]
        else:
            messages = [{"role":"system","content":COMPACT},
                        {"role":"user","content":f"{chat}</chat>\n\n{step}"}]
        tries = []
        for _ in range(self.settings.summary_tries):
            began = time.monotonic()
            result = self.summarize(messages)
            text = result.text.strip()
            metrics = {"seconds":time.monotonic()-began,"bytes":size(text),
                       "model":result.model,"provider":result.provider,"usage":asdict(result.usage)}
            self.store.record_attempt(start,n,metrics)
            if not text:
                raise RuntimeError("Empty summary; original remains durable")
            tries.append(text)
            if size(text) <= self.settings.node_bytes:
                break
            cut = text.encode()[:self.settings.node_bytes].decode("utf-8",errors="ignore")
            messages.extend([{"role":"assistant","content":text},
                             {"role":"user","content":f"That line is {size(text)} bytes; the limit is {self.settings.node_bytes}. It must end where it is cut here:\n{cut}| ← LIMIT"}])
        self.store.save_node(start,n,min(tries,key=size),{"tries":len(tries)})

    def run(self, lane, kind):
        while not self.stop.is_set():
            job = None
            try:
                with self.claim:
                    job = self.store.ready_job(kind,self.busy)
                    if job:
                        self.busy.add(job[:2])
                if job:
                    self.build(*job)
                    self.errors.pop(lane,None)
                    with self.claim:
                        self.busy.discard(job[:2])
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
                if job:
                    with self.claim:
                        self.busy.discard(job[:2])
                continue
            self.wake.wait(1)
            self.wake.clear()

    def settle(self, end):
        self.notify()
        deadline = time.monotonic()+self.settings.settle_seconds
        while True:
            with self.store.lock, self.store.db:
                total = self.store.fit()
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

    def close(self):
        self.stop.set()
        self.wake.set()
        deadline = time.monotonic()+self.settings.summary_timeout+2
        for thread in self.threads:
            thread.join(max(0,deadline-time.monotonic()))
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("Summary request still running; archive lock remains held")
        self.store.close()
