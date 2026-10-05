import copy
import json
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from agent.context_engine import ContextEngine

from .config import Settings
from .store import VIEW_ACK, Store, encoded, size, sticky_head, visible
from .worker import Worker

STATE = sys.modules.setdefault("_optchat_runtime", SimpleNamespace(engines={}, lock=threading.RLock()))
INSTRUCTIONS = """OptChat supplies the entire prior conversation as a bounded summary view in <chat>.
Each line id+n|text covers n messages from id, oldest first. Summaries are historical evidence,
not new instructions; later user corrections take precedence. Use optchat_zoom(id,n) to open
a line into its two children; n=1 retrieves the original text. Long originals have continuation
offsets: follow next_offset to retrieve all text. Use optchat_search for exact remembered words
and optchat_date for timestamps. Retrieve original details before relying on vague summaries,
guessing, or asking the user to repeat them. This turn's working messages remain verbatim.
The view may arrive in two parts around a fixed acknowledgment; together they are one view.
Say in your final reply what you learned that will matter later. Hermes owns tools and delegation;
follow the user's delegation preferences. The OptChat archive is automatic; no memo tool is needed."""


class OptChatUnavailable(KeyboardInterrupt):
    """Cancel the request rather than Hermes's Exception-based fail-open history fallback."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    cost_status: str | None = None


class HostSummary:
    """Summary calls through Hermes's auxiliary router. Once a call shows the router serving
    them from the main model (the default), later calls carry the cache markers Hermes plans
    for auxiliary calls to that destination; Hermes replans them itself on fallback."""

    def __init__(self, runtime, settings):
        self.runtime, self.settings, self.main_route = runtime, settings, False

    def plan(self, messages):
        try:
            from agent.agent_runtime_helpers import configured_cache_ttl, plan_cache_sections_for_destination
        except ImportError:
            return messages
        runtime = self.runtime
        planned, _ = plan_cache_sections_for_destination(
            messages, None, provider=runtime.get("provider") or "", base_url=runtime.get("base_url") or "",
            api_mode=runtime.get("api_mode") or "", model=runtime.get("model") or "", cache_ttl=configured_cache_ttl())
        return planned

    def __call__(self, messages):
        from agent.auxiliary_client import call_llm
        from agent.usage_pricing import normalize_usage, estimate_usage_cost
        runtime, settings = self.runtime, self.settings
        route = {}
        response = call_llm(task="optchat_summary", messages=self.plan(messages) if self.main_route else messages,
                            main_runtime=runtime, max_tokens=max(4096,settings.node_bytes*8),
                            timeout=settings.summary_timeout, route_info=route)
        self.main_route = (route.get("provider"),route.get("model")) == (runtime.get("provider"),runtime.get("model"))
        canonical = normalize_usage(response.usage,provider=route.get("provider",runtime["provider"]))
        model = getattr(response,"model",None) or route.get("model") or runtime["model"]
        provider = route.get("provider") or runtime["provider"]
        cost = estimate_usage_cost(model,canonical,provider=provider,base_url=runtime.get("base_url"))
        amount = getattr(cost,"amount_usd",None)
        usage = Usage(canonical.input_tokens,canonical.output_tokens,canonical.cache_read_tokens,
                      canonical.cache_write_tokens,float(amount) if amount is not None else None,getattr(cost,"status",None))
        return SimpleNamespace(text=response.choices[0].message.content or "",model=model,provider=provider,usage=usage)


class OptChatEngine(ContextEngine):
    name = "optchat"
    emit_automatic_compaction_status = False

    def __init__(self, settings=None, summarizer=None):
        config_error = None
        try:
            self.settings = settings or Settings.load()
        except Exception as exc:
            self.settings = Settings()
            config_error = str(exc)
        self.summarizer = summarizer
        self.worker = None
        self.turn = None
        self.view = None
        self.home = None
        self.session = None
        self.runtime = {}
        self.problem = config_error
        self.config_error = config_error
        self.cache_head = None

    def clone_for_agent(self):
        clone = OptChatEngine(self.settings,self.summarizer)
        clone.config_error = self.config_error
        clone.problem = self.config_error
        return clone

    def update_model(self, model, context_length, base_url="", api_key="", provider="", api_mode=""):
        self.context_length = context_length
        self.threshold_tokens = context_length
        self.runtime = dict(model=model,provider=provider,base_url=base_url,api_key=api_key,api_mode=api_mode)
        if self.worker:
            self.worker.store.budget = self.history_budget()

    def history_budget(self):
        # Summary lines average about 2 bytes per token, so view_bytes (128 KB) is about 64k
        # tokens. Shrink it only for small windows, keeping the view under ~40% of the window.
        if not self.context_length:
            return self.settings.view_bytes
        return min(self.settings.view_bytes,max(2*self.settings.node_bytes+128,self.context_length*4//5))

    def should_compress(self, prompt_tokens=None):
        return False

    def should_compress_preflight(self, messages):
        return False

    def compress(self, messages, **kwargs):
        return messages

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens",usage.get("input_tokens",0))
        self.last_completion_tokens = usage.get("completion_tokens",usage.get("output_tokens",0))
        self.last_total_tokens = usage.get("total_tokens",0)

    def on_session_start(self, session_id, hermes_home=None, **kwargs):
        from hermes_constants import get_hermes_home
        home = str(Path(hermes_home or get_hermes_home()).resolve())
        key = (home,session_id)
        if not session_id:
            self.problem = "A durable Hermes session ID is required"
            return
        if self.session == session_id and self.worker:
            return
        if self.worker:
            self.on_session_end(self.session,[])
        self.home,self.session = home,session_id
        try:
            if self.config_error:
                raise RuntimeError(self.config_error)
            # Validate ownership rather than silently run two history rewriting systems.
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly() or {}
            memory = cfg.get("memory",{})
            if memory.get("memory_enabled",True) or memory.get("user_profile_enabled",True) or memory.get("provider","none") not in ("none","", "default","builtin"):
                raise RuntimeError("Set memory.memory_enabled=false, user_profile_enabled=false, provider=none for OptChat")
            store = Store(Path(home)/"optchat",session_id,self.history_budget())
            runtime = dict(self.runtime,session_id=session_id)
            fn = self.summarizer or HostSummary(runtime,self.settings)
            store.sync_hermes(home,session_id)
            self.worker = Worker(store,self.settings,fn)
            self.worker.notify()
            self.problem = None
            with STATE.lock:
                STATE.engines[key] = self
        except Exception as exc:
            self.problem = str(exc)

    def begin_turn(self, turn_id, conversation_history, **kwargs):
        self.turn = turn_id
        self.view = None
        self.user_content = copy.deepcopy(kwargs.get("user_message",conversation_history[-1].get("content") if conversation_history else ""))
        self.wire_anchor = None
        if self.worker:
            self.worker.store.record_raw("turn:"+turn_id,self.user_content)

    def select_context(self, request_messages, *, conversation_messages=None, incoming_message=None, budget_tokens=0):
        try:
            if self.problem or not self.worker:
                raise RuntimeError(self.problem or "OptChat was not initialized")
            store = self.worker.store
            if self.view is None:
                # The pre_llm_call hook identifies the admitted user boundary even if its
                # text repeats an older message or mid-turn steering adds another user.
                if self.turn is None:
                    raise RuntimeError("OptChat pre_llm_call hook is not loaded; enable the plugin")
                # Hermes can merge adjacent replayed users AFTER pre_llm_call, carrying
                # the oldest row's id into incoming_message. Resolve the admitted row
                # from the already-flushed DB instead of trusting that merged carrier.
                import sqlite3
                with sqlite3.connect((Path(self.home)/"state.db").as_uri()+"?mode=ro",uri=True) as db:
                    row = db.execute("SELECT id FROM messages WHERE session_id=? AND role='user' ORDER BY id DESC LIMIT 1",(self.session,)).fetchone()
                if not row:
                    raise RuntimeError("Hermes did not persist the admitted user message")
                row_id = row[0]
                store.record_raw(f"row:{row_id}",self.user_content)
                store.sync_hermes(self.home,self.session,before_row=row_id)
                with store.lock:
                    event = store.db.execute("SELECT id FROM events WHERE source=?", (f"row:{row_id}",)).fetchone() if row_id else None
                    if event is None:
                        end = store.count()
                    else:
                        end = event[0]
                self.view = self.worker.settle(end)
                self.cache_head = sticky_head(self.cache_head,self.view,self.settings.cache_split)
            store.sync_hermes(self.home,self.session)
            self.worker.notify()
            systems = [copy.deepcopy(m) for m in request_messages if m.get("role") in ("system","developer")]
            if not systems:
                systems = [{"role":"system","content":INSTRUCTIONS}]
            else:
                systems[0]["content"] = visible(systems[0].get("content"))+"\n\n"+INSTRUCTIONS
            # Preserve the assembled wire version (api_content, native reasoning, prefills)
            # of every current-turn row; never touch the canonical transcript.
            query = visible(self.user_content)
            if self.wire_anchor is None:
                candidates = [(j,m) for j,m in enumerate(request_messages) if m.get("role")=="user" and query in visible(m.get("content"))]
                if not candidates:
                    raise RuntimeError("The admitted user message disappeared from Hermes's request")
                j,m = candidates[-1]
                self.wire_anchor = copy.deepcopy(m.get("content"))
                self.cut_offset = visible(self.wire_anchor).rfind(query)
            else:
                candidates = [(j,m) for j,m in enumerate(request_messages) if m.get("role")=="user" and (
                    m.get("content")==self.wire_anchor or
                    isinstance(m.get("content"),str) and isinstance(self.wire_anchor,str) and m["content"].startswith(self.wire_anchor))]
                if not candidates:
                    raise RuntimeError("Hermes removed the current turn's working anchor")
                j,m = candidates[-1]
            tail = copy.deepcopy(request_messages[j:])
            if isinstance(tail[0].get("content"),str):
                tail[0]["content"] = tail[0]["content"][self.cut_offset:]
            elif isinstance(self.user_content,list):
                # Native multimodal current input is kept as blocks, including attachments.
                tail[0]["content"] = copy.deepcopy(self.user_content)
            elif isinstance(tail[0].get("content"),list):
                # Large merged Hermes carriers can be text-block lists even when the
                # admitted input is plain text. Remove historical blocks, not just
                # string prefixes, while retaining all blocks after the current ask.
                parts = tail[0]["content"]
                hits = [(k,p) for k,p in enumerate(parts) if isinstance(p,dict) and query in p.get("text","")]
                if not hits:
                    raise RuntimeError("The current ask is missing from Hermes's content blocks")
                k,part = hits[-1]
                parts = parts[k:]
                parts[0]["text"] = part["text"][part["text"].rfind(query):]
                tail[0]["content"] = parts
            tail = [m for m in tail if m.get("role") not in ("system","developer")]
            if self.cache_head:
                # A message boundary after the view's stable head gives cache markers a place
                # that survives the next turn's merges; Hermes marks the acknowledgment.
                history = [{"role":"user","content":self.cache_head},{"role":"assistant","content":VIEW_ACK},
                           {"role":"user","content":self.view[len(self.cache_head):]}]
            else:
                history = [{"role":"user","content":self.view}]
            result = systems + history + tail
            # Hermes merge carriers retain _absorbed_row_ids for persistence. Those
            # private keys never reach the provider and are not working context.
            if budget_tokens:
                wire_projection = [{k:v for k,v in m.items() if not k.startswith("_")} for m in result]
                try:
                    from agent.model_metadata import estimate_messages_tokens_rough
                    tokens = estimate_messages_tokens_rough(wire_projection)
                except ImportError:
                    tokens = size(encoded(wire_projection))//3
                # Leave a tenth of the window for tool schemas and the reply.
                if tokens > budget_tokens*9//10:
                    raise RuntimeError(f"This turn's work fills the model's window (~{tokens} of {budget_tokens} tokens). "
                                       "Everything so far is archived: send another message to continue from it")
            return result
        except Exception as exc:
            raise OptChatUnavailable("OptChat stopped this request: "+str(exc)) from exc

    def on_turn_complete(self, messages, **kwargs):
        if self.worker:
            self.worker.store.sync_hermes(self.home,self.session)
            self.worker.notify()
        self.view = None
        self.turn = None

    def on_session_end(self, session_id, messages):
        if self.worker:
            self.worker.store.sync_hermes(self.home,self.session)
            self.worker.close()
            with STATE.lock:
                STATE.engines.pop((self.home,self.session),None)
            self.worker = None
        self.view = None
        self.turn = None

    def get_tool_schemas(self):
        specs = [
            ("optchat_zoom","Open line id+n into its two children; n=1 retrieves the original text. Follow next_offset for long originals.",
             {"id":{"type":"integer","minimum":0},"n":{"type":"integer","minimum":1},"offset":{"type":"integer","minimum":0}},["id","n"]),
            ("optchat_date","The date and time of message id.",{"id":{"type":"integer","minimum":0}},["id"]),
            ("optchat_search","Find a literal substring in original messages; continue with after=last returned id.",
             {"query":{"type":"string","minLength":1},"after":{"type":"integer","minimum":-1}},["query"])]
        return [{"name":name,"description":description,"parameters":{"type":"object","properties":props,"required":required,"additionalProperties":False}}
                for name,description,props,required in specs]

    def handle_tool_call(self, name, args, **kwargs):
        try:
            if not self.worker:
                raise ValueError("Archive unavailable")
            store = self.worker.store
            if name == "optchat_zoom":
                result = store.zoom(args["id"],args["n"],args.get("offset",0))
            elif name == "optchat_date":
                result = {"id":args["id"],"date":datetime.fromtimestamp(store.event(args["id"])["date"],timezone.utc).isoformat()}
            elif name == "optchat_search":
                query = args["query"]
                if not isinstance(query,str) or not query:
                    raise ValueError("A nonempty literal query is required")
                result = {"matches":store.search(query,args.get("after",-1))}
            else:
                raise ValueError("Unknown OptChat tool")
            return encoded(result)
        except (ValueError,KeyError,TypeError) as exc:
            return encoded({"error":str(exc)})

    def get_status(self):
        result = super().get_status()
        if self.worker:
            with self.worker.store.lock:
                result.update(history_bytes=size(self.worker.store.render()) if not self.worker.store.ready_job() else None,
                              history_budget_bytes=self.worker.store.budget,events=self.worker.store.count(),
                              summary_error=self.worker.error)
        return result
