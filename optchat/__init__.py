"""Native Hermes ContextEngine and activity observers, loaded through plugin discovery."""
from .engine import OptChatEngine, STATE


def current(session_id):
    from hermes_constants import get_hermes_home
    from pathlib import Path
    with STATE.lock:
        return STATE.engines.get((str(Path(get_hermes_home()).resolve()),session_id))


def pre_turn(session_id, turn_id, conversation_history, **kwargs):
    engine = current(session_id)
    if engine:
        engine.begin_turn(turn_id,conversation_history,**kwargs)


def pre_tool(session_id=None, tool_call_id=None, tool_name=None, args=None, **kwargs):
    engine = current(session_id)
    if engine and tool_call_id:
        store = engine.worker.store
        store.sync_hermes(engine.home,engine.session)
        store.observe_tool("pre",tool_call_id,tool_name=tool_name,args=args,
                           tool_call_uid=kwargs.get("tool_call_uid"),
                           turn_id=kwargs.get("turn_id"),api_request_id=kwargs.get("api_request_id"))


def post_tool(session_id=None, tool_call_id=None, tool_name=None, result=None, **kwargs):
    engine = current(session_id)
    if engine and tool_call_id:
        store = engine.worker.store
        store.observe_tool("post",tool_call_id,tool_name=tool_name,result=result,
                           tool_call_uid=kwargs.get("tool_call_uid"),
                           turn_id=kwargs.get("turn_id"),api_request_id=kwargs.get("api_request_id"))
        store.sync_hermes(engine.home,engine.session)
        engine.worker.notify()


def register(ctx):
    # Dedicated context-engine discovery uses a minimal collector. General discovery
    # supplies the hooks and auxiliary task; both share the selected per-session engine.
    ctx.register_context_engine(OptChatEngine())
    if getattr(ctx,"manifest",None) is None:
        return
    ctx.register_hook("pre_llm_call",pre_turn)
    ctx.register_hook("pre_tool_call",pre_tool)
    ctx.register_hook("post_tool_call",post_tool)
    ctx.register_auxiliary_task("optchat_summary",display_name="OptChat summaries",
                                description="Build durable binary chat summaries with context",
                                defaults={"provider":"auto","timeout":60,"reasoning_effort":"low"})
    from .cli import setup, handle
    ctx.register_cli_command("optchat",help="Set up OptChat or inspect and export its archive",setup_fn=setup,handler_fn=handle)
