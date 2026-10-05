"""Shared configuration for source installation and native plugin setup."""
import os
import shutil
import stat
import uuid
from pathlib import Path

import yaml


def check_compatibility():
    if os.name != "posix":
        raise RuntimeError("OptChat requires macOS or Linux; native Windows is unsupported")
    from agent.context_engine import ContextEngine
    from inspect import signature
    if not hasattr(ContextEngine,"select_context") or "incoming_message" not in signature(ContextEngine.select_context).parameters:
        raise RuntimeError("This Hermes version lacks the required context selection interface")


def prepare_configuration(home, *, check_backup=True):
    check_compatibility()
    config = Path(home)/"config.yaml"
    raw = config.read_text() if config.exists() else ""
    cfg = yaml.safe_load(raw) or {}
    if cfg.get("context",{}).get("engine","compressor") not in ("compressor","optchat"):
        raise ValueError("This home already uses another context engine; choose a separate Hermes home")
    enabled = cfg.setdefault("plugins",{}).setdefault("enabled",[])
    if "optchat" not in enabled:
        enabled.append("optchat")
    cfg.setdefault("context",{})["engine"] = "optchat"
    cfg.setdefault("memory",{}).update(memory_enabled=False,user_profile_enabled=False,provider="none")
    cfg.setdefault("compression",{}).update(enabled=False,idle_compact_after_seconds=0,proactive_prune_tokens=0)
    cfg.setdefault("auxiliary",{}).setdefault("background_review",{})["enabled"] = False
    cfg.setdefault("sessions",{})["max_resume_messages"] = 0
    backup = config.with_name("config.before-optchat.yaml")
    if check_backup and config.exists() and backup.exists():
        raise FileExistsError(f"Preserve the existing backup {backup} before installation or setup")
    return config, cfg


def write_configuration(config, cfg):
    data = yaml.safe_dump(cfg,sort_keys=False).encode("utf-8")
    backup = config.with_name("config.before-optchat.yaml")
    token = uuid.uuid4().hex
    pending = config.with_name(f".optchat-setup-{token}.yaml")
    mode = stat.S_IMODE(config.stat().st_mode) if config.exists() else 0o600
    backup_created = False
    try:
        with pending.open("xb") as out:
            os.chmod(pending,0o600)
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(pending,mode)
        if config.exists():
            # Exclusive creation preserves a backup made by another setup. Move
            # only our own backup aside if publication fails, so retry is possible.
            with backup.open("xb") as out:
                backup_created = True
                os.chmod(backup,0o600)
                with config.open("rb") as original:
                    shutil.copyfileobj(original,out)
                out.flush()
                os.fsync(out.fileno())
        os.replace(pending,config)
    except BaseException as exc:
        if backup_created:
            backup.rename(config.with_name(f".optchat-setup-{token}-backup.yaml"))
        exc.add_note(f"Incomplete setup files, if any, are preserved beside {config} as .optchat-setup-{token}*; the configuration was not replaced.")
        raise
