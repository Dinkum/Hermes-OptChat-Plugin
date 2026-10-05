"""Install into an explicit Hermes home, preserving unrelated settings."""
import argparse
import os
import shutil
from pathlib import Path

import yaml


def check_compatibility():
    if os.name != "posix":
        raise RuntimeError("OptChat requires macOS or Linux; native Windows is unsupported")
    from agent.context_engine import ContextEngine
    from inspect import signature
    if not hasattr(ContextEngine,"select_context") or "incoming_message" not in signature(ContextEngine.select_context).parameters:
        raise RuntimeError("This Hermes version lacks the required context selection interface")


def install(home, source=None):
    check_compatibility()
    home = Path(home).expanduser().resolve()
    source = Path(source or Path(__file__).resolve().parents[1]/"optchat")
    target = home/"plugins"/"optchat"
    if target.exists():
        raise FileExistsError(f"{target} exists; move it aside before installing another version")
    home.mkdir(parents=True,exist_ok=True)
    config = home/"config.yaml"
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
    # No extra packages: this plugin uses Python's standard library and Hermes APIs.
    if config.exists():
        backup = home/"config.before-optchat.yaml"
        if backup.exists():
            raise FileExistsError(f"Preserve the existing backup {backup} before installation")
        shutil.copyfile(config,backup)
    shutil.copytree(source,target,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    print(f"Installed OptChat in {home}. Start Hermes with HERMES_HOME={home} and resume the same session.")
    return home


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--home",type=Path)
    destination.add_argument("--profile",help="Create an isolated named profile cloned from default; do not activate it")
    args = parser.parse_args()
    check_compatibility()
    if args.profile:
        from hermes_cli.profiles import create_profile
        home = create_profile(args.profile,clone_from="default",clone_config=True,no_alias=True,
                              description="Continuous conversations with durable OptChat history.")
        install(home)
        print(f"Launch with: hermes -p {args.profile}. Your active profile is unchanged.")
    else:
        install(args.home)


if __name__ == "__main__":
    main()
