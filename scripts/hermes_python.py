"""Run a repository script through the installed Hermes runtime."""
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path


def run_script(args):
    home, script, *arguments = args
    import hermes_bootstrap
    from hermes_constants import get_hermes_home, set_hermes_home_override

    os.environ["OPTCHAT_RUNTIME_HOME"] = str(get_hermes_home())
    os.environ["OPTCHAT_HERMES_SOURCE"] = str(Path(hermes_bootstrap.__file__).parent)
    if home != "--installed-home":
        target = Path(home)
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.environ["HERMES_HOME"] = str(target)
        set_hermes_home_override(target)
        # Keep test scratch files inside the isolated home as well.
        scratch = target / "cache" / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        for name in ("TMPDIR", "TMP", "TEMP"):
            os.environ[name] = str(scratch)
        import tempfile
        tempfile.tempdir = None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.argv = [script, *arguments]
    runpy.run_path(script, run_name="__main__")


def runtime_command(args, env):
    try:
        result = subprocess.run(
            [env.get("OPTCHAT_HERMES_LAUNCHER", "hermes"), "--print-runtime-command", "--module", "__future__", "--", *args],
            env=env, capture_output=True, text=True, check=True)
        command = json.loads(result.stdout)
        if not isinstance(command, list) or not command or not all(isinstance(a, str) for a in command):
            raise ValueError("Expected a JSON command array")
        # Source launchers return Python -c bootstrap code. Keep that code intact,
        # then run our file in the dependency environment the bootstrap selected.
        # __future__ is a stdlib module with no CLI side effects.
        entry = command.index("-c") + 1
        command[entry] += "; import runpy,sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
        return command
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("OptChat needs a Hermes source-install launcher with "
                           "--print-runtime-command support. Check `hermes --version` "
                           "and the supported installation in README.md.") from exc


def main():
    args = sys.argv[1:]
    if args[:1] == ["--run-script"]:
        run_script(args[1:])
        return
    if os.name != "posix":
        raise SystemExit("OptChat requires macOS or Linux (POSIX file locking); native Windows is unsupported.")
    installed = args[:1] == ["--installed-home"]
    if installed:
        args.pop(0)
    if not args:
        raise SystemExit("Usage: python3 scripts/hermes_python.py [--installed-home] SCRIPT [ARGS...]")
    repo = Path(__file__).resolve().parents[1]
    home = "--installed-home" if installed else str(Path(
        os.environ.get("OPTCHAT_TEST_HOME", repo / ".test-home")).expanduser().resolve())
    args[0] = str(Path(args[0]).expanduser().resolve())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HERMES_DISABLE_LAZY_INSTALLS="1")
    env.setdefault("OPTCHAT_HERMES_LAUNCHER", shutil.which("hermes") or "hermes")
    # Boot the installed dependencies before switching into an isolated test home.
    # Child test processes use the same launch contract and installed runtime home.
    if env.get("OPTCHAT_RUNTIME_HOME"):
        env["HERMES_HOME"] = env["OPTCHAT_RUNTIME_HOME"]
    command = runtime_command([str(Path(__file__).resolve()), "--run-script", home, *args], env)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
