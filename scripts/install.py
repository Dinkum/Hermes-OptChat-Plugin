"""Install into an explicit Hermes home, preserving unrelated settings."""
import argparse
import shutil
import uuid
from pathlib import Path

from optchat.configure import check_compatibility, prepare_configuration, write_configuration


def install(home, source=None):
    check_compatibility()
    home = Path(home).expanduser().resolve()
    source = Path(source or Path(__file__).resolve().parents[1]/"optchat")
    target = home/"plugins"/"optchat"
    if target.exists():
        raise FileExistsError(f"{target} exists; move it aside before installing another version")
    home.mkdir(parents=True,exist_ok=True)
    config, cfg = prepare_configuration(home)
    # No extra packages: this plugin uses Python's standard library and Hermes APIs.
    staging = home/(".optchat-install-"+uuid.uuid4().hex)
    published = False
    try:
        shutil.copytree(source,staging,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
        target.parent.mkdir(parents=True,exist_ok=True)
        staging.rename(target)
        published = True
        write_configuration(config,cfg)
    except BaseException as exc:
        if published:
            target.rename(staging)
        exc.add_note(f"Incomplete plugin files, if any, are preserved at {staging}; installation can be retried.")
        raise
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
        from hermes_cli.profiles import create_profile, get_profile_dir
        # The clone copies config, not its backup. Check engine ownership before
        # publishing a profile with a name that would block a later retry.
        prepare_configuration(get_profile_dir("default"),check_backup=False)
        home = create_profile(args.profile,clone_from="default",clone_config=True,no_alias=True,
                              description="Continuous conversations with durable OptChat history.")
        try:
            install(home)
        except BaseException as exc:
            recovery = home.with_name(f".optchat-failed-{home.name}-{uuid.uuid4().hex}")
            home.rename(recovery)
            exc.add_note(f"The incomplete profile is preserved at {recovery}; the profile name can be retried.")
            raise
        print(f"Launch with: hermes -p {args.profile}. Your active profile is unchanged.")
    else:
        install(args.home)


if __name__ == "__main__":
    main()
