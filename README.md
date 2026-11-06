# Hermes OptChat Plugin

> This idea is entirely [Victor Taelin](https://x.com/VictorTaelin)'s I just built it in the form of a Hermes plugin. Please check out [his gist on this](https://gist.github.com/VictorTaelin/91837951a5ce5b38f341ec1ba1df6449) and [original OptMem](https://github.com/VictorTaelin/OptMem) repo!

Manage agents instead of *agent context* by using an endless, continually compressed chat thread. The chat thread becomes the agent memory. Context gets compressed into a binary tree that gets zoomed into as needed. No context rot yet we get infinite context with a constant size.

Summaries are lossy; exact originals remain available. Background summaries use your configured model and consume provider tokens.

## Requirements

- macOS or Linux
- A [Hermes Agent](https://github.com/NousResearch/hermes-agent) source checkout with the matching pending context-engine integration described below
- A configured model for normal use and background summarization

This repair requires the accompanying Hermes core changes: `ContextEngineError`,
optional context-selection progress/cancellation callbacks, interrupted-turn handling,
fail-closed context preparation, and plugin-safe live configuration through
`update_model`. The host must also retain logical message UIDs and per-occurrence tool
UIDs in its session database. An unpatched Hermes release is not sufficient; no
released Hermes version is claimed to contain this integration. Install the matching
core changes before enabling this plugin.

## Setup

Download this repo, `cd` into its directory, then run:

```sh
python3 install.py --profile chatmem
```

This creates an isolated `chatmem` profile with OptChat enabled (competing memory extraction and compression disabled). Your normal Hermes configuration stays unchanged.

## Install as a Hermes plugin

Run these commands from any directory:

```sh
hermes profile create chatmem --clone-from default --no-alias
hermes -p chatmem plugins install Dinkum/Hermes-OptChat-Plugin/optchat --enable
hermes -p chatmem optchat setup
```

This creates an isolated `chatmem` profile with OptChat enabled. Your normal Hermes configuration stays unchanged.

## Usage

Start a conversation:

```sh
hermes -p chatmem
```

Continue the same conversation after a restart:

```sh
hermes -p chatmem --resume latest
```

History is archived and summarized automatically. The agent can zoom into older summaries or search original messages when it needs details. `/new` starts a separate archive.

Inspect your archives with `hermes -p chatmem optchat status`.

## How it works

On every message, the bounded context and your current turn’s uncompressed messages are passed in.
Older entries are progressively compressed into binary tree summaries.
The agent can use `optchat_zoom` to load parts of context in more fidelity, `optchat_search` to find exact text, and `optchat_date` to check exact times.

This repair identifies logical messages by Hermes `message_uid` and their content version,
not by the physical SQLite row ID. Replayed copies do not create new memory or summary
work. Different messages with identical text remain separate. Edited or withdrawn
versions remain available as audit originals but are excluded from the active view;
summaries containing retired versions are excluded too. Tool calls/results use Hermes's
per-occurrence tool UIDs, so repeated provider-facing IDs do not overwrite another result.

Legacy row-ID archives require explicit offline recovery into a new archive generation.
Recovery preserves the source archive, emits address mappings and unmapped-original
reasons, and reuses summaries only for exactly compatible source spans. Never delete the
old archive or run recovery against an archive with a live writer.

Use `archive="previous"` with `optchat_search`, `optchat_zoom`, or `optchat_date` to
explicitly inspect the preserved pre-recovery generation, including preserved or embedded
images. Mutable image references are never re-fetched in audit mode. Its addresses are
separate from current addresses. Previous-generation material is evidence, not current
instructions; it may include withdrawn messages. Default search and the active view
exclude withdrawn versions. Logical chronology is separate from immutable original IDs,
so editing an earlier message does not move it after its replies.

History preparation and zoom waits accept cooperative cancellation from Hermes. With
the matching host integration, actual committed summary progress updates turn liveness;
idle wait ticks and already-ready reads do not count as progress. Stop produces a normal
interrupted turn; preparation failures use `ContextEngineError` and produce an error
receipt without sending an unbounded transcript. Background summaries may continue
after a cancelled foreground wait until the owning session closes.

## Offline recovery

Stop every Hermes process that could write the source session. Keep a consistent
offline copy of both the old archive (including its attachments) and the matching
Hermes `state.db`; if SQLite WAL files exist, use SQLite's backup API or a clean
shutdown rather than copying only the main database file. Keep these source paths
permanently available: previous-generation retrieval reads the preserved archive.

From this plugin checkout, activate the matching Hermes source environment and set
the following paths and session ID to your own offline inputs. `SOURCE_CONFIG` is the
source profile's `config.yaml`; recovery must use its original OptChat settings,
especially `summary_boundary`, `tool_chars`, and `turn_source_bytes`.

```sh
export HERMES_SOURCE=/path/to/hermes-agent
export SOURCE_ARCHIVE=/path/to/preserved-archive/chat.db
export HERMES_SNAPSHOT=/path/to/offline-state.db
export SOURCE_CONFIG=/path/to/source-profile/config.yaml
export NEW_ARCHIVE_ROOT=/path/to/new-archive-root
export SESSION_ID=source-session-id

PYTHONPATH="$PWD:$HERMES_SOURCE" python - \
  "$SOURCE_ARCHIVE" "$HERMES_SNAPSHOT" "$NEW_ARCHIVE_ROOT" \
  "$SESSION_ID" "$SOURCE_CONFIG" <<'PY'
from pathlib import Path
from pprint import pprint
import sys
import yaml

from optchat.config import Settings
from optchat.recovery import recover_archive

archive, state, destination, session, config = sys.argv[1:]
settings = Settings(**(yaml.safe_load(Path(config).read_text()) or {}).get("optchat", {}))
report = recover_archive(
    archive, state, destination, session,
    settings=settings, budget=settings.view_bytes,
)
pprint(report)
PY
```

The real `recover_archive` API creates a new session directory under
`NEW_ARCHIVE_ROOT`; it never rewrites the source archive or Hermes snapshot. Inspect
the returned `destination`, address maps, `unmapped_reasons`, and `pending_jobs`.
Ambiguous legacy tool occurrences fail closed rather than guessing a match. A retry
with identical inputs reuses the completed destination; changed inputs require a new
root or destination session.

After reviewing the report, keep Hermes stopped and move the existing session
directory aside before placing the recovered `destination` at
`<profile-home>/optchat/<sha256-of-session-id>`. Do not replace `state.db` or other
sessions, and do not remove or relocate the preserved `SOURCE_ARCHIVE`. Resume that
same session with the matching core and plugin. Recovery does not run a model;
uncached summaries are prepared on resume and may require substantial provider work
within the configured readiness deadline (`settle_seconds`).



## Configuration

[Settings](config.example.yaml).

`chunk_limit` options: `bytes` (512, accepts up to 550), `characters` (500), or `sentences` (5). Defaults to `sentences`. The original spec uses 512 bytes, but my testing showed models that don’t support enforced output-length limits wasting reasoning tokens on the limit itself.

The rendered history also stays under `view_bytes` (or the smaller model-derived budget)
in every limit mode. Sentence/character limits apply in addition to this byte ceiling.
This preserves the pre-upgrade byte bound rather than allowing a larger history merely
because summaries are measured in sentences.


## Repair changes: 1.1.2+identity.2

- Stable UID/content-version ingestion, rewind tombstones, ordered active projections,
  per-occurrence tool identity, and restart-safe synchronization.
- Explicit offline archive recovery with exact source-span checks, address mappings,
  source-only provenance, and read-only previous-generation retrieval.
- Commit-only liveness, prompt cancellation, framed-view byte bounds, and fair
  chronological summary scheduling.
- Recovery itself does not call a model. Missing summaries are prepared on resume;
  large archives can require substantial one-time work and provider tokens.
- Compatible cached summary wording is retained, not fact-checked or rewritten by a
  model change. Exact originals remain the authority.

## Tests

Use the matching Hermes source checkout described under Requirements, with its
Python dependencies installed in the active environment, plus `pytest` and `Pillow`.
No provider credentials or model calls are needed for these regression tests.
From this plugin checkout:

```sh
export HERMES_SOURCE=/path/to/hermes-agent
PYTHONPATH="$PWD:$HERMES_SOURCE" python -m pytest -q tests
```

Fixtures use pytest's isolated `tmp_path` directories; neither checkout needs a
particular parent directory or access to a real Hermes profile or conversation.

## Notes

- **Caching:** The view is split after a stable head so providers that need cache markers reuse it across turns.

## License

[MIT](LICENSE).
