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
from utils import fast_safe_load

from optchat.config import Settings
from optchat.recovery import recover_archive

archive, state, destination, session, config = sys.argv[1:]
settings = Settings(**(fast_safe_load(Path(config).read_text()) or {}).get("optchat", {}))
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

The defaults use Taelin's byte-based recipe as the baseline: eager, event-boundary
compression with `chunk_limit: bytes`, `node_bytes: 512`, and a `view_bytes: 128000`
UTF-8-byte history budget. Byte-mode nodes have a **strict ceiling**, including
spaces and punctuation. This is deliberately more conservative than the gist's
allowance for a few extra bytes after retries. Non-ASCII characters can occupy
two to four UTF-8 bytes; five sentences are not an equivalent bound.

`chunk_limit: characters` (500 characters) and `chunk_limit: sentences` (5 sentences)
remain supported as explicit opt-ins. For the optional `user_turn` boundary, byte
mode uses `turn_node_bytes` rather than `node_bytes`.

The rendered history also stays under `view_bytes` (or the smaller model-derived budget)
in every limit mode. Sentence/character limits apply in addition to this byte ceiling.
This preserves the pre-upgrade byte bound rather than allowing a larger history merely
because summaries are measured in sentences.

**Breaking default and byte-limit change:** older versions defaulted to sentences
and accepted byte summaries up to roughly 7.4% above the target. Existing archives
must be checked for compatibility before switching to the recipe. Changing the
configuration or restarting is not an archive migration: incompatible cached nodes
must not be silently reused. Use a controlled migration to a new validated archive,
retaining exact originals and the old archive's audit nodes while incompatible
summaries are regenerated under the selected target. Do not truncate history or
increase the history budget to make the cutover pass.


### Migrating cached summary limits

`optchat.recovery.migrate_summary_limits` creates a new generation while keeping
original event addresses, identities, attachments and the saved view. Conforming
nodes are imported unchanged; only incompatible nodes in that view are
recompressed from their previous summaries. This avoids replaying the full log,
but cannot restore details already omitted from those summaries. Exact originals
remain authoritative.

```python
from dataclasses import replace
from optchat.config import Settings
from optchat.engine import HostSummary
from optchat.recovery import migrate_summary_limits

def migrate_offline(archive, destination_root, runtime):
    settings = replace(Settings.load(), summary_boundary="event",
                       compression_policy="eager", chunk_limit="bytes",
                       node_bytes=512, view_bytes=128000, view_limit=None)
    return migrate_summary_limits(
        archive, destination_root, settings=settings, budget=128000,
        summarize=HostSummary(runtime, settings),
    )
```

`runtime` is the same in-memory provider/model routing dictionary supplied to
`OptChatEngine`, not a credential file to publish. The configured
`optchat_summary` route is retained. Stop the source writer or use a consistent
offline snapshot. The destination must be new. Unrecovered provider failures or
fallback-only summaries prevent publication; supported batch transport can
recover through real singleton calls. Short sources are copied without a call.

Each new generation contains a standalone read-only `audit-source.db` snapshot.
Discarded summary children remain in a separate immutable audit table, never in
the active view. Zoom labels those children as historical and identifies their
source. `archive="previous"` reads the immediate prior snapshot;
`generation=2` follows its predecessor. Current search/view still exclude
withdrawn versions. Keep the recorded older generation paths for deep audit.

Verify the new view and original records before switching the profile archive
link, then reload the owning Hermes process while idle. This is an explicit,
one-time cache-generation change, not a startup rebuild. A normal cold resume
loads the saved view and reuses its existing summary nodes.

### Recipe scope

The history view merges in batches toward half its budget after overflowing;
with the defaults this is 128,000 toward 64,000 bytes. If parents are not ready,
work continues rather than raising the cap. The current turn's working messages
remain separate and are still subject to the model-window guard.

This is not bit-for-bit UniiChat: Hermes supplies tools/provider cache handling
and supports edits/rewinds. Those changes can split the immutable address tree
into separate runs. A sufficiently fragmented cover can require another explicit
generation migration; the plugin does not claim an infinite-history guarantee
under arbitrary transcript mutation, nor UniiChat's measured cache-hit rate.

## Recipe changes: 1.1.2+identity.3

- Byte-based defaults and strict configured node ceilings; incompatible old
  caches fail before their policy or schema is changed.
- Controlled, provider-backed cache migration, sealed audit snapshots and
  historical child navigation without hidden full-tree regeneration.
- Retained summary cover supplies future compaction context even when its
  descendants are uncached; historical boundaries stop before missing lines.
- Required selected-sibling parent work is restored without replaying hidden
  descendants. Smaller summarizer views and main views remain persisted.
- Length-repair requests include the rejected answer and a UTF-8 cut boundary,
  allowing a stateless model call to revise its actual candidate.
- Independent migration compactions share the configured concurrency limit and
  preserve profile context; oversized batch items do not serialize the full import.
- Identity recovery algorithm version 5 rejects reuse of older recovery outputs
  with incompatible cached summaries. Choose a new destination for recovery.


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

Regressions exercise logical message/version identity, current-versus-audit retrieval,
offline recovery, and real session/hook lifecycle. Cancellation tests keep the summary
provider blocked while the caller exits; historical image tests observe filesystem I/O
and retrieve distinct image bytes, rather than patching the plugin's own helpers.
The archive-copy boundary cases use the real 20 MiB limit.

Recovered unambiguous raw tool-result text must be available through search and exact
zoom, not merely stored in a payload column. Cached spans based on pruned text are not
reused after restoring that original. The recovery signature includes the algorithm
version: outputs produced by an older algorithm require a **new destination**, not
idempotent reuse or an in-place rewrite.

## Notes

- **Caching:** The view is split after a stable head so providers that need cache markers reuse it across turns.

## License

[MIT](LICENSE).
