# Hermes OptChat Plugin

> This idea is entirely [Victor Taelin](https://x.com/VictorTaelin)'s I just built it in the form of a Hermes plugin. Please check out [his gist on this](https://gist.github.com/VictorTaelin/91837951a5ce5b38f341ec1ba1df6449) and [original OptMem](https://github.com/VictorTaelin/OptMem) repo!

Manage agents instead of *agent context* by using an endless, continually compressed chat thread. The chat thread becomes the agent memory. Context gets compressed into a binary tree that gets zoomed into as needed. No context rot yet we get infinite context with a constant size.

Summaries are lossy; exact originals remain available. Background summaries use your configured model and consume provider tokens.

## Requirements

- macOS or Linux
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) version `21.5` (source installation)
- A configured model

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

## Configuration

[Settings](config.example.yaml).

`chunk_limit` options: `bytes` (512, accepts up to 550), `characters` (500), or `sentences` (5). Defaults to `sentences`. The original spec uses 512 bytes, but my testing showed models that don’t support enforced output-length limits wasting reasoning tokens on the limit itself.

## Notes

- **Caching:** The view is split after a stable head so providers that need cache markers reuse it across turns.

## License

[MIT](LICENSE).
