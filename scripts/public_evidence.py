"""Redact local machine metadata from synthetic public validation captures."""
import re
from pathlib import Path


def redact(text):
    text = re.sub(r"/Users/[^/\s\"\\<>]+/(?:Documents/Codex/\d{4}-\d{2}-\d{2}/make-me-an-optchat-hermes-plugin|Desktop/Github/Hermes-OptChat-Plugin)",
                  "/workspace/optchat-hermes", text)
    text = re.sub(r"/Users/[^/\s\"\\<>]+", "/home/user", text)
    text = re.sub(r"/private/tmp/optchat-[^/\s\"\\<>]+", "/tmp/optchat-check", text)
    return re.sub(r"Host: macOS \([^)]+\)", "Host: macOS (version redacted)", text)


def sanitize(root):
    changed = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".jsonl", ".html"):
            continue
        raw = path.read_text(encoding="utf-8")
        public = redact(raw)
        if public != raw:
            path.write_text(public, encoding="utf-8")
            changed.append(str(path))
    return changed


if __name__ == "__main__":
    for path in sanitize(Path(__file__).resolve().parents[1] / "evidence"):
        print(path)
