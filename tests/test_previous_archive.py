"""Recovered agents can explicitly retrieve source-only originals without reviving them."""
import base64
import hashlib

import pytest

from optchat.recovery import recover_archive
from test_identity_recovery import (
    PNG_BYTES, SESSION, SETTINGS, engine_at, legacy_archive, message, tool_result, workspace,
)


def test_previous_generation_is_explicit_exact_paginated_and_read_only(workspace, engine_at):
    root, home, conn = workspace
    message(conn, 1, "current", "Current instruction")
    original = "old-original:" + "x" * 40000
    previous = legacy_archive(root, [(0, "row:999", "user", original, "null", 1.0)])
    before = hashlib.sha256(previous.read_bytes()).hexdigest()
    recover_archive(previous, home / "state.db", home / "optchat", SESSION, settings=SETTINGS)
    engine = engine_at()

    current = tool_result(engine, "optchat_zoom", {"id": 0, "n": 1})
    assert current["text"] == "Current instruction"
    old = tool_result(engine, "optchat_zoom", {"id": 0, "n": 1, "archive": "previous"})
    assert old["text"] == original[:30000]
    assert old["next_offset"] == 30000
    tail = tool_result(engine, "optchat_zoom", {"id": 0, "n": 1, "offset": 30000, "archive": "previous"})
    assert tail["text"] == original[30000:]
    assert tail["next_offset"] is None
    assert hashlib.sha256(previous.read_bytes()).hexdigest() == before


def test_previous_search_never_reintroduces_withdrawn_text_into_current_memory(workspace, engine_at):
    root, home, conn = workspace
    message(conn, 1, "current", "unique-current")
    previous = legacy_archive(root, [(0, "row:999", "user", "unique-retired", "null", 1.0)])
    recover_archive(previous, home / "state.db", home / "optchat", SESSION, settings=SETTINGS)
    engine = engine_at()

    assert tool_result(engine, "optchat_search", {"query": "unique-retired"})["matches"] == []
    previous_result = tool_result(engine, "optchat_search", {
        "query": "unique-retired", "archive": "previous",
    })
    assert previous_result["matches"] == [{"id": 0, "kind": "user", "excerpt": "unique-retired"}]
    assert tool_result(engine, "optchat_search", {"query": "unique-retired"})["matches"] == []
    assert "unique-retired" not in engine.worker.store.render()


@pytest.mark.parametrize("size", [20 * 1024 * 1024, 20 * 1024 * 1024 + 1])
def test_historical_embedded_image_respects_archive_copy_limit(engine_at, size):
    # Exercise the real documented cap, not a monkeypatched module constant.
    original = PNG_BYTES + b"\0" * (size - len(PNG_BYTES))
    payload = [{"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(original).decode("ascii"),
    }}]
    engine = engine_at()
    event = engine.worker.store.append(
        "historical-photo", "user", "Preserved photo", payload, historical=True,
    )
    result = tool_result(engine, "optchat_zoom", {"id": event, "n": 1})
    image = result["images"][0]
    assert not image.get("attached", False)
    # Accepted archive copies still exceed the independent native-zoom size cap.
    assert (image["error"] == "SourceTooLarge") is (size > 20 * 1024 * 1024)
