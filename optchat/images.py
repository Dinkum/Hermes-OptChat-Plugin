"""Attachment preservation and bounded native zoom results."""
import asyncio
import base64
import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARCHIVE_BYTES = 20 * 1024 * 1024
PAGE_BYTES = 10 * 1024 * 1024
IMAGE_BYTES = 5 * 1024 * 1024
PAGE_COUNT = 4
WIRE_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def sources(payload):
    if isinstance(payload, list):
        for part in payload:
            yield from sources(part)
    elif isinstance(payload, dict):
        kind = payload.get("type")
        value = None
        if kind in ("image_url", "input_image"):
            value = payload.get("image_url") or payload.get("url")
            if isinstance(value, dict):
                value = value.get("url")
        elif kind == "image":
            source = payload.get("source", {})
            if isinstance(source, dict):
                if source.get("type") == "base64":
                    value = f"data:{source.get('media_type','')};base64,{source.get('data','')}"
                else:
                    value = source.get("url")
        if isinstance(value, str) and value:
            yield value
        elif kind in ("image", "image_url", "input_image"):
            yield ""
        if isinstance(payload.get("content"), list):
            yield from sources(payload["content"])


def local_copy(source):
    from tools.image_source import (
        ResolveContext, SourceNotFound, SourceTooLarge, _finalize,
        _guard_credential_read, _permitted_host_read_target, _resolve_data_url,
    )
    if source.startswith(("http://", "https://")):
        return None
    if source.startswith("data:"):
        if len(source) > ARCHIVE_BYTES * 4 // 3 + 1024:
            raise SourceTooLarge("Image exceeds the 20 MiB archive-copy limit")
        data, mime = _resolve_data_url(source)
    else:
        path = Path(source.removeprefix("file://")).expanduser()
        target = _permitted_host_read_target(path, ResolveContext())
        if target is None:
            return None
        _guard_credential_read(target, source)
        if not target.is_file():
            raise SourceNotFound("Image file is unavailable")
        with target.open("rb") as stream:
            data = stream.read(ARCHIVE_BYTES + 1)
        mime = ""
    if len(data) > ARCHIVE_BYTES:
        raise SourceTooLarge("Image exceeds the 20 MiB archive-copy limit")
    return _finalize(data, mime, "file", source, ("image",))


def resolve(source):
    from tools.image_source import ResolveContext, resolve_image_source
    async def load():
        return await asyncio.wait_for(resolve_image_source(source, ResolveContext()), timeout=10)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(load())
    # Context-engine tools can be dispatched from an already-running async loop.
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(context.run, lambda: asyncio.run(load())).result()


def embedded_copy(source):
    if not source.startswith("data:"):
        raise ValueError("Historical embedded images require a data URL")
    return local_copy(source)


def capture(db, event_id, payload, *, historical=False):
    for ordinal, source in enumerate(sources(payload)):
        try:
            if historical:
                embedded = isinstance(source,str) and source.strip().startswith("data:")
                image = embedded_copy(source.strip()) if embedded else None
                error = None if embedded else "Historical image bytes were not preserved"
            else:
                image = local_copy(source) if source else None
                error = None if source else "Unsupported image reference"
            data, mime = (image.data, image.mime) if image else (None, None)
        except Exception as exc:
            data, mime, error = None, None, type(exc).__name__
        # The original payload retains the reference; embedded/local bytes survive cache cleanup.
        db.execute("INSERT OR IGNORE INTO images VALUES (?,?,?,?,?)",
                   (event_id, ordinal, mime, data, error))


def page(store, event_id, offset, *, read_only=False):
    with store.lock:
        refs = list(sources(json.loads(store.event(event_id)["payload"])))
        has_images = not read_only or store.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='images'").fetchone() is not None
    if type(offset) is not int or offset < 0 or offset > len(refs):
        raise ValueError("Invalid image offset")
    blocks, items, used, stop = [], [], 0, offset
    for ordinal in range(offset, min(len(refs), offset + PAGE_COUNT)):
        with store.lock:
            row = (store.db.execute("SELECT mime,data,error FROM images WHERE event=? AND ordinal=?", (event_id, ordinal)).fetchone()
                   if has_images else None)
        mime, data, error = tuple(row) if row else (None, None, None)
        if data is None and error is None:
            if read_only:
                if isinstance(refs[ordinal],str) and refs[ordinal].strip().startswith("data:"):
                    # Embedded bytes are immutable evidence already in the original
                    # payload, unlike a URL or filename that might now resolve differently.
                    try:
                        image = embedded_copy(refs[ordinal].strip())
                        data,mime = image.data,image.mime
                    except Exception as exc:
                        error = type(exc).__name__
                else:
                    error = "Preserved image data is unavailable in the previous archive"
            else:
                try:
                    image = resolve(refs[ordinal])
                    if len(image.data) > ARCHIVE_BYTES:
                        raise ValueError("Image exceeds the 20 MiB archive-copy limit")
                    mime, data = image.mime, image.data
                    with store.lock, store.db:
                        store.db.execute("INSERT OR IGNORE INTO images VALUES (?,?,?,?,NULL)", (event_id, ordinal, mime, data))
                        store.db.execute("UPDATE images SET mime=?,data=? WHERE event=? AND ordinal=? AND data IS NULL",
                                         (mime, data, event_id, ordinal))
                except Exception as exc:
                    error = type(exc).__name__
        item = {"index": ordinal}
        if error:
            item["error"] = error
        elif mime not in WIRE_MIMES:
            item["error"] = "Image format cannot be attached natively; use vision_analyze with the original reference"
        elif len(data) > IMAGE_BYTES:
            item["error"] = "Image exceeds the 5 MiB zoom limit; use vision_analyze with the original reference"
        elif used + len(data) > PAGE_BYTES:
            break
        else:
            used += len(data)
            blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}})
            item.update(attached=True, mime=mime, bytes=len(data))
        items.append(item)
        stop = ordinal + 1
    return blocks, {"images": items, "image_offset": offset,
                    "next_image_offset": stop if stop < len(refs) else None, "total_images": len(refs)}
