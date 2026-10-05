"""Deterministic, bounded turn leaves with exact original event/character spans."""
from .store import size


def capped(text, limit):
    if len(text) <= limit:
        return text
    half = limit//2
    return text[:half]+f"\n[... {len(text)-2*half} characters cut ...]\n"+text[-half:]


def pieces(event, budget, tool_chars):
    text, kind, event_id = event["text"], event["kind"], event["id"]
    label = kind+": "
    available = budget-size(label)
    if kind in ("tool","echo"):
        source = capped(text,tool_chars)
        if size(source) > available:
            # The span still covers the whole exact event; only model input is capped.
            marker = "\n[... tool text cut; retrieve original ...]\n"
            half = (available-size(marker))//2
            data = source.encode("utf-8")
            source = data[:half].decode("utf-8",errors="ignore")+marker+data[-half:].decode("utf-8",errors="ignore")
        yield event_id,event_id+1,0,len(text),label+source
        return
    offset = 0
    data, byte_offset = text.encode("utf-8"), 0
    while offset < len(text):
        piece = data[byte_offset:byte_offset+available].decode("utf-8",errors="ignore")
        # Prefer a nearby word/newline boundary without dropping any characters.
        if offset+len(piece) < len(text):
            cut = max(piece.rfind("\n"),piece.rfind(" "))+1
            if cut > len(piece)*3//4:
                piece = piece[:cut]
        yield event_id,event_id+1,offset,offset+len(piece),label+piece
        offset += len(piece)
        byte_offset += size(piece)
    if not text:
        yield event_id,event_id+1,0,0,label


def turn_chunks(events, budget, tool_chars):
    """Prefer whole events and keep a tool call/result pair together when it fits."""
    pending = None
    group = []
    def joined(parts):
        first,last = parts[0],parts[-1]
        return first[0],last[1],first[2],last[3],"\n".join(p[4] for p in parts)
    def atoms():
        nonlocal pending
        for event in events:
            for piece in pieces(event,budget,tool_chars):
                if pending is not None:
                    if pending[4].startswith("tool: ") and event["kind"] in ("echo","work") and size(pending[4])+1+size(piece[4]) <= budget:
                        yield [pending,piece]
                        pending = None
                        continue
                    yield [pending]
                pending = piece
        if pending is not None:
            yield [pending]
    for atom in atoms():
        if group and size("\n".join(p[4] for p in group+atom)) > budget:
            yield joined(group)
            group = []
        group.extend(atom)
    if group:
        yield joined(group)
