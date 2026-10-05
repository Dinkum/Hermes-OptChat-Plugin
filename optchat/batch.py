"""Independent summary jobs and a provider-neutral, item-validated JSON envelope."""
import json
from dataclasses import dataclass

from .store import encoded


@dataclass(frozen=True)
class SummaryJob:
    id: str
    context: str
    source: str
    target_bytes: int
    action: str
    limit_unit: str = "bytes"
    target_length: int | None = None


def messages(jobs, prompt):
    return [{"role":"system","content":prompt+"\n\nSummarize each job independently using only its context and source. "
             "Return a JSON object with a summaries array of objects containing id and text. "
             "Return every supplied id exactly once; no extra ids, prose or Markdown fences. "
             "Each text is a nonempty summary within that job's target_length in limit_unit. "
             "Bytes means UTF-8 bytes; characters means Unicode code points including spaces and punctuation; "
             "sentences means sentences, with no separate byte or character limit."},
            {"role":"user","content":encoded({"jobs":[{"id":job.id,"context":job.context,"source":job.source,
                "action":job.action,"limit_unit":job.limit_unit,
                "target_length":job.target_length if job.target_length is not None else job.target_bytes}
                for job in jobs]})}]


def results(text, expected):
    try:
        payload = json.loads(text)
    except (ValueError,TypeError):
        return {}
    if not isinstance(payload,dict) or not isinstance(payload.get("summaries"),list):
        return {}
    found, duplicates = {}, set()
    for item in payload["summaries"]:
        if not isinstance(item,dict) or not isinstance(item.get("id"),str):
            continue
        key = item["id"]
        if key in found:
            duplicates.add(key)
        elif key in expected:
            # Retain invalid text too, so a duplicate cannot overwrite it with a
            # valid-looking value and silently select contradictory responses.
            found[key] = item.get("text")
    return {key:value.strip() for key,value in found.items()
            if key not in duplicates and isinstance(value,str) and value.strip()}
