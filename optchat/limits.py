"""Summary size instructions and local acceptance checks."""
import re
from dataclasses import dataclass


# Count sentence-ending punctuation followed by whitespace/end, ignoring decimal
# points and dots inside identifiers. A remaining fragment counts as a sentence.
SENTENCE_END = re.compile(r'''[.!?]+["')\]”’»]*(?=\s|$)''')
MAX_HISTORY_BYTES = 512000


def measure(text, unit):
    if unit == "bytes":
        return len(text.encode("utf-8"))
    if unit == "characters":
        return len(text)
    ends = list(SENTENCE_END.finditer(text))
    return len(ends)+bool(text[ends[-1].end() if ends else 0:].strip())


@dataclass(frozen=True)
class SummaryLimit:
    unit: str
    target: int

    @property
    def maximum(self):
        # The default byte target remains 512; allow about 7.4% overshoot (550).
        return (self.target*550+511)//512 if self.unit == "bytes" else self.target

    @property
    def output_bytes(self):
        # Characters can occupy four UTF-8 bytes. Sentences have no byte bound;
        # reserve 400 bytes each for batch planning, then fit actual stored sizes.
        return self.maximum if self.unit == "bytes" else self.target*(4 if self.unit == "characters" else 400)

    def measure(self, text):
        return measure(text,self.unit)

    def accepts(self, text):
        return bool(text.strip()) and self.measure(text) <= self.maximum

    def can_copy(self, text):
        return self.accepts(text) and len(text.encode("utf-8")) <= MAX_HISTORY_BYTES

    def clip(self, text):
        if self.unit == "bytes":
            return text.encode("utf-8")[:self.target].decode("utf-8",errors="ignore")
        if self.unit == "characters":
            return text[:self.target]
        ends = list(SENTENCE_END.finditer(text))
        return text[:ends[self.target-1].end()] if len(ends) >= self.target else text

    def fallback(self, source, address):
        label = f"[summary unavailable; zoom {address}] "
        if self.unit in ("bytes","characters"):
            remaining = max(0,self.target-self.measure(label))
            excerpt = SummaryLimit(self.unit,remaining).clip(source)
        else:
            excerpt = self.clip(source)
        text = label+excerpt
        return text.encode("utf-8")[:MAX_HISTORY_BYTES//4].decode("utf-8",errors="ignore").strip()

    def prompt(self, prompt):
        if self.unit == "bytes":
            return prompt
        prompt = re.sub(r"; non-ASCII\s+characters cost 2-4 bytes\.",".",prompt)
        return prompt.replace("within its UTF-8 byte target","within its requested length target")

    def instruction(self, action, source):
        scale = ""
        if self.unit != "sentences":
            example = "-"*self.target
            scale = f"For scale, this line is exactly {self.target} {self.unit}:\n{example}\n\n"
        detail = ", including spaces and punctuation" if self.unit == "characters" else ""
        return scale+f"{action} into one line, in at most {self.target} {self.unit}{detail}:\n{source}"

    def feedback(self, text):
        tolerance = f" (accepted maximum {self.maximum})" if self.unit == "bytes" else ""
        return (f"That line is {self.measure(text)} {self.unit}; the target is {self.target}{tolerance}. "
                "Rewrite the original source more concisely, preserving its essential facts and references.")
