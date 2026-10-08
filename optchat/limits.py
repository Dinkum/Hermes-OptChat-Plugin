"""Summary size instructions and local acceptance checks."""
import re
from dataclasses import dataclass


# Count sentence-ending punctuation followed by whitespace/end, ignoring decimal
# points and dots inside identifiers. A remaining fragment counts as a sentence.
SENTENCE_END = re.compile(r'''[.!?]+["')\]”’»]*(?=\s|$)''')


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
        if self.unit == "bytes":
            return len(text.encode("utf-8"))
        if self.unit == "characters":
            return len(text)
        ends = list(SENTENCE_END.finditer(text))
        return len(ends)+bool(text[ends[-1].end() if ends else 0:].strip())

    def accepts(self, text):
        return bool(text.strip()) and self.measure(text) <= self.maximum

    def can_copy(self, text, source_bytes):
        # Sentence count alone cannot tell whether raw source is already short.
        # Compress long originals even if they contain little punctuation.
        return self.accepts(text) and (self.unit != "sentences" or len(text.encode("utf-8")) <= source_bytes)

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
        if self.unit == "bytes":
            cut = text.encode("utf-8")[:self.target].decode("utf-8",errors="ignore")
        elif self.unit == "characters":
            cut = text[:self.target]
        else:
            ends = list(SENTENCE_END.finditer(text))
            cut = text[:ends[self.target-1].end()]
        tolerance = f" (accepted maximum {self.maximum})" if self.unit == "bytes" else ""
        return (f"That line is {self.measure(text)} {self.unit}; the target is {self.target}{tolerance}. "
                f"It must end where it is cut here:\n{cut}| ← LIMIT")
