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
        return self.target

    @property
    def output_bytes(self):
        # Characters can occupy four UTF-8 bytes. Sentences have no byte bound;
        # reserve 400 bytes each for batch planning, then fit actual stored sizes.
        return self.maximum if self.unit == "bytes" else self.target*(4 if self.unit == "characters" else 400)

    def measure(self, text):
        return measure(text,self.unit)

    def accepts(self, text):
        return bool(text.strip()) and self.measure(text) <= self.maximum

    def can_copy(self, text, *, max_bytes=MAX_HISTORY_BYTES):
        return self.accepts(text) and len(text.encode("utf-8")) <= min(MAX_HISTORY_BYTES,max_bytes)


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
        actual = self.measure(text)
        feedback = (f"That line is {actual} {self.unit}; the hard maximum is {self.target}. "
                    "Rewrite the whole line for the original source, cutting the least valuable details.")
        if self.unit == "bytes":
            # Correct the observed overestimate with a safety margin; stored
            # acceptance still uses the configured ceiling, not this retry goal.
            goal = max(1,self.target*self.target*9//(max(1,actual)*10))
            prefix = text.encode("utf-8")[:goal].decode("utf-8",errors="ignore")
            feedback += (f"\nAim for at most {goal} UTF-8 bytes on this rewrite, below the hard ceiling. "
                         f"Finish before this boundary:\n{prefix}| <- LIMIT")
        return feedback
