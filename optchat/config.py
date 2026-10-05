from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Settings:
    summary_boundary: str = "event"
    compression_policy: str = "eager"
    batching: str = "off"
    chunk_limit: str = "sentences"
    node_bytes: int = 512
    turn_node_bytes: int = 2048
    turn_source_bytes: int = 32000
    compression_trigger: float = 0.8
    batch_input_bytes: int = 128000
    batch_output_bytes: int = 16384
    batch_wait_seconds: float = 0.1
    view_bytes: int = 128000
    settle_seconds: float = 300
    retry_seconds: float = 10
    summary_tries: int = 5
    summary_timeout: float = 60
    tool_chars: int = 30000
    merge_lanes: int = 4
    cache_split: float = 0.3

    def __post_init__(self):
        for name, choices in (("summary_boundary",("event","user_turn")),
                              ("compression_policy",("eager","on_demand")),
                              ("batching",("off","auto")),
                              ("chunk_limit",("bytes","characters","sentences"))):
            if getattr(self,name) not in choices:
                raise ValueError(f"OptChat {name} must be one of {', '.join(choices)}")
        for name in ("node_bytes","view_bytes","summary_tries","tool_chars","merge_lanes",
                     "turn_node_bytes","turn_source_bytes","batch_input_bytes","batch_output_bytes"):
            value = getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int):
                raise ValueError(f"OptChat {name} must be an integer")
        for name in ("settle_seconds","retry_seconds","summary_timeout","cache_split",
                     "compression_trigger","batch_wait_seconds"):
            value = getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f"OptChat {name} must be a finite number")
        if self.node_bytes < 256 or self.view_bytes < 2 * self.summary_bytes + 128:
            raise ValueError("OptChat needs node_bytes >= 256 and room for two selected summary nodes in view_bytes")
        if min(self.settle_seconds, self.retry_seconds, self.summary_timeout, self.summary_tries, self.merge_lanes) <= 0:
            raise ValueError("OptChat timeouts, attempts and lanes must be positive")
        if not 0 <= self.cache_split < 1:
            raise ValueError("OptChat cache_split must be in [0, 1)")
        if self.tool_chars < 1000:
            raise ValueError("OptChat needs tool_chars >= 1000")
        if self.turn_node_bytes < 256 or self.turn_source_bytes < self.turn_node_bytes:
            raise ValueError("OptChat needs turn_source_bytes >= turn_node_bytes >= 256")
        if not 0 < self.compression_trigger <= 1 or self.batch_wait_seconds < 0:
            raise ValueError("OptChat needs 0 < compression_trigger <= 1 and batch_wait_seconds >= 0")
        if self.batch_input_bytes < 1024 or self.batch_output_bytes < self.summary_bytes:
            raise ValueError("OptChat batch budgets must fit a summary and at least 1024 input bytes")

    @property
    def summary_bytes(self):
        return self.turn_node_bytes if self.summary_boundary == "user_turn" else self.node_bytes

    @property
    def summary_limit(self):
        from .limits import SummaryLimit
        target = self.summary_bytes if self.chunk_limit == "bytes" else 500 if self.chunk_limit == "characters" else 5
        return SummaryLimit(self.chunk_limit,target)

    @classmethod
    def load(cls):
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("optchat", {})
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown OptChat settings: {sorted(unknown)}")
        return cls(**raw)
