from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    node_bytes: int = 512
    view_bytes: int = 128000
    settle_seconds: float = 300
    retry_seconds: float = 10
    summary_tries: int = 5
    summary_timeout: float = 60
    tool_chars: int = 30000
    merge_lanes: int = 4
    cache_split: float = 0.3

    def __post_init__(self):
        if self.node_bytes < 256 or self.view_bytes < 2 * self.node_bytes + 128:
            raise ValueError("OptChat needs node_bytes >= 256 and room for two nodes in view_bytes")
        if min(self.settle_seconds, self.retry_seconds, self.summary_timeout, self.summary_tries, self.merge_lanes) <= 0:
            raise ValueError("OptChat timeouts, attempts and lanes must be positive")
        if not 0 <= self.cache_split < 1:
            raise ValueError("OptChat cache_split must be in [0, 1)")
        if self.tool_chars < 1000:
            raise ValueError("OptChat needs tool_chars >= 1000")

    @classmethod
    def load(cls):
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("optchat", {})
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown OptChat settings: {sorted(unknown)}")
        return cls(**raw)
