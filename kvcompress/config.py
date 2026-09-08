"""AI_COMPRESS_* 配置解析。纯 python，禁止 import vllm/torch。"""
from __future__ import annotations

import os
from dataclasses import dataclass

BLOCK_SIZE = 16  # vLLM 默认 block_size（向上对齐用）

_POLICIES = ("rswa", "sink_window")


def _aligned(n: int, block: int = BLOCK_SIZE) -> int:
    return ((n + block - 1) // block) * block


@dataclass(frozen=True)
class CompressConfig:
    enabled: bool = False
    policy: str = "rswa"
    rswa_window: int = 1024
    sink_len: int = 64
    target_archs: tuple[str, ...] = ("Qwen2ForCausalLM",)

    @property
    def policy_obj(self):
        from kvcompress.policy import RSWAPolicy, SinkWindowPolicy  # 延迟导入避免循环
        if self.policy == "rswa":
            return RSWAPolicy(window=self.rswa_window)
        return SinkWindowPolicy(window=self.rswa_window, sink_len=self.sink_len)


def load_config() -> CompressConfig:
    if os.environ.get("AI_COMPRESS_ENABLE", "0") not in ("1", "true", "True"):
        return CompressConfig(enabled=False)

    policy = os.environ.get("AI_COMPRESS_POLICY", "rswa")
    if policy not in _POLICIES:
        raise ValueError(f"AI_COMPRESS_POLICY must be one of {_POLICIES}, got {policy!r}")

    rswa_window = int(os.environ.get("AI_COMPRESS_RSWA_WINDOW", "1024"))
    if rswa_window < BLOCK_SIZE:
        raise ValueError(f"AI_COMPRESS_RSWA_WINDOW must be >= {BLOCK_SIZE}")
    rswa_window = _aligned(rswa_window)

    sink_len = int(os.environ.get("AI_COMPRESS_SINK_LEN", "64"))
    if policy == "sink_window" and sink_len < BLOCK_SIZE:
        raise ValueError(f"AI_COMPRESS_SINK_LEN must be >= {BLOCK_SIZE}")
    sink_len = _aligned(sink_len)

    archs = tuple(
        a.strip()
        for a in os.environ.get(
            "AI_COMPRESS_TARGET_ARCHS", "Qwen2ForCausalLM"
        ).split(",")
        if a.strip()
    )
    if not archs:
        raise ValueError("AI_COMPRESS_TARGET_ARCHS is empty")
    return CompressConfig(
        enabled=True, policy=policy, rswa_window=rswa_window,
        sink_len=sink_len, target_archs=archs,
    )
