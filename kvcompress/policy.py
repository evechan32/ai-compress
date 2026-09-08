"""驱逐策略：给定已计算 token 计数，返回应保留的连续 [start,end) token 区间。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class EvictionPolicy(ABC):
    @abstractmethod
    def retain_ranges(self, num_prefix: int, num_generated: int) -> list[tuple[int, int]]:
        """num_prefix=prompt(prefill) token 数；num_generated=已生成 token 数。"""


class RSWAPolicy(EvictionPolicy):
    """prompt 全保留；生成段只保留末尾 window。语义对齐 vLLM RSWASpec。"""

    def __init__(self, window: int):
        self.window = window

    def retain_ranges(self, num_prefix: int, num_generated: int) -> list[tuple[int, int]]:
        if num_generated <= self.window:
            return [(0, num_prefix + num_generated)]
        gen_start = num_prefix + (num_generated - self.window)
        return [(0, num_prefix), (gen_start, num_prefix + num_generated)]


class SinkWindowPolicy(EvictionPolicy):
    """整段上下文只保留开头 sink 与末尾 window（StreamingLLM 式，v1.x 默认关闭）。"""

    def __init__(self, window: int, sink_len: int):
        self.window = window
        self.sink_len = sink_len

    def retain_ranges(self, num_prefix: int, num_generated: int) -> list[tuple[int, int]]:
        total = num_prefix + num_generated
        if total <= self.sink_len + self.window:
            return [(0, total)]
        return [(0, self.sink_len), (total - self.window, total)]
