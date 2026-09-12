"""adapter：目标架构 → RSWA 受限窗口注意力的运行时注入（零 fork）。

形式 M（依据 docs/spike-r1-findings.md）：
1. 把目标架构模型模块命名空间中的 ``Attention`` 名替换为"可配置 rswa_window 的
   RSWAAttention 工厂"。``RSWAAttention.get_kv_cache_spec`` 返回 RSWASpec
   → 缓存管理器实例化 RSWAManager，逐 decode 步驱逐 prefill 尾与生成窗口间的 gap 块。
2. 覆盖 ``ModelConfig.rswa_window``：底层 HF config 未声明窗口时返回策略窗口，
   且仅当模型架构命中 ``AI_COMPRESS_TARGET_ARCHS``（避免影响非目标模型）。
3. 不支持的架构 / sink_window 策略（v1.x 特性）→ 显式拒绝，避免静默跑全注意力。

约束：本模块内的 vllm 导入全部延迟到函数内，保证纯配置/策略层可本地单测。
"""
from __future__ import annotations

import logging

from kvcompress.config import CompressConfig

logger = logging.getLogger("kvcompress")

# 架构 -> 含该架构 Attention 层的 vllm 模型模块（随支持扩展增加）
_ARCH_TO_MODULE = {
    "Qwen2ForCausalLM": "vllm.model_executor.models.qwen2",
    "Qwen3ForCausalLM": "vllm.model_executor.models.qwen3",
}

_target_archs: frozenset[str] = frozenset()
_window: int = 0
_property_patched: bool = False


def install(cfg: CompressConfig) -> None:
    """在启用时对目标架构注入 RSWA 路径。幂等。"""
    global _target_archs, _window
    if not cfg.enabled:
        return
    if cfg.policy == "sink_window":
        raise NotImplementedError(
            "AI_COMPRESS_POLICY=sink_window 为 v1.x 特性，v1 仅支持 policy=rswa。"
        )
    unsupported = [a for a in cfg.target_archs if a not in _ARCH_TO_MODULE]
    if unsupported:
        raise ValueError(
            f"AI_COMPRESS_TARGET_ARCHS 含不支持的架构 {unsupported}；"
            f"当前支持 {sorted(_ARCH_TO_MODULE)}。"
            "拒绝启动以避免'以为在压缩、实际跑全注意力'的静默错误。"
        )
    _target_archs = frozenset(cfg.target_archs)
    _window = cfg.rswa_window
    _patch_rswa_window_property()
    _patch_backend_guard()
    for arch in cfg.target_archs:
        _patch_attention_factory(_ARCH_TO_MODULE[arch])
    logger.info(
        "kvcompress: RSWA 注入生效 archs=%s window=%d",
        sorted(_target_archs), _window,
    )


_RSWA_OK_BACKENDS = frozenset({"TritonAttentionBackend", "FlexAttentionBackend"})


def _backend_supports_rswa(cls) -> bool:
    """RSWA 掩码仅在实现了它的 backend 生效；FA<4 会静默忽略掩码。"""
    if cls.__name__ in _RSWA_OK_BACKENDS:
        return True
    if "flash_attn" in getattr(cls, "__module__", ""):
        try:
            from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
            return (get_flash_attn_version() or 0) >= 4
        except Exception:
            return False
    return False


def _patch_backend_guard() -> None:
    """拒绝"掩码被忽略但仍释放 KV 块"的 backend（会读已释放显存）。"""
    import importlib
    import os

    mod = importlib.import_module("vllm.model_executor.layers.attention.attention")
    if getattr(mod, "_kvcompress_guard", False):
        return
    orig = mod.get_attn_backend

    def guarded(*args, **kwargs):
        cls = orig(*args, **kwargs)
        if _backend_supports_rswa(cls):
            return cls
        if os.environ.get("AI_COMPRESS_FORCE_BACKEND", "0") == "1":
            from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
            logger.warning(
                "kvcompress: %s 不支持 RSWA 掩码，已替换为 "
                "TritonAttentionBackend（AI_COMPRESS_FORCE_BACKEND=1）",
                cls.__name__,
            )
            return TritonAttentionBackend
        raise RuntimeError(
            f"kvcompress RSWA 需要支持 RSWA 掩码的 attention backend，但选中了 "
            f"{cls.__name__}：它不会应用掩码，却仍会释放 KV 块，导致注意力读取"
            f"已释放/复用的显存（静默错误 + 数据损坏）。"
            f"请加 --attention-backend TRITON_ATTN，"
            f"或设 AI_COMPRESS_FORCE_BACKEND=1 由插件自动替换。"
        )

    mod.get_attn_backend = guarded
    mod._kvcompress_guard = True


def _patch_attention_factory(module_name: str) -> None:
    """把模块命名空间中的 ``Attention`` 名替换为 RSWAAttention 工厂。"""
    import importlib

    from vllm.model_executor.layers.attention.rswa_attention import RSWAAttention

    mod = importlib.import_module(module_name)
    if not hasattr(mod, "Attention"):
        raise ValueError(f"{module_name} 未导出 Attention 名称，无法注入 RSWA 路径")
    window = _window

    def _factory(*args, **kwargs):
        return RSWAAttention(*args, rswa_window=window, **kwargs)

    mod.Attention = _factory
    logger.debug("kvcompress: patched %s.Attention -> RSWAAttention(window=%d)",
                 module_name, window)


def _patch_rswa_window_property() -> None:
    """覆盖 ModelConfig.rswa_window：命中目标架构且底层为 None 时返回策略窗口。"""
    global _property_patched
    if _property_patched:
        return
    from vllm.config.model import ModelConfig

    target = _target_archs
    window = _window
    _orig = (
        ModelConfig.rswa_window.fget
        if isinstance(ModelConfig.rswa_window, property)
        else None
    )

    def _prop(self):
        v = _orig(self) if _orig is not None else None
        if v is not None:
            return v
        archs = getattr(getattr(self, "hf_config", None), "architectures", None) or []
        if any(a in target for a in archs):
            return window
        return None

    ModelConfig.rswa_window = property(_prop)
    _property_patched = True
    logger.debug("kvcompress: patched ModelConfig.rswa_window (target=%s window=%d)",
                 sorted(target), window)
