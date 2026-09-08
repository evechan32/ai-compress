"""kvcompress: vLLM 连续区间 KV 驱逐插件入口。"""
from __future__ import annotations

_ENV_SETS = {
    # sm_120 (RTX 50xx) 上 flashinfer JIT 采样需 CUDA>=12.9；统一走 vLLM 采样器
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}


def _apply_env_defaults() -> None:
    import os
    for k, v in _ENV_SETS.items():
        os.environ.setdefault(k, v)


def entrypoint():
    """vllm.general_plugins 入口：所有进程加载时执行。"""
    _apply_env_defaults()
    from kvcompress import config
    cfg = config.load_config()
    if not cfg.enabled:
        return  # 旁路：零副作用
    from kvcompress import adapter  # noqa: 仅在启用时导入，避免 vllm 耦合提前加载
    adapter.install(cfg)
