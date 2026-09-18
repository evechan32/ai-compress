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


def _pevict_module_order() -> list[str]:
    """按 vLLM 版本决定优先实现。

    vLLM 的 minor 号在 0.11.x 与 0.28.x 之间跨越了 KV 管理器 API 的重构，
    两版实现不通用：0.11.x 用 `pevict_v11`，0.28+ 用 `pevict`。
    """
    import vllm

    parts = str(getattr(vllm, "__version__", "0.28")).split(".")
    try:
        minor = int(parts[1])
    except (IndexError, ValueError):
        minor = 28
    if minor < 28:
        return ["kvcompress.pevict_v11", "kvcompress.pevict"]
    return ["kvcompress.pevict", "kvcompress.pevict_v11"]


def entrypoint():
    """vllm.general_plugins 入口：所有进程加载时执行。"""
    _apply_env_defaults()
    import importlib
    import os
    if os.environ.get("PE_MODE", "off") != "off":
        last = None
        for name in _pevict_module_order():
            try:
                importlib.import_module(name).install()
                last = None
                break
            except Exception as e:
                last = e
        if last is not None:
            print(f"[PE] entrypoint install warn: {type(last).__name__} "
                  f"{str(last)[:160]}", flush=True)
    from kvcompress import config
    cfg = config.load_config()
    if not cfg.enabled:
        return  # 旁路：零副作用
    from kvcompress import adapter  # noqa: 仅在启用时导入，避免 vllm 耦合提前加载
    adapter.install(cfg)
