"""服务器集成测试（需 vllm 0.28 + Qwen2.5 模型）。

1. test_bypass_plugin_inert_when_disabled：插件安装后 ENABLE=0（默认）输出与未启用时一致。
2. test_enabled_rejects_unknown_arch：ENABLE=1 + 未知架构 → 启动期报错并拒绝服务。
"""
import os
import subprocess
import sys
import textwrap

import pytest

REPO = "/root/ai-compress"
MODEL = "/models/qwen2.5-1.5b-instruct"

if not os.path.exists(MODEL):
    pytest.skip("服务器专用测试：缺少 vLLM 运行环境/模型路径", allow_module_level=True)

_CODE = textwrap.dedent(f"""
    import sys
    sys.path.insert(0, {REPO!r})
    from vllm import LLM, SamplingParams
    llm = LLM(model={MODEL!r}, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, enforce_eager=True)
    out = llm.generate(["The capital of France is"],
                       SamplingParams(max_tokens=16, temperature=0.0))
    print("OUT:" + out[0].outputs[0].text)
""")


def _run(extra_env: dict) -> tuple[int, str, str]:
    env = dict(os.environ, **extra_env)
    r = subprocess.run([sys.executable, "-c", _CODE], capture_output=True,
                       text=True, env=env, timeout=600, cwd=REPO)
    out_line = next((ln for ln in r.stdout.splitlines() if ln.startswith("OUT:")), "")
    return r.returncode, out_line, r.stderr


def test_bypass_plugin_inert_when_disabled():
    env = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    rc1, out1, _ = _run(env)                       # 未设 AI_COMPRESS_*（默认关闭）
    assert rc1 == 0, out1
    rc2, out2, _ = _run({**env, "AI_COMPRESS_ENABLE": "0"})  # 显式关闭
    assert rc2 == 0, out2
    assert out1 == out2
    assert out1.startswith("OUT: Paris")           # 与预插件冒烟参考一致（确定性 temp0）


def test_enabled_rejects_unknown_arch():
    env = {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "AI_COMPRESS_ENABLE": "1",
        "AI_COMPRESS_TARGET_ARCHS": "NotARealArch",
    }
    rc, out, err = _run(env)
    assert rc != 0
    assert ("不支持的架构" in err) or ("不支持" in err), err[-1500:]
