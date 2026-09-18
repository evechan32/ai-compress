"""入口点集成检查：不手动 install()，完全依赖 vllm.general_plugins 入口。

若入口点按 vLLM 版本正确分派，应能看到 `[PE11] installed`（0.11.x）并发生真实驱逐。

跑法：
    cd /root/ai-compress
    TMPDIR=/dev/shm/tmp PYTHONPATH=/hy-tmp/t29:/root/ai-compress \
    PE_MODE=chunkkv PE_RATIO=0.5 PE_SINK=0 PE_OBS=32 PE_WIN_AGG=max PE_LOG=1 \
      python3 bench/entrypoint_check.py
"""
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_MODE", "chunkkv")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("PE_MODEL", "/dev/shm/models/Qwen2.5-1.5B-Instruct")


def main():
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.6,
              enforce_eager=True, disable_log_stats=True)
    body = "The harbor master logged every vessel that passed the north pier. " * 90
    prompt = ("Read the document.\n\n" + body
              + "\n\nContinue the document verbatim, do not stop:")
    o = llm.generate([prompt], SamplingParams(max_tokens=64, temperature=0))
    print(f"EP_CHECK mode={os.environ.get('PE_MODE')} "
          f"prompt_tokens={len(llm.get_tokenizer().encode(prompt))} "
          f"out={o[0].outputs[0].text[:60]!r}", flush=True)


if __name__ == "__main__":
    main()
