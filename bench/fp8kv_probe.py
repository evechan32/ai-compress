"""KV 量化探针：在 sm_86 上强开 fp8 KV 路径，量质量与性能。

vLLM 的 fp8 KV 被 `CudaPlatform.supports_fp8() = capability >= 89` 硬门限挡住；
但"低精度存储 + 反量化"在原理上不需要 sm_89 的 fp8 张量核，故强开门限测试：
  - fp8_e5m2：默认 scale=1.0，零 checkpoint 依赖
  - fp8_e4m3：需 scale，用 --calculate-kv-scales 动态算

跑法：
    cd /root/ai-compress
    TMPDIR=/dev/shm/tmp PYTHONPATH=/hy-tmp/t29:/root/ai-compress \
    PE_KVDTYPE=fp8_e5m2 PE_FORCE_FP8=1 python3 bench/fp8kv_probe.py
"""
import os
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

FORCE = os.environ.get("PE_FORCE_FP8", "1") == "1"
if FORCE:
    from vllm.platforms.cuda import CudaPlatform

    CudaPlatform.supports_fp8 = classmethod(lambda cls, *a, **k: True)

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("PE_MODEL", "/dev/shm/models/Qwen2.5-1.5B-Instruct")
KVD = os.environ.get("PE_KVDTYPE", "fp8_e5m2")
CALC = os.environ.get("PE_CALC_SCALES", "1") == "1"

FILLER = "The harbor master logged every vessel that passed the north pier. "


def main():
    kw = {}
    if KVD != "auto":
        kw["kv_cache_dtype"] = KVD
        kw["calculate_kv_scales"] = CALC
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.30,
              enforce_eager=True, max_num_seqs=8, disable_log_stats=True, **kw)
    tok = llm.get_tokenizer()
    body = FILLER * 150
    prompt = ("Read the document.\n\n" + body
              + "\n\nQuestion: What did the harbor master do?\nAnswer:")
    t0 = time.perf_counter()
    outs = llm.generate([prompt] * 4, SamplingParams(max_tokens=48, temperature=0))
    wall = time.perf_counter() - t0
    texts = [o.outputs[0].text[:70] for o in outs]
    same = len(set(texts)) == 1
    print(f"[FP8KV] kvdtype={KVD} calc_scales={CALC} forced={FORCE} "
          f"wall={wall:.2f}s out0={texts[0]!r} all_same={same}", flush=True)


if __name__ == "__main__":
    main()
