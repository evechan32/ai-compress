"""重验 kvcompress RSWA（必须在 TRITON_ATTN 下）。

FA2 下掩码被忽略 → 输出与基线逐字相同（假无损）。这里验证：
1. TRITON_ATTN 下 window 变小确实改变输出（掩码真的生效）；
2. 同批并发 + 重复运行结果确定（没有读已释放块导致的非确定性）。
"""
import hashlib
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ["AI_COMPRESS_ENABLE"] = "1" if os.environ.get("KC_ENABLE") == "1" else "0"
os.environ.setdefault("AI_COMPRESS_RSWA_WINDOW", "64")

import kvcompress  # noqa: E402
import kvcompress.adapter  # noqa: E402
import kvcompress.config  # noqa: E402

_cfg = kvcompress.config.load_config()
if _cfg.enabled:
    kvcompress.adapter.install(_cfg)

from vllm import LLM, SamplingParams  # noqa: E402

TOPICS = ["computing", "maritime trade", "volcanic geology", "jazz history"]


def prompts():
    return [f"Write a long, detailed, continuous essay about {t}. Keep going "
            f"and do not stop." for t in TOPICS]


def main():
    llm = LLM(
        model="/models/qwen2.5-1.5b-instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        max_num_seqs=4,
        disable_log_stats=True,
        attention_backend=os.environ.get("KC_BACKEND") or None,
    )
    sp = SamplingParams(max_tokens=200, temperature=0.0)
    ps = prompts()
    outs = llm.generate(ps, sp)
    texts = [o.outputs[0].text for o in outs]
    hs = [hashlib.sha1(t.encode()).hexdigest()[:10] for t in texts]
    print(f"[V] enabled={_cfg.enabled} window={_cfg.rswa_window} backend="
          f"{os.environ.get('KC_BACKEND')} hashes={hs}", flush=True)
    import json
    dump = os.environ.get("KC_DUMP")
    if dump:
        json.dump(texts, open(dump, "w"))


if __name__ == "__main__":
    main()
