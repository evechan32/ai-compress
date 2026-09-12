"""审计 kvcompress RSWA：在自动选到的 backend 下，掩码是否真的生效。

判据：window=32、生成 200 token。若掩码生效，早期生成 token 不可见 → 输出应与
无插件基线不同；若输出逐字相同，则掩码被忽略（注意力仍在读已释放块）。
"""
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ["AI_COMPRESS_ENABLE"] = "1" if os.environ.get("KC_ENABLE") == "1" else "0"
os.environ.setdefault("AI_COMPRESS_RSWA_WINDOW", "32")

import kvcompress  # noqa: E402
import kvcompress.adapter  # noqa: E402
import kvcompress.config  # noqa: E402

_cfg = kvcompress.config.load_config()
if _cfg.enabled:
    kvcompress.adapter.install(_cfg)

from vllm import LLM, SamplingParams  # noqa: E402

PROMPT = ("Write a detailed, continuous essay about the history of computing. "
          "Begin with the abacus and keep going for a long time. Do not stop.")


def main():
    llm = LLM(
        model="/models/qwen2.5-1.5b-instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        max_num_seqs=1,
        disable_log_stats=True,
    )
    out = llm.generate([PROMPT], SamplingParams(max_tokens=200, temperature=0.0))
    text = out[0].outputs[0].text
    import hashlib
    h = hashlib.sha1(text.encode()).hexdigest()[:12]
    print(f"[KC] enabled={_cfg.enabled} window={_cfg.rswa_window} "
          f"hash={h} len={len(text)}", flush=True)
    print(f"[KC] head={text[:160]!r}", flush=True)
    print(f"[KC] tail={text[-160:]!r}", flush=True)


if __name__ == "__main__":
    main()
