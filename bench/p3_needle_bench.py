"""Needle-in-a-haystack 基准（走我们的 vLLM 插件）。

针（唯一 code）放在不同深度，问 "What is the secret code?"，
检查输出里是否**逐字包含**该 code。这是"答案就藏在某一句"的场景——
和 LongBench（答案不依赖单点）不同，能区分打分的定位能力。

对每种打分/压缩率，报每个深度的命中率与总体命中率。
"""
import json
import os
import random

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_MODE", "off")

if os.environ.get("PE_MODE", "off") != "off":
    import kvcompress.pevict as pevict

    pevict.install()

from vllm import LLM, SamplingParams  # noqa: E402

FILLER = "The harbor master logged every vessel that passed the north pier. "
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
PER_DEPTH = int(os.environ.get("PE_PER_DEPTH", "8"))
TARGET = int(os.environ.get("PE_TARGET", "2048"))


def build_prompts(tok, target=TARGET):
    unit = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    filler = FILLER * (target // unit)
    rng = random.Random(7)
    out = []
    for d in DEPTHS:
        for _ in range(PER_DEPTH):
            code = f"Z{rng.randint(10000, 99999)}"
            pos = max(1, int(len(filler) * d))
            body = filler[:pos] + f" IMPORTANT: the secret vault code is {code}. " + filler[pos:]
            out.append((d, code,
                        "Read the document carefully.\n\n" + body
                        + "\n\nQuestion: What is the secret vault code?\nAnswer:"))
    return out


def main():
    mode = os.environ["PE_MODE"]
    llm = LLM(
        model="/models/qwen2.5-1.5b-instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        max_num_seqs=4,
        disable_log_stats=True,
        attention_backend=os.environ.get("PE_BACKEND") or None,
    )
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=24, temperature=0.0)
    samples = build_prompts(tok)
    outs = llm.generate([p for _, _, p in samples], sp)

    by_depth = {d: [0, 0] for d in DEPTHS}
    hit = 0
    for (d, code, _), o in zip(samples, outs):
        ok = code in o.outputs[0].text
        by_depth[d][0] += int(ok)
        by_depth[d][1] += 1
        hit += int(ok)
    per = {str(d): f"{by_depth[d][0]}/{by_depth[d][1]}" for d in DEPTHS}
    print(f"[NIAH] mode={mode} ratio={os.environ.get('PE_RATIO','1')} "
          f"overall={hit}/{len(samples)} per_depth={per}", flush=True)
    json.dump({"mode": mode, "ratio": os.environ.get("PE_RATIO", "1"),
               "hit": hit, "n": len(samples), "per_depth": per},
              open(f"/root/ai-compress/bench/out/niah-{mode}-"
                   f"{os.environ.get('PE_RATIO','1')}.json", "w"))


if __name__ == "__main__":
    main()
