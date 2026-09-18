"""NIAH 对照（vLLM 0.11.2 端口）——验证打分移植与 0.28 行为一致。

needle 埋在不同深度，问 "What is the secret vault code?"，检查输出是否**逐字包含**该 code。
低保留率（如 0.05）才有区分度：≥5% 保留时任何合理选块都能保住针（历史结论）。

跑法：
    cd /root/ai-compress
    TMPDIR=/dev/shm/tmp PYTHONPATH=/hy-tmp/t29:/root/ai-compress \
    PE_MODE=chunkkv PE_RATIO=0.5 python3 bench/niah_v11.py
"""
import json
import os
import random

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_LOG", "0")
os.environ.setdefault("PE_MODE", "off")

if os.environ.get("PE_MODE", "off") != "off":
    import kvcompress.pevict_v11 as pe11

    pe11.install()

from vllm import LLM, SamplingParams  # noqa: E402

FILLER = "The harbor master logged every vessel that passed the north pier. "
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
PER_DEPTH = int(os.environ.get("PE_PER_DEPTH", "3"))
TARGET = int(os.environ.get("PE_TARGET", "2048"))
MODEL = os.environ.get("PE_MODEL", "/dev/shm/models/Qwen2.5-1.5B-Instruct")


def build_prompts(tok):
    unit = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    filler = FILLER * (TARGET // unit)
    rng = random.Random(7)
    out = []
    for d in DEPTHS:
        for _ in range(PER_DEPTH):
            code = f"Z{rng.randint(10000, 99999)}"
            pos = max(1, int(len(filler) * d))
            body = filler[:pos] + f" IMPORTANT: the secret vault code is {code}. " + filler[pos:]
            out.append((d, code, "Read the document carefully.\n\n" + body
                        + "\n\nQuestion: What is the secret vault code?\nAnswer:"))
    return out


def main():
    mode = os.environ["PE_MODE"]
    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.6,
              enforce_eager=True, max_num_seqs=4, disable_log_stats=True)
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
    print(f"[NIAH11] mode={mode} ratio={os.environ.get('PE_RATIO','1')} "
          f"genwin={os.environ.get('PE_GEN_WINDOW','0')} "
          f"overall={hit}/{len(samples)} per_depth={per}", flush=True)
    os.makedirs("/root/niah-out", exist_ok=True)
    json.dump({"mode": mode, "ratio": os.environ.get("PE_RATIO", "1"), "hit": hit,
               "n": len(samples), "per_depth": per},
              open(f"/root/niah-out/niah11-{mode}-{os.environ.get('PE_RATIO','1')}.json", "w"))


if __name__ == "__main__":
    main()
