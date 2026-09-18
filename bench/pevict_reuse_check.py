"""块复用正确性探针：高并发下被释放的块会被其他请求复用，输出是否仍正确？

原理
----
每个并发请求写入**独一无二的 code**，针埋在不同深度。若压实存在边界错误，
注意力会读到"已被别的请求复用"的块 → 输出里出现**别的请求的 code**。
这正是"释放 + 复用"场景下的致命失效模式，且可精确诊断（而非模糊的掉分）。

指标
----
- hit：输出含**自己的** code
- contamination：输出含**别人的** code（应为 0；任何非零都是严重 bug）
- miss：两者都没有

跑法：
    cd /root/ai-compress
    TMPDIR=/dev/shm/tmp PYTHONPATH=/hy-tmp/t29:/root/ai-compress \
    PE_MODE=chunkkv PE_RATIO=0.3 PE_GEN_WINDOW=256 PE_MAXSEQS=64 PE_GMEM=0.30 \
      python3 bench/pevict_reuse_check.py --n 64 --per-depth 1
"""
import argparse
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


def build(tok, n, target):
    unit = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    filler = FILLER * (target // unit)
    rng = random.Random(11)
    out = []
    for i in range(n):
        code = f"V{i:04d}X"
        pos = max(1, int(len(filler) * rng.random()))
        body = filler[:pos] + f" IMPORTANT: the secret vault code is {code}. " + filler[pos:]
        out.append((code, "Read the document carefully.\n\n" + body
                    + "\n\nQuestion: What is the secret vault code?\nAnswer:"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--target", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()

    mode = os.environ["PE_MODE"]
    llm = LLM(model=args.model, max_model_len=8192,
              gpu_memory_utilization=float(os.environ.get("PE_GMEM", "0.6")),
              enforce_eager=True,
              max_num_seqs=int(os.environ.get("PE_MAXSEQS", "4")),
              disable_log_stats=True)
    tok = llm.get_tokenizer()
    samples = build(tok, args.n, args.target)
    outs = llm.generate([p for _, p in samples],
                        SamplingParams(max_tokens=args.max_new, temperature=0.0))
    codes = [c for c, _ in samples]
    hit = cont = miss = 0
    bad = []
    for (code, _), o in zip(samples, outs):
        text = o.outputs[0].text
        others = [c for c in codes if c != code and c in text]
        if others:
            cont += 1
            bad.append((code, others[:2]))
        elif code in text:
            hit += 1
        else:
            miss += 1
    rec = {"mode": mode, "n": args.n, "max_seqs": int(os.environ.get("PE_MAXSEQS", "4")),
           "gmem": float(os.environ.get("PE_GMEM", "0.6")),
           "ratio": os.environ.get("PE_RATIO", "1"),
           "gen_window": int(os.environ.get("PE_GEN_WINDOW", "0")),
           "hit": hit, "contamination": cont, "miss": miss,
           "bad_examples": bad[:3]}
    os.makedirs("/root/reuse-out", exist_ok=True)
    tag = f"{mode}-gw{os.environ.get('PE_GEN_WINDOW','0')}-m{rec['max_seqs']}"
    json.dump(rec, open(f"/root/reuse-out/reuse-{tag}.json", "w"), indent=2)
    print(f"[REUSE] {json.dumps(rec, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
