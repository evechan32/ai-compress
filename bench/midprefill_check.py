"""prefill 中途驱逐验证：质量（NIAH）+ 峰值（KV 池最小空闲块数）。

关键：必须强制 chunked prefill（`max_num_batched_tokens` 设小），否则 prompt 一次
prefill 完，"中途"驱逐等价于"事后"驱逐，测不出差别。

跑法：
    cd /root/ai-compress
    TMPDIR=/dev/shm/tmp PYTHONPATH=/hy-tmp/t29:/root/ai-compress \
    PE_MODE=chunkkv PE_RATIO=0.3 PE_MIDPREFILL=1 PE_MAXBAT=256 PE_LOG=1 \
      python3 bench/midprefill_check.py --n 40
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
    rng = random.Random(7)
    out = []
    for i in range(n):
        code = f"W{i:04d}Z"
        pos = max(1, int(len(filler) * rng.random()))
        body = filler[:pos] + f" IMPORTANT: the secret vault code is {code}. " + filler[pos:]
        out.append((code, "Read the document carefully.\n\n" + body
                    + "\n\nQuestion: What is the secret vault code?\nAnswer:"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--target", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()

    mode = os.environ["PE_MODE"]
    llm = LLM(model=args.model, max_model_len=8192,
              gpu_memory_utilization=float(os.environ.get("PE_GMEM", "0.30")),
              enforce_eager=True, disable_log_stats=True,
              max_num_seqs=int(os.environ.get("PE_MAXSEQS", "8")),
              max_num_batched_tokens=int(os.environ.get("PE_MAXBAT", "256")))
    tok = llm.get_tokenizer()
    samples = build(tok, args.n, args.target)
    outs = llm.generate([p for _, p in samples],
                        SamplingParams(max_tokens=args.max_new, temperature=0.0))
    codes = [c for c, _ in samples]
    hit = cont = miss = 0
    for (code, _), o in zip(samples, outs):
        text = o.outputs[0].text
        if any(c != code and c in text for c in codes):
            cont += 1
        elif code in text:
            hit += 1
        else:
            miss += 1
    rec = {"mode": mode, "n": args.n, "midprefill": int(os.environ.get("PE_MIDPREFILL", "0")),
           "ratio": os.environ.get("PE_RATIO", "1"), "maxbat": os.environ.get("PE_MAXBAT", "256"),
           "hit": hit, "contamination": cont, "miss": miss}
    print(f"[MIDPF] {json.dumps(rec, ensure_ascii=False)}", flush=True)
    os.makedirs("/root/midpf-out", exist_ok=True)
    json.dump(rec, open(f"/root/midpf-out/midpf-{mode}-{rec['midprefill']}-"
                        f"{rec['ratio']}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
