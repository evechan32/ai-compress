"""吞吐复测（vLLM 0.11.2 端口）——对照 stock 的容量→吞吐曲线。

回答：把 prompt 驱逐 + **生成段窗口**都用上后，容量收益的兑现率能否从 65% 提升？

跑法：
    cd /root/ai-compress
    TMPDIR=/dev/shm/tmp PYTHONPATH=/hy-tmp/t29:/root/ai-compress \
    PE_MODE=chunkkv PE_RATIO=0.3 PE_GEN_WINDOW=256 \
    python3 bench/perf_v11.py --m 128 --prompt-words 576 --gen 512 --tag ...
"""
import argparse
import json
import os
import random
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_MODE", "off")

if os.environ.get("PE_MODE", "off") != "off":
    import kvcompress.pevict_v11 as pe11

    pe11.install()

from vllm import LLM, SamplingParams  # noqa: E402

WORDS = ("committee archive correspondence administrative recurring review "
         "ledger transit harbour lantern quarry meadow ridge copper basin "
         "verdict statute chapter vessel orchard granite ferry prairie").split()


def make_prompts(tok, m, words):
    out = []
    for i in range(m):
        rng = random.Random(1000 + i)
        body = " ".join(rng.choice(WORDS) for _ in range(words))
        out.append(f"Read the document.\n\n{body}\n\nQuestion: What themes "
                   f"recur? (doc {i})\nAnswer:")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--prompt-words", type=int, default=576)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--gmem", type=float, default=0.30)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/root/perf-out")
    args = ap.parse_args()

    mode = os.environ["PE_MODE"]
    llm = LLM(model=args.model, max_model_len=args.max_len,
              gpu_memory_utilization=args.gmem, enforce_eager=True,
              max_num_seqs=args.m, disable_log_stats=True,
              enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    prompts = make_prompts(tok, args.m, args.prompt_words)
    ptoks = len(tok(prompts[0])["input_ids"])
    sp = SamplingParams(max_tokens=args.gen, temperature=0.0)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    wall = time.time() - t0
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    rec = {"tag": args.tag, "mode": mode, "ratio": os.environ.get("PE_RATIO", "1"),
           "gen_window": int(os.environ.get("PE_GEN_WINDOW", "0")),
           "m": args.m, "prompt_tokens": ptoks, "gen": args.gen,
           "gmem": args.gmem, "wall_s": round(wall, 1),
           "out_tokens": out_tokens, "tok_per_s": round(out_tokens / wall, 1)}
    os.makedirs(args.out, exist_ok=True)
    json.dump(rec, open(f"{args.out}/perf11-{args.tag}.json", "w"), indent=2)
    print(f"[PERF11] {json.dumps(rec, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
