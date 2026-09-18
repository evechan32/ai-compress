"""LongBench F1（vLLM 0.11.2 端口）——验证生成段窗口不损质量。

对照：off / chunkkv / chunkkv+genwin，逐样本落盘以便配对比较。
复现：
    cd /root/ai-compress
    TMPDIR=/dev/shm/tmp PYTHONPATH=/hy-tmp/t29:/root/ai-compress \
    PE_MODE=chunkkv PE_RATIO=0.5 PE_GEN_WINDOW=0 \
    python3 bench/f1_v11.py --files qasper.jsonl ... --n 20 --tag r05
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_MODE", "off")

if os.environ.get("PE_MODE", "off") != "off":
    import kvcompress.pevict_v11 as pe11

    pe11.install()

from vllm import LLM, SamplingParams  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.sgl_longbench import TEMPLATE, _f1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="/hy-tmp/longbench/data")
    ap.add_argument("--files", nargs="+", default=["qasper.jsonl"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--max-ctx-chars", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/root/f1-out")
    args = ap.parse_args()

    mode = os.environ["PE_MODE"]
    llm = LLM(model=args.model, max_model_len=args.max_tokens,
              gpu_memory_utilization=0.6, enforce_eager=True, max_num_seqs=4,
              disable_log_stats=True, enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=args.max_new, temperature=0.0)
    res, t0 = {}, time.time()
    for fn in args.files:
        rows = []
        with open(os.path.join(args.data, fn), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= args.n:
                    break
        prompts = [TEMPLATE.format(context=r["context"][: args.max_ctx_chars],
                                   input=r["input"]) for r in rows]
        outs = llm.generate(prompts, sp)
        f1s = [_f1(o.outputs[0].text.strip(), r.get("answers", []))
               for o, r in zip(outs, rows)]
        res[fn] = {"f1": f1s, "mean": round(sum(f1s) / max(1, len(f1s)), 4),
                   "n": len(f1s)}
        print(f"[F1-11] {fn:<22} n={len(f1s)} mean={res[fn]['mean']}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    rec = {"tag": args.tag, "mode": mode, "ratio": os.environ.get("PE_RATIO", "1"),
           "gen_window": int(os.environ.get("PE_GEN_WINDOW", "0")),
           "results": res, "wall_s": round(time.time() - t0, 1)}
    json.dump(rec, open(f"{args.out}/f1-11-{args.tag}.json", "w"), indent=2)
    overall = [v for r in res.values() for v in r["f1"]]
    print(f"[F1-11] MODE={mode} ratio={rec['ratio']} gw={rec['gen_window']} "
          f"overall={sum(overall)/max(1,len(overall)):.4f}", flush=True)


if __name__ == "__main__":
    main()
