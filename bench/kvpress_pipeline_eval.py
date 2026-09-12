"""LongBench subset eval **via the official KVpress pipeline**.

`bench/kvpress_eval.py` drives `generate` inside `with press(model)`, which is
correct for prefill-scoring presses (SnapKV/ChunkKV/TOVA/...) but WRONG for
`KVzipPress`: KVzip applies its compression when the context manager *exits*,
so it must be used through `KVPressTextGenerationPipeline` (prefill context
inside the `with`, generate the question afterwards).

This script uses the pipeline for every method, so KVzip gets a valid number.
Note the pipeline applies the model chat template and asks the raw question,
so its protocol differs from `kvpress_eval.py`; compare within this script only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.sgl_longbench import _f1  # noqa: E402


def build_press(name: str, ratio: float):
    if name == "none":
        return None
    import kvpress

    if name == "chunkkv":
        return kvpress.ChunkKVPress(press=kvpress.SnapKVPress(compression_ratio=ratio))
    mapping = {
        "snapkv": "SnapKVPress", "tova": "TOVAPress", "kvzip": "KVzipPress",
        "expected": "ExpectedAttentionPress", "keydiff": "KeyDiffPress",
    }
    cls = getattr(kvpress, mapping[name])
    try:
        return cls(compression_ratio=ratio)
    except TypeError:
        return cls()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/models/qwen2.5-1.5b-instruct")
    ap.add_argument("--data", default="/hy-tmp/longbench/data")
    ap.add_argument("--files", nargs="+", default=["qasper.jsonl", "multifieldqa_en.jsonl"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--methods", nargs="+", default=["none", "snapkv", "kvzip"])
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--max-ctx-chars", type=int, default=20000)
    ap.add_argument("--out", default="/root/kvpress-out")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvpress import KVPressTextGenerationPipeline

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tok)

    rows_by_file = {}
    for fn in args.files:
        rows = []
        with open(os.path.join(args.data, fn), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= args.n:
                    break
        rows_by_file[fn] = rows

    results = {}
    t0 = time.time()
    for method in args.methods:
        try:
            press = build_press(method, args.ratio)
        except Exception as e:
            print(f"== {method} SKIP build: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        per_file = {}
        for fn, rows in rows_by_file.items():
            f1s = []
            for r in rows:
                try:
                    out = pipe(r["context"][: args.max_ctx_chars], question=r["input"],
                               press=press, max_new_tokens=args.max_new)
                    pred = out["answer"]
                except Exception as e:
                    print(f"  [{method}] {fn} ERR {type(e).__name__}: {str(e)[:140]}", flush=True)
                    f1s.append(0.0)
                    continue
                f1s.append(_f1(pred, r.get("answers", [])))
            per_file[fn] = round(sum(f1s) / max(1, len(f1s)), 4)
            print(f"{method:<10} {fn:<22} n={len(rows)} f1={per_file[fn]}", flush=True)
        mean = round(sum(per_file.values()) / max(1, len(per_file)), 4)
        results[method] = {"per_file": per_file, "mean": mean}
        print(f"== {method:<10} mean_f1={mean}", flush=True)

    payload = {"tag": args.tag, "protocol": "pipeline", "model": args.model,
               "ratio": args.ratio, "n": args.n, "files": args.files,
               "results": results, "wall_s": round(time.time() - t0, 1)}
    path = os.path.join(args.out, f"kvpress-pipe-{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
