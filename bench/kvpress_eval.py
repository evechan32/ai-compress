"""LongBench subset evaluation with NVIDIA KVpress presses (HF path).

Compares KV-eviction methods from `kvpress` against the baseline on the same
tasks/prompts/metrics used by `bench/sgl_longbench.py`, so results line up with
our existing RSWA / scatter / FP8 numbers.

Run on the server with transformers 4.56-5.2 shadowed via PYTHONPATH:

    PYTHONPATH=/root/kvpress-libs /usr/local/bin/python3 bench/kvpress_eval.py \
        --data /hy-tmp/longbench/data \
        --files qasper.jsonl 2wikimqa.jsonl multifieldqa_en.jsonl hotpotqa.jsonl triviaqa.jsonl \
        --n 30 --ratio 0.5 --tag kvpress-r0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.sgl_longbench import TEMPLATE, _f1  # noqa: E402

PRESS_FACTORY = {
    "none": None,
    "snapkv": "SnapKVPress",
    "expected": "ExpectedAttentionPress",
    "keydiff": "KeyDiffPress",
    "tova": "TOVAPress",
    "pyramidkv": "PyramidKVPress",
    "streamingllm": "StreamingLLMPress",
    "kvzip": "KVzipPress",
    "knorm": "KnormPress",
    "observed": "ObservedAttentionPress",
    "leverage": "LeverageScorePress",
    "lagkv": "LagKVPress",
    "qfilter": "QFilterPress",
    "cur": "CURPress",
    "kvzap": "KVzapPress",
    "non_causal": "NonCausalAttnPress",
    "random": "RandomPress",
    "compactor": "CompactorPress",
    "finch": "FinchPress",
    "cap": "CapPress",
}

_SNAP_WRAPPERS = {
    "chunkkv": "ChunkKVPress",
    "chunk": "ChunkPress",
    "ada": "AdaKVPress",
    "critical": "CriticalKVPress",
    "critical_ada": "CriticalAdaKVPress",
    "block": "BlockPress",
    "dms": "DMSPress",
    "merging": "MergingPress",
    "lukv": "LUKVPress",
}


def build_press(name: str, ratio: float):
    if name == "none":
        return None
    import kvpress

    if name in _SNAP_WRAPPERS:
        cls = getattr(kvpress, _SNAP_WRAPPERS[name])
        return cls(press=kvpress.SnapKVPress(compression_ratio=ratio))
    if name == "think":
        return kvpress.ThinKPress(key_channel_compression_ratio=ratio)
    if name == "duo":
        return kvpress.DuoAttentionPress(head_compression_ratio=ratio)
    cls = getattr(kvpress, PRESS_FACTORY[name])
    try:
        return cls(compression_ratio=ratio)
    except TypeError:
        return cls()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/models/qwen2.5-1.5b-instruct")
    ap.add_argument("--data", default="/hy-tmp/longbench/data")
    ap.add_argument("--files", nargs="+",
                    default=["qasper.jsonl", "2wikimqa.jsonl", "multifieldqa_en.jsonl",
                             "hotpotqa.jsonl", "triviaqa.jsonl"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--methods", nargs="+", default=list(PRESS_FACTORY))
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--max-ctx-chars", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", default="bench/out")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    rows_by_file = {}
    for fn in args.files:
        path = os.path.join(args.data, fn)
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= args.n:
                    break
        rows_by_file[fn] = rows

    results = {}
    t0 = time.time()
    path = os.path.join(args.out, f"kvpress-{args.tag}.json")

    def save() -> None:
        payload = {"tag": args.tag, "model": args.model, "ratio": args.ratio,
                   "n": args.n, "files": args.files, "results": results,
                   "wall_s": round(time.time() - t0, 1)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    for method in args.methods:
        try:
            press = build_press(method, args.ratio)
        except Exception as e:
            print(f"== {method:<12} SKIP build: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        per_file = {}
        for fn, rows in rows_by_file.items():
            f1s = []
            for r in rows:
                ctx = r["context"][: args.max_ctx_chars]
                prompt = TEMPLATE.format(context=ctx, input=r["input"])
                inputs = tok(prompt, return_tensors="pt", truncation=True,
                             max_length=args.max_tokens).to(model.device)
                cm = press(model) if press is not None else nullcontext()
                try:
                    with torch.no_grad(), cm:
                        gen = model.generate(
                            **inputs, max_new_tokens=args.max_new, do_sample=False,
                            temperature=None, top_p=None, top_k=None,
                            pad_token_id=tok.eos_token_id,
                        )
                except Exception as e:  # keep sweeping methods even if one fails
                    print(f"  [{method}] {fn} ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
                    f1s.append(0.0)
                    continue
                pred = tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
                f1s.append(_f1(pred, r.get("answers", [])))
            per_file[fn] = round(sum(f1s) / max(1, len(f1s)), 4)
            print(f"{method:<14} {fn:<22} n={len(rows)} f1={per_file[fn]}", flush=True)
        mean = round(sum(per_file.values()) / max(1, len(per_file)), 4)
        results[method] = {"per_file": per_file, "mean": mean}
        print(f"== {method:<12} mean_f1={mean}", flush=True)
        save()

    save()
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
