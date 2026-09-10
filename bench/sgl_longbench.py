"""SGLang LongBench 子集评测：完整注意力 vs 散点 backend（F1 + 答案对比）。"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from types import SimpleNamespace

TEMPLATE = ("Answer the question based on the given documents. Only give me the "
            "answer and do not output any other words.\n\n{context}\n\n"
            "Question: {input}\nAnswer:")


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, golds: list[str]) -> float:
    best = 0.0
    p = _norm(pred).split()
    for g in golds:
        gg = _norm(g).split()
        if not p or not gg:
            best = max(best, 1.0 if p == gg else 0.0)
            continue
        common = set(p) & set(gg)
        if not common:
            continue
        prec = len(common) / len(p)
        rec = len(common) / len(gg)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/qwen2.5-1.5b-instruct")
    ap.add_argument("--data", default="/hy-tmp/longbench")
    ap.add_argument("--files", nargs="+", default=["narrativeqa.jsonl", "2wikimqa.jsonl"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="bench/out")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--attention-backend", default="triton")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--max-ctx-chars", type=int, default=20000)
    args = ap.parse_args()

    if args.attention_backend.startswith("kvx"):
        import sglang_kvx  # noqa: F401
    from bench.sgl_run_eval import SglLLM

    os.makedirs(args.out, exist_ok=True)
    llm = SglLLM(args.model, args.attention_backend, 0.6,
                 kv_cache_dtype=args.kv_cache_dtype)
    sp = SimpleNamespace(max_tokens=32)

    per_file = {}
    t0 = time.time()
    for fn in args.files:
        rows = []
        with open(os.path.join(args.data, fn), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= args.n:
                    break
        f1s = []
        for r in rows:
            ctx = r["context"][: args.max_ctx_chars]
            prompt = TEMPLATE.format(context=ctx, input=r["input"])
            pred = llm.generate([prompt], sp)[0].outputs[0].text.strip()
            score = _f1(pred, r.get("answers", []))
            f1s.append(score)
            rows[rows.index(r)]["pred"] = pred if "pred" not in r else r["pred"]
        per_file[fn] = {"n": len(rows), "f1": round(sum(f1s) / max(1, len(f1s)), 4)}
        print(f"{fn}: n={len(rows)} f1={per_file[fn]['f1']}", flush=True)

    payload = {"tag": args.tag, "backend": args.attention_backend,
               "kv_cache_dtype": args.kv_cache_dtype,
               "per_file": per_file, "wall_s": round(time.time() - t0, 1)}
    path = os.path.join(args.out, f"lb-{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
