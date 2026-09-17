"""惊讶度作为「选择信号」的端到端检验（块级 prompt 压缩）。

设计理由
--------
惊讶度天然是 **prefill 之后** 的信号（需要 lm_head 输出），所以它不能在前向中途驱动剪枝。
与其在 HF 里做易错的 cache 手术（RoPE/position/cache_position），不如用一个语义等价、
且**块粒度与插件一致**的检验：按惊讶度选块保留、删掉其余，再提问。

对照（同一批样本、同一预算，**配对**比较）：
  - `none`      ：完整 context（上界）
  - `surprisal` ：保留惊讶度最高的 k 个块
  - `random`    ：随机保留 k 个块（固定种子）—— **决定性的对照**：信号若 ≈ random 即为废
  - `positional`：保留最前面 k 个块（位置式基线）

指标：LongBench F1（复用 bench.sgl_longbench._f1 与 TEMPLATE）。
输出：每个 (task, method) 的 mean/逐样本 F1，供配对 bootstrap。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.sgl_longbench import TEMPLATE, _f1  # noqa: E402


def surprisal_from_logits(logits, ids):
    """(1,seq,V),(1,seq) -> (seq,) 的 -log p(token_t)；首位置置 0。"""
    logp = torch.log_softmax(logits[0, :-1, :].float(), dim=-1)
    tgt = ids[0, 1:]
    sur = -logp.gather(-1, tgt.view(-1, 1)).squeeze(-1)
    return torch.cat([torch.zeros(1, device=logits.device), sur]).cpu().numpy()


def select_chunks(score, ratio, method, rng):
    n = len(score)
    k = max(1, int(round(n * (1 - ratio))))
    if method == "positional":
        idx = np.arange(k)
    elif method == "random":
        idx = np.array(sorted(rng.sample(range(n), k)))
    else:
        idx = np.array(sorted(np.argsort(-score)[:k]))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/hy-tmp/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="/hy-tmp/longbench/data")
    ap.add_argument("--files", nargs="+", default=["qasper.jsonl"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ratio", type=float, default=0.5, help="被删掉的比例")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-ctx-chars", type=int, default=20000)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--methods", nargs="+",
                    default=["none", "surprisal", "random", "positional"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/root/kvpress-out")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
    ).eval()
    os.makedirs(args.out, exist_ok=True)

    # sanity：重复文本上的 median 惊讶度必须 ≈0，否则拒绝测量
    _ids = tok("The harbor master logged every vessel that passed the north pier. " * 20,
               return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        _o = model(_ids, use_cache=False)
    _med = float(np.median(surprisal_from_logits(_o.logits, _ids)))
    print(f"[SANITY] median surprisal={_med:.4f}", flush=True)
    if not _med < 0.05:
        raise SystemExit("[SANITY] FAILED")
    del _o
    torch.cuda.empty_cache()

    out = {}
    t0 = time.time()
    for fn in args.files:
        rows = []
        with open(os.path.join(args.data, fn), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= args.n:
                    break
        per = {m: [] for m in args.methods}
        for r in rows:
            ctx = r["context"][: args.max_ctx_chars]
            ids_1d = tok(ctx, return_tensors="pt")["input_ids"][0]
            if ids_1d.numel() < 2:
                continue
            cids = ids_1d.unsqueeze(0).cuda()
            with torch.no_grad():
                o = model(cids, use_cache=False)
            sur = surprisal_from_logits(o.logits, cids)
            del o
            torch.cuda.empty_cache()
            chunk = args.chunk
            n_chunks = max(1, ids_1d.numel() // chunk)
            score = sur[: n_chunks * chunk].reshape(n_chunks, chunk).max(1)
            rng = random.Random(1234)

            for m in args.methods:
                if m == "none":
                    short = ctx
                else:
                    idx = select_chunks(score, args.ratio, m, rng)
                    kept = torch.cat([ids_1d[i * chunk:(i + 1) * chunk] for i in idx])
                    short = tok.decode(kept, skip_special_tokens=True)
                prompt = TEMPLATE.format(context=short, input=r["input"])
                ins = tok(prompt, return_tensors="pt", truncation=True,
                          max_length=args.max_tokens).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**ins, max_new_tokens=args.max_new, do_sample=False,
                                         temperature=None, top_p=None, top_k=None,
                                         pad_token_id=tok.eos_token_id)
                pred = tok.decode(gen[0][ins["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
                per[m].append(_f1(pred, r.get("answers", [])))
        out[fn] = {m: {"mean": round(float(np.mean(v)), 4), "n": len(v), "f1": v}
                   for m, v in per.items()}
        for m, v in per.items():
            print(f"[F1] {fn:<22} {m:<11} n={len(v)} f1={np.mean(v):.4f}", flush=True)

    path = os.path.join(args.out, f"surprisal-f1-{args.tag}.json")
    json.dump({"args": vars(args), "results": out, "wall_s": round(time.time() - t0, 1)},
              open(path, "w"), indent=2, ensure_ascii=False)
    print(f"[F1] WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
