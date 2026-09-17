"""代理实验：容量 → 吞吐 曲线（不需要我们的插件）。

问题
----
我们把 prompt KV 压到 keep-r 后，**容量**从 ~1× 涨到 ~2×（3k prompt + 512 生成时），
但实测吞吐只从 1.00× 涨到 1.26×。到底是
  (a) 方法的结构限制（只压 prompt 段，且墙钟被 prefill 占掉大头），还是
  (b) 负载形状选错了（生成太短、并发已饱和）？

做法
----
用 stock vLLM，直接**人工制造不同容量**（扫 `gpu_memory_utilization`），
在几种**负载形状**下测吞吐。这样能在不碰插件的情况下回答：
**"给定一份容量，能兑现多少吞吐？"** 以及它随负载形状怎么变。

如果吞吐随容量近似线性 → 我们的容量收益本应兑现更多 → 值得做 RSWA/生成段联合；
如果很快饱和 → 天花板是 compute，这条路该收手。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

WORDS = ("committee archive correspondence administrative recurring review "
         "ledger transit harbour lantern quarry meadow ridge copper basin "
         "verdict statute chapter vessel orchard granite ferry prairie").split()


def make_prompts(tok, m, target):
    out = []
    for i in range(m):
        rng = random.Random(1000 + i)
        body = " ".join(rng.choice(WORDS) for _ in range(max(1, target // 2)))
        out.append(f"Read the document.\n\n{body}\n\nQuestion: What themes "
                   f"recur? (doc {i})\nAnswer:")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/hy-tmp/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--m", type=int, default=64, help="并发请求数")
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--gen-tokens", type=int, default=512)
    ap.add_argument("--gmem", type=float, default=0.5, help="容量旋钮")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/root/proxy-out")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=args.max_len,
              gpu_memory_utilization=args.gmem, enforce_eager=True,
              max_num_seqs=args.m, disable_log_stats=True,
              enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    prompts = make_prompts(tok, args.m, args.prompt_tokens)
    real_ptoks = len(tok(prompts[0])["input_ids"])
    sp = SamplingParams(max_tokens=args.gen_tokens, temperature=0.0)

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    wall = time.time() - t0
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    rec = {"tag": args.tag, "m": args.m, "prompt_tokens": real_ptoks,
           "gen_tokens": args.gen_tokens, "gmem": args.gmem,
           "wall_s": round(wall, 1), "out_tokens": out_tokens,
           "tok_per_s": round(out_tokens / wall, 1),
           "total_tokens": out_tokens + real_ptoks * args.m}
    os.makedirs(args.out, exist_ok=True)
    json.dump(rec, open(f"{args.out}/proxy-{args.tag}.json", "w"), indent=2)
    print(f"[PROXY] {json.dumps(rec, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
