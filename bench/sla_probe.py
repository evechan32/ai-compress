"""SLA 探针：并发压测 OpenAI 兼容服务，量 TTFT / TPOT 分位数与吞吐。

为什么不用 `vllm bench serve`：0.11.2 的 bench CLI 在 `--no-deps` 安装下没有 console
script，且其参数由 helper 动态添加，逆向成本高于自写。自写还能精确控制"固定 SLA 下
的 max 并发"这一目标指标。

用法：
    TMPDIR=/dev/shm/tmp PYTHONPATH=/hy-tmp/t29 python3 bench/sla_probe.py \
      --url http://127.0.0.1:8100/v1/completions --concurrency 64 \
      --num 128 --prompt-tokens 1024 --max-tokens 1024 --tag off-c64
"""
import argparse
import asyncio
import json
import os
import time

import aiohttp

WORDS = ("committee archive correspondence administrative recurring review "
         "ledger transit harbour lantern quarry meadow ridge copper basin "
         "verdict statute chapter vessel orchard granite ferry prairie").split()


def build_prompt(seed: int, approx_tokens: int) -> str:
    rng = __import__("random").Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(max(1, approx_tokens // 2)))
    return f"Read the document.\n\n{body}\n\nContinue the document verbatim, do not stop:"


async def one_request(session, url, prompt, max_tokens):
    payload = {"model": "m", "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True}
    t0 = time.perf_counter()
    first = None
    chunks = 0
    async with session.post(url, json=payload) as r:
        async for raw in r.content:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            ch = obj.get("choices") or []
            if ch and ch[0].get("text"):
                chunks += 1
                if first is None:
                    first = time.perf_counter()
    t1 = time.perf_counter()
    if first is None:
        return None
    ttft = first - t0
    tpot = (t1 - first) / max(1, chunks - 1)
    return ttft, tpot, chunks, t1 - t0


async def run(args):
    sem = asyncio.Semaphore(args.concurrency)
    results = []
    t_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        async def guarded(i):
            async with sem:
                r = await one_request(s, args.url, build_prompt(i, args.prompt_tokens),
                                      args.max_tokens)
                if r:
                    results.append(r)
        await asyncio.gather(*(guarded(i) for i in range(args.num)))
    wall = time.perf_counter() - t_start

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float("nan")

    if not results:
        print("[SLA] no results", flush=True)
        return
    ttft = [r[0] * 1000 for r in results]
    tpot = [r[1] * 1000 for r in results]
    out_tok = sum(r[2] for r in results)
    rec = {"tag": args.tag, "concurrency": args.concurrency, "num": args.num,
           "prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens,
           "completed": len(results), "out_tokens": out_tok,
           "wall_s": round(wall, 1), "tok_per_s": round(out_tok / wall, 1),
           "ttft_p50_ms": round(pct(ttft, 0.50), 1), "ttft_p99_ms": round(pct(ttft, 0.99), 1),
           "tpot_p50_ms": round(pct(tpot, 0.50), 2), "tpot_p99_ms": round(pct(tpot, 0.99), 2)}
    os.makedirs(args.out, exist_ok=True)
    json.dump(rec, open(f"{args.out}/sla-{args.tag}.json", "w"), indent=2)
    print(f"[SLA] {json.dumps(rec, ensure_ascii=False)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100/v1/completions")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--num", type=int, default=128)
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/root/sla-out")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
