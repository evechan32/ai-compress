"""RSWA 驱逐精度损失评测 v4：单请求超长生成——开头口令在结尾能否复述。

这是驱逐真正生效的场景：一个请求内连续生成远超窗口的 token，生成区头部（含口令）
在滑窗推进中被驱逐；结尾要求复述开头口令 → 被驱逐则失败，保留则成功。

正确性自检：输出开头必须含 'passcode is <CODE>'（口令确实在生成区头部），
总生成需超过窗口才有驱逐发生（对比 w2048 对照组：窗口>总生成 → 不应驱逐 → 应成功）。

用法：
  python bench/longgen_recall.py --tag baseline
  AI_COMPRESS_ENABLE=1 AI_COMPRESS_RSWA_WINDOW=256 python bench/longgen_recall.py --tag plugin-w256
  AI_COMPRESS_ENABLE=1 AI_COMPRESS_RSWA_WINDOW=2048 python bench/longgen_recall.py --tag plugin-w2048
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

CODES = ["K7XQ-ZZ9", "K7XQ-QQ7", "K7XQ-MN3"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/qwen2.5-1.5b-instruct")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="bench/out")
    ap.add_argument("--max-tokens", type=int, default=1700)
    args = ap.parse_args()

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if os.environ.get("AI_COMPRESS_ENABLE") == "1":
        import kvcompress
        kvcompress.entrypoint()

    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm import SamplingParams

    eng_args = EngineArgs(model=args.model, dtype="bfloat16", max_model_len=8192,
                          gpu_memory_utilization=0.85, enforce_eager=True,
                          enable_prefix_caching=False)
    engine = LLMEngine.from_engine_args(eng_args)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    def gen(prompt: str) -> str:
        rid = f"r-{int(time.time() * 1000)}"
        engine.add_request(rid, prompt, sp)
        for _ in range(40000):
            outs = engine.step()
            for o in outs or []:
                if getattr(o, "finished", False):
                    return o.outputs[0].text
        return ""

    rows = []
    t0 = time.time()
    for code in CODES:
        prompt = (
            f"System: You must follow these steps exactly.\n"
            f"Step 1: begin your response with the line 'passcode is {code}'.\n"
            f"Step 2: then immediately continue into a long, detailed story about "
            f"ocean voyages across many countries; do not stop early (aim for at "
            f"least 1000 words).\n"
            f"Step 3: after the story ends, on a new line write RECALL= and then "
            f"the exact passcode from Step 1, nothing else.\n"
            f"User: Begin now.\nAssistant: "
        )
        out = gen(prompt)
        head_ok = f"passcode is {code}" in out[:200]
        m = re.search(r"RECALL\s*=\s*(\S+)", out)
        recalled = m.group(1) if m else ""
        hit = recalled == code
        ntoks = max(1, len(out) // 4)
        rows.append({
            "code": code, "head_ok": head_ok, "approx_tokens": ntoks,
            "recalled": recalled, "hit": hit, "out_tail": out[-120:],
        })
    wall = time.time() - t0

    valid = [r for r in rows if r["head_ok"] and r["approx_tokens"] > 400]
    summary = {
        "n": len(rows),
        "valid": len(valid),
        "recall": round(sum(r["hit"] for r in valid) / len(valid), 3) if valid else None,
        "per_code": rows,
        "eval_wall_s": round(wall, 1),
    }
    path = os.path.join(args.out, f"longgen-{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"tag={args.tag} valid={len(valid)}/{len(rows)} recall={summary['recall']}", flush=True)
    for r in rows:
        print(f"  code={r['code']} head_ok={r['head_ok']} tok~{r['approx_tokens']} "
              f"recalled={r['recalled']!r} hit={r['hit']}", flush=True)
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
