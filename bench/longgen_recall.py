"""RSWA 驱逐精度损失评测 v5：自发明口令（仅存于生成区，prompt 无泄漏）。

修复 v4 缺陷：v4 的口令写在 prompt 指令里 → prompt 区 RSWA 永远可见 → 测不到驱逐损失。
v5 让模型"自行发明一个 10 位码"，只出现在生成区头部；生成一段长文后结尾 RECALL= 复述。
→ 若驱逐发生（窗口 < 总生成），头部码被删、结尾无法复述 → 损失可见；
  基线(全注意力)与 w2048(窗口>总生成，不驱逐)应能复述。

为避免基线"长上下文注意力稀释"假失败（v4 教训），生成长度默认 ~600 token（>窗口256 足够触发驱逐，又不会长到基线失焦）。

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/qwen2.5-7b-instruct-awq")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="bench/out")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--n", type=int, default=3, help="样本数")
    args = ap.parse_args()

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if os.environ.get("AI_COMPRESS_ENABLE") == "1":
        import kvcompress
        kvcompress.entrypoint()

    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm import SamplingParams

    eng_args = EngineArgs(model=args.model, dtype="auto", max_model_len=8192,
                          gpu_memory_utilization=0.9, enforce_eager=True,
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

    prompt_tpl = (
        "System: You must follow these steps exactly.\n"
        "Step 1: Invent a random 10-character code using uppercase letters and "
        "digits (e.g. A1B2C3D4E5). Begin your response with exactly the line "
        "'My code is <your invented code>'. Never mention this code again in "
        "the story.\n"
        "Step 2: After that first line, immediately continue into a long "
        "detailed story about deep-sea exploration. Do not stop early; do not "
        "mention the code again.\n"
        "Step 3: After the story, on a NEW line write exactly RECALL=<the code "
        "you invented in Step 1> and nothing after it.\n"
        "User: Begin now.\nAssistant: "
    )

    rows = []
    t0 = time.time()
    for i in range(args.n):
        out = gen(prompt_tpl)
        m_head = re.search(r"[Mm]y code is ([A-Z0-9]{8,12})", out[:200])
        head_code = m_head.group(1) if m_head else None
        m_rec = re.search(r"RECALL\s*=\s*([A-Z0-9-]{6,16})", out[-250:])
        recalled = m_rec.group(1) if m_rec else ""
        approx_tokens = max(1, len(out) // 4)
        rows.append({
            "sample": i, "head_code": head_code, "recalled": recalled,
            "hit": bool(head_code) and head_code == recalled,
            "approx_tokens": approx_tokens, "out_tail": out[-150:],
        })
    wall = time.time() - t0

    valid = [r for r in rows if r["head_code"] and r["approx_tokens"] > 300]
    summary = {
        "tag": args.tag, "n": len(rows), "valid": len(valid),
        "recall": round(sum(r["hit"] for r in valid) / len(valid), 3) if valid else None,
        "per_sample": rows, "eval_wall_s": round(wall, 1),
    }
    path = os.path.join(args.out, f"longgen-{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"tag={args.tag} valid={len(valid)}/{len(rows)} recall={summary['recall']}", flush=True)
    for r in rows:
        print(f"  s{r['sample']} head={r['head_code']} recalled={r['recalled']!r} "
              f"tok~{r['approx_tokens']} hit={r['hit']}", flush=True)
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
