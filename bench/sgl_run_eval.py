"""SGLang 版评测入口：完整注意力 vs 散点 backend（kvx_scatter）损失量化。

复用 run_eval 的 _run_needle/_run_longqa/_run_multiturn（它们只依赖
llm.generate(list, sampling) -> [obj.outputs[0].text/.token_ids]），用适配层包 SGLang Engine。
输出 json 结构与 run_eval 一致，可直接用 bench/agreement.py 对比。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

from bench.run_eval import _load_scenes, _run_needle, _run_longqa, _run_multiturn


class SglLLM:
    def __init__(self, model: str, attention_backend: str, mem_fraction: float,
                 kv_cache_dtype: str = "auto"):
        from sglang import Engine
        self.e = Engine(model_path=model, dtype="bfloat16",
                        attention_backend=attention_backend,
                        kv_cache_dtype=kv_cache_dtype,
                        mem_fraction_static=mem_fraction,
                        disable_cuda_graph=True)

    def generate(self, prompts, sp):
        res = [self.e.generate(p, {"max_new_tokens": sp.max_tokens, "temperature": 0.0})
               for p in prompts]
        outs = []
        for r in res:
            text = r.get("text", "")
            n = int(r.get("meta_info", {}).get("completion_tokens", max(1, len(text) // 4)))
            outs.append(SimpleNamespace(outputs=[SimpleNamespace(text=text,
                                                                 token_ids=list(range(n)))]))
        return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/qwen2.5-1.5b-instruct")
    ap.add_argument("--data", default="bench/data")
    ap.add_argument("--out", default="bench/out")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--attention-backend", default="triton")
    ap.add_argument("--mem-fraction", type=float, default=0.6)
    ap.add_argument("--max-output-tokens", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.attention_backend.startswith("kvx"):
        import sglang_kvx  # noqa: F401  触发后端注册
    needles, longqas, multies = _load_scenes(args.data)
    llm = SglLLM(args.model, args.attention_backend, args.mem_fraction)
    sampling = SimpleNamespace(max_tokens=args.max_output_tokens)

    t0 = time.time()
    needle_res = _run_needle(llm, sampling, needles)
    longqa_res = _run_longqa(llm, sampling, longqas)
    multi_res = _run_multiturn(llm, sampling, multies)
    wall = time.time() - t0

    metrics = {"tag": args.tag, "backend": args.attention_backend,
               "eval_wall_s": round(wall, 2),
               "needle": needle_res, "longqa": longqa_res, "multiturn": multi_res}
    path = os.path.join(args.out, f"{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
