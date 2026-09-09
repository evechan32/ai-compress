"""评测入口：基线 / 插件共用。同一 tag 启动一次引擎，顺序执行两类任务。

用法：
  python bench/run_eval.py --model /models/qwen2.5-1.5b-instruct \
      --data bench/data --out bench/out --tag baseline
  AI_COMPRESS_ENABLE=1 AI_COMPRESS_RSWA_WINDOW=256 python bench/run_eval.py ... --tag plugin-w256
输出：{out}/{tag}.json —— 各场景答案、正确性、耗时与吞吐。
"""
from __future__ import annotations

import argparse
import json
import os
import time


def _load_scenes(data_dir: str) -> tuple[list[dict], list[dict], list[dict]]:
    def _read(name):
        with open(os.path.join(data_dir, name), encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    return _read("needle.jsonl"), _read("longqa.jsonl"), _read("multiturn.jsonl")


def _run_needle(llm, sampling, scenes) -> dict:
    prompts = [s["prompt"] for s in scenes]
    t0 = time.time()
    outs = llm.generate(prompts, sampling)
    wall = time.time() - t0
    results = []
    for s, o in zip(scenes, outs):
        ans = o.outputs[0].text.strip()
        results.append({
            "depth": s["depth"], "gold": s["answer"],
            "answer": ans, "hit": s["answer"] in ans,
        })
    return {"scenes": results, "wall_s": round(wall, 2),
            "out_tokens": sum(len(o.outputs[0].token_ids) for o in outs)}


def _run_multiturn(llm, sampling, scenes) -> dict:
    all_results = []
    t0 = time.time()
    for conv in scenes:
        convo_results = []
        history = "You are a helpful assistant. Remember facts stated in the conversation.\n"
        for t, turn in enumerate(conv["turns"]):
            history += f"User: {turn['context']}\n"
            history += (f"User: What value is assigned to {turn['subject']}? "
                        f"Answer concisely.\n")
            out = llm.generate([history], sampling)
            ans = out[0].outputs[0].text.strip()
            convo_results.append({
                "turn": t, "subject": turn["subject"],
                "gold": turn["value"], "answer": ans,
                "hit": turn["value"] in ans,
            })
            history += f"Assistant: {ans}\n"
        all_results.append({"conversation": conv["conversation"],
                            "turns": convo_results})
    wall = time.time() - t0
    return {"conversations": all_results, "wall_s": round(wall, 2)}


def _run_longqa(llm, sampling, scenes) -> dict:
    results = []
    wall = 0.0
    for s in scenes:
        t0 = time.time()
        out = llm.generate([s["prompt"]], sampling)[0]
        wall += time.time() - t0
        ans = out.outputs[0].text.strip()
        results.append({
            "doc_tokens": s["doc_tokens"], "gold": s["answer"],
            "answer": ans, "hit": s["answer"] in ans,
        })
    return {"scenes": results, "wall_s": round(wall, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/qwen2.5-1.5b-instruct")
    ap.add_argument("--data", default="bench/data")
    ap.add_argument("--out", default="bench/out")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dtype", default="bfloat16", help="量化模型用 auto")
    ap.add_argument("--max-output-tokens", type=int, default=60)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--prefix-caching", action="store_true",
                    help="启用 vLLM prefix caching（默认关闭以对齐早期评测）")
    args = ap.parse_args()

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    os.makedirs(args.out, exist_ok=True)
    needles, longqas, multies = _load_scenes(args.data)

    t_engine = time.time()
    llm = LLM(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.85, enforce_eager=True,
              enable_prefix_caching=args.prefix_caching)
    engine_load_s = round(time.time() - t_engine, 2)
    sampling = SamplingParams(max_tokens=args.max_output_tokens, temperature=0.0)

    t0 = time.time()
    needle_res = _run_needle(llm, sampling, needles)
    longqa_res = _run_longqa(llm, sampling, longqas)
    multi_res = _run_multiturn(llm, sampling, multies)
    total_wall = time.time() - t0

    out_tok = needle_res["out_tokens"] + sum(
        len(turn["answer"].split()) for conv in multi_res["conversations"]
        for turn in conv["turns"]
    )
    metrics = {
        "tag": args.tag,
        "engine_load_s": engine_load_s,
        "eval_wall_s": round(total_wall, 2),
        "throughput_tok_s": round(out_tok / total_wall, 2) if total_wall else None,
        "needle": needle_res,
        "longqa": longqa_res,
        "multiturn": multi_res,
    }
    path = os.path.join(args.out, f"{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
