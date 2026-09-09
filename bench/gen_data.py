"""评测数据生成：needle-in-haystack + 多轮长对话（自生成，免外部下载）。

纯 python，无 vllm/torch 依赖。token 数按 ~4 字符/token 估算（英文）。
确定性：固定 seed。
"""
from __future__ import annotations

import argparse
import json
import random
import string

_FILLER_SENT = (
    "The harbor master logged every vessel that passed the north pier, "
    "noting cargo, crew size, and estimated arrival time at the outer buoy."
)
_SUBJECTS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
]


def _filler_tokens(n: int) -> str:
    """生成约 n token 的无关填充文本。"""
    return (" ".join([_FILLER_SENT] * max(1, n // 30)))[: n * 4]


def _gen_needle_scenes(rng: random.Random, out_path: str, n_per_depth: int = 2,
                       haystack_tokens: int = 1400) -> None:
    scenes = []
    for depth in (0.25, 0.5, 0.75):
        for _ in range(n_per_depth):
            code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
            fact = f"The secret code for warehouse {depth:.2f} is {code}."
            half = haystack_tokens // 2
            # 在目标深度插入 fact（基于 token 位置近似）
            head_tokens = int(haystack_tokens * depth)
            before = _filler_tokens(head_tokens)
            after = _filler_tokens(haystack_tokens - head_tokens - 15)
            prompt = (
                f"{before}\n{fact}\n{after}\n"
                f"Based on the text above, what is the secret code "
                f"for warehouse {depth:.2f}? Answer with the code only."
            )
            scenes.append({
                "type": "needle", "depth": depth, "answer": code,
                "prompt": prompt, "fact": fact,
            })
    with open(out_path, "w", encoding="utf-8") as f:
        for s in scenes:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"needle: {len(scenes)} scenes -> {out_path}")


def _gen_multiturn_scenes(rng: random.Random, out_path: str,
                          n_convs: int = 2, n_turns: int = 6,
                          ctx_tokens_per_turn: int = 100) -> None:
    scenes = []
    for c in range(n_convs):
        turns = []
        for t in range(n_turns):
            subj = rng.choice(_SUBJECTS)
            val = "".join(rng.choices(string.ascii_lowercase + string.digits, k=5))
            turns.append({
                "subject": subj, "value": val,
                "context": f"User note: assign {subj} to value {val}. "
                           f"{_filler_tokens(ctx_tokens_per_turn)}",
            })
        scenes.append({"type": "multiturn", "conversation": c, "turns": turns})
    with open(out_path, "w", encoding="utf-8") as f:
        for s in scenes:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"multiturn: {len(scenes)} conversations -> {out_path}")


def _gen_longqa_scenes(rng: random.Random, out_path: str) -> None:
    scenes = []
    for toks in (3000, 8000, 12000):
        for _ in range(2):
            code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
            body = _filler_tokens(toks - 40)
            scene_prompt = (
                f"{body}\nIMPORTANT NOTE: The archive key for this vault is {code}.\n"
                f"Question: What is the archive key for this vault? "
                f"Answer with the key only."
            )
            scenes.append({
                "type": "longqa", "doc_tokens": toks, "answer": code,
                "prompt": scene_prompt,
            })
    with open(out_path, "w", encoding="utf-8") as f:
        for s in scenes:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"longqa: {len(scenes)} scenes -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="bench/data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    _gen_needle_scenes(rng, os.path.join(args.out_dir, "needle.jsonl"))
    _gen_multiturn_scenes(rng, os.path.join(args.out_dir, "multiturn.jsonl"))
    _gen_longqa_scenes(rng, os.path.join(args.out_dir, "longqa.jsonl"))


if __name__ == "__main__":
    main()
