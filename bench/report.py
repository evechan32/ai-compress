"""汇总对比报告：基线 vs 插件 tag。

用法：python bench/report.py --results bench/out/baseline.json bench/out/plugin-w256.json [更多...]
"""
from __future__ import annotations

import argparse
import json


def _needle_stats(m):
    hits = sum(s["hit"] for s in m["needle"]["scenes"])
    total = len(m["needle"]["scenes"])
    by_depth = {}
    for s in m["needle"]["scenes"]:
        by_depth.setdefault(s["depth"], [0, 0])
        by_depth[s["depth"]][1] += 1
        by_depth[s["depth"]][0] += 1 if s["hit"] else 0
    return {
        "recall": round(hits / total, 3) if total else None,
        "by_depth": {str(d): f"{h}/{n}" for d, (h, n) in sorted(by_depth.items())},
    }


def _multiturn_stats(m):
    convs = m["multiturn"]["conversations"]
    per_turn = []
    for conv in convs:
        for idx, t in enumerate(conv["turns"]):
            while len(per_turn) <= idx:
                per_turn.append([0, 0])
            per_turn[idx][1] += 1
            per_turn[idx][0] += 1 if t["hit"] else 0
    return {"per_turn_hit": {f"turn{t}": f"{h}/{n}" for t, (h, n) in enumerate(per_turn)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    args = ap.parse_args()

    rows = []
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        ns, ms = _needle_stats(m), _multiturn_stats(m)
        rows.append({
            "tag": m["tag"],
            "needle_recall": ns["recall"],
            "needle_by_depth": ns["by_depth"],
            "multiturn_per_turn": ms["per_turn_hit"],
            "eval_wall_s": m["eval_wall_s"],
            "throughput_tok_s": m["throughput_tok_s"],
        })
        print(f"\n=== {m['tag']} ===")
        print(f"  needle recall: {ns['recall']}  by_depth: {ns['by_depth']}")
        print(f"  multiturn per_turn hit: {ms['per_turn_hit']}")
        print(f"  eval_wall_s: {m['eval_wall_s']}  throughput_tok/s: {m['throughput_tok_s']}")
    with open("bench/out/_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\nsummary -> bench/out/_summary.json")


if __name__ == "__main__":
    main()
