"""基线 vs 插件 回答一致性评测（固定种子，同数据，temperature=0）。

判据：同一批任务（needle / 多轮对话），baseline 与启用插件的回答应基本一致。
报告精确一致率 + 平均相似度（difflib）+ 差异样本。

输入为 bench/run_eval.py 产出的 {tag}.json（相同 bench/data，seed 相同）。

用法：python bench/agreement.py bench/out/baseline.json bench/out/plugin-w1024.json [更多]
"""
from __future__ import annotations

import argparse
import difflib
import json


def _answers(m: dict) -> list[str]:
    """按固定顺序抽取所有回答文本（needle 场景 + 多轮各 turn）。"""
    out: list[str] = []
    for s in m["needle"]["scenes"]:
        out.append(s["answer"])
    for conv in m["multiturn"]["conversations"]:
        for t in conv["turns"]:
            out.append(t["answer"])
    return out


def _norm(s: str) -> str:
    return " ".join(s.strip().split())


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def compare(base_path: str, other_path: str) -> dict:
    with open(base_path, encoding="utf-8") as f:
        base = json.load(f)
    with open(other_path, encoding="utf-8") as f:
        other = json.load(f)
    ab, ao = _answers(base), _answers(other)
    assert len(ab) == len(ao), f"answer count mismatch: {len(ab)} vs {len(ao)}"
    exact = sum(1 for x, y in zip(ab, ao) if _norm(x) == _norm(y))
    sims = [_ratio(_norm(x), _norm(y)) for x, y in zip(ab, ao)]
    diffs = [
        {"i": i, "base": x[:200], "other": y[:200], "sim": round(s, 3)}
        for i, (x, y, s) in enumerate(zip(ab, ao, sims)) if s < 1.0
    ]
    return {
        "base_tag": base["tag"], "other_tag": other["tag"],
        "n": len(ab),
        "exact_match": round(exact / len(ab), 3),
        "mean_sim": round(sum(sims) / len(sims), 3),
        "min_sim": round(min(sims), 3),
        "n_diff": len(diffs),
        "sample_diffs": diffs[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="第一个为基线，其余为插件/对比")
    args = ap.parse_args()
    base = args.results[0]
    for other in args.results[1:]:
        r = compare(base, other)
        print(f"\n=== {r['base_tag']} vs {r['other_tag']} ===")
        print(f"  n={r['n']} exact_match={r['exact_match']} "
              f"mean_sim={r['mean_sim']} min_sim={r['min_sim']} n_diff={r['n_diff']}")
        for d in r["sample_diffs"]:
            print(f"  [diff #{d['i']} sim={d['sim']}]")
            print(f"    base : {d['base']!r}")
            print(f"    other: {d['other']!r}")


if __name__ == "__main__":
    main()
