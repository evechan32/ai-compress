"""配对精度评测：baseline vs 我们的方法，带 bootstrap 置信区间。

两种用法：
  1) 采集：PE_TAG=xxx PE_MODE=... 跑一遍，逐样本 F1 存盘
  2) 比较：PE_COMPARE=baseline PE_TAG=ours  -> 配对 bootstrap 的 ΔF1 + 95% CI
"""
import json
import os
import re
import string

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_MODE", "off")

if os.environ.get("PE_MODE", "off") != "off":
    import kvcompress.pevict as pevict

    pevict.install()

import numpy as np  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

TEMPLATE = ("Answer the question based on the given documents. Only give me the "
            "answer and do not output any other words.\n\n{context}\n\n"
            "Question: {input}\nAnswer:")
FILES = ["qasper.jsonl", "2wikimqa.jsonl", "narrativeqa.jsonl",
         "hotpotqa.jsonl", "multifieldqa_en.jsonl"]
N = int(os.environ.get("PE_N", "200"))
OUT = "/root/ai-compress/bench/out"


def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred, golds):
    best = 0.0
    p = _norm(pred).split()
    for g in golds:
        gg = _norm(g).split()
        if not p or not gg:
            best = max(best, 1.0 if p == gg else 0.0)
            continue
        common = set(p) & set(gg)
        if not common:
            continue
        prec, rec = len(common) / len(p), len(common) / len(gg)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def collect():
    tag = os.environ["PE_TAG"]
    llm = LLM(model="/models/qwen2.5-1.5b-instruct", max_model_len=8192,
              gpu_memory_utilization=0.6, enforce_eager=True, max_num_seqs=8,
              disable_log_stats=True,
              attention_backend=os.environ.get("PE_BACKEND") or None)
    sp = SamplingParams(max_tokens=32, temperature=0.0)
    allf1 = []
    for fn in FILES:
        rows = []
        with open(os.path.join("/hy-tmp/longbench/data", fn), encoding="utf-8") as f:
            for line in f:
                if len(rows) >= N:
                    break
                rows.append(json.loads(line))
        outs = llm.generate(
            [TEMPLATE.format(context=r["context"][:20000], input=r["input"]) for r in rows], sp)
        f1s = [_f1(o.outputs[0].text.strip(), r.get("answers", []))
               for o, r in zip(outs, rows)]
        allf1.extend(f1s)
        print(f"[PAIR] {fn}: n={len(f1s)} f1={sum(f1s)/len(f1s):.4f}", flush=True)
    json.dump(allf1, open(f"{OUT}/f1s-{tag}.json", "w"))
    print(f"[PAIR] tag={tag} overall={sum(allf1)/len(allf1):.4f} "
          f"n={len(allf1)} saved", flush=True)


def compare():
    ref = np.array(json.load(open(f"{OUT}/f1s-{os.environ['PE_COMPARE']}.json")))
    cur = np.array(json.load(open(f"{OUT}/f1s-{os.environ['PE_TAG']}.json")))
    n = min(len(ref), len(cur))
    ref, cur = ref[:n], cur[:n]
    d = cur - ref
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, n, n)].mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"[PAIR] {os.environ['PE_COMPARE']}={ref.mean():.4f} "
          f"{os.environ['PE_TAG']}={cur.mean():.4f} dF1={d.mean():+.4f} "
          f"95%CI=[{lo:+.4f},{hi:+.4f}] n={n} "
          f"significant={'YES' if (lo > 0 or hi < 0) else 'NO'}", flush=True)


if __name__ == "__main__":
    compare() if os.environ.get("PE_COMPARE") else collect()
