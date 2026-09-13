"""对 near-tie 鲁棒的质量指标：decode 第 1 步的下一 token 分布距离。

第 1 步两个配置的上下文完全相同（都是 prompt），所以分布差异纯粹来自驱逐。
每个配置导出 top-k logprob，之后离线比较：
  - top1 同/异率
  - 参考 top1 token 在另一配置下的 logprob 差（正值 = 退化）
  - 前向 KL（在 top-k 并集上归一化）
用法：
  PE_MODE=... PE_TAG=ref    python -m bench.p3_metric   # 导出 metric-ref.json
  PE_MODE=... PE_TAG=evict  python -m bench.p3_metric   # 导出 metric-evict.json
  PE_REF_TAG=ref PE_TAG=evict python -m bench.p3_metric compare
"""
import json
import math
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_MODE", "off")

if os.environ.get("PE_MODE", "off") != "off":
    import kvcompress.pevict as pevict

    pevict.install()

from vllm import LLM, SamplingParams  # noqa: E402

TEMPLATE = ("Answer the question based on the given documents. Only give me the "
            "answer and do not output any other words.\n\n{context}\n\n"
            "Question: {input}\nAnswer:")
FILES = ["qasper.jsonl", "2wikimqa.jsonl", "narrativeqa.jsonl",
         "hotpotqa.jsonl", "multifieldqa_en.jsonl"]
N = int(os.environ.get("PE_N", "60"))
TOPK = int(os.environ.get("PE_TOPK", "20"))
STEPS = int(os.environ.get("PE_STEPS", "16"))
OUT_DIR = "/root/ai-compress/bench/out"


def load_samples():
    out = []
    for fn in FILES:
        rows = []
        with open(os.path.join("/hy-tmp/longbench/data", fn), encoding="utf-8") as f:
            for line in f:
                if len(rows) >= N:
                    break
                rows.append(json.loads(line))
        out.append((fn, rows))
    return out


def export():
    tag = os.environ.get("PE_TAG", "x")
    llm = LLM(
        model="/models/qwen2.5-1.5b-instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        max_num_seqs=4,
        disable_log_stats=True,
        attention_backend=os.environ.get("PE_BACKEND") or None,
        max_num_batched_tokens=int(os.environ.get("PE_MAXBAT", "0")) or None,
    )
    sp = SamplingParams(max_tokens=STEPS, temperature=0.0, logprobs=TOPK)
    samples = load_samples()
    res = {}
    for fn, rows in samples:
        prompts = [TEMPLATE.format(context=r["context"][:20000], input=r["input"])
                   for r in rows]
        outs = llm.generate(prompts, sp)
        per = []
        for o in outs:
            per.append([{int(k): float(v.logprob) for k, v in step.items()}
                        for step in o.outputs[0].logprobs])
        res[fn] = per
        print(f"[M] {fn}: n={len(per)}", flush=True)
    path = os.path.join(OUT_DIR, f"metric-{tag}.json")
    json.dump(res, open(path, "w"))
    print(f"[M] wrote {path}", flush=True)


def kl(p, q):
    keys = set(p) | set(q)
    lp = {k: p.get(k, -25.0) for k in keys}
    lq = {k: q.get(k, -25.0) for k in keys}
    sp_ = sum(math.exp(v) for v in lp.values())
    sq = sum(math.exp(v) for v in lq.values())
    total = 0.0
    for k in keys:
        pp = math.exp(lp[k]) / sp_
        qq = math.exp(lq[k]) / sq
        if pp > 0:
            total += pp * math.log(pp / qq)
    return total


def compare():
    ref = json.load(open(os.path.join(OUT_DIR,
                f"metric-{os.environ['PE_REF_TAG']}.json")))
    cur = json.load(open(os.path.join(OUT_DIR, f"metric-{os.environ['PE_TAG']}.json")))
    n = div = 0
    first_divs, kl_agree, kl_div_step = [], [], []
    for fn in ref:
        for a, b in zip(ref[fn], cur[fn]):
            n += 1
            T = min(len(a), len(b))
            diverged = False
            for t in range(T):
                d = kl(a[t], b[t])
                ta = max(a[t], key=a[t].get)
                tb = max(b[t], key=b[t].get)
                if ta != tb:
                    diverged = True
                    first_divs.append(t)
                    kl_div_step.append(d)
                    break
                kl_agree.append(d)
            if diverged:
                div += 1
    import statistics as st
    print(f"[M] n={n} div_rate={div/max(1,n):.4f} "
          f"mean_first_div={st.mean(first_divs) if first_divs else -1:.2f} "
          f"mean_KL_agreeing={st.mean(kl_agree) if kl_agree else 0:.5f} "
          f"mean_KL_at_div={st.mean(kl_div_step) if kl_div_step else 0:.5f}",
          flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare()
    else:
        export()
