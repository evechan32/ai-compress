"""LongBench 任务精度（F1）：无驱逐 vs 驱逐 vs 位置式（走 vLLM）。"""
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

from vllm import LLM, SamplingParams  # noqa: E402

TEMPLATE = ("Answer the question based on the given documents. Only give me the "
            "answer and do not output any other words.\n\n{context}\n\n"
            "Question: {input}\nAnswer:")
FILES = ["qasper.jsonl", "2wikimqa.jsonl", "narrativeqa.jsonl",
         "hotpotqa.jsonl", "multifieldqa_en.jsonl"]
N = int(os.environ.get("PE_N", "60"))
MAXNEW = int(os.environ.get("PE_MAXNEW", "32"))


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
        prec = len(common) / len(p)
        rec = len(common) / len(gg)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def main():
    mode = os.environ["PE_MODE"]
    _kw = {}
    if os.environ.get("PE_ASYNC") is not None:
        _kw["async_scheduling"] = os.environ["PE_ASYNC"] == "1"
    llm = LLM(
        model=os.environ.get("PE_MODEL", "/models/qwen2.5-1.5b-instruct"),
        max_model_len=int(os.environ.get("PE_MAXLEN", "8192")),
        gpu_memory_utilization=float(os.environ.get("PE_GMEM", "0.6")),
        enforce_eager=True,
        max_num_seqs=4,
        disable_log_stats=True,
        attention_backend=os.environ.get("PE_BACKEND") or None,
        tensor_parallel_size=int(os.environ.get("PE_TP", "1")),
        **_kw,
    )
    sp = SamplingParams(max_tokens=MAXNEW, temperature=0.0)
    per_task, agg = {}, []
    for fn in FILES:
        rows = []
        with open(os.path.join("/hy-tmp/longbench/data", fn), encoding="utf-8") as f:
            for line in f:
                if len(rows) >= N:
                    break
                rows.append(json.loads(line))
        prompts = [TEMPLATE.format(context=r["context"][:20000], input=r["input"])
                   for r in rows]
        outs = llm.generate(prompts, sp)
        f1s = [_f1(o.outputs[0].text.strip(), r.get("answers", []))
               for o, r in zip(outs, rows)]
        per_task[fn] = round(sum(f1s) / max(1, len(f1s)), 4)
        agg.extend(f1s)
        print(f"[F1] {fn}: n={len(f1s)} f1={per_task[fn]}", flush=True)
    print(f"[F1] mode={mode} ratio={os.environ.get('PE_RATIO','0')} "
          f"overall={round(sum(agg)/len(agg), 4)} per_task={per_task}", flush=True)
    json.dump(per_task,
              open(f"/root/ai-compress/bench/out/f1-{mode}-"
                   f"{os.environ.get('PE_RATIO','0')}.json", "w"))


if __name__ == "__main__":
    main()
