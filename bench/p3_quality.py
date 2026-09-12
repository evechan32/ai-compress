"""B 质量对齐：teacher-forced 答案 NLL（低方差）。

对每条 LongBench 样本，把 [context][question][answer] 作为整段 prefill，
obs 覆盖 question+answer，保留决策由 Q/A 对上下文的注意力驱动；
量答案 token 的平均 NLL。基线 vs chunkkv 的差值即驱逐损失。
"""
import json
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_LOG", "0")
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


def answer_nll(llm, tok, context, question, answer):
    prefix = TEMPLATE.format(context=context[:20000], input=question)
    full = prefix + " " + answer
    ids_p = tok.encode(prefix)
    ids_f = tok.encode(full)
    p = 0
    while p < len(ids_p) and p < len(ids_f) and ids_p[p] == ids_f[p]:
        p += 1
    if p >= len(ids_f):
        return None
    out = llm.generate(
        [full],
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
    )[0]
    plp = out.prompt_logprobs
    nlls = []
    for i in range(p, len(ids_f)):
        entry = plp[i] if plp is not None and i < len(plp) else None
        if not entry:
            continue
        hit = entry.get(ids_f[i])
        if hit is None:
            continue
        nlls.append(-float(hit.logprob))
    return sum(nlls) / len(nlls) if nlls else None


def main():
    llm = LLM(
        model="/models/qwen2.5-1.5b-instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        max_num_seqs=4,
        disable_log_stats=True,
        attention_backend=os.environ.get("PE_BACKEND") or None,
    )
    tok = llm.get_tokenizer()
    n = int(os.environ.get("PE_N", "40"))
    agg, per_task = [], {}
    for fn in FILES:
        vals = []
        with open(os.path.join("/hy-tmp/longbench/data", fn), encoding="utf-8") as f:
            for line in f:
                if len(vals) >= n:
                    break
                r = json.loads(line)
                ans = (r.get("answers") or [""])[0]
                v = answer_nll(llm, tok, r["context"], r["input"], ans)
                if v is not None:
                    vals.append(v)
        if vals:
            per_task[fn] = round(sum(vals) / len(vals), 4)
            agg.extend(vals)
        print(f"[Q] {fn}: n={len(vals)} nll={per_task.get(fn)}", flush=True)
    print(f"[Q] mode={os.environ['PE_MODE']} ratio={os.environ.get('PE_RATIO','0')} "
          f"overall_nll={round(sum(agg)/len(agg), 4)} per_task={per_task}", flush=True)


if __name__ == "__main__":
    main()
