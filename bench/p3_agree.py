"""B 质量对齐（decode 侧）：baseline vs 驱逐的逐 token 一致率 + F1。

驱逐在 prefill 之后生效，只影响 decode。所以对同一批样本，
分别用 baseline 与 PE_MODE 生成 16 token（贪心），比较生成 token 的一致率。
先跑 PE_MODE=off 写出基线，再跑驱逐模式对比。
"""
import json
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
MAXNEW = int(os.environ.get("PE_MAX_NEW", "16"))
OUT_DIR = "/root/ai-compress/bench/out"


def load_samples():
    samples = []
    for fn in FILES:
        rows = []
        with open(os.path.join("/hy-tmp/longbench/data", fn), encoding="utf-8") as f:
            for line in f:
                if len(rows) >= N:
                    break
                rows.append(json.loads(line))
        samples.append((fn, rows))
    return samples


def main():
    mode = os.environ["PE_MODE"]
    llm = LLM(
        model="/models/qwen2.5-1.5b-instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        max_num_seqs=4,
        disable_log_stats=True,
        attention_backend=os.environ.get("PE_BACKEND") or None,
        max_num_batched_tokens=int(os.environ.get("PE_MAXBAT","0")) or None,
    )
    sp = SamplingParams(max_tokens=MAXNEW, temperature=0.0)
    samples = load_samples()
    results = {}
    for fn, rows in samples:
        prompts = [TEMPLATE.format(context=r["context"][:20000], input=r["input"])
                   for r in rows]
        outs = llm.generate(prompts, sp)
        results[fn] = [list(o.outputs[0].token_ids) for o in outs]
        print(f"[G] {fn}: n={len(rows)}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"gen-{mode or 'off'}.json")
    json.dump(results, open(path, "w"))
    print(f"[G] mode={mode} wrote {path}", flush=True)

    if mode != "off":
        base = json.load(open(os.path.join(OUT_DIR, "gen-off.json")))
        tot = same = 0
        for fn, _ in samples:
            for a, b in zip(base[fn], results[fn]):
                m = min(len(a), len(b))
                tot += m
                same += sum(1 for i in range(m) if a[i] == b[i])
        print(f"[G] mode={mode} token_agree={same/max(1,tot):.4f} "
              f"({same}/{tot})", flush=True)


if __name__ == "__main__":
    main()
