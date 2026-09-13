"""物理释放的吞吐收益：并发量超过不驱逐时的 KV 容量。

对比 passthrough（装插件但不驱逐，显存占用同基线）vs chunkkv（真实释放块）。
M 个长 prompt 并发；M 取得足够大，使不驱逐时 KV 池被占满（触发 preemption）。
"""
import json
import os
import random
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_MODE", "off")

if os.environ.get("PE_MODE", "off") != "off":
    import kvcompress.pevict as pevict

    pevict.install()

from vllm import LLM, SamplingParams  # noqa: E402

WORDS = ("committee archive correspondence administrative recurring review "
         "ledger transit harbour lantern quarry meadow ridge copper basin "
         "verdict statute chapter vessel orchard granite ferry prairie").split()


def make_prompts(tok, m, target=3000):
    out = []
    for i in range(m):
        rng = random.Random(1000 + i)
        body = " ".join(rng.choice(WORDS) for _ in range(target // 3))
        out.append(f"Read the document.\n\n{body}\n\nQuestion: What themes "
                   f"recur? (doc {i})\nAnswer:")
    return out


def main():
    mode = os.environ["PE_MODE"]
    m = int(os.environ.get("PE_M", "64"))
    t = int(os.environ.get("PE_MAXNEW", "32"))
    llm = LLM(
        model="/models/qwen2.5-1.5b-instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        max_num_seqs=m,
        disable_log_stats=True,
        enable_prefix_caching=False,
        attention_backend=os.environ.get("PE_BACKEND") or None,
    )
    prompts = make_prompts(llm.get_tokenizer(), m)
    sp = SamplingParams(max_tokens=t, temperature=0.0)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    wall = time.time() - t0
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[PERF] mode={mode} M={m} T={t} wall={wall:.1f}s "
          f"out_tokens={out_tokens} tok_per_s={out_tokens / wall:.1f}", flush=True)
    json.dump({"mode": mode, "M": m, "T": t, "wall": wall,
               "out_tokens": out_tokens, "tok_per_s": out_tokens / wall},
              open(f"/root/ai-compress/bench/out/perf-{mode or 'off'}.json", "w"))


if __name__ == "__main__":
    main()
