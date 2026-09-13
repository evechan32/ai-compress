"""B 加固验证：多请求一致性 + prefix caching。

每个 prompt 中段放一个唯一 code。分别“单请求跑”和“批量跑”，
若两者输出逐字一致 → 逐请求驱逐无串扰。
PE_PC=1 开启 prefix caching 以暴露块签名碰撞。
"""
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_LOG", "1")
os.environ.setdefault("PE_MODE", "off")

if os.environ.get("PE_MODE", "off") != "off":
    import kvcompress.pevict as pevict

    pevict.install()

from vllm import LLM, SamplingParams  # noqa: E402

FILLER = "The history of the ancient library is long and complex. "
CODES = ["74829", "13579", "24680", "98765"]
FRACS = [0.15, 0.35, 0.55, 0.75]


def make_prompt(tok, code, frac=0.5, target=3000):
    unit = tok(FILLER, add_special_tokens=False)["input_ids"]
    filler = FILLER * (target // len(unit))
    pos = int(len(filler) * frac)
    body = filler[:pos] + f" The secret code is {code}. " + filler[pos:]
    return ("Read the document carefully.\n\n" + body
            + "\n\nQuestion: What is the secret code?\nAnswer:")


def main():
    llm = LLM(
        model="/models/qwen2.5-1.5b-instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        max_num_seqs=4,
        disable_log_stats=True,
        enable_prefix_caching=os.environ.get("PE_PC", "0") == "1",
        attention_backend=os.environ.get("PE_BACKEND") or None,
        max_num_batched_tokens=int(os.environ.get("PE_MAXBAT","0")) or None,
    )
    tok = llm.get_tokenizer()
    prompts = [make_prompt(tok, c, f) for c, f in zip(CODES, FRACS)]
    sp = SamplingParams(max_tokens=int(os.environ.get("PE_MAX_NEW", "10")),
                        temperature=0.0)

    singles = [o.outputs[0].text for o in llm.generate(prompts, sp)]
    batched = [o.outputs[0].text for o in llm.generate(prompts, sp)]

    ok = 0
    for i, c in enumerate(CODES):
        hit = c in batched[i]
        same = singles[i] == batched[i]
        ok += int(hit and same)
        print(f"[VAL] code={c} single={singles[i]!r} batch={batched[i]!r} "
              f"hit={hit} same={same}", flush=True)
    print(f"[VAL] mode={os.environ['PE_MODE']} pc={os.environ.get('PE_PC','0')} "
          f"pass={ok}/{len(CODES)}", flush=True)


if __name__ == "__main__":
    main()
