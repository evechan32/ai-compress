"""P2 判定实验：needle 在中段，验证驱逐是否真的影响注意力。"""
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_LOG", "1")
os.environ.setdefault("PE_MODE", "off")

if os.environ.get("PE_MODE", "off") != "off":
    import kvcompress.pevict as pevict

    pevict.install()

from vllm import LLM, SamplingParams  # noqa: E402

NEEDLE = " The secret code is 74829. "
FILLER = "The history of the ancient library is long and complex. "


def make_prompt(tok, target_tokens=3500):
    unit = tok(FILLER, add_special_tokens=False)["input_ids"]
    n = target_tokens // len(unit)
    filler = FILLER * n
    half = len(filler) // 2
    body = filler[:half] + NEEDLE + filler[half:]
    return ("Read the document carefully.\n\n" + body
            + "\n\nQuestion: What is the secret code?\nAnswer:")


def main():
    llm = LLM(
        model="/models/qwen2.5-1.5b-instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        max_num_seqs=1,
        disable_log_stats=True,
        attention_backend=os.environ.get("PE_BACKEND") or None,
    )
    prompt = make_prompt(llm.get_tokenizer())
    ntok = len(llm.get_tokenizer().encode(prompt))
    out = llm.generate([prompt], SamplingParams(max_tokens=24, temperature=0.0))
    print(f"[NEEDLE] tokens={ntok} mode={os.environ['PE_MODE']} "
          f"text={out[0].outputs[0].text!r}", flush=True)


if __name__ == "__main__":
    main()
