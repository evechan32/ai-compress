"""S1 冒烟：vLLM 0.11.2 + pepoch_v11（position 模式，验证真释放 + 压实）。

跑法（3090，vLLM 0.11.2 target 环境）：
    cd /root/ai-compress
    TMPDIR=/dev/shm/tmp PYTHONPATH=/hy-tmp/t29 VLLM_USE_FLASHINFER_SAMPLER=0 PE_LOG=1 \
      python3 bench/s1_smoke_v11.py
"""
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_LOG", "1")

MODE = os.environ.get("PE_MODE", "on")
if MODE != "off":
    import kvcompress.pevict_v11 as pe11

    pe11.install()

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("PE_MODEL", "/dev/shm/models/Qwen2.5-1.5B-Instruct")


def main():
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.6,
              enforce_eager=True, disable_log_stats=True)
    body = "The harbor master logged every vessel that passed the north pier. " * 90
    prompt = ("Read the document.\n\n" + body
              + "\n\nQuestion: What did the harbor master do?\nAnswer:")
    ntok = len(llm.get_tokenizer().encode(prompt))
    cap = os.environ.get("PE_MAXNEW", "400")
    o = llm.generate([prompt], SamplingParams(max_tokens=int(cap), temperature=0))
    import hashlib
    text = o[0].outputs[0].text
    h = hashlib.sha1(text.encode()).hexdigest()[:12]
    toks = len(o[0].outputs[0].token_ids)
    print(f"S1 mode={MODE} genwin={os.environ.get('PE_GEN_WINDOW','0')} "
          f"prompt_tokens={ntok} out_tokens={toks} sha1={h} out={text[:70]!r}", flush=True)


if __name__ == "__main__":
    main()
