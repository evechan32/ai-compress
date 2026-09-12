"""P2 spike 运行脚本：位置式 prompt 驱逐，验证物理释放与输出。"""
import json
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PE_LOG", "1")
os.environ.setdefault("PE_MODE", "off")

if os.environ.get("PE_MODE", "off") != "off":
    import kvcompress.pevict as pevict

    pevict.install()

from vllm import LLM, SamplingParams  # noqa: E402

MAX_NEW = int(os.environ.get("PE_MAX_NEW", "64"))


def long_prompt():
    with open("/hy-tmp/longbench/data/qasper.jsonl", encoding="utf-8") as f:
        rec = json.loads(f.readline())
    return (
        "Answer the question based on the given documents. Only give me the "
        "answer and do not output any other words.\n\n" + rec["context"][:20000]
        + "\n\nQuestion: " + rec["input"] + "\nAnswer:"
    )


def main():
    llm = LLM(
        model="/models/qwen2.5-1.5b-instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        max_num_seqs=1,
        disable_log_stats=True,
    )
    prompt = long_prompt()
    ntok = len(llm.get_tokenizer().encode(prompt))
    out = llm.generate([prompt], SamplingParams(max_tokens=MAX_NEW,
                                                temperature=0.0))
    print(f"[RESULT] prompt_tokens={ntok} mode={os.environ['PE_MODE']} "
          f"text={out[0].outputs[0].text!r}", flush=True)


if __name__ == "__main__":
    main()
