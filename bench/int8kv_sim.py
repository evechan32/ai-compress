"""int8 KV 量化的数值等效模拟：量化→反量化 hook（不碰内核，先拿质量与开销）。

原理：在 `k_proj` / `v_proj` 的输出上插 forward hook，做
    q = clamp(round(x / s), -127, 127);  x' = q * s
这与"KV 池存 int8 + 读取时反量化"的**数值路径完全一致**（只差显存占用是真的省 2×）。
KIVI 式粒度：**K 按 channel、V 按 token**（K 的离群点按 channel，V 按 token）。

对照：无量化 / per-tensor / K-channel+V-token，报 NIAH 命中率与耗时。

跑法：
    cd /root/ai-compress
    PYTHONPATH=/root/kvpress-libs PE_KVQ=perchan python3 bench/int8kv_sim.py
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.sgl_longbench import TEMPLATE, _f1  # noqa: E402

MODEL = os.environ.get("PE_MODEL", "/dev/shm/models/Qwen2.5-1.5B-Instruct")
QE = os.environ.get("PE_KVQ", "none")   # none | pertensor | perchan
FILLER = "The harbor master logged every vessel that passed the north pier. "


FIXED_SCALE = {}  # 全局标量模式：完全复刻 vLLM（每层一个标量、只算一次）


def qdq(x, dim, key=None):
    """对称 int8 量化→反量化。

    dim=None   → 整个张量一个标量（= per-layer 标量，粗粒度）
    dim=-1     → 每 token 一个标量
    dim=-2     → 每 channel 一个标量
    key 不为空且 FIXED_SCALE 已缓存 → 用缓存的全局标量（复刻 vLLM 的"只算一次"）
    """
    if dim is None:
        if key is not None and key in FIXED_SCALE:
            s = FIXED_SCALE[key]
        else:
            s = x.abs().amax().clamp_min(1e-8) / 127.0
            if key is not None:
                FIXED_SCALE[key] = s
    else:
        s = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.clamp(torch.round(x / s), -127, 127)
    return q * s


STATS = [0, 0.0, 0.0]  # [调用次数, 绝对误差累加, 原值绝对幅度累加]


def install_hooks(model, mode):
    if mode == "none":
        return []
    handles = []
    for name, mod in model.named_modules():
        if not name.endswith(("k_proj", "v_proj")):
            continue
        is_k = name.endswith("k_proj")
        # KIVI：K 按 channel(-1)，V 按 token(-2)；perchan 模式用这个，pertensor 用全张量
        if mode == "layer_fixed":
            dim = None
        elif mode == "layer_percall":
            dim = None
        elif mode == "pertoken":
            dim = (-1,)
        else:  # perchan：K 按 channel、V 按 token
            dim = (-2,) if is_k else (-2,)
        _key = (name, mode) if mode == "layer_fixed" else None

        def hook(m, inp, out, dim=dim, _key=_key):
            y = qdq(out, dim, _key)
            STATS[0] += 1
            STATS[1] += float((y - out).abs().mean())
            STATS[2] += float(out.abs().mean())
            return y

        handles.append(mod.register_forward_hook(hook))
    print(f"[INT8KV] hooks_installed={len(handles)} mode={mode} STATS={STATS}", flush=True)
    return handles


def build_prompts(tok, n, target=1536):
    unit = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    filler = FILLER * (target // unit)
    rng = random.Random(5)
    out = []
    for i in range(n):
        code = f"Q{i:04d}Y"
        pos = max(1, int(len(filler) * rng.random()))
        body = filler[:pos] + f" IMPORTANT: the secret vault code is {code}. " + filler[pos:]
        out.append((code, "Read the document carefully.\n\n" + body
                    + "\n\nQuestion: What is the secret vault code?\nAnswer:"))
    return out


def main():
    n = int(os.environ.get("PE_N", "20"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).cuda().eval()
    handles = install_hooks(model, QE)

    samples = build_prompts(tok, n)
    hit = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for code, prompt in samples:
            ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
            gen = model.generate(ids, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
            hit += int(code in text)
    wall = time.perf_counter() - t0
    for h in handles:
        h.remove()
    if QE != "none":
        print(f"[INT8KV] calls={STATS[0]} mean_abs_err={STATS[1]/max(1,STATS[0]):.6f} "
              f"mean_abs_val={STATS[2]/max(1,STATS[0]):.6f} "
              f"rel={100*(STATS[1]/max(1,STATS[2])):.2f}%", flush=True)
    print(f"[INT8KV] mode={QE} n={n} hit={hit}/{n} wall={wall:.1f}s "
          f"per_sample={wall/n:.2f}s", flush=True)

    if os.environ.get("PE_F1") == "1":
        data = "/hy-tmp/longbench/data"
        for fn in ["qasper.jsonl", "multifieldqa_en.jsonl"]:
            rows = []
            with open(os.path.join(data, fn), encoding="utf-8") as f:
                for line in f:
                    if len(rows) >= n:
                        break
                    rows.append(json.loads(line))
            t0 = time.perf_counter()
            f1s = []
            with torch.no_grad():
                for r in rows:
                    prompt = TEMPLATE.format(context=r["context"][:20000], input=r["input"])
                    ids = tok(prompt, return_tensors="pt", truncation=True,
                              max_length=8192)["input_ids"].cuda()
                    gen = model.generate(ids, max_new_tokens=32, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                    pred = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
                    f1s.append(_f1(pred, r.get("answers", [])))
            print(f"[INT8KV-F1] mode={QE} {fn} n={len(f1s)} "
                  f"f1={sum(f1s)/max(1,len(f1s)):.4f}", flush=True)


if __name__ == "__main__":
    main()
