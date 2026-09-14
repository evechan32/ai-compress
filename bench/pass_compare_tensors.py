"""对比"第一遍"与"C 的第二遍"的张量：hidden / q(pre-RoPE) / k / v。

逐个 token 比 cosine 相似度：pass1 的 token j vs pass2 的 token j（第二遍里 context 的副本）。
同时测两种第二遍用法：
  - 不给 position_ids（transformers 默认从 0 开始 → 与第一遍位置重合）
  - 给 position_ids = [L, 2L)（正确定位）
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/models/qwen2.5-1.5b-instruct"
S = 384
LAYER = 27
FILLER = "The harbor master logged every vessel that passed the north pier. "
NEEDLE = " IMPORTANT: the secret vault code is K7XQ21. "


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda()
    m.eval()

    unit = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    n = S // unit
    text = FILLER * (n // 2) + NEEDLE + FILLER * (n - n // 2)
    ids = tok(text, return_tensors="pt")["input_ids"][:, :S].cuda()
    L = ids.shape[1]
    needle = len(tok(FILLER * (n // 2) + NEEDLE[:-len("K7XQ21. ")],
                     add_special_tokens=False)["input_ids"])
    nd = slice(needle, needle + 6)

    layer = m.model.layers[LAYER].self_attn
    cap = {}

    def mk_out(name):
        return lambda mod, i, o: cap.__setitem__(
            name, o.detach().reshape(-1, o.shape[-1]).float().cpu())

    def mk_in(name):
        return lambda mod, i: cap.__setitem__(
            name, i[0].detach().reshape(-1, i[0].shape[-1]).float().cpu())

    hs = [layer.q_proj.register_forward_pre_hook(mk_in("h")),
          layer.q_proj.register_forward_hook(mk_out("q")),
          layer.k_proj.register_forward_hook(mk_out("k")),
          layer.v_proj.register_forward_hook(mk_out("v"))]

    with torch.no_grad():
        o1 = m(ids, use_cache=True)
    cache = o1.past_key_values
    del o1
    p1 = {k: v.clone() for k, v in cap.items()}

    runs = {}
    with torch.no_grad():
        cap.clear()
        m(ids, past_key_values=cache, use_cache=False)
        runs["no_posids"] = {k: v.clone() for k, v in cap.items()}
        cap.clear()
        pos = torch.arange(L, 2 * L).unsqueeze(0).cuda()
        m(ids, past_key_values=cache, use_cache=False, position_ids=pos)
        runs["posids_L..2L"] = {k: v.clone() for k, v in cap.items()}

    for tag, p2 in runs.items():
        print(f"\n=== second pass: {tag} ===", flush=True)
        for name in ("h", "q", "k", "v"):
            a, b = p1[name], p2[name]
            c = cos(a, b)
            print(f"  {name:<2} cos(all)={c.mean():.4f} min={c.min():.4f} "
                  f"| needle={c[nd].mean():.4f} filler={c[:64].mean():.4f}",
                  flush=True)
    print(f"\nneedle_pos={needle} L={L} layer={LAYER}", flush=True)


if __name__ == "__main__":
    main()
