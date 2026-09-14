"""能否用第一遍的 Q/K 模拟"第二遍自查"，省掉 extra prefill？

第二遍的本质 = token j 作为 query 去看缓存里的 j（非因果自查）。
第一遍已经算出全部 Q、K（post-RoPE），所以可以离线算
    softmax(Q_ctx K_ctxᵀ/√d)   ← 不加因果掩码
再对 query 取 max → 每-token 重要性 → needle rank。

对比：第一遍因果注意力（sum/max）、真第二遍（带 cache 再喂一次）。
"""
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/models/qwen2.5-1.5b-instruct"
S = 512
FILLER = "The harbor master logged every vessel that passed the north pier. "
NEEDLE = " IMPORTANT: the secret vault code is K7XQ21. "


def rope(x, pos, theta):
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))
    f = torch.outer(pos.float().to(x.device), inv)
    emb = torch.cat([f, f], dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    half = d // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rot * sin


def rank_of(imp, pos):
    return int((imp > imp[pos]).sum())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda()
    m.eval()
    cfg = m.config
    Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.hidden_size // Hq
    rep = Hq // Hkv
    theta = cfg.rope_parameters.get("rope_theta", 1e6)

    unit = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    n = S // unit
    text = FILLER * (n // 2) + NEEDLE + FILLER * (n - n // 2)
    ids = tok(text, return_tensors="pt")["input_ids"][:, :S].cuda()
    L = ids.shape[1]
    needle = len(tok(FILLER * (n // 2) + NEEDLE[:-len("K7XQ21. ")],
                     add_special_tokens=False)["input_ids"])

    qv = {}
    layer = m.model.layers[-1].self_attn
    layer.q_proj.register_forward_hook(
        lambda mod, i, o: qv.__setitem__("q", o.detach()))
    with torch.no_grad():
        out = m(ids, output_attentions=True, use_cache=True)
    cache = out.past_key_values
    a_causal = out.attentions[-1][0].float().cpu()
    del out

    q_rope = rope(qv["q"].reshape(L, Hq, D).float().cpu(),
                  torch.arange(L), theta)
    k = cache.layers[-1].keys[0][:, :L, :].float().cpu()
    kx = k.permute(1, 0, 2).repeat_interleave(rep, dim=1)
    sc = torch.einsum("ihd,jhd->hij", q_rope, kx) / (D ** 0.5)
    p = torch.softmax(sc, dim=-1)

    for k in (1, 8, 32, 128):
        sub = a_causal[:, -k:, :]
        print(f"[NC] last{k:<3} max      needle_rank={rank_of(sub.amax(dim=-2).mean(0).numpy(), needle)}",
              flush=True)
        print(f"[NC] last{k:<3} sum      needle_rank={rank_of(sub.sum(dim=-2).mean(0).numpy(), needle)}",
              flush=True)
    for name, imp in (
        ("nc_max(1st-pass)", p.amax(dim=-2).mean(0).numpy()),
        ("nc_sum(1st-pass)", p.sum(dim=-2).mean(0).numpy()),
        ("causal_max", a_causal.amax(dim=-2).mean(0).numpy()),
        ("causal_sum", a_causal.sum(dim=-2).mean(0).numpy()),
    ):
        print(f"[NC] {name:<17} needle_rank={rank_of(imp, needle)}/{L} "
              f"raw={imp[needle]:.5f}", flush=True)

    pids = tok(text, return_tensors="pt")["input_ids"].cuda()
    off = pids.shape[1] - L
    with torch.no_grad():
        o2 = m(pids, past_key_values=cache, output_attentions=True)
    a2 = o2.attentions[-1][0].float().cpu()
    imp2 = a2[:, off:off + L, :L].amax(dim=-2).mean(0).numpy()
    print(f"[NC] {'2nd-pass(resend)':<17} needle_rank={rank_of(imp2, needle)}/{L} "
          f"raw={imp2[needle]:.5f}", flush=True)
    print(f"[NC] needle_pos={needle} L={L}", flush=True)


if __name__ == "__main__":
    main()
