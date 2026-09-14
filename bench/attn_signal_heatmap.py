"""同一段 context 上三种"打分信号"的注意力热点图。

1) LM attention   : 模型自己的 Q（prefill 时的注意力，因果）
2) KVzip 重建     : 用"请复述上文"探针带着 cache 跑一遍，读到的注意力
3) K 当 Q         : softmax(K K^T)，自相似

输出: 每-token 重要性曲线（三种对齐）+ 各自的热力图；标注 needle 位置。
"""
import os

import matplotlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL = "/models/qwen2.5-1.5b-instruct"
OUT = "/root/attn-signal"
S = 512
FILLER = "The harbor master logged every vessel that passed the north pier. "
NEEDLE = " IMPORTANT: the secret vault code is K7XQ21. "


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda()
    m.eval()

    unit = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    n = S // unit
    pre = FILLER * (n // 2) + NEEDLE
    text = pre + FILLER * (n - n // 2)
    ids = tok(text, return_tensors="pt")["input_ids"][:, :S].cuda()
    L = ids.shape[1]
    needle_pos = len(tok(FILLER * (n // 2) + NEEDLE[:-len("K7XQ21. ")],
                         add_special_tokens=False)["input_ids"])

    with torch.no_grad():
        out1 = m(ids, output_attentions=True, use_cache=True)
    A1 = out1.attentions[-1][0].float().cpu()          # [Hq, L, L]  LM (causal)
    cache = out1.past_key_values
    del out1

    probe = "\n\nRepeat the previous context exactly." + text
    pids = tok(probe, return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        out2 = m(pids, past_key_values=cache, output_attentions=True)
    A2 = out2.attentions[-1][0].float().cpu()          # [Hq, P, L+P]
    del out2

    keys = cache.layers[-1].keys[0][:, :L, :].float().cpu()   # [Hkv, L, D]
    vals = cache.layers[-1].values[0][:, :L, :].float().cpu()  # [Hkv, L, D]
    kq = torch.einsum("hid,hjd->hij", keys, keys)
    A3 = torch.softmax(kq / keys.shape[-1] ** 0.5, dim=-1)   # [Hkv, L, L]  K-as-Q

    vnorm = vals.norm(dim=-1).mean(0).numpy()                # [L]  ‖V‖
    imp_lm = A1.sum(dim=-2).mean(0).numpy()            # sum over queries
    imp_kv = A2[:, :, :L].amax(dim=-2).mean(0).numpy()  # max over probe queries
    imp_ka = A3.sum(dim=-2).mean(0).numpy()
    imp_kvf = imp_kv * vnorm                           # KVzip 最终分 = attn * ‖V‖

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra -= ra.mean()
        rb -= rb.mean()
        return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

    for n1, v1 in (("lm", imp_lm), ("kvzip", imp_kv), ("kasq", imp_ka),
                   ("vnorm", vnorm), ("kvzip*V", imp_kvf)):
        row = []
        for n2, v2 in (("lm", imp_lm), ("kvzip", imp_kv), ("kasq", imp_ka),
                       ("vnorm", vnorm), ("kvzip*V", imp_kvf)):
            row.append(f"{spearman(v1, v2):+.2f}")
        print(f"[SPEARMAN] {n1:>8}: " + " ".join(row), flush=True)

    signals = [("lm", imp_lm), ("kvzip", imp_kv), ("kasq", imp_ka),
               ("vnorm", vnorm), ("kvzip*V", imp_kvf)]
    np.savez(f"{OUT}/imp.npz", needle=needle_pos, L=L,
             **{k: v for k, v in signals})
    for name, v in signals:
        t = v.sum()
        print(f"COVER {name}: sink64={v[:64].sum()/t:.3f} "
              f"win128={v[-128:].sum()/t:.3f} "
              f"mid={v[64:L-128].sum()/t:.3f} "
              f"top10%={np.sort(v)[::-1][:int(0.1*L)].sum()/t:.3f} "
              f"needle_rank={int((v > v[needle_pos]).sum())}", flush=True)

    x = np.arange(L)
    plt.figure(figsize=(13, 4))
    for name, v, c in (("LM (model's own Q)", imp_lm, "tab:blue"),
                       ("KVzip (repeat probe)", imp_kv, "tab:red"),
                       ("K-as-Q (self-sim)", imp_ka, "tab:green"),
                       ("||V|| norm", vnorm, "tab:orange"),
                       ("KVzip*||V||", imp_kvf, "tab:purple")):
        plt.plot(x, v / v.sum(), label=name, color=c, lw=1.2)
    plt.axvline(needle_pos, color="k", ls="--", lw=1, label="needle")
    plt.xlabel("context position")
    plt.ylabel("normalized importance")
    plt.title(f"Per-token importance: three query-agnostic signals (layer 27, L={L})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/signals_curve.png", dpi=120)
    plt.close()

    for name, mat in (("lm", A1), ("kvzip", A2[:, :, :L]), ("kasq", A3)):
        mm = mat.mean(0).numpy()                        # mean over heads
        if name == "kvzip":
            mm = mm[-L:]                                # probe queries for the context
        plt.figure(figsize=(7, 6))
        plt.imshow(mm, aspect="auto", cmap="viridis")
        plt.colorbar(label="attn")
        plt.axvline(needle_pos, color="r", ls="--", lw=1)
        plt.title(f"{name}: attention map (queries x keys), layer 27")
        plt.xlabel("key position")
        plt.ylabel("query position")
        plt.tight_layout()
        plt.savefig(f"{OUT}/map_{name}.png", dpi=120)
        plt.close()
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
