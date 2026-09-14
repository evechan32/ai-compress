"""复述探针的成分消融：指令到底起多大作用？

对同一段 context，用同一份 KV cache，分别用不同的"探针输入"跑一次前向，
读它对 context 的注意力，算每-token 重要性 + needle rank：

  A. "Repeat the previous context exactly." + context   (KVzip 原版)
  B. "Continue the text below." + context               (中性指令)
  C. context 原样再喂一遍（无指令）

若 A 明显优于 B/C → 指令确实切换了模型的"任务"；
若三者接近 → 起作用的只是"第二次前向 / 非因果可见"，不是指令。
"""
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/models/qwen2.5-1.5b-instruct"
S = 512
FILLER = "The harbor master logged every vessel that passed the north pier. "
NEEDLE = " IMPORTANT: the secret vault code is K7XQ21. "


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
    needle_pos = len(tok(FILLER * (n // 2) + NEEDLE[:-len("K7XQ21. ")],
                         add_special_tokens=False)["input_ids"])

    with torch.no_grad():
        out1 = m(ids, output_attentions=True, use_cache=True)
    cache = out1.past_key_values
    lm = out1.attentions[-1][0].float().cpu()
    del out1
    lm_imp = lm.sum(dim=-2).mean(0).numpy()
    lm_max = lm.amax(dim=-2).mean(0).numpy()
    del lm

    probes = {
        "A_repeat": "\n\nRepeat the previous context exactly.",
        "B_continue": "\n\nContinue the text below.",
        "C_none": "",
    }
    rows = []
    for name, instr in probes.items():
        pids = tok(instr + text, return_tensors="pt")["input_ids"].cuda()
        off = pids.shape[1] - L
        pos = torch.arange(L, L + pids.shape[1]).unsqueeze(0).cuda()
        with torch.no_grad():
            o = m(pids, past_key_values=cache, output_attentions=True,
                  position_ids=pos)
        att = o.attentions[-1][0].float().cpu()
        del o
        imp = att[..., -L:, :L].amax(dim=-2).mean(0).numpy()
        own = att[:, off:off + L, :L].diagonal(dim1=-2, dim2=-1).mean(0).numpy()
        nd = slice(needle_pos, needle_pos + 6)
        filler_mask = np.ones(L, dtype=bool)
        filler_mask[nd] = False
        am = att[:, off:off + L, :L].argmax(dim=-1).float().mean(0).numpy()
        src = np.arange(L) * 1.0
        copy_nd = float((am[nd] == src[nd]).mean())
        copy_fl = float((am[filler_mask] == src[filler_mask]).mean())
        print(f"[RETR] {name:<11} self-attn needle={own[nd].mean():.4f} "
              f"filler={own[filler_mask].mean():.4f} | copy_rate needle={copy_nd:.3f} "
              f"filler={copy_fl:.3f}", flush=True)
        rank = int((imp > imp[needle_pos]).sum())
        share = imp[needle_pos] / imp.sum()
        top = np.argsort(-imp)[:10]
        above = np.argsort(-imp)[:rank + 1]
        print(f"[PROBE] {name:<11} needle_rank={rank}/{L} share={share*100:.3f}% "
              f"needle_raw={imp[needle_pos]:.5f} top_above={sorted(above[:-1].tolist())}",
              flush=True)
        rows.append((name, rank, share))

    for nm, v in (("lm_sum", lm_imp), ("lm_max", lm_max)):
        rank = int((v > v[needle_pos]).sum())
        print(f"[PROBE] {nm:<11} needle_rank={rank}/{L} share={v[needle_pos]/v.sum()*100:.3f}%",
              flush=True)

    print(f"[PROBE] needle_pos={needle_pos} L={L}", flush=True)


if __name__ == "__main__":
    main()
