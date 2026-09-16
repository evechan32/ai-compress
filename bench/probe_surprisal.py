"""惊讶度信号探针（含反例控制）。

问题：第一遍 logits 的 -log p(token) 能不能挑出"信息 token"？

**关键控制组**（否则会自欺）：
  1. `code_marked`   随机码 + "IMPORTANT:" 标记 → 惊讶度必然占优（作弊）
  2. `fact_marked`   自然语言事实 + 标记
  3. `fact_plain`    自然语言事实、**无标记** → 真正的考验
  4. `repeat_core`   内容取自填充句（**按构造低惊讶**）→ 惊讶度的**反例**

并且分别报告 **core 段（真正区分的内容）** 与 **整段 needle** 的 rank，
避免被 "IMPORTANT:" 这种共享标记刷分（早期版本正是被它刷出了假的 rank 0）。

同时报告成本：算全位置 logits 相对普通 prefill 的墙钟比。
"""
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("PE_MODEL", "/models/qwen2.5-1.5b-instruct")
S = int(os.environ.get("PE_S", "512"))
BLOCK = int(os.environ.get("PE_BLOCK", "16"))
FILLER = "The harbor master logged every vessel that passed the north pier. "

# (前置标记文本, 真正区分的内容, 后置)
NEEDLES = {
    "code_marked": (" IMPORTANT: the secret vault code is ", "K7XQ21", ". "),
    "fact_marked": (" IMPORTANT: the vault was sealed in the spring of ",
                    "seventeen ninety two", ". "),
    "fact_plain": (" The vault was sealed in the spring of ",
                   "seventeen ninety two", ". "),
    "repeat_core": (" The log mentions ", "north pier", " again. "),
}


def surprisal_from_logits(logits, ids):
    """logits: (1, seq, vocab), ids: (1, seq) -> (seq,) 的 -log p(token_t)；首位置置 0。

    必须 gather(ids[0,1:])（真实下一个 token），不能用位置下标当词表下标。
    """
    logp = torch.log_softmax(logits[0, :-1, :].float(), dim=-1)
    tgt = ids[0, 1:]
    sur = -logp.gather(-1, tgt.view(-1, 1)).squeeze(-1)
    return torch.cat([torch.zeros(1, device=logits.device), sur]).cpu().numpy()


def rank_of(v, lo, hi):
    m = v[lo:hi].max()
    return int((v > m).sum())


def sanity_or_die(m, tok, max_median=0.05):
    """环境自检：重复文本上的 median 惊讶度必须 ≈0，否则拒绝测量。"""
    ids = tok(FILLER * 20, return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        o = m(ids, use_cache=False)
    med = float(np.median(surprisal_from_logits(o.logits, ids)))
    print(f"[SANITY] repeated-filler median surprisal={med:.4f} (需 <{max_median})", flush=True)
    if not med < max_median:
        raise SystemExit(f"[SANITY] FAILED median={med} -> 拒绝在此状态下测量")
    del o
    torch.cuda.empty_cache()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True
    ).cuda().eval()
    sanity_or_die(m, tok)

    unit = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    n = S // unit
    prefix = FILLER * (n // 2)
    print(f"[SUR] model={MODEL} S={S} block={BLOCK} unit={unit} n={n}", flush=True)

    for name, (pre, core, post) in NEEDLES.items():
        text = prefix + pre + core + post + FILLER * (n - n // 2)
        ids = tok(text, return_tensors="pt")["input_ids"][:, :S].cuda()
        L = ids.shape[1]
        a = len(tok(prefix + pre, add_special_tokens=False)["input_ids"])
        b = len(tok(prefix + pre + core, add_special_tokens=False)["input_ids"])
        c = len(tok(prefix + pre + core + post, add_special_tokens=False)["input_ids"])
        core_lo, core_hi = min(a, L - 1), min(b, L)
        full_hi = min(c, L)

        with torch.no_grad():
            o = m(ids, output_attentions=True, use_cache=False)
        lm = o.attentions[-1][0].float().cpu()
        lm_sum = lm.sum(dim=-2).mean(0).numpy()
        lm_max = lm.amax(dim=-2).mean(0).numpy()
        sur = surprisal_from_logits(o.logits, ids)
        del o, lm
        torch.cuda.empty_cache()

        nb = L // BLOCK
        sur_blk = sur[:nb * BLOCK].reshape(nb, BLOCK).max(1)
        blo = core_lo // BLOCK

        print(
            f"[SUR] {name:<12} L={L} core=[{core_lo},{core_hi}) full_hi={full_hi} | "
            f"CORE rank: lm_sum={rank_of(lm_sum, core_lo, core_hi)} "
            f"lm_max={rank_of(lm_max, core_lo, core_hi)} "
            f"sur={rank_of(sur, core_lo, core_hi)} "
            f"sur_blk={int((sur_blk > sur_blk[blo:blo + 1]).sum())}/{nb} | "
            f"core_sur={sur[core_lo:core_hi].max():.2f} "
            f"core_med={float(np.median(sur[core_lo:core_hi])):.2f} "
            f"seq_med={float(np.median(sur)):.3f}",
            flush=True)

    ids = tok(FILLER * 20, return_tensors="pt")["input_ids"].cuda()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = m(ids, use_cache=False)
    torch.cuda.synchronize()
    print(f"[COST] 1.5B L={ids.shape[1]} full-logits prefill={time.time() - t0:.3f}s", flush=True)


if __name__ == "__main__":
    main()
