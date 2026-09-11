"""Attention + KV-cache distribution sweep across models and input lengths.

For each (model, input_length) pair:
  * run one HF eager forward over a needle-in-the-middle prompt;
  * capture per-(layer, head) attention statistics via forward hooks and
    immediately drop the full attention matrix (so long lengths do not OOM);
  * record attention-mass distribution over KV positions and the real KV cache
    memory layout (per layer / per KV head, from the returned cache).

Then writes per-run NPZ + PNG, plus cross-run aggregate plots (coverage vs
length, KV bytes vs length, usage distribution vs length).

Run on the GPU server, e.g.:

    /usr/local/bin/python3 bench/attn_sweep.py \
        --models /models/qwen2.5-1.5b-instruct \
        --lengths 512 1024 2048 4096 \
        --out /hy-tmp/attn_sweep

Offline analysis only: no server / generation is started.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FILLER = "The harbor master logged every vessel that passed the north pier. "
NEEDLE = " IMPORTANT: the secret vault code is K7XQ21. "

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #
def build_ids(tok, length: int, needle_frac: float = 0.5):
    """Build exactly `length` token ids with the needle placed at needle_frac."""
    filler_ids = tok(FILLER, add_special_tokens=False).input_ids
    needle_ids = tok(NEEDLE, add_special_tokens=False).input_ids
    if not filler_ids:
        raise ValueError("tokenizer produced empty filler")

    bos = tok.bos_token_id
    prefix = [bos] if bos is not None else []
    body = length - len(prefix)
    if body <= len(needle_ids) + 2:
        raise ValueError(f"length {length} too small for needle")

    n_head = int((body - len(needle_ids)) * needle_frac)
    head = (filler_ids * (n_head // len(filler_ids) + 1))[:n_head]
    tail_len = body - len(needle_ids) - len(head)
    tail = (filler_ids * (tail_len // len(filler_ids) + 1))[:tail_len]

    ids = prefix + head + needle_ids + tail
    ids = ids[:length]
    start = len(prefix) + len(head)
    needle_pos = list(range(start, min(start + len(needle_ids), len(ids))))
    return torch.tensor([ids], dtype=torch.long), needle_pos


# --------------------------------------------------------------------------- #
# attention capture
# --------------------------------------------------------------------------- #
_CURRENT: "Optional[AttnCapture]" = None


def _chunked_eager_attention_forward(module, query, key, value, attention_mask,
                                     scaling, dropout: float = 0.0, **kwargs):
    """Query-chunked replacement for HF's eager attention.

    Peak memory is O(H * chunk * Lk) instead of O(H * Lq * Lk), so long
    sequences no longer OOM.  Attention statistics are accumulated into the
    active `AttnCapture`; attn_weights is never returned, so HF cannot retain
    a per-layer matrix.
    """
    cap = _CURRENT
    groups = getattr(module, "num_key_value_groups", 1)
    if groups and groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    B, H, Lq, _ = query.shape
    Lk = key.shape[-2]
    Dv = value.shape[-1]
    out = torch.empty(B, H, Lq, Dv, device=query.device, dtype=query.dtype)

    li = cap.next_index() if cap is not None else None
    store_full = cap is not None and cap.store_full and li in cap.full_layers
    full = np.empty((H, Lq, Lk), dtype=np.float16) if store_full else None
    chunk = cap.chunk if cap is not None else 1024
    colsum = rowsum = last = None

    for start in range(0, Lq, chunk):
        end = min(start + chunk, Lq)
        scores = torch.matmul(query[:, :, start:end, :], key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            m = attention_mask[..., start:end, :Lk]
            if m.dtype == torch.bool:
                scores = scores.masked_fill(~m, torch.finfo(scores.dtype).min)
            else:
                scores = scores + m
        w = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
        out[:, :, start:end, :] = torch.matmul(w.to(value.dtype), value)
        if cap is not None:
            colsum = w.sum(dim=2) if colsum is None else colsum + w.sum(dim=2)
            rowsum = w.sum(dim=-1) if rowsum is None else rowsum + w.sum(dim=-1)
            if end == Lq:
                last = w[:, :, -1, :].detach().cpu().numpy()  # [B, H, Lk]
            if full is not None:
                full[:, start:end, :] = w[0].to(torch.float16).cpu().numpy()
        del scores, w

    if cap is not None:
        cap.record(li, last, colsum, rowsum, full)
    return out.transpose(1, 2).contiguous(), None


class AttnCapture:
    """Installs a query-chunked eager attention and collects KV statistics.

    Unlike a forward hook (which cannot stop HF from retaining the full
    per-layer attention matrix on transformers 5.x), this replaces the eager
    attention function in the registry, so the [H, Lq, Lk] tensor is never
    materialized.
    """

    def __init__(self, n_layers: int, full_layers: Optional[List[int]] = None,
                 chunk: int = 1024, store_full: bool = False):
        self.n_layers = n_layers
        self.full_layers = set(full_layers or [])
        self.chunk = chunk
        self.store_full = store_full
        self._idx = 0
        self.last: Dict[int, np.ndarray] = {}
        self.colsum: Dict[int, np.ndarray] = {}
        self.rowsum: Dict[int, np.ndarray] = {}
        self.full: Dict[int, np.ndarray] = {}
        self._prev = None

    def next_index(self) -> int:
        i = self._idx
        self._idx += 1
        return i

    def record(self, li, last, colsum, rowsum, full) -> None:
        if last is not None:
            self.last[li] = last[0]  # [H, Lk]
        if colsum is not None:
            self.colsum[li] = colsum[0].cpu().numpy()
            self.rowsum[li] = rowsum[0].cpu().numpy()
        if full is not None:
            self.full[li] = full

    def attach(self, model) -> None:
        global _CURRENT
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as R

        self._prev = R.get("eager", None)
        try:
            R.register("eager", _chunked_eager_attention_forward)
        except Exception:
            R._global_mapping["eager"] = _chunked_eager_attention_forward
        for attr in ("_local_mapping", "_global_mapping"):
            m = getattr(R, attr, None)
            if isinstance(m, dict):
                m["eager"] = _chunked_eager_attention_forward
        _CURRENT = self

    def detach(self) -> None:
        global _CURRENT
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as R

        try:
            if self._prev is not None:
                R.register("eager", self._prev)
            else:
                R.pop("eager", None)
        except Exception:
            pass
        _CURRENT = None

    # -- shape helpers ----------------------------------------------------- #
    def stacked(self, key: str) -> np.ndarray:
        d = getattr(self, key)
        return np.stack([d[i] for i in range(self.n_layers)])  # [L, H, ...]



# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def coverage_row(row: np.ndarray, L: int, sink: int, window: int) -> Dict[str, float]:
    total = float(row.sum())
    if total <= 0:
        return {}
    return {
        "sink": float(row[:sink].sum()) / total,
        "window": float(row[max(0, L - window):].sum()) / total,
        "middle": float(row[sink:max(0, L - window)].sum()) / total,
        "top10pct": float(np.sort(row)[::-1][: max(1, int(0.10 * L))].sum()) / total,
    }


def summarize(cap: AttnCapture, L: int, needle_pos: List[int], sink: int, window: int):
    me = cap.stacked("last").mean(1)  # [L, L] mean over heads
    out: Dict[str, dict] = {}
    out["last_layer"] = coverage_row(me[-1], L, sink, window)
    out["mean_over_layers"] = coverage_row(me.mean(0), L, sink, window)
    if needle_pos:
        valid = [p for p in needle_pos if 0 <= p < me.shape[1]]
        for name, row in (("last_layer", me[-1]), ("mean_over_layers", me.mean(0))):
            tot = float(row.sum())
            out[name]["needle_mass"] = float(row[valid].sum()) / tot if tot > 0 else 0.0
            if valid:
                order = np.argsort(row)[::-1]
                rank_of = {int(pos): i + 1 for i, pos in enumerate(order)}
                out[name]["needle_rank"] = min(rank_of[p] for p in valid)
    return me, out


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_run(me: np.ndarray, cap: AttnCapture, L: int, sink: int, window: int,
             outdir: str, tag: str, max_full: int) -> None:
    plt = _plt()
    n_layers = cap.n_layers

    fig, ax = plt.subplots(figsize=(12, 4))
    for li in (0, n_layers // 2, n_layers - 1):
        ax.plot(me[li], lw=1, label=f"layer {li}")
    ax.axvspan(0, sink, color="grey", alpha=0.15, label=f"sink {sink}")
    ax.axvspan(max(0, L - window), L, color="red", alpha=0.12, label=f"window {window}")
    ax.legend()
    ax.set_xlabel("KV position")
    ax.set_ylabel("attn (mean over heads)")
    ax.set_title(f"{tag} L={L}: last-query attention vs KV position")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "attn_positions.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    c = np.cumsum(me.mean(0))
    ax.plot(c / c[-1])
    ax.axvline(sink, color="grey", ls="--", label=f"sink {sink}")
    ax.axvline(max(0, L - window), color="red", ls="--", label="window start")
    ax.legend()
    ax.set_xlabel("KV position")
    ax.set_ylabel("cumulative attn")
    ax.set_title(f"{tag} L={L}: cumulative attention coverage")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "attn_cumulative.png"), dpi=120)
    plt.close(fig)

    if L <= max_full:
        for li in (0, n_layers // 2, n_layers - 1):
            m = cap.last[li]
            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.imshow(m, aspect="auto", cmap="viridis")
            fig.colorbar(im, ax=ax, label="attn")
            ax.set_title(f"{tag} L={L} layer {li}: head x KV position")
            ax.set_xlabel("KV position")
            ax.set_ylabel("head")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"attn_heads_L{li}.png"), dpi=120)
            plt.close(fig)


def plot_aggregate(summaries: List[dict], out: str) -> None:
    plt = _plt()
    if not summaries:
        return
    models = sorted({s["model"] for s in summaries})

    # coverage vs length
    metrics = ["sink", "window", "middle", "top10pct", "needle_mass"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        for m in models:
            pts = sorted(
                [(s["length"], s["coverage"].get("mean_over_layers", {}).get(metric))
                 for s in summaries if s["model"] == m],
                key=lambda p: p[0],
            )
            xs = [p[0] for p in pts if p[1] is not None]
            ys = [p[1] for p in pts if p[1] is not None]
            if xs:
                ax.plot(xs, ys, marker="o", label=m)
        ax.set_title(metric)
        ax.set_xlabel("input length")
        ax.set_xscale("log", base=2)
    np.atleast_1d(axes)[0].set_ylabel("attention fraction")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.suptitle("Attention coverage vs input length (last query, mean layers)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "aggregate_coverage.png"), dpi=120)
    plt.close(fig)

    # KV bytes vs length
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        pts = sorted([(s["length"], s["kv"]["total_bytes"]) for s in summaries if s["model"] == m])
        xs = [p[0] for p in pts]
        ys = [p[1] / 1024**2 for p in pts]
        if xs:
            ax.plot(xs, ys, marker="o", label=m)
    ax.set_xlabel("input length")
    ax.set_ylabel("KV cache (MiB)")
    ax.set_title("KV cache size vs input length")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "kv_bytes_vs_length.png"), dpi=120)
    plt.close(fig)

    # per-layer KV bytes (one representative run per model)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        s = next(x for x in summaries if x["model"] == m)
        lay = np.array(s["kv"]["per_layer_bytes"]) / 1024**2
        ax.plot(range(len(lay)), lay, marker=".", label=f"{m} (L={s['length']})")
    ax.set_xlabel("layer")
    ax.set_ylabel("KV per layer (MiB)")
    ax.set_title("KV cache distribution across layers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "kv_per_layer.png"), dpi=120)
    plt.close(fig)

    # usage distribution per model (mean over layers/heads), normalized
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in models:
        for s in sorted([x for x in summaries if x["model"] == m], key=lambda x: x["length"]):
            usage = np.load(s["usage_npy"])
            ax.plot(usage, lw=0.8, label=f"{m} L={s['length']}")
    ax.set_xlabel("KV position")
    ax.set_ylabel("normalized attention received")
    ax.set_title("KV usage distribution (attention mass per position)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "kv_usage_distribution.png"), dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def kv_stats(model, cache, L: int) -> dict:
    """Real cache memory layout from the returned KV cache."""
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    elem = next(model.parameters()).element_size()

    keys = getattr(cache, "key_cache", None) if cache is not None else None
    if keys:
        per_layer = [int(k.numel() * k.element_size() * 2) for k in keys]  # K + V
        measured = True
    else:
        per_layer = [int(L * n_kv * head_dim * elem * 2)] * n_layers
        measured = False
    total = int(sum(per_layer))
    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "elem_size": elem,
        "per_layer_bytes": per_layer,
        "total_bytes": total,
        "per_kv_head_bytes": (total // (n_layers * n_kv)) if n_layers and n_kv else 0,
        "gqa_group": (n_heads // n_kv) if n_kv else 0,
        "measured": measured,
    }


def run_one(model_list, lengths, out_root: str, sink: int, window: int,
            needle_frac: float, dtype: torch.dtype, max_full: int,
            device: str) -> List[dict]:
    summaries: List[dict] = []
    for mpath in model_list:
        name = os.path.basename(mpath.rstrip("/"))
        print(f"[model] {name} ({mpath})", flush=True)
        tok = AutoTokenizer.from_pretrained(mpath)
        model = AutoModelForCausalLM.from_pretrained(
            mpath, dtype=dtype, attn_implementation="eager"
        ).to(device)
        model.eval()
        n_layers = model.config.num_hidden_layers
        full_layers = [0, n_layers // 2, n_layers - 1] if max_full else []
        for L in lengths:
            try:
                ids, needle_pos = build_ids(tok, L, needle_frac)
                ids = ids.to(device)
                cap = AttnCapture(n_layers, full_layers=full_layers,
                                  store_full=(ids.shape[1] <= max_full))
                cap.attach(model)
                try:
                    with torch.no_grad():
                        out = model(ids, use_cache=True, output_attentions=True)
                    cache = out.past_key_values
                finally:
                    cap.detach()

                me, cov = summarize(cap, ids.shape[1], needle_pos, sink, window)
                colsum = cap.stacked("colsum")  # [L, H, Lk]
                usage = colsum.sum(axis=(0, 1))  # attention received per KV position
                usage = usage / max(usage.sum(), 1e-9)

                tag = f"{name}_L{ids.shape[1]}"
                outdir = os.path.join(out_root, name, f"L{ids.shape[1]}")
                os.makedirs(outdir, exist_ok=True)
                usage_npy = os.path.join(outdir, "usage.npy")
                np.save(usage_npy, usage)
                np.savez_compressed(
                    os.path.join(outdir, "attn.npz"),
                    last=cap.stacked("last"),
                    colsum=colsum,
                    **{f"full_L{k}": v for k, v in cap.full.items()},
                )
                plot_run(me, cap, ids.shape[1], sink, window, outdir, tag, max_full)

                kv = kv_stats(model, cache, ids.shape[1])
                rec = {
                    "model": name,
                    "path": mpath,
                    "length": int(ids.shape[1]),
                    "needle_pos": needle_pos,
                    "sink": sink,
                    "window": window,
                    "coverage": cov,
                    "kv": kv,
                    "usage_npy": usage_npy,
                    "outdir": outdir,
                }
                with open(os.path.join(outdir, "stats.json"), "w") as f:
                    json.dump(rec, f, indent=2)
                summaries.append(rec)
                print(
                    f"  L={L:>6} sink={cov['mean_over_layers']['sink']:.3f} "
                    f"win={cov['mean_over_layers']['window']:.3f} "
                    f"mid={cov['mean_over_layers']['middle']:.3f} "
                    f"top10={cov['mean_over_layers']['top10pct']:.3f} "
                    f"needle={cov['mean_over_layers'].get('needle_mass', float('nan')):.4f} "
                    f"kv={kv['total_bytes']/1024**2:.1f}MiB",
                    flush=True,
                )
                del cap, me, colsum, out, cache
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(
                    f"  L={L:>6} SKIPPED (CUDA OOM): eager attention matrix too "
                    f"large; lower --max-full or use a smaller head count",
                    flush=True,
                )
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        del model
        gc.collect()
        torch.cuda.empty_cache()
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--lengths", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    ap.add_argument("--out", default="/hy-tmp/attn_sweep")
    ap.add_argument("--sink", type=int, default=64)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--needle-frac", type=float, default=0.5)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-full", type=int, default=2048,
                    help="store full head x position matrices only for L <= this")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summaries = run_one(
        args.models, sorted(args.lengths), args.out, args.sink, args.window,
        args.needle_frac, DTYPES[args.dtype], args.max_full, args.device,
    )
    plot_aggregate(summaries, args.out)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
