"""Layer-level attention concentration analysis over a completed attn sweep.

Reads the per-run `attn.npz` (colsum: attention received per KV position,
per layer/head) produced by `bench/attn_sweep.py` and answers: *how
compressible is each layer?* — i.e. which layers can be given a small KV
budget and which must be kept (nearly) full.

Outputs, under <root>/analysis:
  * layer_concentration.json  — per (model, length, layer) coverage stats
  * layer_top10_vs_depth.png  — top-10% coverage by layer depth
  * layer_mid_vs_depth.png    — middle (long-range) mass by layer depth
  * layer_usage_heatmap.png   — layer x position usage (downsampled), L=largest
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def coverage(r: np.ndarray, sink: int, window: int) -> dict:
    L = r.shape[0]
    return {
        "sink": float(r[:sink].sum()),
        "window": float(r[max(0, L - window):].sum()),
        "middle": float(r[sink:max(0, L - window)].sum()),
        "top10": float(np.sort(r)[::-1][: max(1, int(0.10 * L))].sum()),
    }


def analyse(root: str, sink: int, window: int):
    records = []
    for stats_path in sorted(glob.glob(os.path.join(root, "*", "L*", "stats.json"))):
        with open(stats_path) as f:
            rec = json.load(f)
        npz_path = os.path.join(os.path.dirname(stats_path), "attn.npz")
        if not os.path.exists(npz_path):
            continue
        colsum = np.load(npz_path)["colsum"]  # [n_layers, H, Lk]
        n_layers, _, Lk = colsum.shape
        rows = colsum.sum(axis=1)  # [n_layers, Lk]
        rows = rows / np.maximum(rows.sum(axis=1, keepdims=True), 1e-9)

        per_layer = []
        for li in range(n_layers):
            c = coverage(rows[li], sink, window)
            c["layer"] = li
            per_layer.append(c)
        records.append(
            {
                "model": rec["model"],
                "length": rec["length"],
                "n_layers": n_layers,
                "per_layer": per_layer,
            }
        )
        # stash rows for the heatmap of the largest length per model
        records[-1]["_rows"] = rows
    return records


def plot(records, out: str, sink: int, window: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({r["model"] for r in records})
    lengths = sorted({r["length"] for r in records})

    for metric, fname, title in [
        ("top10", "layer_top10_vs_depth.png", "top-10% attention mass by layer depth"),
        ("middle", "layer_mid_vs_depth.png", "middle (long-range) mass by layer depth"),
    ]:
        ncol = len(lengths)
        fig, axes = plt.subplots(1, ncol, figsize=(4 * ncol, 3.4), sharey=True)
        for ax, L in zip(np.atleast_1d(axes), lengths):
            for m in models:
                r = next((x for x in records if x["model"] == m and x["length"] == L), None)
                if not r:
                    continue
                ys = [p[metric] for p in r["per_layer"]]
                ax.plot(range(len(ys)), ys, marker=".", ms=4, label=m)
            ax.set_title(f"L={L}")
            ax.set_xlabel("layer")
        np.atleast_1d(axes)[0].set_ylabel(metric)
        np.atleast_1d(axes)[-1].legend(fontsize=7)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=120)
        plt.close(fig)

    # layer x position usage heatmap for the largest length per model
    big = max(lengths)
    subset = [r for r in records if r["length"] == big]
    fig, axes = plt.subplots(len(subset), 1, figsize=(11, 2.4 * len(subset)))
    for ax, r in zip(np.atleast_1d(axes), subset):
        rows = r["_rows"]
        step = max(1, rows.shape[1] // 512)
        ax.imshow(rows[:, ::step], aspect="auto", cmap="magma",
                  extent=[0, rows.shape[1], r["n_layers"], 0])
        ax.set_title(f"{r['model']} L={big}: per-layer KV usage")
        ax.set_xlabel("KV position")
        ax.set_ylabel("layer")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "layer_usage_heatmap.png"), dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="sweep output root (contains <model>/L<n>/)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sink", type=int, default=64)
    ap.add_argument("--window", type=int, default=256)
    args = ap.parse_args()

    out = args.out or os.path.join(args.root, "analysis")
    os.makedirs(out, exist_ok=True)
    records = analyse(args.root, args.sink, args.window)
    if not records:
        raise SystemExit("no runs found")

    clean = [{k: v for k, v in r.items() if k != "_rows"} for r in records]
    with open(os.path.join(out, "layer_concentration.json"), "w") as f:
        json.dump(clean, f, indent=2)

    plot(records, out, args.sink, args.window)

    # quick console table: mean top10 + spread of top10 across layers (per model/L)
    print(f"{'model':<24}{'L':>6}{'top10_mean':>12}{'top10_min':>11}{'top10_max':>11}{'mid_mean':>10}")
    for r in records:
        t = np.array([p["top10"] for p in r["per_layer"]])
        m = np.array([p["middle"] for p in r["per_layer"]])
        print(f"{r['model']:<24}{r['length']:>6}{t.mean():>12.3f}{t.min():>11.3f}{t.max():>11.3f}{m.mean():>10.3f}")
    print("LAYER_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
