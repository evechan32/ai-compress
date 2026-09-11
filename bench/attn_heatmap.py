import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/models/qwen2.5-1.5b-instruct"
OUT = "/hy-tmp/plots"
L_MAX = 1024


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda()
    model.eval()

    filler = "The harbor master logged every vessel that passed the north pier. " * 120
    text = filler + " IMPORTANT: the secret vault code is K7XQ21. " + filler
    enc = tok(text, return_tensors="pt", truncation=True, max_length=L_MAX)
    ids = enc["input_ids"].cuda()
    L = ids.shape[1]

    with torch.no_grad():
        out = model(ids, output_attentions=True)

    attn = [a[0].float().cpu().numpy() for a in out.attentions]
    n_layers = len(attn)
    last = np.stack([a[:, -1, :] for a in attn])
    mean_head = last.mean(axis=1)
    np.save(f"{OUT}/attn_last_query.npy", mean_head)

    print(f"seq_len={L} layers={n_layers} heads={attn[0].shape[0]}", flush=True)
    for name, row in [("last_layer", mean_head[-1]), ("mean_over_layers", mean_head.mean(0))]:
        total = float(row.sum())
        sink = float(row[:64].sum()) / total
        win = float(row[-256:].sum()) / total
        top10 = float(np.sort(row)[::-1][: int(0.10 * L)].sum()) / total
        mid = float(row[64 : L - 256].sum()) / total
        print(f"COVER {name}: sink64={sink:.3f} window256={win:.3f} "
              f"middle={mid:.3f} top10%={top10:.3f}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for li in (0, n_layers // 2, n_layers - 1):
        m = attn[li][:, -1, :]
        plt.figure(figsize=(12, 4))
        plt.imshow(m, aspect="auto", cmap="viridis")
        plt.colorbar(label="attn")
        plt.title(f"Layer {li}: last-query attention over {L} KV positions (heads x pos)")
        plt.xlabel("KV position")
        plt.ylabel("head")
        plt.tight_layout()
        plt.savefig(f"{OUT}/attn_heads_L{li}.png", dpi=120)
        plt.close()

    plt.figure(figsize=(12, 4))
    for li, c in [(0, "tab:blue"), (n_layers // 2, "tab:orange"), (n_layers - 1, "tab:green")]:
        plt.plot(mean_head[li], label=f"layer {li}", color=c, lw=1)
    plt.axvspan(0, 64, color="grey", alpha=0.15, label="sink 64")
    plt.axvspan(L - 256, L, color="red", alpha=0.12, label="window 256 (from end)")
    plt.legend()
    plt.xlabel("KV position")
    plt.ylabel("attn (mean over heads)")
    plt.title("Last-query attention distribution vs KV position")
    plt.tight_layout()
    plt.savefig(f"{OUT}/attn_positions.png", dpi=120)
    plt.close()

    plt.figure(figsize=(10, 4))
    c = np.cumsum(mean_head.mean(0))
    c = c / c[-1]
    plt.plot(c)
    plt.axvline(64, color="grey", ls="--", label="sink 64")
    plt.axvline(L - 256, color="red", ls="--", label="window start")
    plt.legend()
    plt.xlabel("KV position")
    plt.ylabel("cumulative attn")
    plt.title("Cumulative attention vs KV position (mean over layers/heads)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/attn_cumulative.png", dpi=120)
    plt.close()

    m = mean_head.mean(0)
    plt.figure(figsize=(6, 6))
    plt.imshow(np.tile(m, (L, 1)), aspect="auto", cmap="magma")
    plt.colorbar(label="attn (broadcast)")
    plt.title("Attention weight per KV position (mean layers/heads)")
    plt.xlabel("KV position")
    plt.ylabel("query position (broadcast)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/attn_heatmap_broadcast.png", dpi=120)
    plt.close()

    print("PLOTS_DONE", flush=True)


if __name__ == "__main__":
    main()
