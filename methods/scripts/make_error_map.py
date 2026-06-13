"""Generate unfolded FP/FN/class-swap error maps (baseline vs m+s+k).

Renders, for a regular tunnel (1-1) and a complex tunnel (4-1), three panels:
ground-truth semantics, SAM4Tun baseline error categories, and GPT m+s+k error
categories, in the unfolded (theta, h) plane. Saved as a high-resolution PDF.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = "/media/boringtao/Expansion/R4Tun-AIC!/data"
OUT = "/media/boringtao/Expansion/R4Tun-AIC!/methods/papers/figs/error_map.pdf"

# (label, baseline final.csv, m+s+k final.csv, schema max_id)
CASES = [
    ("Regular (1-1)", f"{ROOT}/sam4tun/1-1/final.csv",
     f"{ROOT}/ablation_gpt/memory+state+knowledge/1-1/final.csv", 6),
    ("Complex (4-1)", f"{ROOT}/sam4tun/4-1/final.csv",
     f"{ROOT}/ablation_gpt/memory+state+knowledge/4-1/final.csv", 7),
]

# Error categories
CAT_CORRECT = 0
CAT_FN = 1       # GT block -> pred background (missed block / under-seg)
CAT_FP = 2       # GT background -> pred block (noise -> key block)
CAT_SWAP = 3     # GT block -> wrong block class
CAT_UNMAP = 4    # pred 7 (unmapped) or 8 (synthetic upsample), excluded from mIoU

ERR_COLORS = ["#cccccc", "#2166ac", "#b2182b", "#f4a300", "#efe8d6"]
ERR_LABELS = [
    "Correct",
    "FN: block\u2192background",
    "FP: background\u2192block",
    "Class swap",
    "Unmapped / synthetic (excl.)",
]
# Draw order (back to front): synthetic, correct, then the errors on top.
DRAW_ORDER = [CAT_UNMAP, CAT_CORRECT, CAT_FN, CAT_FP, CAT_SWAP]
CAP_PER_CAT = 90_000  # subsample cap per category for tractable vector output


def error_categories(seg, pred, max_id):
    cat = np.full(seg.shape, CAT_CORRECT, dtype=np.int8)
    unmapped = (pred == 7) | (pred == 8) | (pred > max_id)
    g_bg = seg == 0
    p_bg = pred == 0
    g_blk = (seg > 0) & (seg <= max_id)
    p_blk = (pred > 0) & (pred <= max_id) & (pred != 7) & (pred != 8)
    cat[g_blk & p_bg] = CAT_FN
    cat[g_bg & p_blk] = CAT_FP
    cat[g_blk & p_blk & (seg != pred)] = CAT_SWAP
    cat[unmapped] = CAT_UNMAP
    return cat


def subsample(n, k, rng):
    if n <= k:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def draw_error_panel(ax, theta, h, cat, rng):
    """Scatter error categories back-to-front so FP/FN/swap sit on top."""
    err_cmap = ListedColormap(ERR_COLORS)
    for c in DRAW_ORDER:
        sel = np.where(cat == c)[0]
        if sel.size == 0:
            continue
        sel = sel[subsample(sel.size, CAP_PER_CAT, rng)]
        size = 0.6 if c in (CAT_UNMAP, CAT_CORRECT) else 1.4
        ax.scatter(theta[sel], h[sel], c=[ERR_COLORS[c]] * sel.size,
                   s=size, marker=".", linewidths=0, rasterized=True)
    return err_cmap


def load(path):
    return pd.read_csv(path, usecols=["theta", "h", "segment", "pred"])


def main():
    rng = np.random.default_rng(0)
    gt_cmap = ListedColormap(
        ["#d9d9d9", "#e41a1c", "#377eb8", "#4daf4a",
         "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
    )

    fig, axes = plt.subplots(
        len(CASES), 3, figsize=(13, 4.2 * len(CASES)), constrained_layout=True
    )
    if len(CASES) == 1:
        axes = axes[None, :]

    for r, (label, base_p, msk_p, max_id) in enumerate(CASES):
        df_b = load(base_p)
        df_k = load(msk_p)

        seg_b = df_b["segment"].to_numpy()
        pred_b = df_b["pred"].to_numpy()
        cat_b = error_categories(seg_b, pred_b, max_id)

        seg_k = df_k["segment"].to_numpy()
        pred_k = df_k["pred"].to_numpy()
        cat_k = error_categories(seg_k, pred_k, max_id)

        # Ground truth panel uses baseline rows (identical GT geometry).
        idx_b = subsample(len(df_b), 250_000, rng)

        tb, hb = df_b["theta"].to_numpy(), df_b["h"].to_numpy()
        tk, hk = df_k["theta"].to_numpy(), df_k["h"].to_numpy()

        ax = axes[r, 0]
        gt_clip = np.clip(seg_b[idx_b], 0, 7)
        ax.scatter(tb[idx_b], hb[idx_b], c=gt_clip, cmap=gt_cmap, vmin=0, vmax=7,
                   s=0.6, marker=".", linewidths=0, rasterized=True)
        ax.set_title(f"{label}\nGround truth", fontsize=11)
        ax.set_ylabel("h (axial)", fontsize=9)

        for ax_i, cat, t, h, sub in [
            (axes[r, 1], cat_b, tb, hb, "SAM4Tun baseline"),
            (axes[r, 2], cat_k, tk, hk, "GPT m+s+k"),
        ]:
            draw_error_panel(ax_i, t, h, cat, rng)
            frac = np.bincount(cat, minlength=5) / cat.size * 100
            sub_t = (f"{sub}\nFN {frac[CAT_FN]:.0f}%  "
                     f"FP {frac[CAT_FP]:.0f}%  swap {frac[CAT_SWAP]:.0f}%  "
                     f"unmap {frac[CAT_UNMAP]:.0f}%")
            ax_i.set_title(sub_t, fontsize=10)

        for c in range(3):
            axes[r, c].set_xlabel(r"$\theta$ (circumferential)", fontsize=9)
            axes[r, c].tick_params(labelsize=7)

    handles = [Patch(facecolor=ERR_COLORS[i], label=ERR_LABELS[i]) for i in range(5)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
