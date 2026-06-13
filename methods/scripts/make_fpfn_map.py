"""Simplest FP/FN error map, laid out as a ring x theta raster.

Each ring is one horizontal row, theta on the x-axis. Points are coloured only as
correct / FN (block->bg) / FP (bg->block) / class swap. Per-ring K positions are
overlaid (GT vs predicted) and a density sidebar is attached, so the four causes
of class confusion are directly readable:
  - density sidebar              -> one global preprocessing setting cannot fit all rings
  - GT vs predicted K marks      -> staggered, moving anchor; mismatch rotates a row
  - swap colour at theta far from K -> fixed-offset drift at the ring ends
  - a fully mirrored row         -> reversed handedness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = "/media/boringtao/Expansion/R4Tun-AIC!/data"
OUT = "/media/boringtao/Expansion/R4Tun-AIC!/methods/papers/figs/error_map.pdf"

CASES = [
    ("Regular (1-1)", f"{ROOT}/sam4tun/1-1/final.csv",
     f"{ROOT}/ablation_gpt/memory+state+knowledge/1-1/final.csv", 6),
    ("Complex (4-1)", f"{ROOT}/sam4tun/4-1/final.csv",
     f"{ROOT}/ablation_gpt/memory+state+knowledge/4-1/final.csv", 7),
]

C_CORRECT = "#d9d9d9"
C_FN = "#2166ac"   # block -> background
C_FP = "#d62728"   # background -> block
C_SWAP = "#f4a300"  # wrong block class
CAP_CORRECT = 60_000
CAP_ERR = 80_000


def load(path, max_id):
    df = pd.read_csv(path, usecols=["theta", "segment", "pred", "ring"])
    df = df[(df["pred"] != 7) & (df["pred"] != 8) & (df["pred"] <= max_id)
            & np.isfinite(df["ring"])].copy()
    df["ring"] = df["ring"].astype(int)
    seg, pred = df["segment"].to_numpy(), df["pred"].to_numpy()
    cat = np.zeros(len(df), dtype=np.int8)            # 0 correct
    cat[(seg > 0) & (pred == 0)] = 1                  # FN
    cat[(seg == 0) & (pred > 0)] = 2                  # FP
    cat[(seg > 0) & (pred > 0) & (seg != pred)] = 3   # swap
    df["cat"] = cat
    return df


def k_positions(df):
    gt = df[df["segment"] == 1].groupby("ring")["theta"].median()
    pr = df[df["pred"] == 1].groupby("ring")["theta"].median()
    return gt, pr


def draw_raster(ax, df, rng, title):
    for c, color, s in [(0, C_CORRECT, 1.5), (1, C_FN, 3), (2, C_FP, 3),
                        (3, C_SWAP, 3)]:
        sub = df[df["cat"] == c]
        cap = CAP_CORRECT if c == 0 else CAP_ERR
        if len(sub) > cap:
            sub = sub.sample(cap, random_state=0)
        # jitter ring rows slightly so bands have visible thickness
        y = sub["ring"].to_numpy() + rng.uniform(-0.35, 0.35, len(sub))
        ax.scatter(sub["theta"], y, s=s, c=color, marker=".", linewidths=0,
                   rasterized=True)
    gt, pr = k_positions(df)
    ax.plot(gt.values, gt.index, "+", color="black", ms=6, mew=1.2,
            label="GT K")
    ax.plot(pr.values, pr.index, "x", color="#7b3294", ms=5, mew=1.2,
            label="pred K")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"circumferential angle $\theta$", fontsize=9)
    ax.tick_params(labelsize=7)


def draw_density(ax, df):
    cnt = df.groupby("ring").size()
    ax.barh(cnt.index, cnt.values / 1000.0, height=0.8, color="#777")
    ax.set_xlabel("pts/ring\n(k)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.invert_xaxis()


def main():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(
        len(CASES), 3, figsize=(13, 4.6 * len(CASES)),
        gridspec_kw={"width_ratios": [0.16, 1, 1]}, constrained_layout=True)

    for r, (label, base_p, msk_p, max_id) in enumerate(CASES):
        db = load(base_p, max_id)
        dk = load(msk_p, max_id)
        draw_density(axes[r, 0], dk)
        axes[r, 0].set_ylabel(f"{label}\nring index", fontsize=9)

        for ax, df, sub in [(axes[r, 1], db, "SAM4Tun baseline"),
                            (axes[r, 2], dk, "GPT m+s+k")]:
            frac = np.bincount(df["cat"], minlength=4) / len(df) * 100
            t = (f"{sub}: FN {frac[1]:.0f}%  FP {frac[2]:.0f}%  "
                 f"swap {frac[3]:.0f}%")
            draw_raster(ax, df, rng, t)
        # share y limits across the row
        ylim = (min(db["ring"].min(), dk["ring"].min()) - 1,
                max(db["ring"].max(), dk["ring"].max()) + 1)
        for c in range(3):
            axes[r, c].set_ylim(ylim)

    handles = [
        Patch(facecolor=C_CORRECT, label="correct"),
        Patch(facecolor=C_FN, label="FN: block\u2192background"),
        Patch(facecolor=C_FP, label="FP: background\u2192block"),
        Patch(facecolor=C_SWAP, label="class swap"),
        Line2D([0], [0], marker="+", color="black", ls="", label="GT K"),
        Line2D([0], [0], marker="x", color="#7b3294", ls="", label="pred K"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
