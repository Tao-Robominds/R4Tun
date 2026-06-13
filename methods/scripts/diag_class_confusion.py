"""Diagnose the source of class confusion on regular tunnels.

Tests two hypotheses for why GT block points get the wrong block CLASS while
boundaries are broadly found:
  H1  per-ring rotational/phase offset (predicted K position shifted vs GT)
  H2  unmodeled staggering (alternate rings rotated by ~half a block, but the
      pipeline applies the same circumferential layout to every ring)

Evidence printed:
  * per-ring class accuracy (look for odd/even alternation -> stagger)
  * GT K-block centroid theta per ring (look for two-level alternation)
  * GT vs predicted circumferential class ordering / offset
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

CASES = {
    "sam4tun/1-1": "/media/boringtao/Expansion/R4Tun-AIC!/data/sam4tun/1-1/final.csv",
    "msk/1-1": "/media/boringtao/Expansion/R4Tun-AIC!/data/ablation_gpt/memory+state+knowledge/1-1/final.csv",
}
NAMES = {0: "BG", 1: "K", 2: "B1", 3: "A1", 4: "A2", 5: "A3", 6: "B2", 7: "B2b"}


def load(path):
    df = pd.read_csv(path, usecols=["segment", "ring", "theta", "pred"])
    df = df[(df["pred"] != 7) & (df["pred"] != 8)]  # mapped points only
    return df


def per_ring_accuracy(df):
    blk = df[df["segment"] > 0]
    rows = []
    for ring, g in blk.groupby("ring"):
        acc = float((g["segment"] == g["pred"]).mean())
        rows.append((int(ring), len(g), acc))
    return sorted(rows)


def k_theta_per_ring(df):
    k = df[df["segment"] == 1]
    out = {}
    for ring, g in k.groupby("ring"):
        out[int(ring)] = float(g["theta"].median())
    return out


def circ_order(df, col):
    """Median theta per class -> circumferential ordering of classes."""
    blk = df[df[col] > 0]
    med = blk.groupby(col)["theta"].median()
    return med.sort_values()


def main():
    for name, path in CASES.items():
        print(f"\n{'='*64}\n{name}\n{'='*64}")
        df = load(path)
        blk = df[df["segment"] > 0]
        overall = float((blk["segment"] == blk["pred"]).mean())
        print(f"GT-block class accuracy (mapped): {overall:.3f}  "
              f"n_block={len(blk):,}  rings={df['ring'].nunique()}")

        # --- H2: per-ring accuracy alternation ---
        pr = per_ring_accuracy(df)
        accs = np.array([a for _, _, a in pr])
        rings = np.array([r for r, _, _ in pr])
        even = accs[rings % 2 == 0]
        odd = accs[rings % 2 == 1]
        print("\nPer-ring GT-block accuracy (ring: acc):")
        print("  " + "  ".join(f"{r}:{a:.2f}" for r, _, a in pr))
        if len(even) and len(odd):
            print(f"  even-ring mean acc={even.mean():.3f}  "
                  f"odd-ring mean acc={odd.mean():.3f}  "
                  f"|diff|={abs(even.mean()-odd.mean()):.3f}")

        # --- stagger: GT K-block theta alternation across rings ---
        kth = k_theta_per_ring(df)
        ks = [kth[r] for r in sorted(kth)]
        print("\nGT K-block median theta per ring:")
        print("  " + "  ".join(f"{r}:{kth[r]:.3f}" for r in sorted(kth)))
        if len(ks) >= 3:
            diffs = np.diff(ks)
            print(f"  consecutive-ring K-theta deltas: "
                  + " ".join(f"{d:+.3f}" for d in diffs))

        # --- H1/ordering: GT vs pred circumferential class ordering ---
        print("\nGT circumferential ordering (class: median theta):")
        for c, t in circ_order(df, "segment").items():
            print(f"  {NAMES.get(int(c), c):4s} {t:.3f}")
        print("Predicted circumferential ordering (class: median theta):")
        for c, t in circ_order(df, "pred").items():
            print(f"  {NAMES.get(int(c), c):4s} {t:.3f}")

        # --- per-ring sector alignment: GT vs pred block centroid theta,
        #     re-centred on each ring's GT K so stagger is removed ---
        print("\nPer-ring block centroids relative to GT K (theta units):")
        for ring in sorted(df["ring"].unique()):
            g = df[(df["ring"] == ring) & (df["segment"] > 0)]
            if len(g) == 0:
                continue
            k_theta = g[g["segment"] == 1]["theta"].median()
            if not np.isfinite(k_theta):
                continue
            gt_c = (g.groupby("segment")["theta"].median() - k_theta)
            pr = g[g["pred"] > 0]
            pr_c = (pr.groupby("pred")["theta"].median() - k_theta)
            acc = float((g["segment"] == g["pred"]).mean())
            def fmt(series):
                return " ".join(
                    f"{NAMES.get(int(c), c)}:{v:+.2f}" for c, v in series.items()
                )
            print(f"  ring {int(ring):>2} acc={acc:.2f}")
            print(f"      GT  : {fmt(gt_c)}")
            print(f"      pred: {fmt(pr_c)}")


if __name__ == "__main__":
    main()
