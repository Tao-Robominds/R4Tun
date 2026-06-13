"""Separate K-position rotation from walk-direction (handedness) flips.

For each ring we order the block classes by their signed circumferential offset
from the GT key block. We then test whether the PREDICTED ordering is:
  * a rotation of the GT ordering            -> pure K-offset (rotation) error
  * a rotation of the REVERSED GT ordering   -> ordering / +- direction flip
  * neither                                  -> mixed / detection garbage
Point-weighted recovered accuracy under the best rotation (with vs without a
direction flip) quantifies how much each effect contributes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CASES = {
    "sam4tun/1-1": "/media/boringtao/Expansion/R4Tun-AIC!/data/sam4tun/1-1/final.csv",
    "msk/1-1": "/media/boringtao/Expansion/R4Tun-AIC!/data/ablation_gpt/memory+state+knowledge/1-1/final.csv",
}
NAMES = {0: "BG", 1: "K", 2: "B1", 3: "A1", 4: "A2", 5: "A3", 6: "B2", 7: "B2b"}


def load(path):
    df = pd.read_csv(path, usecols=["segment", "ring", "theta", "pred"])
    return df[(df["pred"] != 7) & (df["pred"] != 8)]


def signed_offset(theta, k, period):
    return ((theta - k + period / 2.0) % period) - period / 2.0


def order_by_offset(df_ring, col, k_theta, period):
    blk = df_ring[df_ring[col] > 0]
    med = blk.groupby(col)["theta"].median()
    off = {int(c): signed_offset(med[c], k_theta, period) for c in med.index}
    return [c for c, _ in sorted(off.items(), key=lambda kv: kv[1])], off


def is_rotation(a, b):
    """True if list b is a cyclic rotation of list a (same length, same set)."""
    if sorted(a) != sorted(b) or len(a) == 0:
        return False
    aa = a + a
    return any(aa[i:i + len(a)] == b for i in range(len(a)))


def main():
    for name, path in CASES.items():
        print(f"\n{'='*66}\n{name}\n{'='*66}")
        df = load(path)
        period = float(df["theta"].max() - df["theta"].min())
        print(f"theta period (arc span) ~ {period:.2f}")

        n_rot = n_flip = n_other = 0
        for ring in sorted(df["ring"].unique()):
            g = df[(df["ring"] == ring) & (df["segment"] > 0)]
            kk = g[g["segment"] == 1]["theta"]
            if len(g) == 0 or len(kk) == 0:
                continue
            k_theta = float(kk.median())
            gt_order, _ = order_by_offset(g, "segment", k_theta, period)
            pr_order, _ = order_by_offset(g, "pred", k_theta, period)
            common = [c for c in gt_order if c in pr_order]
            gt_c = [c for c in gt_order if c in common]
            pr_c = [c for c in pr_order if c in common]

            rot = is_rotation(gt_c, pr_c)
            flip = is_rotation(gt_c[::-1], pr_c)
            tag = "rotation" if rot else ("FLIP" if flip else "other")
            if rot:
                n_rot += 1
            elif flip:
                n_flip += 1
            else:
                n_other += 1

            acc = float((g["segment"] == g["pred"]).mean())
            gname = "".join(f"{NAMES[c]} " for c in gt_order)
            pname = "".join(f"{NAMES[c]} " for c in pr_order)
            print(f"  ring {int(ring):>2} acc={acc:.2f}  [{tag}]")
            print(f"      GT order  : {gname}")
            print(f"      pred order: {pname}")

        print(f"\n  summary: rotation-only={n_rot}  direction-FLIP={n_flip}  "
              f"other={n_other}")


if __name__ == "__main__":
    main()
