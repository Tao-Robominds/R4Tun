"""Confirm cumulative drift from the fixed A/B offset mechanism.

If blocks are placed at K +/- n * (fixed AB sector), any mismatch between the
fixed sector size and the true block angular size accumulates with n. So even on
rings where K is correctly located, per-class recall should DECREASE with the
block's ordinal distance from K. We measure this on K-aligned rings only, to
isolate the sector-size effect from K-mislocation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CASES = {
    "msk/1-1": "/media/boringtao/Expansion/R4Tun-AIC!/data/ablation_gpt/memory+state+knowledge/1-1/final.csv",
    "sam4tun/1-1": "/media/boringtao/Expansion/R4Tun-AIC!/data/sam4tun/1-1/final.csv",
}
NAMES = {1: "K", 2: "B1", 3: "A1", 4: "A2", 5: "A3", 6: "B2"}
# ordinal circumferential distance from K (from the per-ring centroid geometry)
DIST = {1: 0, 2: 1, 6: 1, 3: 2, 5: 2, 4: 3}


def load(path):
    df = pd.read_csv(path, usecols=["segment", "ring", "theta", "pred"])
    df = df[(df["pred"] != 7) & (df["pred"] != 8) & np.isfinite(df["ring"])]
    return df


def k_offset(g):
    gk = g[g["segment"] == 1]["theta"]
    pk = g[g["pred"] == 1]["theta"]
    if len(gk) == 0 or len(pk) == 0:
        return np.nan
    return abs(float(pk.median()) - float(gk.median()))


def main():
    for name, path in CASES.items():
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        df = load(path)
        # pick K-aligned rings (small predicted-K registration error)
        aligned = []
        for ring, g in df.groupby("ring"):
            if k_offset(g) < 1.5:  # < ~half an AB sector
                aligned.append(ring)
        print(f"K-aligned rings (|pred_K - gt_K| < 1.5): "
              f"{[int(r) for r in aligned]}")
        sub = df[df["ring"].isin(aligned)]
        blk = sub[sub["segment"] > 0]

        print("\nPer-class recall on K-aligned rings (by distance from K):")
        rows = []
        for c in [1, 2, 6, 3, 5, 4]:
            cls = blk[blk["segment"] == c]
            if len(cls) == 0:
                continue
            rec = float((cls["pred"] == c).mean())
            rows.append((DIST[c], NAMES[c], rec, len(cls)))
        for dist, nm, rec, n in sorted(rows):
            bar = "#" * int(round(rec * 30))
            print(f"  dist {dist}  {nm:3s}  recall={rec:.3f}  n={n:>8,}  {bar}")

        # aggregate recall by distance
        print("\nMean recall by distance-from-K:")
        bydist = {}
        for dist, nm, rec, n in rows:
            bydist.setdefault(dist, []).append((rec, n))
        for dist in sorted(bydist):
            vals = bydist[dist]
            wmean = sum(r * n for r, n in vals) / sum(n for _, n in vals)
            print(f"  distance {dist}: weighted recall = {wmean:.3f}")


if __name__ == "__main__":
    main()
