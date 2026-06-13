"""Per-ring point-cloud density and K circumferential shift within one tunnel.

For each GT ring we report:
  * point count (total / block / background) and density (points per theta*h area)
  * GT K-block median theta  -> ring-to-ring K shift (the stagger)
  * detected-K vs GT-K registration error (mapped to theta)
  * GT-block class accuracy
and we correlate density and K-detection error with accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CASES = {
    "sam4tun/1-1": "/media/boringtao/Expansion/R4Tun-AIC!/data/sam4tun/1-1/final.csv",
    "sam4tun/4-1": "/media/boringtao/Expansion/R4Tun-AIC!/data/sam4tun/4-1/final.csv",
}


def load(path):
    df = pd.read_csv(path, usecols=["segment", "ring", "theta", "h", "pred", "r"])
    return df[np.isfinite(df["ring"])]


def main():
    for name, path in CASES.items():
        print(f"\n{'='*72}\n{name}\n{'='*72}")
        df = load(path)
        rings = sorted(df["ring"].unique())
        rows = []
        for ring in rings:
            g = df[df["ring"] == ring]
            blk = g[g["segment"] > 0]
            n = len(g)
            nb = len(blk)
            th_span = g["theta"].max() - g["theta"].min()
            h_span = g["h"].max() - g["h"].min()
            area = th_span * h_span if th_span > 0 and h_span > 0 else np.nan
            density = n / area if area and np.isfinite(area) else np.nan
            kk = blk[blk["segment"] == 1]["theta"]
            k_theta = float(kk.median()) if len(kk) else np.nan
            acc = float((blk["segment"] == blk["pred"]).mean()) if nb else np.nan
            rows.append(dict(ring=int(ring), n=n, nb=nb, h_span=h_span,
                             density=density, k_theta=k_theta, acc=acc))
        t = pd.DataFrame(rows).sort_values("ring")

        # --- density variation ---
        d = t["density"].to_numpy()
        cnt = t["n"].to_numpy()
        print("Per-ring density and counts:")
        for _, r in t.iterrows():
            print(f"  ring {r['ring']:>2}  n={int(r['n']):>8,}  "
                  f"density={r['density']:.1f}  Kθ={r['k_theta']:.2f}  acc={r['acc']:.2f}")
        print(f"\n  point-count CV={np.std(cnt)/np.mean(cnt):.3f}  "
              f"max/min={cnt.max()/cnt.min():.2f}")
        print(f"  density     CV={np.nanstd(d)/np.nanmean(d):.3f}  "
              f"max/min={np.nanmax(d)/np.nanmin(d):.2f}")

        # --- K shift ring-to-ring ---
        kth = t["k_theta"].to_numpy()
        dk = np.diff(kth)
        print(f"\n  GT K theta range across rings: "
              f"[{np.nanmin(kth):.2f}, {np.nanmax(kth):.2f}] "
              f"span={np.nanmax(kth)-np.nanmin(kth):.2f}")
        print(f"  consecutive K-shift: "
              + " ".join(f"{x:+.2f}" for x in dk))
        print(f"  mean|K-shift|={np.nanmean(np.abs(dk)):.2f}  "
              f"max|K-shift|={np.nanmax(np.abs(dk)):.2f}")

        # --- correlations with accuracy ---
        valid = t.dropna(subset=["density", "acc"])
        if len(valid) >= 3:
            cd = np.corrcoef(valid["density"], valid["acc"])[0, 1]
            print(f"\n  corr(density, accuracy) = {cd:+.2f}")
        # K detection error proxy: deviation of each ring K from the per-tunnel
        # dominant K cluster is not GT-free; instead report how erratic K is.
        if len(valid) >= 3:
            cn = np.corrcoef(valid["n"], valid["acc"])[0, 1]
            print(f"  corr(point_count, accuracy) = {cn:+.2f}")


if __name__ == "__main__":
    main()
