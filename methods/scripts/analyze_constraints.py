"""Quantify the four structural constraints of the fixed SAM4Tun labelling rule.

Consolidates the per-constraint diagnostics (density/K-shift, class confusion,
order-vs-offset, A/B drift, handedness) into a single, tunnel-level analysis that
compares the SAM4Tun baseline against the Opus-4.6 m+s+k adapted outputs.

Constraints
  C1  Non-uniform point density   -> per-ring count spread + corr(density, acc)
  C2  Moving K-anchor             -> K-theta shift, K-mislocation rate, acc drop
  C3  Fixed segment-offset template -> recall vs ordinal distance from K (K-aligned)
  C4  Hard-coded walk direction    -> rotation / direction-flip / other ring counts

Usage
  python analyze_constraints.py --gate          # 1-1 and 4-1 only
  python analyze_constraints.py --all            # all 30 tunnels -> csv + md
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

BASE_ROOT = "/media/boringtao/Expansion/R4Tun-AIC!/data/sam4tun"
MSK_ROOT = "data/ablation_anthropic/memory+state+knowledge"
OUT_DIR = "methods/reviews/v2/analysis"

REGULAR = ["1-1", "1-2", "1-3", "1-4", "1-5",
           "2-1", "2-2", "2-3", "2-4", "2-5",
           "3-1-1", "3-1-2", "3-1-3"]
COMPLEX = ["4-1", "4-2", "4-3", "4-4", "4-5", "4-6", "4-7", "4-8", "4-9", "4-10",
           "5-1", "5-2", "5-3", "5-4", "5-5", "5-6", "5-7"]


def base_path(tid: str) -> str:
    return f"{BASE_ROOT}/{tid}/final.csv"


def msk_path(tid: str) -> str:
    return f"{MSK_ROOT}/{tid}/final.csv"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["segment", "ring", "theta", "h", "pred", "r"])
    # Original GT-labelled points only (synthetic upsampled points carry no GT).
    df = df[np.isfinite(df["segment"]) & np.isfinite(df["ring"])]
    return df


def signed_offset(theta, k, period):
    return ((theta - k + period / 2.0) % period) - period / 2.0


def block_acc(df: pd.DataFrame) -> float:
    blk = df[df["segment"] > 0]
    return float((blk["segment"] == blk["pred"]).mean()) if len(blk) else np.nan


def error_composition(df: pd.DataFrame, max_id: int) -> dict:
    """Decompose every GT-labelled point into correct / FN / FP / swap / unmapped.

    Fractions are over all ground-truth points (background + blocks):
      correct  : pred == segment
      FN       : GT block -> predicted background (missed block / under-seg)
      FP       : GT background -> predicted block
      swap     : GT block -> wrong block class
      unmapped : pred is synthetic/unassigned (8) or out of schema (> max_id)
    """
    seg = df["segment"].to_numpy()
    pred = df["pred"].to_numpy()
    g_bg = seg == 0
    g_blk = seg > 0
    unmapped = (pred == 8) | (pred > max_id)
    p_bg = (pred == 0)
    p_blk = (pred > 0) & (pred <= max_id) & (~unmapped)
    n = len(seg)
    correct = (pred == seg) & (~unmapped)
    fn = g_blk & p_bg
    fp = g_bg & p_blk
    swap = g_blk & p_blk & (seg != pred)
    return {
        "correct": float(correct.mean()),
        "fn": float(fn.mean()),
        "fp": float(fp.mean()),
        "swap": float(swap.mean()),
        "unmapped": float(unmapped.mean()),
    }


def class_distance_from_k(df: pd.DataFrame, period: float, sector: float) -> dict:
    """Circumferential distance of each class from K in sector units (0 = K).

    Binning by |offset|/sector groups the symmetric blocks on either side of K
    (e.g. B1/B2) into the same ordinal distance, matching the fixed
    detect-K-then-step-by-one-sector labelling template.
    """
    off = {}
    for ring, g in df.groupby("ring"):
        kk = g[g["segment"] == 1]["theta"]
        if not len(kk):
            continue
        k_theta = float(kk.median())
        blk = g[g["segment"] > 0]
        med = blk.groupby("segment")["theta"].median()
        for c, t in med.items():
            off.setdefault(int(c), []).append(abs(signed_offset(t, k_theta, period)))
    mean_abs = {c: float(np.median(v)) for c, v in off.items() if v}
    return {c: int(round(v / sector)) for c, v in mean_abs.items()}


def analyze(tid: str, category: str) -> dict:
    bp, mp = base_path(tid), msk_path(tid)
    if not os.path.exists(bp) or not os.path.exists(mp):
        return {"tunnel": tid, "category": category, "missing": True}

    base = load(bp)
    msk = load(mp)
    max_id = int(msk["segment"].max())
    comp_base = error_composition(base, max_id)
    comp_msk = error_composition(msk, max_id)
    period = float(msk["theta"].max() - msk["theta"].min())
    n_blocks = max_id  # K + (max_id-1) others around the ring
    sector = period / n_blocks
    rings = sorted(msk["ring"].unique())

    # ---- per-ring tables (msk) ----
    per_ring = []
    for ring in rings:
        g = msk[msk["ring"] == ring]
        blk = g[g["segment"] > 0]
        if not len(blk):
            continue
        th_span = g["theta"].max() - g["theta"].min()
        h_span = g["h"].max() - g["h"].min()
        area = th_span * h_span if th_span > 0 and h_span > 0 else np.nan
        n = len(g)
        density = n / area if area and np.isfinite(area) else np.nan
        kg = g[g["segment"] == 1]["theta"]
        kp = g[g["pred"] == 1]["theta"]
        k_gt = float(kg.median()) if len(kg) else np.nan
        k_pred = float(kp.median()) if len(kp) else np.nan
        if np.isfinite(k_gt) and np.isfinite(k_pred):
            k_off = abs(signed_offset(k_pred, k_gt, period))
        else:
            k_off = np.nan
        per_ring.append(dict(tunnel=tid, category=category, ring=int(ring),
                             n=n, density=density, k_gt=k_gt, k_off=k_off,
                             k_off_sectors=(k_off / sector) if np.isfinite(k_off) else np.nan,
                             acc=float((blk["segment"] == blk["pred"]).mean())))
    pr = pd.DataFrame(per_ring)

    # ---- C1 density ----
    cnt = pr["n"].to_numpy(dtype=float)
    dens = pr["density"].to_numpy(dtype=float)
    c1_count_ratio = float(np.nanmax(cnt) / np.nanmin(cnt))
    c1_density_cv = float(np.nanstd(dens) / np.nanmean(dens))
    valid = pr.dropna(subset=["density", "acc"])
    c1_corr = (float(np.corrcoef(valid["density"], valid["acc"])[0, 1])
               if len(valid) >= 3 else np.nan)

    # ---- C2 moving K-anchor ----
    kgt = pr["k_gt"].dropna().to_numpy()
    k_span = float(np.nanmax(kgt) - np.nanmin(kgt)) if len(kgt) else np.nan
    k_meanshift = float(np.nanmean(np.abs(np.diff(kgt)))) if len(kgt) >= 2 else np.nan
    aligned_mask = pr["k_off"] < 0.5 * sector
    c2_misloc_frac = float((~aligned_mask).mean())
    acc_aligned = float(pr.loc[aligned_mask, "acc"].mean()) if aligned_mask.any() else np.nan
    acc_misloc = float(pr.loc[~aligned_mask, "acc"].mean()) if (~aligned_mask).any() else np.nan

    # ---- C3 A/B fixed-offset drift (K-aligned rings only) ----
    aligned_rings = pr.loc[aligned_mask, "ring"].tolist()
    sub = msk[msk["ring"].isin(aligned_rings)]
    dist_map = class_distance_from_k(msk, period, sector)
    recall_by_dist = {}
    blk = sub[sub["segment"] > 0]
    for c in sorted(dist_map):
        if c == 0:
            continue
        cls = blk[blk["segment"] == c]
        if not len(cls):
            continue
        d = dist_map[c]
        rec = float((cls["pred"] == c).mean())
        recall_by_dist.setdefault(d, []).append((rec, len(cls)))
    c3 = {}
    for d in sorted(recall_by_dist):
        vals = recall_by_dist[d]
        c3[d] = sum(r * n for r, n in vals) / sum(n for _, n in vals)
    c3_near = c3.get(1, np.nan)               # nearest non-K ring (dist 1)
    c3_far = c3.get(max(c3), np.nan) if c3 else np.nan

    # ---- C4 handedness (rotation vs direction-flip) ----
    def order_of(df_ring, col, k_theta):
        b = df_ring[df_ring[col] > 0]
        med = b.groupby(col)["theta"].median()
        offs = {int(c): signed_offset(med[c], k_theta, period) for c in med.index}
        return [c for c, _ in sorted(offs.items(), key=lambda kv: kv[1])]

    def is_rotation(a, b):
        if sorted(a) != sorted(b) or not a:
            return False
        aa = a + a
        return any(aa[i:i + len(a)] == b for i in range(len(a)))

    n_rot = n_flip = n_other = 0
    flip_accs = []
    for ring in rings:
        g = msk[(msk["ring"] == ring) & (msk["segment"] > 0)]
        kk = g[g["segment"] == 1]["theta"]
        if not len(g) or not len(kk):
            continue
        k_theta = float(kk.median())
        gt_o = order_of(g, "segment", k_theta)
        pr_o = order_of(g, "pred", k_theta)
        common = [c for c in gt_o if c in pr_o]
        gt_c = [c for c in gt_o if c in common]
        pr_c = [c for c in pr_o if c in common]
        if is_rotation(gt_c, pr_c):
            n_rot += 1
        elif is_rotation(gt_c[::-1], pr_c):
            n_flip += 1
            flip_accs.append(float((g["segment"] == g["pred"]).mean()))
        else:
            n_other += 1
    n_eval = n_rot + n_flip + n_other

    return {
        "tunnel": tid, "category": category, "missing": False,
        "max_id": max_id, "n_rings": len(rings),
        "acc_base": block_acc(base), "acc_msk": block_acc(msk),
        # FP/FN/swap composition (fraction of GT points)
        "base_correct": comp_base["correct"], "base_fn": comp_base["fn"],
        "base_fp": comp_base["fp"], "base_swap": comp_base["swap"],
        "base_unmapped": comp_base["unmapped"],
        "msk_correct": comp_msk["correct"], "msk_fn": comp_msk["fn"],
        "msk_fp": comp_msk["fp"], "msk_swap": comp_msk["swap"],
        "msk_unmapped": comp_msk["unmapped"],
        # C1
        "c1_count_ratio": c1_count_ratio, "c1_density_cv": c1_density_cv,
        "c1_corr_dens_acc": c1_corr,
        # C2
        "c2_k_span": k_span, "c2_k_meanshift": k_meanshift,
        "c2_misloc_frac": c2_misloc_frac,
        "c2_acc_aligned": acc_aligned, "c2_acc_misloc": acc_misloc,
        # C3
        "c3_recall_near": c3_near, "c3_recall_far": c3_far,
        "c3_recall_by_dist": c3,
        # C4
        "c4_n_rings": n_eval, "c4_rot": n_rot, "c4_flip": n_flip,
        "c4_other": n_other,
        "c4_flip_acc": float(np.mean(flip_accs)) if flip_accs else np.nan,
        "per_ring": per_ring,
    }


def print_row(r):
    if r.get("missing"):
        print(f"  [MISSING] {r['tunnel']}")
        return
    print(f"\n{'='*64}\n{r['tunnel']}  ({r['category']}, {r['max_id']}-class, "
          f"{r['n_rings']} rings)\n{'='*64}")
    print(f"  block acc: baseline={r['acc_base']:.3f} -> m+s+k={r['acc_msk']:.3f}")
    print(f"  FP/FN composition (GT points): "
          f"baseline correct={r['base_correct']*100:.0f}% FN={r['base_fn']*100:.0f}% "
          f"FP={r['base_fp']*100:.0f}% swap={r['base_swap']*100:.0f}% "
          f"unmap={r['base_unmapped']*100:.0f}%")
    print(f"     m+s+k correct={r['msk_correct']*100:.0f}% FN={r['msk_fn']*100:.0f}% "
          f"FP={r['msk_fp']*100:.0f}% swap={r['msk_swap']*100:.0f}% "
          f"unmap={r['msk_unmapped']*100:.0f}%")
    print(f"  C1 density: count max/min={r['c1_count_ratio']:.2f}  "
          f"CV={r['c1_density_cv']:.3f}  corr(density,acc)={r['c1_corr_dens_acc']:+.2f}")
    print(f"  C2 K-anchor: GT K-theta span={r['c2_k_span']:.2f}  "
          f"mean|shift|={r['c2_k_meanshift']:.2f}  mislocated={r['c2_misloc_frac']*100:.0f}%")
    print(f"     acc aligned={r['c2_acc_aligned']:.3f}  misloc={r['c2_acc_misloc']:.3f}")
    print(f"  C3 A/B drift (K-aligned): recall by dist={ {k: round(v,3) for k,v in r['c3_recall_by_dist'].items()} }")
    print(f"     near(d1)={r['c3_recall_near']:.3f}  far={r['c3_recall_far']:.3f}")
    print(f"  C4 handedness: rings={r['c4_n_rings']}  rotation={r['c4_rot']}  "
          f"FLIP={r['c4_flip']}  other={r['c4_other']}  flip-acc={r['c4_flip_acc']}")


def aggregate(rows, cat):
    sel = [r for r in rows if not r.get("missing") and r["category"] == cat]
    if not sel:
        return None
    def m(key):
        vals = [r[key] for r in sel if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else np.nan
    # recall by distance pooled (mean of per-tunnel weighted recalls)
    dists = {}
    for r in sel:
        for d, v in r["c3_recall_by_dist"].items():
            dists.setdefault(d, []).append(v)
    c3 = {d: float(np.mean(v)) for d, v in dists.items()}
    return {
        "category": cat, "n": len(sel),
        "acc_base": m("acc_base"), "acc_msk": m("acc_msk"),
        "base_correct": m("base_correct"), "base_fn": m("base_fn"),
        "base_fp": m("base_fp"), "base_swap": m("base_swap"),
        "base_unmapped": m("base_unmapped"),
        "msk_correct": m("msk_correct"), "msk_fn": m("msk_fn"),
        "msk_fp": m("msk_fp"), "msk_swap": m("msk_swap"),
        "msk_unmapped": m("msk_unmapped"),
        "c1_count_ratio": m("c1_count_ratio"), "c1_density_cv": m("c1_density_cv"),
        "c1_corr_dens_acc": m("c1_corr_dens_acc"),
        "c2_k_span": m("c2_k_span"), "c2_k_meanshift": m("c2_k_meanshift"),
        "c2_misloc_frac": m("c2_misloc_frac"),
        "c2_acc_aligned": m("c2_acc_aligned"), "c2_acc_misloc": m("c2_acc_misloc"),
        "c3_recall_by_dist": c3,
        "c4_rot": sum(r["c4_rot"] for r in sel),
        "c4_flip": sum(r["c4_flip"] for r in sel),
        "c4_other": sum(r["c4_other"] for r in sel),
        "c4_flip_acc": m("c4_flip_acc"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.gate:
        cases = [("1-1", "regular"), ("4-1", "complex")]
    else:
        cases = [(t, "regular") for t in REGULAR] + [(t, "complex") for t in COMPLEX]

    rows = []
    for tid, cat in cases:
        r = analyze(tid, cat)
        rows.append(r)
        print_row(r)

    if args.all:
        import json
        os.makedirs(OUT_DIR, exist_ok=True)
        flat, ring_rows = [], []
        for r in rows:
            if r.get("missing"):
                continue
            ring_rows.extend(r["per_ring"])
            d = {k: v for k, v in r.items()
                 if k not in ("c3_recall_by_dist", "per_ring")}
            for dist, val in r["c3_recall_by_dist"].items():
                d[f"c3_recall_d{dist}"] = val
            flat.append(d)
        df = pd.DataFrame(flat)
        csv_p = f"{OUT_DIR}/constraint_contributions.csv"
        df.to_csv(csv_p, index=False)
        pd.DataFrame(ring_rows).to_csv(f"{OUT_DIR}/per_ring.csv", index=False)
        print(f"\nwrote {csv_p}  ({len(df)} tunnels)")
        print(f"wrote {OUT_DIR}/per_ring.csv  ({len(ring_rows)} rings)")

        aggs = {}
        for cat in ["regular", "complex"]:
            agg = aggregate(rows, cat)
            if agg:
                aggs[cat] = agg
                print(f"\n### {cat.upper()} (n={agg['n']}) aggregate")
                for k, v in agg.items():
                    print(f"   {k}: {v}")
        with open(f"{OUT_DIR}/aggregate.json", "w") as f:
            json.dump(aggs, f, indent=2)
        print(f"wrote {OUT_DIR}/aggregate.json")
        return rows
    return rows


if __name__ == "__main__":
    main()
