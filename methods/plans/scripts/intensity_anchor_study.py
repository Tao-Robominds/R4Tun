#!/usr/bin/env python3
"""Empirical study: which intensity-based intrinsic feature correlates with mIoU?

We have ~900 candidates (30 held-out tunnel-4/5 rings × ~30 candidates each)
cached under ``logs/iterative_reflection_proof_v4/candidate_labelmaps/``. For
each candidate we compute several intensity-derived features by projecting
points-in-class via ``pixel_to_point.pkl`` and the candidate's labelmap.

Features evaluated:
  - I_high_minus_low  : mean intensity of predicted {K, A4, B2} minus mean
                        intensity of predicted {A1, A2, A3, B1}.
  - intensity_corr    : Pearson correlation of predicted-class intensity
                        vector vs calibration intensity vector.
  - intensity_spearman: Spearman rank correlation of predicted vs calibration.
  - class_separation  : std of per-class intensity means / mean of within-
                        class intensity stds (higher = better-separated).
  - per_class_tightness : mean of (1 / (1 + within-class std))
  - group_purity      : within the high-group {K,A4,B2}, fraction of points
                        whose intensity > median of the held-out ring; same
                        for low-group <= median; combined.
  - r_std_match       : compare predicted-class r_std vector to calibration.

We rank features by Spearman correlation with mIoU, pooled and per-tunnel.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
CAND_ROOT = REPO_ROOT / "logs" / "iterative_reflection_proof_v4" / "candidate_labelmaps"
HELDOUT_ROOT = REPO_ROOT / "logs" / "iterative_reflection_proof_v4" / "heldout_iterative_reflection"
PANEL_ROOT = REPO_ROOT / "logs" / "iterative_reflection_proof_v4" / "panel" / "r0"

CALIB_DIRS = {
    "4": REPO_ROOT / "logs" / "detection_boundary_structural_panel_v3" / "artifacts",
    "5": REPO_ROOT / "logs" / "detection_boundary_structural_panel_v3" / "artifacts",
}
CALIB_RINGS = {
    "4-3": "logs/detection_boundary_structural_panel_v3/artifacts/4-3/r179/best/4-3/r179",
    "4-4": "logs/detection_boundary_structural_panel_v3/artifacts/4-4/r215/best/4-4/r215",
    "4-5": "logs/detection_boundary_structural_panel_v3/artifacts/4-5/r249/best/4-5/r249",
    "4-6": "logs/detection_boundary_structural_panel_v3/artifacts/4-6/r283/best/4-6/r283",
    "5-1": "logs/detection_boundary_structural_panel_v3/artifacts/5-1/r116/best/5-1/r116",
    "5-6": "logs/detection_boundary_structural_panel_v3/artifacts/5-6/r285/best/5-6/r285",
    "5-7": "logs/detection_boundary_structural_panel_v3/artifacts/5-7/r321/best/5-7/r321",
}

# Held-out tunnels that aren't in CALIB_RINGS borrow the "nearest" calibration.
TUNNEL_FALLBACK = {
    "4-1": "4-3", "4-2": "4-3", "4-7": "4-6", "4-8": "4-6", "4-9": "4-6", "4-10": "4-6",
    "5-2": "5-1", "5-3": "5-1", "5-4": "5-1", "5-5": "5-6",
}

# block-name -> labelmap class id (taken from labelmap_meta.json convention)
BLOCK_TO_CLS = {"K": 1, "B1": 2, "A1": 3, "A2": 4, "A3": 5, "A4": 6, "B2": 7}
HIGH_GROUP_CLS = {1, 6, 7}    # K, A4, B2
LOW_GROUP_CLS = {2, 3, 4, 5}  # B1, A1, A2, A3
ALL_FG_CLS = list(range(1, 8))


def _calib_signature(ring_dir: Path, template_path: Path) -> dict[int, dict[str, float]]:
    """Build per-labelmap-class intensity stats from a calibration ring.

    The calibration ring's ``unwrapped.csv`` carries GT ``segment`` ints.
    The ``template.json`` carries the GT block ordering, which we use to
    map segment id (1..7 in template y-order) to block name and then to
    labelmap class id.
    """

    df = pd.read_csv(ring_dir / "unwrapped.csv")
    tpl = json.loads(template_path.read_text())
    seg_to_block = {i + 1: comp[1] for i, comp in enumerate(tpl["components"])}
    out: dict[int, dict[str, float]] = {}
    for seg in [1, 2, 3, 4, 5, 6, 7]:
        sub = df[df["segment"] == seg]
        if len(sub) < 30:
            continue
        block = seg_to_block.get(int(seg), "?")
        cid = BLOCK_TO_CLS.get(block)
        if cid is None:
            continue
        out[cid] = {
            "intensity_mean": float(sub["intensity"].mean()),
            "intensity_std": float(sub["intensity"].std()),
            "r_std": float(sub["r"].std()),
            "n_pts": len(sub),
            "block": block,
        }
    return out


def _per_class_intensity(
    labelmap: np.ndarray,
    df_pts: pd.DataFrame,
    ptp: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """Map each held-out point to its predicted class via labelmap and
    aggregate per-class intensity / r-std.
    """

    H, W = labelmap.shape
    pp = ptp[(ptp["pixel_y"] >= 0) & (ptp["pixel_y"] < H) & (ptp["pixel_x"] >= 0) & (ptp["pixel_x"] < W)].copy()
    if pp.empty:
        return {}
    pp["cls"] = labelmap[pp["pixel_y"].to_numpy(), pp["pixel_x"].to_numpy()]
    int_by_idx = df_pts["intensity"].to_dict()
    r_by_idx = df_pts["r"].to_dict()
    pp["intensity"] = pp["index"].map(int_by_idx)
    pp["r"] = pp["index"].map(r_by_idx)
    pp = pp.dropna(subset=["intensity"])
    out: dict[int, dict[str, float]] = {}
    for cls in sorted(pp["cls"].unique()):
        cid = int(cls)
        if cid == 0:
            continue
        sub = pp[pp["cls"] == cid]
        if len(sub) < 30:
            continue
        out[cid] = {
            "intensity_mean": float(sub["intensity"].mean()),
            "intensity_std": float(sub["intensity"].std()),
            "intensity_median": float(sub["intensity"].median()),
            "r_mean": float(sub["r"].mean()),
            "r_std": float(sub["r"].std()),
            "n_pts": int(len(sub)),
        }
    return out


def _features_for_candidate(
    pred_sig: dict[int, dict[str, float]],
    calib_sig: dict[int, dict[str, float]],
    ring_intensity_median: float,
    df_pts: pd.DataFrame,
    ptp: pd.DataFrame,
    labelmap: np.ndarray,
) -> dict[str, float]:
    feats: dict[str, float] = {}

    pred_int = []
    pred_int_std = []
    calib_int = []
    calib_int_present = []
    pred_int_present = []
    for cid in ALL_FG_CLS:
        if cid in pred_sig and cid in calib_sig:
            pred_int_present.append(pred_sig[cid]["intensity_mean"])
            calib_int_present.append(calib_sig[cid]["intensity_mean"])
        if cid in pred_sig:
            pred_int.append(pred_sig[cid]["intensity_mean"])
            pred_int_std.append(pred_sig[cid]["intensity_std"])

    # group separation
    high_means = [pred_sig[c]["intensity_mean"] for c in HIGH_GROUP_CLS if c in pred_sig]
    low_means = [pred_sig[c]["intensity_mean"] for c in LOW_GROUP_CLS if c in pred_sig]
    if high_means and low_means:
        feats["I_high_minus_low"] = float(np.mean(high_means) - np.mean(low_means))
    else:
        feats["I_high_minus_low"] = float("nan")

    # correlation with calibration
    if len(pred_int_present) >= 4:
        ca = np.asarray(calib_int_present, dtype=float)
        pa = np.asarray(pred_int_present, dtype=float)
        if ca.std() > 0 and pa.std() > 0:
            feats["intensity_corr"] = float(np.corrcoef(ca, pa)[0, 1])
            feats["intensity_spearman"] = float(
                np.corrcoef(pd.Series(ca).rank().to_numpy(), pd.Series(pa).rank().to_numpy())[0, 1]
            )
        else:
            feats["intensity_corr"] = float("nan")
            feats["intensity_spearman"] = float("nan")
    else:
        feats["intensity_corr"] = float("nan")
        feats["intensity_spearman"] = float("nan")

    # class separation: std of per-class means / mean of within-class stds
    if len(pred_int) >= 3 and any(s > 0 for s in pred_int_std):
        feats["class_separation"] = float(np.std(pred_int) / max(1e-9, np.mean(pred_int_std)))
    else:
        feats["class_separation"] = float("nan")

    # per-class tightness: mean of 1 / (1 + std)
    feats["per_class_tightness"] = (
        float(np.mean([1.0 / (1.0 + s) for s in pred_int_std])) if pred_int_std else float("nan")
    )

    # group purity: in predicted high-group regions, fraction of points
    # with intensity > ring median; in predicted low-group regions,
    # fraction <= ring median.
    H, W = labelmap.shape
    pp = ptp[(ptp["pixel_y"] >= 0) & (ptp["pixel_y"] < H) & (ptp["pixel_x"] >= 0) & (ptp["pixel_x"] < W)].copy()
    pp["cls"] = labelmap[pp["pixel_y"].to_numpy(), pp["pixel_x"].to_numpy()]
    int_by_idx = df_pts["intensity"].to_dict()
    pp["intensity"] = pp["index"].map(int_by_idx)
    pp = pp.dropna(subset=["intensity"])
    if len(pp) > 100 and np.isfinite(ring_intensity_median):
        high_mask = pp["cls"].isin(HIGH_GROUP_CLS)
        low_mask = pp["cls"].isin(LOW_GROUP_CLS)
        if high_mask.any() and low_mask.any():
            n_high_correct = int(((high_mask) & (pp["intensity"] > ring_intensity_median)).sum())
            n_high = int(high_mask.sum())
            n_low_correct = int(((low_mask) & (pp["intensity"] <= ring_intensity_median)).sum())
            n_low = int(low_mask.sum())
            purity_h = n_high_correct / max(1, n_high)
            purity_l = n_low_correct / max(1, n_low)
            feats["group_purity"] = float(0.5 * (purity_h + purity_l))
            feats["group_purity_high"] = float(purity_h)
            feats["group_purity_low"] = float(purity_l)
        else:
            feats["group_purity"] = float("nan")
            feats["group_purity_high"] = float("nan")
            feats["group_purity_low"] = float("nan")
    else:
        feats["group_purity"] = float("nan")
        feats["group_purity_high"] = float("nan")
        feats["group_purity_low"] = float("nan")

    # r_std rank match
    pred_rstd_present = []
    calib_rstd_present = []
    for cid in ALL_FG_CLS:
        if cid in pred_sig and cid in calib_sig:
            pred_rstd_present.append(pred_sig[cid]["r_std"])
            calib_rstd_present.append(calib_sig[cid]["r_std"])
    if len(pred_rstd_present) >= 4:
        ca = np.asarray(calib_rstd_present, dtype=float)
        pa = np.asarray(pred_rstd_present, dtype=float)
        if ca.std() > 0 and pa.std() > 0:
            feats["rstd_spearman"] = float(
                np.corrcoef(pd.Series(ca).rank().to_numpy(), pd.Series(pa).rank().to_numpy())[0, 1]
            )
        else:
            feats["rstd_spearman"] = float("nan")
    else:
        feats["rstd_spearman"] = float("nan")
    return feats


def _process_ring(
    ring_key: str,
    cand_subdir: Path,
) -> list[dict[str, Any]]:
    tunnel = ring_key.split("/")[0]
    calib_tunnel = tunnel if tunnel in CALIB_RINGS else TUNNEL_FALLBACK.get(tunnel)
    if calib_tunnel is None:
        return []
    calib_dir = REPO_ROOT / CALIB_RINGS[calib_tunnel]
    template_path = (
        REPO_ROOT
        / "logs"
        / "detection_boundary_structural_panel_v3"
        / "templates_gt_calibrated"
        / calib_tunnel
        / calib_dir.name
        / "template.json"
    )
    if not template_path.exists():
        return []
    calib_sig = _calib_signature(calib_dir, template_path)

    held_dir = HELDOUT_ROOT / ring_key.replace("/", "/") / "A2_iterative_intrinsic_reflection"
    if not held_dir.exists():
        return []
    df_held_full = pd.read_csv(held_dir / "context_unwrapped.csv")
    target_ring = int(df_held_full["ring"].mode()[0])
    df_held = df_held_full[df_held_full["ring"] == target_ring].reset_index(drop=True)
    ptp_path = held_dir / "pixel_to_point.pkl"
    if not ptp_path.exists():
        return []
    with open(ptp_path, "rb") as f:
        ptp = pd.DataFrame(pickle.load(f))
    ring_intensity_median = float(df_held["intensity"].median()) if len(df_held) else float("nan")

    rows: list[dict[str, Any]] = []
    for d in sorted(cand_subdir.iterdir()):
        if not (d / "labelmap.npy").exists() or not (d / "metrics.json").exists():
            continue
        try:
            metrics = json.loads((d / "metrics.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        miou = metrics.get("miou")
        if miou is None or not np.isfinite(miou):
            continue
        labelmap = np.load(d / "labelmap.npy")
        pred_sig = _per_class_intensity(labelmap, df_held, ptp)
        feats = _features_for_candidate(
            pred_sig=pred_sig,
            calib_sig=calib_sig,
            ring_intensity_median=ring_intensity_median,
            df_pts=df_held,
            ptp=ptp,
            labelmap=labelmap,
        )
        feats.update(
            {
                "ring_key": ring_key,
                "tunnel_id": tunnel,
                "candidate_dir": d.name,
                "candidate_kind": metrics.get("candidate_kind"),
                "miou": float(miou),
                "J_reflect": metrics.get("J_reflect"),
                "G_structural": metrics.get("G_structural"),
                "guardrail_pass": metrics.get("guardrail_pass"),
            }
        )
        rows.append(feats)
    return rows


def main() -> int:
    out_rows: list[dict[str, Any]] = []
    for tunnel_dir in sorted(CAND_ROOT.iterdir()):
        if not tunnel_dir.is_dir():
            continue
        for ring_dir in sorted(tunnel_dir.iterdir()):
            ring_key = f"{tunnel_dir.name}/{ring_dir.name}"
            try:
                rows = _process_ring(ring_key, ring_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"  skipping {ring_key}: {exc}")
                continue
            if rows:
                print(f"  {ring_key}: {len(rows)} candidates")
            out_rows.extend(rows)

    df = pd.DataFrame(out_rows)
    df.to_csv(PANEL_ROOT / "intensity_anchor_study.csv", index=False)
    print(f"\nTotal {len(df)} candidate rows written.")

    feat_cols = [
        "I_high_minus_low",
        "intensity_corr",
        "intensity_spearman",
        "class_separation",
        "per_class_tightness",
        "group_purity",
        "group_purity_high",
        "group_purity_low",
        "rstd_spearman",
        "G_structural",
        "J_reflect",
    ]
    print("\nPooled Spearman vs mIoU (across all candidates, all rings):")
    rows_pool = []
    for col in feat_cols:
        d = df[[col, "miou"]].dropna()
        if len(d) >= 5:
            sp = float(d[col].corr(d["miou"], method="spearman"))
            pe = float(d[col].corr(d["miou"], method="pearson"))
        else:
            sp = pe = float("nan")
        rows_pool.append({"feature": col, "spearman_pooled": sp, "pearson_pooled": pe, "n": len(d)})
        print(f"  {col:24s} n={len(d):4d} spearman={sp:+.3f} pearson={pe:+.3f}")
    pd.DataFrame(rows_pool).to_csv(PANEL_ROOT / "intensity_anchor_pooled_corr.csv", index=False)

    print("\nPer-ring mean Spearman vs mIoU (only rings with at least 8 candidates):")
    per_ring_rows = []
    for col in feat_cols:
        spm = []
        for rk, sub in df.groupby("ring_key"):
            d = sub[[col, "miou"]].dropna()
            if len(d) >= 8 and d[col].std() > 0 and d["miou"].std() > 0:
                spm.append(float(d[col].corr(d["miou"], method="spearman")))
        if spm:
            mean_sp = float(np.mean(spm))
            median_sp = float(np.median(spm))
            n_pos = int(np.sum([s > 0.2 for s in spm]))
            n_total = int(len(spm))
        else:
            mean_sp = median_sp = float("nan")
            n_pos = n_total = 0
        per_ring_rows.append(
            {
                "feature": col,
                "mean_per_ring_spearman": mean_sp,
                "median_per_ring_spearman": median_sp,
                "n_rings_pos_correlation": n_pos,
                "n_rings_total": n_total,
            }
        )
        print(f"  {col:24s} mean={mean_sp:+.3f} median={median_sp:+.3f} pos_corr_rings={n_pos}/{n_total}")
    pd.DataFrame(per_ring_rows).to_csv(PANEL_ROOT / "intensity_anchor_per_ring_corr.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
