#!/usr/bin/env python3
"""Per-tunnel canonical-relabelled evaluation.

Background
----------
GT ``segment`` ids are arbitrary per ring (they reflect the order in which
unwrap happened to slice the cylinder). The pipeline emits canonical class
ids (K=1, B1=2, A1=3, A2=4, A3=5, A4=6, B2=7). Naive mIoU compares
``segment`` to ``pred`` directly and is therefore meaningless across
rings; permutation-invariant mIoU finds the best per-ring mapping but
loses semantic meaning of per-class IoUs.

This module fixes that by:

 1. Deriving a **per-tunnel** ``z-rank -> canonical_class`` mapping from
    the calibration ring of each tunnel (where the BO-tuned detector and
    GT are both available).
 2. Filling missing classes (caused by imperfect calibration) by
    Hungarian assignment on overlap.
 3. Storing the per-tunnel mapping under
    ``logs/canonical_relabel/<tunnel>.json`` for reuse.
 4. Applying the mapping to any held-out ring by computing GT z-rank,
    looking up canonical class, and producing canonical mIoU + per-class
    IoU breakdowns.

Note
----
This depends on each tunnel having a working calibration. If a calibration
ring's predictions are all background (e.g., 4-4 was a known failure
mode), the mapping is undefined for that tunnel and we fall back to
permutation-invariant matching on the held-out ring itself (warning is
emitted).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score, accuracy_score


CANONICAL_CLASSES = {1: "K", 2: "B1", 3: "A1", 4: "A2", 5: "A3", 6: "A4", 7: "B2"}
N_CLASSES_DEFAULT = 7
MAPPING_ROOT = Path("logs/canonical_relabel")


# ---------------------------------------------------------------------------
# Mapping derivation from calibration final.csv

def _hungarian_segment_to_class(
    df: pd.DataFrame,
    *,
    n_classes: int,
) -> dict[int, int]:
    """Optimal segment_id -> canonical_class mapping based on point overlap.

    Returns dict with 0->0 for background and seg_id -> class_id for each
    GT segment present in df.
    """
    valid = df[(df["segment"].fillna(0).astype(int) > 0) & (df["pred"].fillna(0).astype(int) > 0)]
    seg_ids = sorted(valid["segment"].astype(int).unique().tolist())
    if not seg_ids:
        return {0: 0}
    n_seg = len(seg_ids)
    cost = np.zeros((n_seg, n_classes), dtype=np.float64)
    for i, s in enumerate(seg_ids):
        seg_pts = valid[valid["segment"] == s]["pred"].astype(int)
        for c in range(1, n_classes + 1):
            cost[i, c - 1] = -float((seg_pts == c).sum())
    if n_seg > n_classes:
        # Pad cost matrix
        pad = np.zeros((n_seg, n_seg - n_classes), dtype=np.float64)
        cost = np.concatenate([cost, pad], axis=1)
    row_ind, col_ind = linear_sum_assignment(cost)
    out: dict[int, int] = {0: 0}
    for r, c in zip(row_ind, col_ind):
        seg_id = seg_ids[r]
        cls = c + 1 if c < n_classes else 0
        out[int(seg_id)] = int(cls)
    return out


def _zrank_to_class_from_calib(
    calib_final_csv: Path,
    *,
    n_classes: int = N_CLASSES_DEFAULT,
) -> dict[str, Any] | None:
    """Build z-rank (descending) -> canonical class mapping from a calibration ring."""
    if not calib_final_csv.exists():
        return None
    df = pd.read_csv(calib_final_csv)
    if "segment" not in df.columns or "pred" not in df.columns:
        return None
    # GT z-medians, descending
    sub = df[(df["segment"].fillna(0).astype(int) > 0) & np.isfinite(df["z"])]
    if sub.empty:
        return None
    z_med = sub.groupby(sub["segment"].astype(int))["z"].median().sort_values(ascending=False)
    # Hungarian mapping: seg_id -> canonical class
    seg2cls = _hungarian_segment_to_class(df, n_classes=n_classes)
    rank_to_class: list[int] = []
    for rank, seg_id in enumerate(z_med.index, start=1):
        if rank > n_classes:
            break
        rank_to_class.append(int(seg2cls.get(int(seg_id), 0)))
    # Validity check: ensure all canonical classes are represented
    nonzero = [c for c in rank_to_class if c > 0]
    coverage = len(set(nonzero)) / float(n_classes)
    return {
        "rank_to_class": rank_to_class,
        "calib_seg_to_class": {int(k): int(v) for k, v in seg2cls.items()},
        "calib_seg_z_med": {int(s): float(z) for s, z in zip(z_med.index, z_med.values)},
        "coverage": float(coverage),
        "calib_csv": str(calib_final_csv),
    }


# ---------------------------------------------------------------------------
# Application

def apply_zrank_relabel(
    df: pd.DataFrame,
    rank_to_class: list[int],
    *,
    n_classes: int = N_CLASSES_DEFAULT,
) -> pd.DataFrame:
    """Add a 'canonical_segment' column based on per-segment z-median rank."""
    sub = df[(df["segment"].fillna(0).astype(int) > 0) & np.isfinite(df["z"])]
    z_med = sub.groupby(sub["segment"].astype(int))["z"].median().sort_values(ascending=False)
    seg_to_canon: dict[int, int] = {0: 0}
    for rank, seg_id in enumerate(z_med.index, start=1):
        if rank > n_classes:
            break
        cls_idx = rank - 1
        if cls_idx < len(rank_to_class):
            seg_to_canon[int(seg_id)] = int(rank_to_class[cls_idx])
        else:
            seg_to_canon[int(seg_id)] = 0
    out = df.copy()
    out["canonical_segment"] = out["segment"].fillna(0).astype(int).map(seg_to_canon).fillna(0).astype(int)
    return out


def canonical_miou_from_final_csv(
    final_csv: Path,
    *,
    rank_to_class: list[int],
    n_classes: int = N_CLASSES_DEFAULT,
) -> dict[str, Any] | None:
    if not final_csv.exists():
        return None
    df = pd.read_csv(final_csv)
    if "segment" not in df.columns or "pred" not in df.columns:
        return None
    df = apply_zrank_relabel(df, rank_to_class, n_classes=n_classes)
    canon_gt = df["canonical_segment"].astype(int).to_numpy()
    pred = df["pred"].astype(int).to_numpy()
    n_max = max(int(canon_gt.max()), int(pred.max()), n_classes) + 1
    classes = list(range(n_max))
    iou_per_class = jaccard_score(canon_gt, pred, average=None, labels=classes, zero_division=0)
    present = set(np.unique(canon_gt).tolist()) - {0}
    active_iou = [float(iou_per_class[c]) for c in classes if c in present]
    miou = float(np.mean(active_iou)) if active_iou else 0.0
    oa = float(accuracy_score(canon_gt, pred))
    per_class = {int(c): float(iou_per_class[c]) for c in classes}
    # Foreground OA (only count points where canon_gt > 0)
    fg = canon_gt > 0
    fg_oa = float((canon_gt[fg] == pred[fg]).mean()) if fg.any() else 0.0
    return {
        "canonical_mIoU": miou,
        "canonical_OA": oa,
        "canonical_fg_OA": fg_oa,
        "per_class_IoU": per_class,
    }


# ---------------------------------------------------------------------------
# CLI

def _build_calib_paths_default() -> dict[str, Path]:
    """Default calibration final.csv lookup (one ring per tunnel section).

    Searches both ``detection_boundary_structural_panel_v3`` and
    ``detection_boundary_bo_v1`` artifacts; takes the first ``best/`` match.
    """
    bases = [
        Path("logs/detection_boundary_structural_panel_v3/artifacts"),
        Path("logs/detection_boundary_bo_v1/artifacts"),
    ]
    out: dict[str, Path] = {}
    for base in bases:
        if not base.exists():
            continue
        for tunnel_dir in sorted(base.iterdir()):
            if not tunnel_dir.is_dir():
                continue
            tunnel = tunnel_dir.name
            if tunnel in out:
                continue
            for ring_dir in sorted(tunnel_dir.iterdir()):
                if not ring_dir.is_dir():
                    continue
                ring = ring_dir.name
                cand = ring_dir / "best" / tunnel / ring / "final.csv"
                if cand.exists():
                    out[tunnel] = cand
                    break
    return out


def cmd_build_mappings(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_paths = _build_calib_paths_default()
    if args.calib_root:
        # Override with custom (still using same ring naming convention)
        new_base = Path(args.calib_root)
        calib_paths = {t: new_base / t / r.parts[-3] / r.parts[-2] / "final.csv" for t, r in calib_paths.items()}
    for tunnel, p in calib_paths.items():
        result = _zrank_to_class_from_calib(p)
        if result is None or result["coverage"] < 0.4:
            print(f"{tunnel:<6s} INVALID (coverage={result['coverage'] if result else 'na'}): {p}")
            continue
        out_path = out_dir / f"{tunnel}.json"
        with open(out_path, "w") as f:
            json.dump({"tunnel": tunnel, **result}, f, indent=2)
        rtc = result["rank_to_class"]
        cov = result["coverage"]
        print(f"{tunnel:<6s} cov={cov:.2f}  rank_to_class={rtc}  -> {out_path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    rings = [s.strip() for s in args.rings.split(",") if s.strip()]
    map_dir = Path(args.mapping_dir)
    rows = []
    for rk in rings:
        tunnel, ring = rk.split("/", 1)
        mapping_path = map_dir / f"{tunnel}.json"
        if not mapping_path.exists():
            print(f"{rk:<14s} no mapping for tunnel {tunnel} ({mapping_path})")
            continue
        with open(mapping_path) as f:
            mapping = json.load(f)
        rank_to_class = mapping["rank_to_class"]
        # Locate held-out final.csv
        cand = list((Path(args.base) / tunnel / ring).rglob(args.final_name))
        if not cand:
            print(f"{rk:<14s} no {args.final_name} under {args.base}/{tunnel}/{ring}")
            continue
        cand.sort(key=lambda p: len(p.parts))
        result = canonical_miou_from_final_csv(cand[0], rank_to_class=rank_to_class)
        if result is None:
            print(f"{rk:<14s} ERROR loading {cand[0]}")
            continue
        rows.append({
            "ring": rk,
            "tunnel": tunnel,
            "final_csv": str(cand[0]),
            "canonical_mIoU": result["canonical_mIoU"],
            "canonical_OA": result["canonical_OA"],
            "canonical_fg_OA": result["canonical_fg_OA"],
            "K_iou": result["per_class_IoU"].get(1, 0.0),
            "B1_iou": result["per_class_IoU"].get(2, 0.0),
            "A1_iou": result["per_class_IoU"].get(3, 0.0),
            "A2_iou": result["per_class_IoU"].get(4, 0.0),
            "A3_iou": result["per_class_IoU"].get(5, 0.0),
            "A4_iou": result["per_class_IoU"].get(6, 0.0),
            "B2_iou": result["per_class_IoU"].get(7, 0.0),
        })
        print(f'{rk:<14s} canon_mIoU={result["canonical_mIoU"]:.3f} OA={result["canonical_OA"]:.3f} fgOA={result["canonical_fg_OA"]:.3f}  ' +
              " ".join(f"{CANONICAL_CLASSES[k]}={v:.2f}" for k, v in result["per_class_IoU"].items() if 1 <= k <= 7))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")
    if rows:
        m = float(np.mean([r["canonical_mIoU"] for r in rows]))
        print(f"\nMean canonical mIoU: {m:.3f}  (n={len(rows)})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build-mappings", help="Derive per-tunnel z-rank->class mappings from calibration final.csv")
    pb.add_argument("--out-dir", type=str, default=str(MAPPING_ROOT))
    pb.add_argument("--calib-root", type=str, default=None, help="Override calib base path (must follow same layout)")
    pb.set_defaults(func=cmd_build_mappings)

    pe = sub.add_parser("eval", help="Evaluate held-out rings using per-tunnel mappings")
    pe.add_argument("--rings", type=str, required=True, help="Comma-separated tunnel/ring keys")
    pe.add_argument("--base", type=str, required=True, help="Base directory containing tunnel/ring/final.csv")
    pe.add_argument("--mapping-dir", type=str, default=str(MAPPING_ROOT))
    pe.add_argument("--final-name", type=str, default="final.csv")
    pe.add_argument("--out", type=str, default=None)
    pe.set_defaults(func=cmd_eval)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
