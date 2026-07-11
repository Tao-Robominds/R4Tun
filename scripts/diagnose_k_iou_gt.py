#!/usr/bin/env python3
"""Ground-truth decomposition of K-block IoU for tunnel SAM runs."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

K_LABEL = 1
CLASS_NAMES = {
    0: "Background",
    1: "K",
    2: "B1",
    3: "A1",
    4: "A2",
    5: "A3",
    6: "B2",
}
# Hardcoded K trapezoid vertical extent in mm (sam.py generate_template_mask)
TEMPLATE_K_SPAN_MM = 2 * max(619.16, 460.77)
RESOLUTION = 0.005


def build_gt_k_map(pixel_to_point: list, segments: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    gt = np.zeros((h, w), dtype=np.uint8)
    for entry in pixel_to_point:
        idx = int(entry["index"])
        if segments[idx] != K_LABEL:
            continue
        py, px = int(entry["pixel_y"]), int(entry["pixel_x"])
        if 0 <= py < h and 0 <= px < w:
            gt[py, px] = 1
    return gt


def build_pred_k_map(pixel_to_point: list, pred_labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    pred = np.zeros((h, w), dtype=np.uint8)
    for entry in pixel_to_point:
        idx = int(entry["index"])
        if pred_labels[idx] != K_LABEL:
            continue
        py, px = int(entry["pixel_y"]), int(entry["pixel_x"])
        if 0 <= py < h and 0 <= px < w:
            pred[py, px] = 1
    return pred


def vertical_span_y(mask: np.ndarray) -> float:
    ys, _ = np.where(mask)
    if len(ys) == 0:
        return 0.0
    return float(ys.max() - ys.min())


def k_metrics(gt_k: np.ndarray, pred_k: np.ndarray) -> dict:
    tp = int(np.logical_and(gt_k, pred_k).sum())
    fp = int(np.logical_and(~gt_k, pred_k).sum())
    fn = int(np.logical_and(gt_k, ~pred_k).sum())
    union = tp + fp + fn
    iou = tp / union if union else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "union": union, "iou": iou, "precision": prec, "recall": rec}


def point_k_metrics(df: pd.DataFrame) -> dict:
    gt = df["gt_labels"].values == K_LABEL
    pr = df["pred_labels"].values == K_LABEL
    tp = int((gt & pr).sum())
    fp = int((~gt & pr).sum())
    fn = int((gt & ~pr).sum())
    union = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "union": union,
        "iou": tp / union if union else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "gt_k": int(gt.sum()),
        "pred_k": int(pr.sum()),
    }


def label_breakdown(series_gt_k: pd.Series, pred_col: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for lbl, cnt in series_gt_k[pred_col].value_counts().items():
        out[CLASS_NAMES.get(int(lbl), f"class_{lbl}")] = int(cnt)
    return out


def assign_buckets(labelled: pd.DataFrame, multi_column_rings: list[int]) -> dict:
    """Assign each K union-error pixel to exactly one root-cause bucket A–E."""
    buckets = {
        "A_oversize": 0,
        "B_theta_seam": 0,
        "C_edge": 0,
        "D_class_leakage": 0,
        "E_other": 0,
    }
    seam_rings: set[int] = set()
    k_labelled = labelled[labelled["gt_labels"] == K_LABEL]
    for gt_ring in multi_column_rings:
        sub = k_labelled[k_labelled["gt_rings"] == gt_ring]
        if sub.empty:
            continue
        counts = sub["pred_rings"].value_counts()
        if len(counts) >= 2 and counts.iloc[1] / counts.sum() >= 0.05:
            seam_rings.add(int(gt_ring))

    interior_oversize_cols: set[int] = set()
    for col in range(2, 9):
        sub = labelled[labelled["pred_rings"] == col]
        gt_k = int((sub["gt_labels"] == K_LABEL).sum())
        pred_k = int((sub["pred_labels"] == K_LABEL).sum())
        if gt_k and pred_k / gt_k > 1.2:
            interior_oversize_cols.add(col)

    fn_rows = labelled[(labelled["gt_labels"] == K_LABEL) & (labelled["pred_labels"] != K_LABEL)]
    fp_rows = labelled[(labelled["pred_labels"] == K_LABEL) & (labelled["gt_labels"] != K_LABEL)]

    def bucket_one(row: pd.Series, is_fp: bool) -> str:
        col = int(row["pred_rings"])
        if col in (0, 9):
            return "C_edge"
        if is_fp:
            if int(row["gt_labels"]) in (2, 6):
                return "D_class_leakage"
            if col in interior_oversize_cols:
                return "A_oversize"
            if col == 1:
                return "B_theta_seam"
            return "E_other"
        gt_ring = int(row["gt_rings"])
        if col == 1 or gt_ring in seam_rings:
            return "B_theta_seam"
        if int(row["pred_labels"]) in (2, 6):
            return "D_class_leakage"
        return "E_other"

    for _, row in fn_rows.iterrows():
        buckets[bucket_one(row, is_fp=False)] += 1
    for _, row in fp_rows.iterrows():
        buckets[bucket_one(row, is_fp=True)] += 1

    total_union_err = sum(buckets.values())
    shares = {k: (v / total_union_err if total_union_err else 0.0) for k, v in buckets.items()}
    interior_fp = int(fp_rows[fp_rows["pred_rings"].isin(interior_oversize_cols)].shape[0])
    interior_fn = int(fn_rows[fn_rows["pred_rings"].isin(interior_oversize_cols)].shape[0])
    return {
        "counts": buckets,
        "shares": shares,
        "total_union_error": total_union_err,
        "seam_rings": sorted(seam_rings),
        "interior_oversize_cols": sorted(interior_oversize_cols),
        "interior_oversize_fp": interior_fp,
        "interior_oversize_total_error": interior_fp + interior_fn,
    }


def run_diagnosis(tunnel_dir: Path, out_dir: Path, resolution: float = RESOLUTION) -> dict:
    tunnel_dir = tunnel_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    only = pd.read_csv(tunnel_dir / "only_label.csv")
    labelled = only[only["pred_rings"] >= 0].copy()

    with open(tunnel_dir / "pixel_to_point.pkl", "rb") as f:
        ptp = pickle.load(f)
    final = pd.read_csv(tunnel_dir / "final.csv")
    segments = final["segment"].values
    pred_point = final["pred"].values if "pred" in final.columns else None

    depth = np.load(tunnel_dir / "depth_map.npy")
    h, w = depth.shape

    # Align pred on points via only_label merge on index order — use final pred column
    if pred_point is None:
        raise ValueError("final.csv missing pred column")

    global_m = point_k_metrics(labelled)
    fn_rows = labelled[(labelled["gt_labels"] == K_LABEL) & (labelled["pred_labels"] != K_LABEL)]
    fp_rows = labelled[(labelled["pred_labels"] == K_LABEL) & (labelled["gt_labels"] != K_LABEL)]
    global_fn_breakdown = label_breakdown(fn_rows, "pred_labels")
    global_fp_breakdown = label_breakdown(fp_rows, "gt_labels")

    per_column: list[dict] = []
    for col in sorted(labelled["pred_rings"].unique()):
        sub = labelled[labelled["pred_rings"] == col]
        m = point_k_metrics(sub)
        ratio = m["pred_k"] / m["gt_k"] if m["gt_k"] else None
        per_column.append(
            {
                "column": int(col),
                **m,
                "pred_gt_ratio": ratio,
                "fn_breakdown": label_breakdown(
                    sub[(sub["gt_labels"] == K_LABEL) & (sub["pred_labels"] != K_LABEL)],
                    "pred_labels",
                ),
                "fp_breakdown": label_breakdown(
                    sub[(sub["pred_labels"] == K_LABEL) & (sub["gt_labels"] != K_LABEL)],
                    "gt_labels",
                ),
            }
        )

    # GT ring -> pred_rings crosstab for K pixels
    k_labelled = labelled[labelled["gt_labels"] == K_LABEL]
    ring_crosstab: dict[str, dict] = {}
    multi_column_rings: list[int] = []
    for gt_ring in sorted(k_labelled["gt_rings"].unique()):
        sub = k_labelled[k_labelled["gt_rings"] == gt_ring]
        counts = {int(k): int(v) for k, v in sub["pred_rings"].value_counts().items()}
        ring_crosstab[str(int(gt_ring))] = counts
        if len(counts) > 1:
            multi_column_rings.append(int(gt_ring))

    # Raster maps
    gt_map = build_gt_k_map(ptp, segments, (h, w))
    pred_map = build_pred_k_map(ptp, pred_point, (h, w))
    raster_m = k_metrics(gt_map.astype(bool), pred_map.astype(bool))

    # Per-column raster spans using column bands from initial_points
    init = pd.read_csv(tunnel_dir / "initial_points.csv")
    half_w = 1264 / (2 * resolution * 1000)  # segment_width default
    template_span_px = TEMPLATE_K_SPAN_MM / (resolution * 1000)
    geom_rows = []
    for i, row in init.iterrows():
        cx = float(row["X"])
        band = (np.arange(w) >= cx - half_w) & (np.arange(w) <= cx + half_w)
        band2d = np.tile(band, (h, 1))
        gt_col = gt_map & band2d
        pr_col = pred_map & band2d
        gt_span = vertical_span_y(gt_col)
        pr_span = vertical_span_y(pr_col)
        rm = k_metrics(gt_col.astype(bool), pr_col.astype(bool))
        geom_rows.append(
            {
                "column": i,
                "det_y": float(row["Y"]),
                "gt_span_px": gt_span,
                "pred_span_px": pr_span,
                "template_span_px": template_span_px,
                "span_overshoot_px": pr_span - gt_span if gt_span else None,
                "raster_iou": rm["iou"],
            }
        )
        if i < len(per_column):
            per_column[i]["gt_span_px"] = gt_span
            per_column[i]["pred_span_px"] = pr_span
            per_column[i]["template_span_px"] = template_span_px

    buckets = assign_buckets(labelled, multi_column_rings)

    summary = {
        "tunnel_dir": str(tunnel_dir),
        "global_point": global_m,
        "global_fn_breakdown": global_fn_breakdown,
        "global_fp_breakdown": global_fp_breakdown,
        "raster_global": raster_m,
        "per_column": per_column,
        "ring_k_crosstab": ring_crosstab,
        "multi_column_gt_rings": multi_column_rings,
        "geometry_per_column": geom_rows,
        "buckets": buckets,
        "template_k_span_mm": TEMPLATE_K_SPAN_MM,
    }

    # --- Figures ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    cols = [c["column"] for c in per_column]
    ious = [c["iou"] for c in per_column]
    axes[0, 0].bar(cols, ious, color="steelblue")
    axes[0, 0].axhline(0.65, color="r", ls="--", label="target 0.65")
    axes[0, 0].set_xlabel("SAM column")
    axes[0, 0].set_ylabel("K IoU")
    axes[0, 0].set_title("Per-column K IoU (point-level)")
    axes[0, 0].legend()

    ratios = [c["pred_gt_ratio"] if c["pred_gt_ratio"] else 0 for c in per_column]
    axes[0, 1].bar(cols, ratios, color="coral")
    axes[0, 1].axhline(1.0, color="k", lw=0.8)
    axes[0, 1].set_xlabel("SAM column")
    axes[0, 1].set_ylabel("pred_K / gt_K")
    axes[0, 1].set_title("K area ratio")

    err_cols = [c["column"] for c in per_column]
    fp_vals = [c["fp"] for c in per_column]
    fn_vals = [c["fn"] for c in per_column]
    x = np.array(err_cols)
    axes[1, 0].bar(x - 0.15, fp_vals, width=0.3, label="FP", color="#d62728")
    axes[1, 0].bar(x + 0.15, fn_vals, width=0.3, label="FN", color="#2166ac")
    axes[1, 0].set_xlabel("SAM column")
    axes[1, 0].set_title("K false positive / false negative pixels")
    axes[1, 0].legend()

    bucket_names = list(buckets["counts"].keys())
    bucket_vals = list(buckets["counts"].values())
    axes[1, 1].bar(range(len(bucket_names)), bucket_vals, tick_label=[b.replace("_", "\n") for b in bucket_names])
    axes[1, 1].set_title("Union-error pixels by root-cause bucket")
    plt.tight_layout()
    fig.savefig(out_dir / "k_iou_per_column.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Error overlay on depth map (subsample for visibility)
    tp_m = gt_map & pred_map
    fp_m = (~gt_map.astype(bool)) & pred_map.astype(bool)
    fn_m = gt_map.astype(bool) & (~pred_map.astype(bool))
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    try:
        base = plt.imread(str(tunnel_dir / "depth_map.png"))
        if base.ndim == 3:
            overlay = base[:, :, :3].astype(np.float32)
            if overlay.max() > 1:
                overlay /= 255.0
    except Exception:
        overlay[:] = 0.3

    overlay[tp_m] = [0.7, 0.7, 0.7]
    overlay[fn_m] = [0.1, 0.3, 0.9]
    overlay[fp_m] = [0.9, 0.2, 0.1]
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    ax2.imshow(overlay)
    ax2.set_title("K errors: gray=TP, blue=FN, red=FP")
    fig2.savefig(out_dir / "k_error_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Zoom columns 1 and 9
    for col_idx, title in [(1, "col1_seam"), (9, "col9_edge")]:
        if col_idx >= len(init):
            continue
        cx = float(init.iloc[col_idx]["X"])
        x0, x1 = max(0, int(cx - 200)), min(w, int(cx + 200))
        band = (np.arange(w) >= cx - half_w) & (np.arange(w) <= cx + half_w)
        col_mask = np.tile(band, (h, 1))
        ys = np.where((gt_map | pred_map) & col_mask)[0]
        y0, y1 = (max(0, int(ys.min()) - 40), min(h, int(ys.max()) + 40)) if len(ys) else (1300, 1700)
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.imshow(overlay[y0:y1, x0:x1])
        ax3.set_title(f"Zoom column {col_idx} ({title})")
        fig3.savefig(out_dir / f"k_error_zoom_{title}.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)

    with open(out_dir / "k_iou_gt_diagnostics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tunnel_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--resolution", type=float, default=RESOLUTION)
    args = parser.parse_args()
    out = args.out_dir or Path("data") / f"{args.tunnel_dir.name}_k_iou_gt"
    summary = run_diagnosis(args.tunnel_dir, out, args.resolution)
    g = summary["global_point"]
    print(f"Global K IoU={g['iou']:.4f} prec={g['precision']:.3f} rec={g['recall']:.3f}")
    print(f"pred/gt ratio={g['pred_k']/g['gt_k']:.2f} TP={g['tp']} FP={g['fp']} FN={g['fn']}")
    print("Buckets:", summary["buckets"]["counts"])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
