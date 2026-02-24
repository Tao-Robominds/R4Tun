"""
GT-derived shape analysis for complex staggered tunnels (4-1, 5-1).

Uses ground truth (segment, ring) from the point cloud and pixel_to_point
to build a GT label map in depth-map space, then extracts the actual
proximate shape of each block instance (contour/polygon). Computes shape
descriptors and aggregates by block type to find patterns (e.g. K/B1/B2
vs A-blocks, rectangularity, vertex count).

Outputs:
  - data/<tunnel_id>/gt_shape_analysis/ per-instance and per-type stats
  - data/<tunnel_id>/gt_shape_analysis/report.md summary and patterns
  - Optional: polygon JSON or contour visualizations
"""

import os
import sys
import json
import pickle
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import cv2

# 7-seg: segment 1=K, 2=B1, 3=A1, 4=A2, 5=A3, 6=A4, 7=B2 (evaluation convention)
SEGMENT_TO_BLOCK_7 = {
    1: "K",
    2: "B1",
    3: "A1",
    4: "A2",
    5: "A3",
    6: "A4",
    7: "B2",
}


def load_gt_label_map(
    tunnel_dir: str,
    depth_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build GT label map (segment) and ring map from enhanced.csv + pixel_to_point.
    Pixels with no mapping stay 0 / -1. Ties per pixel resolved by mode.
    """
    enhanced_path = os.path.join(tunnel_dir, "enhanced.csv")
    p2p_path = os.path.join(tunnel_dir, "pixel_to_point.pkl")
    if not os.path.exists(enhanced_path) or not os.path.exists(p2p_path):
        raise FileNotFoundError(f"Need {enhanced_path} and {p2p_path}")

    df = pd.read_csv(enhanced_path, usecols=["segment", "ring"])
    df = df.reset_index(drop=True)
    with open(p2p_path, "rb") as f:
        pixel_to_point = pickle.load(f)

    # Vectorized: one row per mapping; restrict df to indices that appear in pixel_to_point
    p2p_df = pd.DataFrame(pixel_to_point)
    need_idx = p2p_df["index"].unique()
    df_small = df.loc[df.index.intersection(need_idx)]
    p2p_df = p2p_df.merge(
        df_small[["segment", "ring"]],
        left_on="index",
        right_index=True,
        how="inner",
    )
    p2p_df = p2p_df[(p2p_df["segment"] >= 1) & (p2p_df["segment"] <= 7)]
    if p2p_df.empty:
        height, width = depth_shape
        return np.zeros((height, width), dtype=np.int32), np.full((height, width), -1, dtype=np.int32)

    # First occurrence per pixel (drop_duplicates keeps first)
    mode_df = p2p_df.drop_duplicates(subset=["pixel_y", "pixel_x"], keep="first")[
        ["pixel_y", "pixel_x", "segment", "ring"]
    ]

    height, width = depth_shape
    gt_segment = np.zeros((height, width), dtype=np.int32)
    gt_ring = np.full((height, width), -1, dtype=np.int32)
    py = mode_df["pixel_y"].values.astype(int)
    px = mode_df["pixel_x"].values.astype(int)
    valid = (py >= 0) & (py < height) & (px >= 0) & (px < width)
    seg_vals = mode_df["segment"].values.astype(np.int32)
    ring_vals = mode_df["ring"].values.astype(np.int32)
    gt_segment[py[valid], px[valid]] = seg_vals[valid]
    gt_ring[py[valid], px[valid]] = ring_vals[valid]
    return gt_segment, gt_ring


def extract_instance_masks(
    gt_segment: np.ndarray,
    gt_ring: np.ndarray,
) -> Dict[Tuple[int, int], np.ndarray]:
    """For each (segment, ring) with segment in 1..7, binary mask of that instance."""
    instances = {}
    for seg in range(1, 8):
        for ring in np.unique(gt_ring[gt_ring >= 0]):
            mask = ((gt_segment == seg) & (gt_ring == ring)).astype(np.uint8)
            if np.sum(mask) < 10:  # skip tiny fragments
                continue
            instances[(seg, int(ring))] = mask
    return instances


def contour_to_polygon(contour: np.ndarray, epsilon_ratio: float = 0.02) -> np.ndarray:
    """Approximate contour to polygon (fewer vertices)."""
    if len(contour) < 3:
        return contour
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(2.0, perimeter * epsilon_ratio)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def shape_descriptors(mask: np.ndarray) -> Dict[str, float]:
    """Compute rectangularity, aspect ratio, vertex count, area, perimeter.
    Uses convex hull of instance pixels when mask is fragmented (many small contours).
    """
    points = np.column_stack(np.where(mask > 0))  # (y, x)
    if len(points) < 10:
        return {}
    # Convex hull of all instance pixels gives one contiguous shape
    hull = cv2.convexHull(points.astype(np.float32))
    area = float(cv2.contourArea(hull))
    if area < 1:
        return {}
    perimeter = cv2.arcLength(hull, True)
    rect = cv2.minAreaRect(hull)
    (_, (w, h), _) = rect
    box_w, box_h = max(w, h), min(w, h)
    box_area = box_w * box_h
    rectangularity = area / box_area if box_area > 0 else 0.0
    aspect_ratio = box_w / box_h if box_h > 0 else 0.0
    approx = contour_to_polygon(hull)
    n_vertices = len(approx)

    return {
        "area_px": area,
        "perimeter_px": perimeter,
        "rectangularity": rectangularity,
        "aspect_ratio": aspect_ratio,
        "n_vertices": n_vertices,
        "bbox_width_px": box_w,
        "bbox_height_px": box_h,
    }


def run_analysis(tunnel_id: str, base_dir: str = "data") -> Dict[str, Any]:
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    depth_path = os.path.join(tunnel_dir, "depth_map.png")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    img = cv2.imread(depth_path)
    height, width = img.shape[:2]

    gt_segment, gt_ring = load_gt_label_map(tunnel_dir, (height, width))
    instances = extract_instance_masks(gt_segment, gt_ring)

    # Per-instance descriptors
    instance_stats = []
    for (seg, ring), mask in instances.items():
        block = SEGMENT_TO_BLOCK_7.get(seg, f"S{seg}")
        desc = shape_descriptors(mask)
        if not desc:
            continue
        desc["segment"] = seg
        desc["ring"] = ring
        desc["block"] = block
        instance_stats.append(desc)

    if not instance_stats:
        return {"tunnel_id": tunnel_id, "n_instances": 0, "by_block": {}, "instances": []}

    df_inst = pd.DataFrame(instance_stats)

    # Aggregate by block type
    by_block = {}
    for block in df_inst["block"].unique():
        sub = df_inst[df_inst["block"] == block]
        by_block[block] = {
            "count": len(sub),
            "area_px_mean": float(sub["area_px"].mean()),
            "area_px_std": float(sub["area_px"].std()) if len(sub) > 1 else 0.0,
            "rectangularity_mean": float(sub["rectangularity"].mean()),
            "rectangularity_std": float(sub["rectangularity"].std()) if len(sub) > 1 else 0.0,
            "aspect_ratio_mean": float(sub["aspect_ratio"].mean()),
            "n_vertices_mean": float(sub["n_vertices"].mean()),
            "n_vertices_std": float(sub["n_vertices"].std()) if len(sub) > 1 else 0.0,
        }

    return {
        "tunnel_id": tunnel_id,
        "n_instances": len(instance_stats),
        "by_block": by_block,
        "instances": instance_stats,
        "df": df_inst,
    }


def write_report(results_4_1: Dict, results_5_1: Dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report.md")
    lines = [
        "# GT-derived block shape analysis (4-1, 5-1)",
        "",
        "Actual proximate shapes extracted from ground truth (segment, ring) in the point cloud.",
        "",
        "## 4-1",
    ]
    if results_4_1.get("n_instances", 0) == 0:
        lines.append("- No instances extracted.")
    else:
        lines.append(f"- Instances: {results_4_1['n_instances']}")
        lines.append("")
        lines.append("| Block | Count | Rectangularity (mean±std) | Aspect ratio | Vertices (mean±std) |")
        lines.append("|-------|-------|----------------------------|--------------|----------------------|")
        for block, s in results_4_1.get("by_block", {}).items():
            r = s["rectangularity_mean"]
            r_std = s["rectangularity_std"]
            ar = s["aspect_ratio_mean"]
            v = s["n_vertices_mean"]
            v_std = s["n_vertices_std"]
            lines.append(f"| {block} | {s['count']} | {r:.3f} ± {r_std:.3f} | {ar:.2f} | {v:.1f} ± {v_std:.1f} |")

    lines.append("")
    lines.append("## 5-1")
    if results_5_1.get("n_instances", 0) == 0:
        lines.append("- No instances extracted.")
    else:
        lines.append(f"- Instances: {results_5_1['n_instances']}")
        lines.append("")
        lines.append("| Block | Count | Rectangularity (mean±std) | Aspect ratio | Vertices (mean±std) |")
        lines.append("|-------|-------|----------------------------|--------------|----------------------|")
        for block, s in results_5_1.get("by_block", {}).items():
            r = s["rectangularity_mean"]
            r_std = s["rectangularity_std"]
            ar = s["aspect_ratio_mean"]
            v = s["n_vertices_mean"]
            v_std = s["n_vertices_std"]
            lines.append(f"| {block} | {s['count']} | {r:.3f} ± {r_std:.3f} | {ar:.2f} | {v:.1f} ± {v_std:.1f} |")

    lines.append("")
    lines.append("## Patterns")
    lines.append("")
    lines.append("- **Rectangularity** 1.0 = perfect rectangle; <1 = irregular (e.g. trapezoid, curved).")
    lines.append("- **Vertices** 4 = rectangle; >4 suggests non-rectangular (K, B1, B2) vs A-blocks.")
    lines.append("- Compare K, B1, B2 (often keystone/side blocks) vs A1–A4 (arch blocks) across tunnels.")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {path}")


def main():
    parser = argparse.ArgumentParser(description="GT shape analysis for 4-1 and 5-1")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--output-dir", default=None, help="Override output dir (default: data/<tunnel_id>/gt_shape_analysis)")
    parser.add_argument("--tunnels", nargs="+", default=["4-1", "5-1"], help="Tunnel ids")
    args = parser.parse_args()

    all_results = {}
    for tunnel_id in args.tunnels:
        tunnel_dir = os.path.join(args.data_dir, tunnel_id)
        if not os.path.isdir(tunnel_dir):
            print(f"Skipping {tunnel_id}: no directory {tunnel_dir}")
            continue
        print(f"Analyzing {tunnel_id}...")
        try:
            out_dir = args.output_dir or os.path.join(tunnel_dir, "gt_shape_analysis")
            res = run_analysis(tunnel_id, base_dir=args.data_dir)
            all_results[tunnel_id] = res
            os.makedirs(out_dir, exist_ok=True)
            # Save per-tunnel JSON (without df for size)
            save = {k: v for k, v in res.items() if k != "df"}
            with open(os.path.join(out_dir, "shape_stats.json"), "w") as f:
                json.dump(save, f, indent=2)
            if res.get("df") is not None:
                res["df"].to_csv(os.path.join(out_dir, "instance_descriptors.csv"), index=False)
            print(f"  Instances: {res.get('n_instances', 0)}")
        except Exception as e:
            print(f"  Error: {e}")
            all_results[tunnel_id] = {"tunnel_id": tunnel_id, "error": str(e)}

    if "4-1" in all_results and "5-1" in all_results:
        report_dir = os.path.join(args.data_dir, "gt_shape_analysis")
        write_report(
            all_results.get("4-1", {}),
            all_results.get("5-1", {}),
            report_dir,
        )


if __name__ == "__main__":
    main()
