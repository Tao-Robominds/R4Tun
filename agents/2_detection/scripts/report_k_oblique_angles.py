#!/usr/bin/env python3
"""
Report the oblique line angle parameters and the actual angles of lines crossing
each ring's K vertical (center x). One K per ring; oblique lines that cross that x
define the K block.

Usage (from repo root):
  python -m agents.irregular.2_detection.scripts.report_k_oblique_angles --tunnel 4-1 [--base-dir data]
"""

import argparse
import os
import sys

os.environ.setdefault("TQDM_DISABLE", "1")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import importlib
_det = importlib.import_module("agents.irregular.2_detection.2_detection")
detect_lines = _det.detect_lines
load_parameters = _det.load_parameters
detect_k_dbscan = _det.detect_k_dbscan
apply_k_regulator = _det.apply_k_regulator
_segment_length_and_angle = _det._segment_length_and_angle


def line_angle_deg(x1, y1, x2, y2):
    """Same as in detect_lines: angle in degrees, positive = upward-right, negative = downward-right."""
    return float(np.degrees(np.arctan2(-(y2 - y1), x2 - x1)))


def main():
    ap = argparse.ArgumentParser(description="Report K-block oblique line angles")
    ap.add_argument("--tunnel", default="4-1")
    ap.add_argument("--base-dir", default="data")
    args = ap.parse_args()

    tunnel_dir = os.path.join(args.base_dir, args.tunnel)
    path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(path):
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)
    depth_map = np.load(path)
    L, W = depth_map.shape
    ring_count = int(open(os.path.join(tunnel_dir, "ring_count.txt")).read())
    params, _ = load_parameters(args.tunnel, args.base_dir)

    angle_lo = float(params.get("angle_oblique_min", 6.0))
    angle_hi = float(params.get("angle_oblique_max", 9.0))
    print("Oblique angle parameters (detection filter):")
    print(f"  angle_oblique_min = {angle_lo} deg  (positive band lower bound)")
    print(f"  angle_oblique_max = {angle_hi} deg  (positive band upper bound)")
    print(f"  negative band = [-{angle_hi}, -{angle_lo}] deg")
    print()

    line_data = detect_lines(depth_map, params)
    pos_lines = line_data.get("positive_lines", [])
    neg_lines = line_data.get("negative_lines", [])

    print("Per ring: K vertical x, then oblique lines crossing that x (angle deg, y at crossing).")
    print("One K per ring; once identified, oblique lines at that x define the K block angle.\n")

    for r in range(ring_count):
        vx = (r + 0.5) * (W / ring_count)
        angles_pos = []
        angles_neg = []
        ys_pos = []
        ys_neg = []
        for (x1, y1, x2, y2) in pos_lines:
            if x1 == x2:
                continue
            if min(x1, x2) <= vx <= max(x1, x2):
                t = (vx - x1) / (x2 - x1)
                y_cross = y1 + t * (y2 - y1)
                ang = line_angle_deg(x1, y1, x2, y2)
                angles_pos.append(ang)
                ys_pos.append(y_cross)
        for (x1, y1, x2, y2) in neg_lines:
            if x1 == x2:
                continue
            if min(x1, x2) <= vx <= max(x1, x2):
                t = (vx - x1) / (x2 - x1)
                y_cross = y1 + t * (y2 - y1)
                ang = line_angle_deg(x1, y1, x2, y2)
                angles_neg.append(ang)
                ys_neg.append(y_cross)

        n_pos = len(angles_pos)
        n_neg = len(angles_neg)
        print(f"Ring {r}: K vertical x = {vx:.1f}")
        print(f"  Positive oblique lines crossing: {n_pos}")
        if angles_pos:
            print(f"    angles (deg): {[round(a, 2) for a in angles_pos]}")
            print(f"    y at crossing: {[round(y, 1) for y in ys_pos]}")
        print(f"  Negative oblique lines crossing: {n_neg}")
        if angles_neg:
            print(f"    angles (deg): {[round(a, 2) for a in angles_neg]}")
            print(f"    y at crossing: {[round(y, 1) for y in ys_neg]}")
        if angles_pos or angles_neg:
            all_ang = angles_pos + angles_neg
            print(f"  Detected K-block oblique angles at this ring: {[round(a, 2) for a in all_ang]}")
        print()

    # --- Angles of the oblique lines actually chosen for K (one K per ring) ---
    print("=" * 60)
    print("Oblique line angles of the CHOSEN K-block lines (used by regulator):")
    print("(One K per ring; these are the pos/neg lines that were selected.)")
    k_height_px = float(params.get("k_expected_height_px", 500))
    k_df = detect_k_dbscan(line_data, ring_count, params)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        k_df, used_pos_indices, used_neg_indices = apply_k_regulator(
            k_df, line_data, ring_count, k_height_px, params
        )
    used_pos = used_pos_indices or set()
    used_neg = used_neg_indices or set()
    pos_lines = line_data["positive_lines"]
    neg_lines = line_data["negative_lines"]
    angles_pos_used = []
    for idx in sorted(used_pos):
        if idx < len(pos_lines):
            _, ang = _segment_length_and_angle(pos_lines[idx])
            angles_pos_used.append(round(ang, 2))
    angles_neg_used = []
    for idx in sorted(used_neg):
        if idx < len(neg_lines):
            _, ang = _segment_length_and_angle(neg_lines[idx])
            angles_neg_used.append(round(ang, 2))
    print(f"  Positive oblique (used): {angles_pos_used}")
    print(f"  Negative oblique (used): {angles_neg_used}")
    if angles_pos_used:
        print(f"  Positive range: {min(angles_pos_used):.2f}° to {max(angles_pos_used):.2f}°")
    if angles_neg_used:
        print(f"  Negative range: {min(angles_neg_used):.2f}° to {max(angles_neg_used):.2f}°")


if __name__ == "__main__":
    main()
