#!/usr/bin/env python3
"""
Auto-analyze GT (all_segments_gt.csv) to produce block_structure.json for any tunnel.

Computes:
- ordered absolute angular distances from K (d1..d6 as circumference fractions)
- K-adjacent blocks per ring (neg_side, pos_side)
- circular block order per ring
- mean distances and statistics for BO search bounds

Output: data/<tunnel_id>/block_structure.json

Run from repo root:
  python scripts/analyze_block_structure.py <tunnel_id> [--data-dir data]
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

EXPANSION_BLOCKS = ["B1", "B2", "A1", "A2", "A3", "A4"]


def wrap(dy: float, circ: float) -> float:
    """Wrap offset to [-circ/2, circ/2] for circumferential distance."""
    half = circ / 2.0
    while dy > half:
        dy -= circ
    while dy < -half:
        dy += circ
    return dy


def analyze_block_structure(tunnel_id: str, data_dir: str = "data") -> dict:
    tunnel_dir = os.path.join(data_dir, tunnel_id)
    gt_path = os.path.join(tunnel_dir, "all_segments_gt.csv")
    depth_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT not found: {gt_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth map not found: {depth_path} (needed for circumference)")

    gt = pd.read_csv(gt_path)
    if "ring" in gt.columns and "Ring" not in gt.columns:
        gt = gt.rename(columns={"ring": "Ring"})
    if "segment_name" in gt.columns and "Block" not in gt.columns:
        gt = gt.rename(columns={"segment_name": "Block"})

    depth_map = np.load(depth_path)
    circ = int(depth_map.shape[0])

    rings = sorted(gt["Ring"].unique())
    n_rings = len(rings)

    # Per-ring: ordered absolute distances from K, K-adjacent blocks, circular order
    ordered_dists_px = []  # list of 6-tuples per ring
    ordered_dists_frac = []
    k_adjacent = []  # list of {neg_side, pos_side, neg_dist_frac, pos_dist_frac} per ring
    circular_orders = []  # list of block order (K removed) per ring

    for ring in rings:
        rdf = gt[gt["Ring"] == ring]
        k_row = rdf[rdf["Block"] == "K"]
        if len(k_row) == 0:
            continue
        ky = float(k_row["Y"].iloc[0])

        dists = []
        for block in EXPANSION_BLOCKS:
            b_row = rdf[rdf["Block"] == block]
            if len(b_row) == 0:
                continue
            by = float(b_row["Y"].iloc[0])
            d_cw = (by - ky) % circ
            d_ccw = circ - d_cw
            dists.append(min(d_cw, d_ccw))

        dists.sort()
        if len(dists) != 6:
            continue
        ordered_dists_px.append(dists)
        ordered_dists_frac.append([d / circ for d in dists])

        # K-adjacent: nearest block on negative side (above K) and positive side (below K)
        neg_nearest = None
        pos_nearest = None
        neg_dist = 1e9
        pos_dist = 1e9
        for block in EXPANSION_BLOCKS:
            b_row = rdf[rdf["Block"] == block]
            if len(b_row) == 0:
                continue
            by = float(b_row["Y"].iloc[0])
            off = wrap(by - ky, circ)
            if off < 0 and abs(off) < neg_dist:
                neg_dist = abs(off)
                neg_nearest = block
            if off > 0 and off < pos_dist:
                pos_dist = off
                pos_nearest = block
        k_adjacent.append({
            "neg_side": neg_nearest,
            "pos_side": pos_nearest,
            "neg_dist_frac": neg_dist / circ if neg_nearest else None,
            "pos_dist_frac": pos_dist / circ if pos_nearest else None,
        })

        # Circular order (all 7 blocks sorted by offset from K, then drop K)
        positions = []
        for _, row in rdf.iterrows():
            off = wrap(row["Y"] - ky, circ)
            positions.append((row["Block"], off))
        positions.sort(key=lambda x: x[1])
        order = [p[0] for p in positions if p[0] != "K"]
        circular_orders.append(order)

    # Aggregate statistics
    arr = np.array(ordered_dists_frac)
    mean_dists_frac = np.mean(arr, axis=0).tolist()
    std_dists_frac = np.std(arr, axis=0).tolist()
    min_dists_frac = np.min(arr, axis=0).tolist()
    max_dists_frac = np.max(arr, axis=0).tolist()

    # Unique circular orderings (stagger phases)
    unique_orders = []
    seen = set()
    for order in circular_orders:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            unique_orders.append(list(order))

    # Ring X positions (for edge detection)
    ring_x = []
    for ring in rings:
        rdf = gt[gt["Ring"] == ring]
        k_row = rdf[rdf["Block"] == "K"]
        if len(k_row) > 0:
            ring_x.append(float(k_row["X"].iloc[0]))
        else:
            ring_x.append(None)

    out = {
        "tunnel_id": tunnel_id,
        "circumference_px": circ,
        "n_rings": n_rings,
        "rings": [int(r) for r in rings],
        "ordered_distances": {
            "mean_frac": mean_dists_frac,
            "std_frac": std_dists_frac,
            "min_frac": min_dists_frac,
            "max_frac": max_dists_frac,
            "mean_px": (np.array(mean_dists_frac) * circ).tolist(),
        },
        "k_adjacent_per_ring": k_adjacent,
        "circular_order_per_ring": circular_orders,
        "unique_circular_orders": unique_orders,
        "ring_x": ring_x,
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze GT block structure, output block_structure.json")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--output", default=None, help="Output path (default: data/<tunnel_id>/block_structure.json)")
    args = parser.parse_args()

    out = analyze_block_structure(args.tunnel_id, args.data_dir)
    if args.output is None:
        args.output = os.path.join(args.data_dir, args.tunnel_id, "block_structure.json")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
