"""
Universal A/B block position detection for complex_staggered tunnels.

Uses: groove-line boundary detection + 6-distance structural model + stagger detection
+ fusion/regulators. Produces all_segments.csv (Ring, Block, X, Y, quality).

No tunnel-specific hardcoding; parameters are circumference fractions and pixel distances.
"""
import os
import sys
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_agents_detection = os.path.join(_project_root, "agents", "irregular", "2_detection", "2_detection.py")
import importlib.util
_spec = importlib.util.spec_from_file_location("detection", _agents_detection)
_detection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detection)

detect_lines = _detection.detect_lines
line_segment_vertical_intersection = _detection.line_segment_vertical_intersection

EXPANSION_BLOCKS = ["B1", "B2", "A1", "A2", "A3", "A4"]


def _wrap_dy(y1: float, y2: float, circ: float) -> float:
    d = abs(y1 - y2)
    return min(d, circ - d)


def _wrap_offset(dy: float, circ: float) -> float:
    half = circ / 2.0
    while dy > half:
        dy -= circ
    while dy < -half:
        dy += circ
    return dy


# -----------------------------------------------------------------------------
# Step 1: Groove boundary detection
# -----------------------------------------------------------------------------
def get_line_intersections_at_x(
    line_data: Dict,
    x: float,
    circ: float,
) -> Tuple[List[float], List[float]]:
    """At vertical column x, get Y intersections of positive and negative lines."""
    pos_lines = line_data.get("positive_lines", [])
    neg_lines = line_data.get("negative_lines", [])
    pos_ys = []
    for seg in pos_lines:
        y = line_segment_vertical_intersection(x, seg)
        if y is not None:
            pos_ys.append(y % circ)
    neg_ys = []
    for seg in neg_lines:
        y = line_segment_vertical_intersection(x, seg)
        if y is not None:
            neg_ys.append(y % circ)
    return pos_ys, neg_ys


def merge_boundaries(ys: List[float], merge_dist: float, circ: float) -> List[float]:
    """Merge Y positions within merge_dist; return sorted list."""
    if not ys:
        return []
    ys = sorted(ys)
    merged = []
    cluster = [ys[0]]
    for i in range(1, len(ys)):
        if ys[i] - cluster[0] <= merge_dist:
            cluster.append(ys[i])
        else:
            merged.append(np.mean(cluster))
            cluster = [ys[i]]
    merged.append(np.mean(cluster))
    return merged


def groove_boundaries_at_ring(
    line_data: Dict,
    kx: float,
    circ: float,
    merge_dist: float,
) -> List[float]:
    """
    At ring's K X position, collect all line Y-intersections (pos + neg),
    merge within merge_dist, return sorted boundary candidates.
    """
    pos_ys, neg_ys = get_line_intersections_at_x(line_data, kx, circ)
    all_ys = pos_ys + neg_ys
    if not all_ys:
        return []
    return merge_boundaries(all_ys, merge_dist, circ)


def groove_boundaries_all_rings(
    line_data: Dict,
    k_positions: pd.DataFrame,
    circ: float,
    merge_dist: float,
) -> List[List[float]]:
    """Per-ring list of groove boundary Y positions."""
    boundaries_per_ring = []
    for _, row in k_positions.iterrows():
        kx = float(row["X"])
        b = groove_boundaries_at_ring(line_data, kx, circ, merge_dist)
        boundaries_per_ring.append(b)
    return boundaries_per_ring


# -----------------------------------------------------------------------------
# Step 2: Structural distance model
# -----------------------------------------------------------------------------
def structural_block_ys(
    ky: float,
    circ: float,
    d_fracs: List[float],
    pos_indices: Tuple[int, ...],
) -> List[float]:
    """
    Place 6 blocks at distances d_fracs from K.
    pos_indices: which of 0..5 go on positive (below K) side; rest negative.
    Returns 6 Y positions (wrapped to [0, circ)).
    """
    ys = []
    for i in range(6):
        dist_px = d_fracs[i] * circ
        if i in pos_indices:
            y = (ky + dist_px) % circ
        else:
            y = (ky - dist_px) % circ
        if y < 0:
            y += circ
        ys.append(y)
    return ys


# -----------------------------------------------------------------------------
# Step 3: Stagger detection (score C(6,3) combos against groove boundaries)
# -----------------------------------------------------------------------------
def score_stagger_combo(
    pred_ys: List[float],
    boundary_ys: List[float],
    circ: float,
) -> float:
    """Lower is better: total min distance from each pred to nearest boundary."""
    if not boundary_ys:
        return float("inf")
    total = 0.0
    for py in pred_ys:
        best = min(_wrap_dy(py, by, circ) for by in boundary_ys)
        total += best
    return total


def best_stagger_for_ring(
    ky: float,
    circ: float,
    d_fracs: List[float],
    boundary_ys: List[float],
) -> Tuple[Tuple[int, ...], float]:
    """Try all C(6,3) assignments; return (best_pos_indices, best_score)."""
    best_combo = (0, 1, 2)
    best_score = float("inf")
    for pos_indices in combinations(range(6), 3):
        pred_ys = structural_block_ys(ky, circ, d_fracs, pos_indices)
        score = score_stagger_combo(pred_ys, boundary_ys, circ)
        if score < best_score:
            best_score = score
            best_combo = pos_indices
    return best_combo, best_score


def stagger_with_fallback(
    k_positions: pd.DataFrame,
    circ: float,
    d_fracs: List[float],
    boundaries_per_ring: List[List[float]],
    min_boundaries_for_detect: int = 4,
) -> List[Tuple[int, ...]]:
    """
    Per-ring best stagger (pos_indices). If a ring has < min_boundaries_for_detect
    boundaries, copy stagger from nearest ring that has enough.
    """
    n_rings = len(k_positions)
    best_per_ring: List[Optional[Tuple[int, ...]]] = [None] * n_rings

    for i in range(n_rings):
        ky = float(k_positions.iloc[i]["Y"])
        b = boundaries_per_ring[i] if i < len(boundaries_per_ring) else []
        if len(b) >= min_boundaries_for_detect:
            combo, _ = best_stagger_for_ring(ky, circ, d_fracs, b)
            best_per_ring[i] = combo
        else:
            best_per_ring[i] = None

    # Fallback: use nearest well-detected ring
    for i in range(n_rings):
        if best_per_ring[i] is not None:
            continue
        best_dist = 1e9
        fallback = (0, 1, 2)
        for j in range(n_rings):
            if best_per_ring[j] is None:
                continue
            dist = abs(i - j)
            if dist < best_dist:
                best_dist = dist
                fallback = best_per_ring[j]
        best_per_ring[i] = fallback

    return list(best_per_ring)


# -----------------------------------------------------------------------------
# Step 4: Fusion + regulators
# -----------------------------------------------------------------------------
def nearest_boundary(y_pred: float, boundary_ys: List[float], circ: float, max_radius: float) -> Optional[float]:
    """Return nearest boundary Y within max_radius, or None."""
    if not boundary_ys:
        return None
    best = None
    best_d = max_radius + 1
    for by in boundary_ys:
        d = _wrap_dy(y_pred, by, circ)
        if d <= max_radius and d < best_d:
            best_d = d
            best = by
    return best


def fuse_and_regulate(
    k_positions: pd.DataFrame,
    circ: float,
    d_fracs: List[float],
    stagger_per_ring: List[Tuple[int, ...]],
    boundaries_per_ring: List[List[float]],
    groove_blend: float,
    groove_search_radius: float,
    dx_global: float,
    edge_threshold: float,
    edge_scale: float,
) -> List[Dict[str, Any]]:
    """
    For each ring, compute 6 block Y positions (structural + optional groove blend),
    apply edge damping, set X = K_X + dx_global.
    Returns list of dicts: {ring_idx, block_idx_0_to_5, x, y} (unlabeled).
    """
    rows = []
    n_rings = len(k_positions)

    for i in range(n_rings):
        kx = float(k_positions.iloc[i]["X"])
        ky = float(k_positions.iloc[i]["Y"])
        x_block = kx + dx_global
        is_edge = kx < edge_threshold
        scale = edge_scale if is_edge else 1.0
        d_fracs_scaled = [f * scale for f in d_fracs]
        pos_indices = stagger_per_ring[i] if i < len(stagger_per_ring) else (0, 1, 2)
        pred_ys = structural_block_ys(ky, circ, d_fracs_scaled, pos_indices)
        boundaries = boundaries_per_ring[i] if i < len(boundaries_per_ring) else []

        for block_idx, y_pred in enumerate(pred_ys):
            y_final = y_pred
            near = nearest_boundary(y_pred, boundaries, circ, groove_search_radius)
            if near is not None and 0 <= groove_blend <= 1:
                y_final = (1.0 - groove_blend) * y_pred + groove_blend * near
            y_final = y_final % circ
            if y_final < 0:
                y_final += circ
            rows.append({
                "ring_idx": i,
                "block_idx": block_idx,
                "X": x_block,
                "Y": y_final,
            })
    return rows


# -----------------------------------------------------------------------------
# Step 5: Block labeling (B1, B2, A1-A4 by angular position / size)
# -----------------------------------------------------------------------------
def label_blocks(
    rows: List[Dict],
    k_positions: pd.DataFrame,
    circ: float,
    b_size_ratio: float,
) -> pd.DataFrame:
    """
    Rows have ring_idx, block_idx, X, Y. block_idx 0..5 are in order of d1..d6.
    The two K-adjacent (smallest distances) are B1, B2; the other four are A1-A4.
    We assign by angular position: sort 6 blocks by Y, then label B1, B2 (two nearest to K), A1-A4 (rest).
    Actually the structural model already orders by distance: indices 0,1 are the two nearest (d1,d2).
    So we need to map (ring_idx, block_idx) -> block name. block_idx corresponds to one of d1..d6.
    The two with smallest d are B-blocks; the four with larger d are A-blocks.
    So: block_idx 0,1 -> B1, B2 (order by Y: lower Y first is B1 or B2 depending on tunnel; we use B1, B2 arbitrarily by order)
    block_idx 2,3,4,5 -> A1, A2, A3, A4 (order by Y).
    """
    out_rows = []
    by_ring: Dict[int, List[Dict]] = {}
    for r in rows:
        ri = r["ring_idx"]
        if ri not in by_ring:
            by_ring[ri] = []
        by_ring[ri].append(r)

    for ring_idx in sorted(by_ring.keys()):
        ring_blocks = by_ring[ring_idx]
        k_row = k_positions.iloc[ring_idx]
        ky = float(k_row["Y"])
        kx = float(k_row["X"])
        conf = float(k_row.get("Confidence", 1.0))

        out_rows.append({
            "Ring": ring_idx,
            "Block": "K",
            "X": kx,
            "Y": ky,
            "quality": conf,
        })

        # Sort by block_idx: 0,1 are B; 2,3,4,5 are A
        ring_blocks.sort(key=lambda x: x["block_idx"])
        b_blocks = [ring_blocks[0], ring_blocks[1]]
        a_blocks = [ring_blocks[2], ring_blocks[3], ring_blocks[4], ring_blocks[5]]
        b_blocks.sort(key=lambda x: x["Y"])
        a_blocks.sort(key=lambda x: x["Y"])
        labels_b = ["B1", "B2"]
        labels_a = ["A1", "A2", "A3", "A4"]
        for b, lb in zip(b_blocks, labels_b):
            out_rows.append({
                "Ring": ring_idx,
                "Block": lb,
                "X": b["X"],
                "Y": round(b["Y"], 1),
                "quality": conf,
            })
        for a, la in zip(a_blocks, labels_a):
            out_rows.append({
                "Ring": ring_idx,
                "Block": la,
                "X": a["X"],
                "Y": round(a["Y"], 1),
                "quality": conf,
            })

    return pd.DataFrame(out_rows, columns=["Ring", "Block", "X", "Y", "quality"])


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def run_ab_detection(
    depth_map: np.ndarray,
    k_positions: pd.DataFrame,
    line_data: Optional[Dict] = None,
    params: Optional[Dict] = None,
    circ: Optional[int] = None,
    merge_dist: float = 50.0,
    d_fracs: Optional[List[float]] = None,
    groove_blend: float = 0.0,
    groove_search_radius: float = 100.0,
    dx_global: float = 0.0,
    edge_threshold: float = 400.0,
    edge_scale: float = 1.0,
    b_size_ratio: float = 0.5,
    min_boundaries_for_stagger: int = 4,
) -> pd.DataFrame:
    """
    Run full A/B detection pipeline.

    - depth_map: used if line_data is None (we call detect_lines).
    - k_positions: DataFrame with X, Y, (optional Confidence). One row per ring.
    - line_data: from detect_lines; if None, computed from depth_map and params.
    - params: detection params for detect_lines (required if line_data is None).
    - circ: circumference (depth map height). If None, depth_map.shape[0].
    - merge_dist, d_fracs, groove_*, dx_global, edge_*, b_size_ratio: BO params.
    - d_fracs: 6 ordered circumference fractions. If None, use default [0.10, 0.11, 0.24, 0.29, 0.36, 0.41].
    """
    if circ is None:
        circ = int(depth_map.shape[0])
    if line_data is None:
        if params is None:
            params = {}
        line_data = detect_lines(depth_map, params)
    if d_fracs is None:
        d_fracs = [0.10, 0.11, 0.24, 0.29, 0.36, 0.41]
    if len(d_fracs) != 6:
        raise ValueError("d_fracs must have length 6")

    boundaries_per_ring = groove_boundaries_all_rings(line_data, k_positions, circ, merge_dist)
    stagger_per_ring = stagger_with_fallback(
        k_positions, circ, d_fracs, boundaries_per_ring, min_boundaries_for_stagger
    )
    rows = fuse_and_regulate(
        k_positions, circ, d_fracs, stagger_per_ring, boundaries_per_ring,
        groove_blend, groove_search_radius, dx_global, edge_threshold, edge_scale,
    )
    df = label_blocks(rows, k_positions, circ, b_size_ratio)
    return df


def run_ab_detection_from_tunnel(
    tunnel_id: str,
    base_dir: str = "data",
    k_positions: Optional[pd.DataFrame] = None,
    params: Optional[Dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load depth map, ring count, optionally K positions and params from tunnel dir.
    Then run_ab_detection. If k_positions is None, load from detected_k_dbscan.csv or similar.
    """
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    depth_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Missing {depth_path}")
    depth_map = np.load(depth_path)
    circ = int(depth_map.shape[0])

    if k_positions is None:
        for name in ["detected_k_dbscan.csv", "detected_k_groove_pair.csv", "detected_k_banded.csv"]:
            k_path = os.path.join(tunnel_dir, name)
            if os.path.exists(k_path):
                k_positions = pd.read_csv(k_path)
                if "Ring" not in k_positions.columns:
                    k_positions.insert(0, "Ring", range(len(k_positions)))
                break
        if k_positions is None:
            raise FileNotFoundError(f"No K positions CSV found in {tunnel_dir}")

    if params is None:
        from importlib.util import spec_from_file_location, module_from_spec
        _det_spec = spec_from_file_location("detection", _agents_detection)
        _det_mod = module_from_spec(_det_spec)
        _det_spec.loader.exec_module(_det_mod)
        params, _ = _det_mod.load_parameters(tunnel_id, base_dir)
        if params is None:
            params = {}

    line_data = detect_lines(depth_map, params)
    return run_ab_detection(
        depth_map, k_positions, line_data=line_data, params=params, circ=circ, **kwargs
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Universal A/B block detection")
    parser.add_argument("tunnel_id", help="e.g. 4-1")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=None, help="Output CSV (default: data/<tunnel_id>/all_segments_geo_ab.csv)")
    parser.add_argument("--merge-dist", type=float, default=50.0)
    parser.add_argument("--groove-blend", type=float, default=0.0)
    parser.add_argument("--groove-radius", type=float, default=100.0)
    parser.add_argument("--dx-global", type=float, default=0.0)
    parser.add_argument("--edge-threshold", type=float, default=400.0)
    parser.add_argument("--edge-scale", type=float, default=1.0)
    args = parser.parse_args()
    df = run_ab_detection_from_tunnel(
        args.tunnel_id, base_dir=args.data_dir,
        merge_dist=args.merge_dist,
        groove_blend=args.groove_blend,
        groove_search_radius=args.groove_radius,
        dx_global=args.dx_global,
        edge_threshold=args.edge_threshold,
        edge_scale=args.edge_scale,
    )
    out = args.output or os.path.join(args.data_dir, args.tunnel_id, "all_segments_geo_ab.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} segments to {out}")
