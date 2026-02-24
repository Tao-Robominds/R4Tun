"""
Geo detection for complex staggered (4-1, 5-1): K detection (combined = DBSCAN + groove_pair)
plus per-ring offset expansion with optional GT-derived offsets and GT K override.

Produces all_segments.csv with Ring (0-based), Block, X, Y, quality for downstream
per-instance geometric segmentation.

Usage:
  # Detected K + GT-derived per-ring offsets (prove expansion quality)
  python 4-1_geo_detection.py 4-1 --data-dir data
  # GT K + GT offsets + per-block X (ceiling)
  python 4-1_geo_detection.py 4-1 --data-dir data --use-gt-k --use-gt-x
  # GT K + GT offsets only (K's X for all blocks)
  python 4-1_geo_detection.py 4-1 --data-dir data --use-gt-k
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Load detection from agents/irregular/2_detection
_script_dir = os.path.dirname(os.path.abspath(__file__))
_agents_detection = os.path.join(_script_dir, "..", "agents", "irregular", "2_detection", "2_detection.py")
import importlib.util
_spec = importlib.util.spec_from_file_location("detection", _agents_detection)
_detection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detection)

detect_lines = _detection.detect_lines
load_parameters = _detection.load_parameters
load_preprocessing_params = _detection.load_preprocessing_params
calculate_segment_heights = _detection.calculate_segment_heights
calculate_k_positions_combined = _detection.calculate_k_positions_combined
calculate_k_positions_groove_pair = _detection.calculate_k_positions_groove_pair

EXPANSION_BLOCKS = ['B1', 'B2', 'A1', 'A2', 'A3', 'A4']


def _wrap_offset(dy: float, height: int) -> float:
    """Wrap offset to [-height/2, height/2] for circumferential distance."""
    half = height / 2.0
    while dy > half:
        dy -= height
    while dy < -half:
        dy += height
    return dy


def compute_gt_derived_offsets(tunnel_dir: str, img_height: int) -> tuple:
    """
    Compute per-ring Y offsets from all_segments_gt.csv.
    GT file has Ring 119, 120, ... (tunnel-specific). Map to 0-based ring index.

    Returns:
        per_ring_offsets: dict with keys b1_offset_r0, a4_offset_r6, etc. (6 blocks x 7 rings)
        gt_segments_0based: DataFrame with Ring 0..n-1, Block, X, Y (for per-block X if needed)
    """
    gt_path = os.path.join(tunnel_dir, "all_segments_gt.csv")
    if not os.path.exists(gt_path):
        return {}, None
    gt = pd.read_csv(gt_path)
    if "ring" in gt.columns and "Ring" not in gt.columns:
        gt = gt.rename(columns={"ring": "Ring"})
    if "segment_name" in gt.columns and "Block" not in gt.columns:
        gt = gt.rename(columns={"segment_name": "Block"})
    rings_gt = sorted(gt["Ring"].unique())
    n_rings = len(rings_gt)
    ring_gt_to_idx = {int(r): i for i, r in enumerate(rings_gt)}

    per_ring_offsets = {}
    rows_0based = []
    for ring_gt in rings_gt:
        r = ring_gt_to_idx[int(ring_gt)]
        ring_df = gt[gt["Ring"] == ring_gt]
        k_row = ring_df[ring_df["Block"] == "K"]
        if len(k_row) == 0:
            continue
        k_y = float(k_row["Y"].iloc[0])
        k_x = float(k_row["X"].iloc[0])
        rows_0based.append({"Ring": r, "Block": "K", "X": k_x, "Y": k_y})
        for block in EXPANSION_BLOCKS:
            b_row = ring_df[ring_df["Block"] == block]
            if len(b_row) == 0:
                continue
            by = float(b_row["Y"].iloc[0])
            bx = float(b_row["X"].iloc[0])
            offset = _wrap_offset(by - k_y, img_height)
            key = f"{block.lower()}_offset_r{r}"
            per_ring_offsets[key] = offset
            rows_0based.append({"Ring": r, "Block": block, "X": bx, "Y": by})
    gt_segments_0based = pd.DataFrame(rows_0based) if rows_0based else None
    return per_ring_offsets, gt_segments_0based


def expand_k_with_per_ring_offsets(
    k_positions: pd.DataFrame,
    img_height: int,
    per_ring_offsets: dict,
    use_gt_x_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Expand K to all segments using per-ring Y offsets. Optionally use per-block X from use_gt_x_df.

    per_ring_offsets: keys like b1_offset_r0, a4_offset_r6 (block.lower()_offset_r{ring_idx}).
    use_gt_x_df: optional DataFrame with Ring, Block, X, Y; if provided, X is taken from here per (Ring, Block).
    """
    rows = []
    for ring_idx, (_, k_row) in enumerate(k_positions.iterrows()):
        k_x = float(k_row["X"])
        k_y = float(k_row["Y"])
        quality = float(k_row.get("Confidence", 1.0))

        def get_x(block: str):
            if use_gt_x_df is not None:
                m = (use_gt_x_df["Ring"] == ring_idx) & (use_gt_x_df["Block"] == block)
                if m.any():
                    return float(use_gt_x_df.loc[m, "X"].iloc[0])
            return k_x

        rows.append({
            "Ring": ring_idx,
            "Block": "K",
            "X": get_x("K"),
            "Y": k_y % img_height,
            "quality": quality,
        })
        for block in EXPANSION_BLOCKS:
            key = f"{block.lower()}_offset_r{ring_idx}"
            offset = per_ring_offsets.get(key, 0.0)
            y = (k_y + offset) % img_height
            if y < 0:
                y += img_height
            rows.append({
                "Ring": ring_idx,
                "Block": block,
                "X": get_x(block),
                "Y": round(y, 1),
                "quality": quality,
            })
    return pd.DataFrame(rows, columns=["Ring", "Block", "X", "Y", "quality"])


def get_k_positions_and_height(tunnel_id: str, base_dir: str) -> tuple:
    """
    Compute K positions and image height (no GT, no offsets). For BO: call once, then use
    expand_k_with_per_ring_offsets with trial offsets.
    Returns (k_positions: DataFrame with X, Y, Confidence), img_height: int.
    """
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    depth_map_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    ring_count_path = os.path.join(tunnel_dir, "ring_count.txt")
    if not os.path.exists(depth_map_path):
        raise FileNotFoundError(f"depth_map_outlier.npy not found at {depth_map_path}. Run preprocessing first.")
    if not os.path.exists(ring_count_path):
        raise FileNotFoundError(f"ring_count.txt not found at {ring_count_path}")
    depth_map_outlier = np.load(depth_map_path)
    ring_count = int(open(ring_count_path, "r").read())
    L, W = depth_map_outlier.shape
    params, _ = load_parameters(tunnel_id, base_dir)
    preprocessing_params = load_preprocessing_params(tunnel_id, base_dir)
    tunnel_diameter = preprocessing_params.get("tunnel_diameter", 5.5)
    resolution = preprocessing_params.get("depth_map_resolution", 0.005)
    k_height_mm, ab_height_mm = calculate_segment_heights(tunnel_diameter)
    line_data = detect_lines(depth_map_outlier, params)
    method = params.get("k_detection_method", "combined")
    if method == "combined":
        k_positions = calculate_k_positions_combined(
            line_data, ring_count, k_height_mm, resolution, params
        )
    else:
        k_positions = calculate_k_positions_groove_pair(
            line_data, ring_count, k_height_mm, resolution, params
        )
    if params.get("reverse_ring_order", False):
        k_positions = k_positions.iloc[::-1].reset_index(drop=True)
    return k_positions, L


def run_geo_detection_with_offsets(
    tunnel_id: str,
    base_dir: str,
    per_ring_offsets: dict,
    k_positions: pd.DataFrame = None,
    output_file: str = "all_segments.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run detection with explicit per-ring offsets and optional K positions.
    No GT file read for offsets. For BO: pass precomputed k_positions and trial offsets.

    per_ring_offsets: dict with keys b1_offset_r0, a4_offset_r6, etc. (6 blocks x n_rings).
    k_positions: DataFrame with columns X, Y, Confidence (and optional Type). If None, K is
        computed via line detection + groove_pair/combined from parameters_detection.json.
    """
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    depth_map_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    ring_count_path = os.path.join(tunnel_dir, "ring_count.txt")
    if not os.path.exists(depth_map_path):
        raise FileNotFoundError(f"depth_map_outlier.npy not found at {depth_map_path}. Run preprocessing first.")
    if not os.path.exists(ring_count_path):
        raise FileNotFoundError(f"ring_count.txt not found at {ring_count_path}")

    depth_map_outlier = np.load(depth_map_path)
    ring_count = int(open(ring_count_path, "r").read())
    L, W = depth_map_outlier.shape

    if k_positions is None:
        params, _ = load_parameters(tunnel_id, base_dir)
        preprocessing_params = load_preprocessing_params(tunnel_id, base_dir)
        tunnel_diameter = preprocessing_params.get("tunnel_diameter", 5.5)
        resolution = preprocessing_params.get("depth_map_resolution", 0.005)
        k_height_mm, ab_height_mm = calculate_segment_heights(tunnel_diameter)
        line_data = detect_lines(depth_map_outlier, params)
        method = params.get("k_detection_method", "combined")
        if method == "combined":
            k_positions = calculate_k_positions_combined(
                line_data, ring_count, k_height_mm, resolution, params
            )
        else:
            k_positions = calculate_k_positions_groove_pair(
                line_data, ring_count, k_height_mm, resolution, params
            )
        if params.get("reverse_ring_order", False):
            k_positions = k_positions.iloc[::-1].reset_index(drop=True)
        if verbose:
            print(f"[geo_detection_with_offsets] K positions ({method}): {len(k_positions)} rings")

    all_segments = expand_k_with_per_ring_offsets(
        k_positions, L, per_ring_offsets, use_gt_x_df=None
    )
    out_path = os.path.join(tunnel_dir, output_file)
    all_segments.to_csv(out_path, index=False)
    if verbose:
        print(f"  Saved: {out_path} ({len(all_segments)} segments)")
    return all_segments


def run_geo_detection(
    tunnel_id: str,
    base_dir: str = "data",
    use_gt_k: bool = False,
    use_gt_x: bool = False,
    output_file: str = "all_segments.csv",
) -> pd.DataFrame:
    """
    Run detection: line detection + K (combined or from GT) + per-ring offset expansion.
    Offsets are always GT-derived from all_segments_gt.csv when that file exists.
    """
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    depth_map_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    ring_count_path = os.path.join(tunnel_dir, "ring_count.txt")
    if not os.path.exists(depth_map_path):
        raise FileNotFoundError(f"depth_map_outlier.npy not found at {depth_map_path}. Run preprocessing first.")
    if not os.path.exists(ring_count_path):
        raise FileNotFoundError(f"ring_count.txt not found at {ring_count_path}")

    depth_map_outlier = np.load(depth_map_path)
    ring_count = int(open(ring_count_path, "r").read())
    L, W = depth_map_outlier.shape

    params, _ = load_parameters(tunnel_id, base_dir)
    preprocessing_params = load_preprocessing_params(tunnel_id, base_dir)
    tunnel_diameter = preprocessing_params.get("tunnel_diameter", 5.5)
    resolution = preprocessing_params.get("depth_map_resolution", 0.005)
    k_height_mm, ab_height_mm = calculate_segment_heights(tunnel_diameter)

    # Step 1: Line detection
    print(f"[Step 1] Detecting lines...")
    line_data = detect_lines(depth_map_outlier, params)
    print(f"  Positive: {len(line_data['positive_lines'])}, Negative: {len(line_data['negative_lines'])}")

    # Step 2: K positions (detected or from GT)
    if use_gt_k:
        gt_path = os.path.join(tunnel_dir, "all_segments_gt.csv")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"--use-gt-k requires {gt_path}")
        gt = pd.read_csv(gt_path)
        if "ring" in gt.columns and "Ring" not in gt.columns:
            gt = gt.rename(columns={"ring": "Ring"})
        if "segment_name" in gt.columns and "Block" not in gt.columns:
            gt = gt.rename(columns={"segment_name": "Block"})
        rings_gt = sorted(gt["Ring"].unique())
        k_rows = []
        for i, ring_gt in enumerate(rings_gt):
            k_row = gt[(gt["Ring"] == ring_gt) & (gt["Block"] == "K")]
            if len(k_row) == 0:
                continue
            k_rows.append(("gt_k", float(k_row["X"].iloc[0]), float(k_row["Y"].iloc[0]), 1.0))
        k_positions = pd.DataFrame(k_rows, columns=["Type", "X", "Y", "Confidence"])
        print(f"[Step 2] Using GT K positions ({len(k_positions)} rings)")
    else:
        method = params.get("k_detection_method", "combined")
        if method == "combined":
            k_positions = calculate_k_positions_combined(
                line_data, ring_count, k_height_mm, resolution, params
            )
        else:
            k_positions = calculate_k_positions_groove_pair(
                line_data, ring_count, k_height_mm, resolution, params
            )
        if params.get("reverse_ring_order", False):
            k_positions = k_positions.iloc[::-1].reset_index(drop=True)
        print(f"[Step 2] K positions ({method}): {len(k_positions)} rings")

    # Step 3: GT-derived per-ring offsets
    per_ring_offsets, gt_segments_0based = compute_gt_derived_offsets(tunnel_dir, L)
    if not per_ring_offsets:
        raise RuntimeError("GT-derived offsets require all_segments_gt.csv in tunnel dir")
    print(f"[Step 3] Per-ring offsets: {len(per_ring_offsets)} keys (GT-derived)")

    # Step 4: Expand to all segments
    use_gt_x_df = gt_segments_0based if use_gt_x else None
    all_segments = expand_k_with_per_ring_offsets(
        k_positions, L, per_ring_offsets, use_gt_x_df=use_gt_x_df
    )
    out_path = os.path.join(tunnel_dir, output_file)
    all_segments.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(all_segments)} segments)")
    return all_segments


def main():
    parser = argparse.ArgumentParser(description="Geo detection: K + per-ring offsets -> all_segments.csv")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--use-gt-k", action="store_true", help="Use K positions from all_segments_gt.csv")
    parser.add_argument("--use-gt-x", action="store_true", help="Use per-block X from GT (implies GT-derived offsets)")
    parser.add_argument("--output", default="all_segments.csv", help="Output filename")
    args = parser.parse_args()
    run_geo_detection(
        args.tunnel_id,
        base_dir=args.data_dir,
        use_gt_k=args.use_gt_k,
        use_gt_x=args.use_gt_x,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
