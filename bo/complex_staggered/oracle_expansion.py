"""
Oracle analysis for per-ring expansion geometry on tunnel 5-1 (complex_staggered).

Goal:
- Use the existing detected K positions (from the 283px baseline run)
  and the ground-truth all_segments_gt.csv to estimate the *best possible*
  mean segment distance for several expansion models, without BO.

Models:
- Model 0 (baseline): global k_to_b_px and ab_step_px (2 params)
- Model 1 (per-ring Y offsets): global k_to_b_px, ab_step_px + per-ring Y offsets
- Model 2 (per-ring steps): per-ring k_to_b_px[r], ab_step_px[r]

This script is read-only on the data: it loads from data/ and data/bo/, writes
no outputs, and simply prints the best mean distance for each model.
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, differential_evolution


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def wrap_aware_distance(x1: float, y1: float, x2: float, y2: float, img_height: int) -> float:
    """Euclidean distance with Y wrap-around."""
    dx = x1 - x2
    dy = abs(y1 - y2)
    dy = min(dy, img_height - dy)
    return float(np.hypot(dx, dy))


def match_segments(pred_df: pd.DataFrame, gt_df: pd.DataFrame, img_height: int):
    """
    Match predicted to GT segments by nearest position per block type
    using Hungarian assignment with Y wrap-around.
    Returns:
        List of (pred_idx, gt_idx, distance),
        list of unmatched GT indices,
        list of unmatched pred indices.
    """
    all_matches: List[Tuple[int, int, float]] = []
    all_unmatched_gt: List[int] = []
    all_unmatched_pred: List[int] = []

    block_types = set(gt_df["Block"].unique()) | set(pred_df["Block"].unique())

    for block in block_types:
        gt_block = gt_df[gt_df["Block"] == block].reset_index(drop=True)
        pred_block = pred_df[pred_df["Block"] == block].reset_index(drop=True)

        n_gt, n_pred = len(gt_block), len(pred_block)
        if n_gt == 0 and n_pred == 0:
            continue
        if n_gt == 0:
            all_unmatched_pred.extend(pred_block.index.tolist())
            continue
        if n_pred == 0:
            all_unmatched_gt.extend(gt_block.index.tolist())
            continue

        cost = np.zeros((n_gt, n_pred), dtype=np.float64)
        for i in range(n_gt):
            for j in range(n_pred):
                cost[i, j] = wrap_aware_distance(
                    gt_block.loc[i, "X"],
                    gt_block.loc[i, "Y"],
                    pred_block.loc[j, "X"],
                    pred_block.loc[j, "Y"],
                    img_height,
                )

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_gt = set()
        matched_pred = set()
        for r, c in zip(row_ind, col_ind):
            d = float(cost[r, c])
            all_matches.append((int(pred_block.index[c]), int(gt_block.index[r]), d))
            matched_gt.add(r)
            matched_pred.add(c)

        for i in range(n_gt):
            if i not in matched_gt:
                all_unmatched_gt.append(int(gt_block.index[i]))
        for j in range(n_pred):
            if j not in matched_pred:
                all_unmatched_pred.append(int(pred_block.index[j]))

    return all_matches, all_unmatched_gt, all_unmatched_pred


def load_data(tunnel_id: str = "5-1") -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Load detected K positions, GT segments, and image height."""
    k_path = PROJECT_ROOT / "data" / "bo" / tunnel_id / "detected.csv"
    gt_path = PROJECT_ROOT / "data" / tunnel_id / "all_segments_gt.csv"
    depth_path = PROJECT_ROOT / "data" / tunnel_id / "depth_map_outlier.npy"

    if not k_path.exists():
        raise FileNotFoundError(f"K detections not found at {k_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT segments not found at {gt_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth map not found at {depth_path}")

    k_positions = pd.read_csv(k_path)
    gt_segments = pd.read_csv(gt_path)
    depth_map = np.load(depth_path)
    img_height = depth_map.shape[0]

    return k_positions, gt_segments, img_height


def expand_baseline_global(
    k_positions: pd.DataFrame,
    img_height: int,
    k_to_b: float,
    ab_step: float,
    walk_order=None,
) -> pd.DataFrame:
    """Baseline expansion: single global k_to_b and ab_step for all rings."""
    if walk_order is None:
        walk_order = [("K", 0), ("B1", 1), ("A1", 1), ("A2", 1), ("A3", 1), ("A4", 1), ("B2", -1)]

    forward_blocks = [(b, d) for b, d in walk_order if d >= 0]
    reverse_blocks = [(b, d) for b, d in walk_order if d == -1]

    rows = []
    for ring_idx, (_, k_row) in enumerate(k_positions.iterrows()):
        k_x = k_row["X"]
        k_y = k_row["Y"]
        quality = float(k_row.get("Confidence", 1.0))

        # Forward pass: K then downward blocks
        map_y = k_y
        for idx, (block, _direction) in enumerate(forward_blocks):
            if block == "K":
                map_y = k_y
            elif idx == 1:
                map_y = k_y + k_to_b
            else:
                map_y = map_y + ab_step

            rows.append(
                {
                    "Ring": ring_idx,
                    "Block": block,
                    "X": k_x,
                    "Y": map_y % img_height,
                    "quality": quality,
                }
            )

        # Reverse pass: upward blocks from K
        map_y = k_y
        for idx, (block, _direction) in enumerate(reverse_blocks):
            if idx == 0:
                map_y = k_y - k_to_b
            else:
                map_y = map_y - ab_step

            rows.append(
                {
                    "Ring": ring_idx,
                    "Block": block,
                    "X": k_x,
                    "Y": map_y % img_height,
                    "quality": quality,
                }
            )

    return pd.DataFrame(rows, columns=["Ring", "Block", "X", "Y", "quality"])


def expand_per_ring_offset(
    k_positions: pd.DataFrame,
    img_height: int,
    k_to_b: float,
    ab_step: float,
    offsets: List[float],
    walk_order=None,
) -> pd.DataFrame:
    """Global steps but per-ring Y-offset applied to K before expansion."""
    assert len(offsets) == len(k_positions), "offsets must match number of rings"
    k_mod = k_positions.copy()
    for idx in range(len(k_mod)):
        k_mod.loc[idx, "Y"] = (k_mod.loc[idx, "Y"] + offsets[idx]) % img_height
    return expand_baseline_global(k_mod, img_height, k_to_b, ab_step, walk_order=walk_order)


def expand_per_ring_steps(
    k_positions: pd.DataFrame,
    img_height: int,
    k_to_b_per_ring: List[float],
    ab_step_per_ring: List[float],
    walk_order=None,
) -> pd.DataFrame:
    """Per-ring k_to_b and ab_step."""
    assert len(k_to_b_per_ring) == len(k_positions)
    assert len(ab_step_per_ring) == len(k_positions)

    if walk_order is None:
        walk_order = [("K", 0), ("B1", 1), ("A1", 1), ("A2", 1), ("A3", 1), ("A4", 1), ("B2", -1)]

    forward_blocks = [(b, d) for b, d in walk_order if d >= 0]
    reverse_blocks = [(b, d) for b, d in walk_order if d == -1]

    rows = []
    for ring_idx, (_, k_row) in enumerate(k_positions.iterrows()):
        k_x = k_row["X"]
        k_y = k_row["Y"]
        quality = float(k_row.get("Confidence", 1.0))
        k_to_b = k_to_b_per_ring[ring_idx]
        ab_step = ab_step_per_ring[ring_idx]

        # Forward pass
        map_y = k_y
        for idx, (block, _direction) in enumerate(forward_blocks):
            if block == "K":
                map_y = k_y
            elif idx == 1:
                map_y = k_y + k_to_b
            else:
                map_y = map_y + ab_step

            rows.append(
                {
                    "Ring": ring_idx,
                    "Block": block,
                    "X": k_x,
                    "Y": map_y % img_height,
                    "quality": quality,
                }
            )

        # Reverse pass
        map_y = k_y
        for idx, (block, _direction) in enumerate(reverse_blocks):
            if idx == 0:
                map_y = k_y - k_to_b
            else:
                map_y = map_y - ab_step

            rows.append(
                {
                    "Ring": ring_idx,
                    "Block": block,
                    "X": k_x,
                    "Y": map_y % img_height,
                    "quality": quality,
                }
            )

    return pd.DataFrame(rows, columns=["Ring", "Block", "X", "Y", "quality"])


def evaluate_mean_distance(pred_df: pd.DataFrame, gt_df: pd.DataFrame, img_height: int) -> float:
    """Compute mean segment distance using Hungarian matching."""
    matches, _, _ = match_segments(pred_df, gt_df, img_height)
    if not matches:
        return float("inf")
    dists = [d for (_, _, d) in matches]
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Helpers for baseline global model (Model 0) - kept for reference
# ---------------------------------------------------------------------------

def random_search_model_0(k_positions, gt_segments, img_height, n_samples: int = 200) -> Dict:
    """Global k_to_b, ab_step."""
    rng = np.random.default_rng(42)
    best = {"mean_dist": float("inf"), "params": None}

    # Include baseline as a seed
    seeds = [
        (418.660427, 422.223351),
        (400.0, 400.0),
        (500.0, 450.0),
    ]
    for k_to_b, ab_step in seeds:
        pred = expand_baseline_global(k_positions, img_height, k_to_b, ab_step)
        md = evaluate_mean_distance(pred, gt_segments, img_height)
        if md < best["mean_dist"]:
            best = {"mean_dist": md, "params": {"k_to_b": k_to_b, "ab_step": ab_step}}

    for _ in range(n_samples):
        k_to_b = rng.uniform(200.0, 800.0)
        ab_step = rng.uniform(200.0, 800.0)
        pred = expand_baseline_global(k_positions, img_height, k_to_b, ab_step)
        md = evaluate_mean_distance(pred, gt_segments, img_height)
        if md < best["mean_dist"]:
            best = {"mean_dist": md, "params": {"k_to_b": k_to_b, "ab_step": ab_step}}

    return best


def derive_steps_from_gt(gt_segments: pd.DataFrame, img_height: int) -> Dict[str, List[float]]:
    """
    Derive per-ring K->B1 and mean AB step from ground truth geometry.

    Returns:
        dict with keys 'k_to_b_per_ring', 'ab_step_per_ring'
    """
    rings = sorted(gt_segments["Ring"].unique())
    k_to_b_list = []
    ab_step_list = []

    for ring in rings:
        ring_gt = gt_segments[gt_segments["Ring"] == ring].set_index("Block")
        blocks = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]
        ys = {}
        for b in blocks:
            if b in ring_gt.index:
                ys[b] = ring_gt.loc[b, "Y"]

        # K->B1
        if "K" in ys and "B1" in ys:
            dy = ys["B1"] - ys["K"]
            if dy < -img_height / 2:
                dy += img_height
            if dy > img_height / 2:
                dy -= img_height
            k_to_b_list.append(float(dy))
        else:
            k_to_b_list.append(500.0)

        # AB steps (B1->A1, A1->A2, A2->A3, A3->A4, A4->B2)
        ab_steps = []
        for i in range(len(blocks) - 1):
            b0, b1 = blocks[i], blocks[i + 1]
            if b0 in ys and b1 in ys:
                dy = ys[b1] - ys[b0]
                if dy < -img_height / 2:
                    dy += img_height
                if dy > img_height / 2:
                    dy -= img_height
                ab_steps.append(float(dy))
        if ab_steps:
            ab_step_list.append(float(np.mean(ab_steps)))
        else:
            ab_step_list.append(500.0)

    return {"k_to_b_per_ring": k_to_b_list, "ab_step_per_ring": ab_step_list}


def de_optimize_model_2(
    k_positions: pd.DataFrame,
    gt_segments: pd.DataFrame,
    img_height: int,
    maxiter: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Proper oracle for Model 2 using differential_evolution.

    Params vector x has length 2 * n_rings:
        x[0:n_rings]   = k_to_b_per_ring
        x[n_rings:2n]  = ab_step_per_ring
    """
    n_rings = len(k_positions)

    # Bounds for each parameter
    bounds = [(100.0, 900.0)] * (2 * n_rings)

    # Objective: mean segment distance
    def objective(x: np.ndarray) -> float:
        k_to_b = x[:n_rings]
        ab_step = x[n_rings:]
        pred = expand_per_ring_steps(
            k_positions,
            img_height,
            k_to_b_per_ring=k_to_b.tolist(),
            ab_step_per_ring=ab_step.tolist(),
        )
        return evaluate_mean_distance(pred, gt_segments, img_height)

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        tol=1e-6,
        seed=seed,
        polish=True,
    )

    x_best = result.x
    k_to_b_best = x_best[:n_rings].tolist()
    ab_step_best = x_best[n_rings:].tolist()

    return {
        "mean_dist": float(result.fun),
        "params": {
            "k_to_b_per_ring": k_to_b_best,
            "ab_step_per_ring": ab_step_best,
        },
    }


def main():
    tunnel_id = "5-1"
    k_positions, gt_segments, img_height = load_data(tunnel_id)

    print("Oracle analysis for per-ring expansion (tunnel 5-1)")
    print(f"  K positions: {len(k_positions)}")
    print(f"  GT segments: {len(gt_segments)}")
    print(f"  img_height: {img_height}")

    # Reference: Model 0 (global) with random search
    print("\n[Model 0] Global k_to_b_px, ab_step_px (random search, 200 samples)")
    m0 = random_search_model_0(k_positions, gt_segments, img_height, n_samples=200)
    print(f"  Best mean_dist: {m0['mean_dist']:.1f}px")
    print(f"  Params: {json.dumps(m0['params'], indent=2)}")

    # Direct GT-derived per-ring steps (upper bound for Model 2)
    print("\n[Model 2-GT] Per-ring steps derived directly from GT")
    gt_steps = derive_steps_from_gt(gt_segments, img_height)
    gt_pred = expand_per_ring_steps(
        k_positions,
        img_height,
        gt_steps["k_to_b_per_ring"],
        gt_steps["ab_step_per_ring"],
    )
    gt_md = evaluate_mean_distance(gt_pred, gt_segments, img_height)
    print(f"  Mean_dist with GT-derived steps: {gt_md:.1f}px")
    print(f"  k_to_b_per_ring: {[round(v, 1) for v in gt_steps['k_to_b_per_ring']]}")
    print(f"  ab_step_per_ring: {[round(v, 1) for v in gt_steps['ab_step_per_ring']]}")

    # Proper oracle for Model 2 using differential evolution
    print("\n[Model 2-DE] Per-ring k_to_b_px[r], ab_step_px[r] (differential_evolution)")
    m2 = de_optimize_model_2(k_positions, gt_segments, img_height, maxiter=1000, seed=42)
    print(f"  Best mean_dist: {m2['mean_dist']:.1f}px")
    print(f"  Params (per-ring k_to_b): {[round(v, 1) for v in m2['params']['k_to_b_per_ring']]}")
    print(f"  Params (per-ring ab_step): {[round(v, 1) for v in m2['params']['ab_step_per_ring']]}")


if __name__ == "__main__":
    main()

