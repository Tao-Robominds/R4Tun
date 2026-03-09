"""
Bayesian Optimization for Detection Parameters with All-Segments Objective

Optimizes detection parameters (line detection + segment expansion) to minimize
mean position error against ground truth all_segments_gt.csv.

Search space includes line detection params + k_to_b_px/ab_step_px for fixed geometry expansion.
Uses forest_minimize (Random Forest surrogate).
Detection reads depth_map_outlier.npy from preprocessing stage (no enhancing step).
"""

import os
import sys
import json
import glob
import time
import argparse
import importlib.util
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from skopt import forest_minimize
from skopt.space import Real, Integer
from scipy.optimize import linear_sum_assignment

# Add project root to path
# BO script is now in: bo/{agent_type}/
# Project root is 3 levels up
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Extract agent_type from directory name
# Script is in bo/{agent_type}/, so parent.name gives agent_type
DEFAULT_AGENT_TYPE = Path(__file__).parent.name
# Complex_staggered tunnels (4-1, 5-1) use agents/irregular/ for detection code and params
AGENT_DIR = 'irregular' if DEFAULT_AGENT_TYPE == 'complex_staggered' else DEFAULT_AGENT_TYPE

# Import detection functions
# Detection script is in: agents/{AGENT_DIR}/2_detection/
detection_dir = PROJECT_ROOT / 'agents' / AGENT_DIR / '2_detection'
sys.path.insert(0, str(detection_dir))

spec = importlib.util.spec_from_file_location(
    "detection",
    os.path.join(detection_dir, "2_detection.py")
)
detection_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detection_module)

run_detection = detection_module.run_detection
load_parameters = detection_module.load_parameters
get_param = detection_module.get_param


# =============================================================================
# Search Space Definition (13 parameters)
# =============================================================================

def get_detection_dimensions(
    tunnel_id: str = None,
    agent_type: str = DEFAULT_AGENT_TYPE
) -> Tuple[List, List[str], Dict]:
    """
    Define search space for detection parameters (detection-only, 14D).
    
    Loads per-tunnel config from bo/{agent_type}/configs/detect_{tunnel_id}.json if it exists.
    If config exists:
        - Excludes parameters listed in fixed_params from search space
        - Overrides bounds from narrowed_bounds
    If no config: returns full 14D default space.
    
    Note: Negative angles are derived symmetrically from positive angles.
    Enhancing parameters (target_distances, curvature_neighbors, depth_map_resolution, interpolation_window)
    are NOT in this search space - they belong to preprocessing BO.
    
    Args:
        tunnel_id: Tunnel identifier (e.g., '1-4')
        agent_type: Agent type (e.g., 'complex_staggered')
    
    Returns:
        Tuple of (dimensions list, parameter names list, fixed_params dict)
    """
    # Default bounds (14D base + 15D complex-specific + 14D per-ring geometry)
    default_bounds = {
        # Base detection parameters (14D)
        'binary_threshold': (80, 200),
        'dilation_kernel_size': (2, 5),
        'dilation_iterations': (1, 4),
        'hough_oblique_threshold': (20, 120),
        'hough_oblique_min_length': (40, 150),
        'hough_oblique_max_gap': (20, 80),
        'angle_min': (4.0, 7.0),
        'angle_max': (7.5, 12.0),
        'hough_vertical_threshold': (200, 800),
        'hough_horizontal_threshold': (20, 100),
        'hough_horizontal_min_length': (50, 150),
        'hough_horizontal_max_gap': (3, 30),
        'horizontal_angle_tolerance': (0.5, 3.0),
        'merge_distance_threshold': (1, 10),
        # Geometric K: ring X position (tunable when using k_detection_method=geometric)
        'ring_offset': (50.0, 400.0),
        'ring_spacing_px': (300.0, 500.0),
        # Complex-specific parameters (8D)
        'complex_hough_threshold': (10, 50),
        'complex_hough_min_length': (15, 100),
        'complex_hough_max_gap': (50, 150),
        'complex_angle_pos_min': (3.0, 6.0),
        'complex_angle_pos_max': (10.0, 15.0),
        'complex_angle_neg_min': (-15.0, -10.0),
        'complex_angle_neg_max': (-6.0, -3.0),
        'complex_min_y_span': (5, 50),
        'complex_min_x_span': (5, 50),
        'complex_eps_primary': (0.02, 0.10),
        'complex_eps_secondary': (0.05, 0.15),
        'complex_subdivision_threshold': (1.0, 2.5),
        'complex_max_subdivisions': (2, 5),
        'complex_conf_midpoint': (0.5, 0.9),
        'complex_conf_intersection': (0.7, 1.0),
        # Per-ring expansion geometry (for up to 7 rings, 14D)
        'k_to_b_r0': (100.0, 900.0),
        'k_to_b_r1': (100.0, 900.0),
        'k_to_b_r2': (100.0, 900.0),
        'k_to_b_r3': (100.0, 900.0),
        'k_to_b_r4': (100.0, 900.0),
        'k_to_b_r5': (100.0, 900.0),
        'k_to_b_r6': (100.0, 900.0),
        'ab_step_r0': (100.0, 900.0),
        'ab_step_r1': (100.0, 900.0),
        'ab_step_r2': (100.0, 900.0),
        'ab_step_r3': (100.0, 900.0),
        'ab_step_r4': (100.0, 900.0),
        'ab_step_r5': (100.0, 900.0),
        'ab_step_r6': (100.0, 900.0),
    }
    
    # Load per-tunnel config if it exists
    fixed_params = {}
    narrowed_bounds = {}
    
    if tunnel_id:
        config_file = PROJECT_ROOT / 'bo' / agent_type / 'configs' / f'detect_{tunnel_id}.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            fixed_params = config.get('fixed_params', {})
            narrowed_bounds = config.get('narrowed_bounds', {})
    
    # Build bounds dict (narrowed overrides default)
    bounds = default_bounds.copy()
    # Convert JSON arrays to tuples for compatibility
    narrowed_bounds_converted = {}
    for key, value in narrowed_bounds.items():
        if isinstance(value, list):
            narrowed_bounds_converted[key] = tuple(value)
        else:
            narrowed_bounds_converted[key] = value
    bounds.update(narrowed_bounds_converted)
    
    # Build dimensions and param_names, excluding fixed params
    dimensions = []
    param_names = []
    
    param_defs = [
        # Base detection parameters
        ('binary_threshold', Integer, bounds['binary_threshold']),
        ('dilation_kernel_size', Integer, bounds['dilation_kernel_size']),
        ('dilation_iterations', Integer, bounds['dilation_iterations']),
        ('hough_oblique_threshold', Integer, bounds['hough_oblique_threshold']),
        ('hough_oblique_min_length', Integer, bounds['hough_oblique_min_length']),
        ('hough_oblique_max_gap', Integer, bounds['hough_oblique_max_gap']),
        ('angle_min', Real, bounds['angle_min']),
        ('angle_max', Real, bounds['angle_max']),
        ('hough_vertical_threshold', Integer, bounds['hough_vertical_threshold']),
        ('hough_horizontal_threshold', Integer, bounds['hough_horizontal_threshold']),
        ('hough_horizontal_min_length', Integer, bounds['hough_horizontal_min_length']),
        ('hough_horizontal_max_gap', Integer, bounds['hough_horizontal_max_gap']),
        ('horizontal_angle_tolerance', Real, bounds['horizontal_angle_tolerance']),
        ('merge_distance_threshold', Real, bounds['merge_distance_threshold']),
        ('ring_offset', Real, bounds['ring_offset']),
        ('ring_spacing_px', Real, bounds['ring_spacing_px']),
        # Complex-specific parameters
        ('complex_hough_threshold', Integer, bounds['complex_hough_threshold']),
        ('complex_hough_min_length', Integer, bounds['complex_hough_min_length']),
        ('complex_hough_max_gap', Integer, bounds['complex_hough_max_gap']),
        ('complex_angle_pos_min', Real, bounds['complex_angle_pos_min']),
        ('complex_angle_pos_max', Real, bounds['complex_angle_pos_max']),
        ('complex_angle_neg_min', Real, bounds['complex_angle_neg_min']),
        ('complex_angle_neg_max', Real, bounds['complex_angle_neg_max']),
        ('complex_min_y_span', Integer, bounds['complex_min_y_span']),
        ('complex_min_x_span', Integer, bounds['complex_min_x_span']),
        ('complex_eps_primary', Real, bounds['complex_eps_primary']),
        ('complex_eps_secondary', Real, bounds['complex_eps_secondary']),
        ('complex_subdivision_threshold', Real, bounds['complex_subdivision_threshold']),
        ('complex_max_subdivisions', Integer, bounds['complex_max_subdivisions']),
        ('complex_conf_midpoint', Real, bounds['complex_conf_midpoint']),
        ('complex_conf_intersection', Real, bounds['complex_conf_intersection']),
        # Per-ring expansion geometry (always defined; unused rings are ignored)
        ('k_to_b_r0', Real, bounds['k_to_b_r0']),
        ('k_to_b_r1', Real, bounds['k_to_b_r1']),
        ('k_to_b_r2', Real, bounds['k_to_b_r2']),
        ('k_to_b_r3', Real, bounds['k_to_b_r3']),
        ('k_to_b_r4', Real, bounds['k_to_b_r4']),
        ('k_to_b_r5', Real, bounds['k_to_b_r5']),
        ('k_to_b_r6', Real, bounds['k_to_b_r6']),
        ('ab_step_r0', Real, bounds['ab_step_r0']),
        ('ab_step_r1', Real, bounds['ab_step_r1']),
        ('ab_step_r2', Real, bounds['ab_step_r2']),
        ('ab_step_r3', Real, bounds['ab_step_r3']),
        ('ab_step_r4', Real, bounds['ab_step_r4']),
        ('ab_step_r5', Real, bounds['ab_step_r5']),
        ('ab_step_r6', Real, bounds['ab_step_r6']),
    ]
    
    for name, param_type, (low, high) in param_defs:
        if name not in fixed_params:
            dimensions.append(param_type(low, high, name=name))
            param_names.append(name)
    
    return dimensions, param_names, fixed_params


def params_to_detection_json(params: List, param_names: List[str], fixed_params: Dict = None) -> Dict:
    """
    Convert BO parameters to detection JSON structure (detection-only, no enhancing).
    
    Derives negative angles symmetrically from positive angles.
    Merges fixed_params into output (fixed params are not in param_names/params).
    
    Args:
        params: List of BO-tuned parameter values
        param_names: List of parameter names corresponding to params
        fixed_params: Dict of fixed parameter values (not in BO search space)
    
    Returns:
        Detection parameters dict (no enhancing parameters)
    """
    if fixed_params is None:
        fixed_params = {}
    
    param_dict = dict(zip(param_names, params))
    
    # Merge fixed params into param_dict (fixed params take precedence)
    param_dict.update(fixed_params)
    
    # Derive negative angles from positive
    angle_min = param_dict['angle_min']
    angle_max = param_dict['angle_max']
    
    result = {
        # Base detection parameters
        'binary_threshold': int(param_dict['binary_threshold']),
        'dilation_kernel_size': int(param_dict['dilation_kernel_size']),
        'dilation_iterations': int(param_dict['dilation_iterations']),
        'hough_oblique_threshold': int(param_dict['hough_oblique_threshold']),
        'hough_oblique_min_length': int(param_dict['hough_oblique_min_length']),
        'hough_oblique_max_gap': int(param_dict['hough_oblique_max_gap']),
        'angle_positive_min': float(angle_min),
        'angle_positive_max': float(angle_max),
        'angle_negative_min': -float(angle_max),  # Symmetric
        'angle_negative_max': -float(angle_min),  # Symmetric
        'hough_vertical_threshold': int(param_dict['hough_vertical_threshold']),
        'hough_horizontal_threshold': int(param_dict.get('hough_horizontal_threshold', 50)),
        'hough_horizontal_min_length': int(param_dict.get('hough_horizontal_min_length', 100)),
        'hough_horizontal_max_gap': int(param_dict.get('hough_horizontal_max_gap', 10)),
        'horizontal_angle_tolerance': float(param_dict.get('horizontal_angle_tolerance', 1.0)),
        'merge_distance_threshold': float(param_dict.get('merge_distance_threshold', 3.0)),
    }
    
    # Add complex-specific parameters (flat keys)
    if 'complex_hough_threshold' in param_dict:
        result.update({
            'complex_hough_threshold': int(param_dict['complex_hough_threshold']),
            'complex_hough_min_length': int(param_dict.get('complex_hough_min_length', 50)),
            'complex_hough_max_gap': int(param_dict.get('complex_hough_max_gap', 100)),
            'complex_angle_pos_min': float(param_dict.get('complex_angle_pos_min', 4.0)),
            'complex_angle_pos_max': float(param_dict.get('complex_angle_pos_max', 12.0)),
            'complex_angle_neg_min': float(param_dict.get('complex_angle_neg_min', -12.0)),
            'complex_angle_neg_max': float(param_dict.get('complex_angle_neg_max', -4.0)),
            'complex_min_y_span': int(param_dict.get('complex_min_y_span', 30)),
            'complex_min_x_span': int(param_dict.get('complex_min_x_span', 30)),
            'complex_eps_primary': float(param_dict.get('complex_eps_primary', 0.05)),
            'complex_eps_secondary': float(param_dict.get('complex_eps_secondary', 0.10)),
            'complex_subdivision_threshold': float(param_dict.get('complex_subdivision_threshold', 1.5)),
            'complex_max_subdivisions': int(param_dict.get('complex_max_subdivisions', 4)),
            'complex_conf_midpoint': float(param_dict.get('complex_conf_midpoint', 0.7)),
            'complex_conf_intersection': float(param_dict.get('complex_conf_intersection', 0.9)),
        })
    
    # Expansion geometry
    # Global (legacy) geometry if present
    if 'k_to_b_px' in param_dict:
        result['k_to_b_px'] = float(param_dict['k_to_b_px'])
    if 'ab_step_px' in param_dict:
        result['ab_step_px'] = float(param_dict['ab_step_px'])

    # Per-ring geometry for up to 7 rings (used by expand_k_per_ring_steps)
    for ring_idx in range(7):
        key_k = f'k_to_b_r{ring_idx}'
        key_ab = f'ab_step_r{ring_idx}'
        if key_k in param_dict and key_ab in param_dict:
            result[key_k] = float(param_dict[key_k])
            result[key_ab] = float(param_dict[key_ab])
    
    # Geometric K params (for k_detection_method=geometric)
    if 'ring_offset' in param_dict:
        result['ring_offset'] = float(param_dict['ring_offset'])
    if 'ring_spacing_px' in param_dict:
        result['ring_spacing_px'] = float(param_dict['ring_spacing_px'])

    # Per-ring per-block Y offsets (expansion_method=offsets)
    for key, val in param_dict.items():
        if '_offset_r' in key:
            result[key] = float(val)

    # Pass through any fixed_params not in result (e.g. reverse_ring_order)
    for key, val in fixed_params.items():
        if key not in result:
            result[key] = val
    
    return result


# =============================================================================
# All-Segments Score Computation
# =============================================================================

def _wrap_aware_distance(x1, y1, x2, y2, img_height):
    """Euclidean distance with Y wrap-around."""
    dx = x1 - x2
    dy = abs(y1 - y2)
    dy = min(dy, img_height - dy)
    return np.sqrt(dx**2 + dy**2)


def match_segments(pred_df, gt_df, img_height):
    """
    Match predicted to GT segments by nearest position per block type
    using Hungarian assignment with Y wrap-around.
    
    Returns:
        List of (pred_idx, gt_idx, distance) for matched pairs,
        list of unmatched GT indices, list of unmatched pred indices.
    """
    all_matches = []
    all_unmatched_gt = []
    all_unmatched_pred = []
    
    block_types = set(gt_df['Block'].unique()) | set(pred_df['Block'].unique())
    
    for block in block_types:
        gt_block = gt_df[gt_df['Block'] == block].reset_index(drop=True)
        pred_block = pred_df[pred_df['Block'] == block].reset_index(drop=True)
        
        n_gt = len(gt_block)
        n_pred = len(pred_block)
        
        if n_gt == 0:
            all_unmatched_pred.extend(pred_block.index.tolist())
            continue
        if n_pred == 0:
            all_unmatched_gt.extend(gt_block.index.tolist())
            continue
        
        # Build cost matrix with wrap-around Y
        cost = np.zeros((n_gt, n_pred))
        for i in range(n_gt):
            for j in range(n_pred):
                cost[i, j] = _wrap_aware_distance(
                    gt_block.loc[i, 'X'], gt_block.loc[i, 'Y'],
                    pred_block.loc[j, 'X'], pred_block.loc[j, 'Y'],
                    img_height
                )
        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        matched_gt = set()
        matched_pred = set()
        for r, c in zip(row_ind, col_ind):
            all_matches.append((pred_block.index[c], gt_block.index[r], cost[r, c]))
            matched_gt.add(r)
            matched_pred.add(c)
        
        for i in range(n_gt):
            if i not in matched_gt:
                all_unmatched_gt.append(i)
        for j in range(n_pred):
            if j not in matched_pred:
                all_unmatched_pred.append(j)
    
    return all_matches, all_unmatched_gt, all_unmatched_pred


def compute_kposition_f1(
    detected: pd.DataFrame,
    gt: pd.DataFrame,
    threshold: float = 150.0,
    img_height: int = None,
) -> Dict:
    """
    Compute K-Position Weighted F1 score using Hungarian matching.
    
    Args:
        detected: DataFrame with 'X', 'Y' columns (detected K positions)
        gt: DataFrame with 'X', 'Y' columns (ground truth K positions)
        threshold: Distance threshold in pixels for a "correct" match
        img_height: Image height for wrap-around (if None, no wrap-around)
    
    Returns:
        Dictionary with F1, precision, recall, position_bonus, and detailed metrics
    """
    if len(detected) == 0 or len(gt) == 0:
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'position_bonus': 0.0,
            'weighted_f1': 0.0,
            'tp': 0,
            'fp': len(detected),
            'fn': len(gt),
            'matched_distances': [],
            'mean_distance': 9999.0,
        }
    
    n_gt = len(gt)
    n_det = len(detected)
    
    distances = np.zeros((n_gt, n_det))
    for i, (_, gt_row) in enumerate(gt.iterrows()):
        for j, (_, det_row) in enumerate(detected.iterrows()):
            if img_height is not None:
                distances[i, j] = _wrap_aware_distance(
                    gt_row['X'], gt_row['Y'],
                    det_row['X'], det_row['Y'],
                    img_height
                )
            else:
                dx = gt_row['X'] - det_row['X']
                dy = gt_row['Y'] - det_row['Y']
                distances[i, j] = np.sqrt(dx**2 + dy**2)
    
    row_indices, col_indices = linear_sum_assignment(distances)
    
    tp = 0
    matched_distances = []
    matched_gt_indices = set()
    matched_det_indices = set()
    
    for i, j in zip(row_indices, col_indices):
        dist = distances[i, j]
        if dist <= threshold:
            tp += 1
            matched_distances.append(dist)
            matched_gt_indices.add(i)
            matched_det_indices.add(j)
    
    fn = n_gt - len(matched_gt_indices)
    fp = n_det - len(matched_det_indices)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    if matched_distances:
        mean_distance = np.mean(matched_distances)
        position_bonus = max(0.0, 1.0 - mean_distance / threshold)
    else:
        position_bonus = 0.0
    
    weighted_f1 = f1 * (0.5 + 0.5 * position_bonus)
    
    return {
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'position_bonus': float(position_bonus),
        'weighted_f1': float(weighted_f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'matched_distances': [float(d) for d in matched_distances],
        'mean_distance': float(np.mean(matched_distances)) if matched_distances else 9999.0,
    }


def compute_all_segments_score(
    all_segments: pd.DataFrame,
    gt_segments: pd.DataFrame,
    img_height: int,
    k_f1_threshold: float = 150.0,
    max_dist: float = 500.0,
    penalty_dist: float = 500.0,
) -> Dict:
    """
    Composite objective: K-match F1 * 0.3 + position accuracy * 0.7.
    
    Matches predicted to GT segments per block type using Hungarian matching
    with wrap-around-aware distances. Unmatched segments (both GT and pred)
    receive a penalty distance.
    
    Args:
        all_segments: Predicted segments DataFrame (Ring, Block, X, Y)
        gt_segments: Ground truth segments DataFrame (Ring, Block, X, Y)
        img_height: Image height for Y wrap-around
        k_f1_threshold: Distance threshold for K-position F1
        max_dist: Normalization distance for position accuracy component
        penalty_dist: Penalty distance for unmatched segments
    
    Returns:
        Dict with composite score and detailed metrics
    """
    # K-position F1 component
    pred_k = all_segments[all_segments['Block'] == 'K'][['X', 'Y']].reset_index(drop=True)
    gt_k = gt_segments[gt_segments['Block'] == 'K'][['X', 'Y']].reset_index(drop=True)
    k_results = compute_kposition_f1(pred_k, gt_k, k_f1_threshold, img_height)
    
    # All-segments matching
    matches, unmatched_gt, unmatched_pred = match_segments(
        all_segments, gt_segments, img_height
    )
    
    matched_dists = [d for _, _, d in matches]
    penalty_dists = [penalty_dist] * (len(unmatched_gt) + len(unmatched_pred))
    all_dists = matched_dists + penalty_dists
    
    if all_dists:
        mean_dist = float(np.mean(all_dists))
    else:
        mean_dist = max_dist
    
    position_accuracy = max(0.0, 1.0 - mean_dist / max_dist)
    
    composite_score = k_results['weighted_f1'] * 0.3 + position_accuracy * 0.7
    
    return {
        'composite_score': float(composite_score),
        'k_weighted_f1': k_results['weighted_f1'],
        'k_f1': k_results['f1'],
        'k_precision': k_results['precision'],
        'k_recall': k_results['recall'],
        'k_tp': k_results['tp'],
        'k_fp': k_results['fp'],
        'k_fn': k_results['fn'],
        'k_mean_distance': k_results['mean_distance'],
        'position_accuracy': float(position_accuracy),
        'mean_segment_distance': mean_dist,
        'num_matched': len(matches),
        'num_unmatched_gt': len(unmatched_gt),
        'num_unmatched_pred': len(unmatched_pred),
        'num_pred_segments': len(all_segments),
        'num_gt_segments': len(gt_segments),
        'matched_distances': [float(d) for d in matched_dists],
    }


def fix_ring_order_to_gt(
    tunnel_dir: str,
    segments_file: str = "all_segments.csv",
    gt_file: str = "all_segments_gt.csv",
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Relabel detection ring indices so that ring i in output = the detected ring that
    was matched to GT ring at index i (by Hungarian on K positions). Use after
    detection so downstream and same-index comparisons see aligned ring identity.
    """
    seg_path = os.path.join(tunnel_dir, segments_file)
    gt_path = os.path.join(tunnel_dir, gt_file)
    if not os.path.exists(seg_path) or not os.path.exists(gt_path):
        return pd.DataFrame()
    segments = pd.read_csv(seg_path)
    if "ring" in segments.columns and "Ring" not in segments.columns:
        segments = segments.rename(columns={"ring": "Ring"})
    gt = pd.read_csv(gt_path)
    if "ring" in gt.columns and "Ring" not in gt.columns:
        gt = gt.rename(columns={"ring": "Ring"})
    block_col = "Block" if "Block" in segments.columns else "Type"
    gt_k = gt[gt["Block"] == "K"].sort_values("Ring").reset_index(drop=True)
    det_k = segments[segments[block_col] == "K"].sort_values("Ring").reset_index(drop=True)
    n_gt, n_det = len(gt_k), len(det_k)
    if n_gt == 0 or n_det == 0:
        return segments
    # Image height for wrap-aware distance
    depth_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    H = np.load(depth_path).shape[0] if os.path.exists(depth_path) else 5000
    cost = np.zeros((n_gt, n_det))
    for i in range(n_gt):
        for j in range(n_det):
            cost[i, j] = _wrap_aware_distance(
                float(gt_k.iloc[i]["X"]), float(gt_k.iloc[i]["Y"]),
                float(det_k.iloc[j]["X"]), float(det_k.iloc[j]["Y"]), H,
            )
    row_ind, col_ind = linear_sum_assignment(cost)
    # det ring index j was matched to gt at row_ind[col_ind==j]; col_ind[k] is det index for gt k
    # So for gt index i, det index is col_ind[i]. So det_to_gt[col_ind[i]] = i (0-based).
    det_to_gt_index = np.zeros(max(segments["Ring"].max() + 1, n_det), dtype=int)
    for i in range(len(row_ind)):
        det_to_gt_index[col_ind[i]] = row_ind[i]
    segments["Ring"] = segments["Ring"].map(lambda r: det_to_gt_index[int(r)] if int(r) < len(det_to_gt_index) else r)
    segments = segments.sort_values(["Ring", block_col]).reset_index(drop=True)
    out_path = os.path.join(tunnel_dir, output_file or segments_file)
    segments.to_csv(out_path, index=False)
    return segments


def compute_k_mean_distance_hungarian(
    k_positions: pd.DataFrame,
    gt_k: pd.DataFrame,
    img_height: int,
    penalty_dist: float = 500.0,
) -> Dict:
    """
    K-only objective: mean wrap-aware distance under optimal assignment (Hungarian).
    Ring-order agnostic; minimizes mean distance over best matching.
    """
    n_gt = len(gt_k)
    n_pred = len(k_positions)
    if n_gt == 0:
        return {"mean_k_distance": penalty_dist, "k_score": 0.0, "num_matched": 0, "k_distances": []}
    cost = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            cost[i, j] = _wrap_aware_distance(
                float(gt_k.iloc[i]["X"]), float(gt_k.iloc[i]["Y"]),
                float(k_positions.iloc[j]["X"]), float(k_positions.iloc[j]["Y"]), img_height,
            )
    row_ind, col_ind = linear_sum_assignment(cost)
    dists = [float(cost[r, c]) for r, c in zip(row_ind, col_ind)]
    for _ in range(n_gt - len(dists)):
        dists.append(penalty_dist)
    mean_k_distance = float(np.mean(dists))
    max_dist = 500.0
    k_score = max(0.0, 1.0 - mean_k_distance / max_dist)
    return {
        "mean_k_distance": mean_k_distance,
        "k_score": k_score,
        "num_matched": len(dists),
        "k_distances": dists[:n_gt],
    }


def compute_k_centroid_score(
    k_positions: pd.DataFrame,
    gt_k: pd.DataFrame,
    img_height: int,
    penalty_dist: float = 500.0,
) -> Dict:
    """
    K-only objective: mean wrap-aware distance between detected and GT K (by ring order).
    Both are assumed to be 7 rows, ring 0..6 in same order. Missing K get penalty_dist.
    """
    n_gt = len(gt_k)
    n_pred = len(k_positions)
    if n_gt == 0:
        return {'mean_k_distance': penalty_dist, 'k_score': 0.0, 'num_matched': 0}
    dists = []
    for i in range(min(n_gt, n_pred)):
        gx, gy = gt_k.iloc[i]['X'], gt_k.iloc[i]['Y']
        dx, dy = k_positions.iloc[i]['X'], k_positions.iloc[i]['Y']
        d = _wrap_aware_distance(gx, gy, dx, dy, img_height)
        dists.append(d)
    for _ in range(n_gt - n_pred):
        dists.append(penalty_dist)
    mean_k_distance = float(np.mean(dists))
    max_dist = 500.0
    k_score = max(0.0, 1.0 - mean_k_distance / max_dist)  # higher is better, for logging
    return {
        'mean_k_distance': mean_k_distance,
        'k_score': k_score,
        'num_matched': min(n_gt, n_pred),
        'k_distances': dists[:n_gt],
    }


# Offset block names (must match detection.OFFSET_BLOCKS)
SEGMENTS_RING_BLOCKS = ['B1', 'B2', 'A1', 'A2', 'A3', 'A4']


def _wrap_aware_y_distance(y1: float, y2: float, img_height: int) -> float:
    """Y-only wrap-aware distance (for spacing penalty)."""
    dy = abs(y1 - y2)
    return min(dy, img_height - dy)


def compute_segments_ring_score(
    all_segments: pd.DataFrame,
    gt_segments: pd.DataFrame,
    img_height: int,
    ring_index: int,
    min_spacing_px: float = 147.0,
    spacing_penalty_weight: float = 2.0,
) -> Dict:
    """
    Ring-specific objective: mean wrap-aware distance for the 6 non-K blocks
    in the ring vs GT, plus soft spacing penalty for block overlap.

    Penalty: for all 21 pairs among 7 blocks in the ring, if wrap-aware Y distance
    < min_spacing_px, add weight * (min_spacing_px - distance) to the score.
    """
    pred_ring = all_segments[all_segments['Ring'] == ring_index].copy()
    gt_ring = gt_segments[gt_segments['Ring'] == ring_index].copy()
    if len(pred_ring) == 0 or len(gt_ring) == 0:
        return {
            'mean_distance': 9999.0,
            'spacing_penalty': 0.0,
            'total_score': 9999.0,
            'num_matched': 0,
        }

    # Mean distance for 6 non-K blocks (match by block name)
    dists = []
    for block in SEGMENTS_RING_BLOCKS:
        p = pred_ring[pred_ring['Block'] == block]
        g = gt_ring[gt_ring['Block'] == block]
        if len(p) and len(g):
            d = _wrap_aware_distance(
                float(p.iloc[0]['X']), float(p.iloc[0]['Y']),
                float(g.iloc[0]['X']), float(g.iloc[0]['Y']),
                img_height,
            )
            dists.append(d)
    mean_distance = float(np.mean(dists)) if dists else 9999.0

    # Spacing penalty: 7 block Y positions (K + 6 others)
    ys = []
    for block in ['K'] + SEGMENTS_RING_BLOCKS:
        row = pred_ring[pred_ring['Block'] == block]
        if len(row):
            ys.append(float(row.iloc[0]['Y']))
    if len(ys) < 7:
        spacing_penalty = 1000.0  # missing blocks -> large penalty
    else:
        violation_sum = 0.0
        for i in range(7):
            for j in range(i + 1, 7):
                d = _wrap_aware_y_distance(ys[i], ys[j], img_height)
                if d < min_spacing_px:
                    violation_sum += (min_spacing_px - d)
        spacing_penalty = spacing_penalty_weight * violation_sum

    total_score = mean_distance + spacing_penalty
    return {
        'mean_distance': mean_distance,
        'spacing_penalty': spacing_penalty,
        'total_score': total_score,
        'num_matched': len(dists),
    }


# =============================================================================
# Objective Function
# =============================================================================

class DetectionObjective:
    """
    Objective function that evaluates detection parameters using composite
    all-segments score (K-match F1 * 0.3 + position accuracy * 0.7).
    Detection reads depth_map_outlier.npy from preprocessing (no enhancing step).
    """
    
    def __init__(
        self,
        tunnel_id: str,
        data_dir: str = 'data',
        verbose: bool = True,
        eval_offset: int = 0,
        agent_type: str = DEFAULT_AGENT_TYPE,
        objective_type: str = 'composite',
        ring_index: Optional[int] = None,
        min_spacing_px: float = 147.0,
        spacing_penalty_weight: float = 2.0,
    ):
        self.tunnel_id = tunnel_id
        self.data_dir = data_dir
        self.verbose = verbose
        self.agent_type = agent_type
        self.objective_type = objective_type  # 'composite', 'k_only', or 'segments_ring'
        self.ring_index = ring_index
        self.min_spacing_px = min_spacing_px
        self.spacing_penalty_weight = spacing_penalty_weight

        self.tunnel_dir = os.path.join(data_dir, tunnel_id)
        agent_dir = AGENT_DIR if agent_type == 'complex_staggered' else agent_type
        self.params_dir = os.path.join(
            PROJECT_ROOT,
            'agents', agent_dir, '2_detection',
            'parameters', tunnel_id
        )
        os.makedirs(self.params_dir, exist_ok=True)
        
        # Load ground truth from all_segments_gt.csv
        gt_file = os.path.join(self.tunnel_dir, 'all_segments_gt.csv')
        if not os.path.exists(gt_file):
            raise FileNotFoundError(
                f"all_segments_gt.csv not found: {gt_file}. "
                f"Create it by copying the ground-truth all_segments.csv."
            )
        
        self.gt_segments = pd.read_csv(gt_file)
        required_cols = {'Ring', 'Block', 'X', 'Y'}
        if not required_cols.issubset(self.gt_segments.columns):
            raise ValueError(f"all_segments_gt.csv missing columns: {required_cols - set(self.gt_segments.columns)}")
        
        # Load image height from depth map for wrap-around calculations
        depth_map_file = os.path.join(self.tunnel_dir, 'depth_map_outlier.npy')
        if not os.path.exists(depth_map_file):
            raise FileNotFoundError(
                f"depth_map_outlier.npy not found at {depth_map_file}. "
                f"Run preprocessing first to generate depth maps."
            )
        depth_map = np.load(depth_map_file)
        self.img_height = depth_map.shape[0]
        
        # Get search space (loads config if exists)
        self.dimensions, self.param_names, self.fixed_params = get_detection_dimensions(
            tunnel_id=tunnel_id,
            agent_type=agent_type
        )
        
        # Tracking
        self.eval_offset = eval_offset
        self.eval_count = 0
        self.best_score = np.inf if objective_type in ('k_only', 'segments_ring') else -np.inf  # minimize distance for k_only and segments_ring
        self.best_params = None
        self.history = []
        self.logs_dir = os.path.join(
            PROJECT_ROOT,
            'bo', agent_type, 'logs'
        )
        os.makedirs(self.logs_dir, exist_ok=True)

        self.gt_k = self.gt_segments[self.gt_segments['Block'] == 'K'].sort_values('Ring').reset_index(drop=True) if objective_type == 'k_only' else None

        if verbose:
            gt_blocks = self.gt_segments['Block'].value_counts().to_dict()
            print(f"Detection BO for tunnel {tunnel_id}")
            print(f"GT segments: {len(self.gt_segments)} (from all_segments_gt.csv)")
            print(f"GT block counts: {gt_blocks}")
            print(f"Image height: {self.img_height}px (for wrap-around)")
            print(f"Parameters: {len(self.param_names)} (fixed: {len(self.fixed_params)})")
            if self.fixed_params:
                print(f"Fixed parameters: {list(self.fixed_params.keys())}")
            print(f"Eval numbering starts at: {self.eval_offset + 1}")
            print(f"Logs directory: {self.logs_dir}")
            obj_desc = objective_type
            if objective_type == 'k_only':
                obj_desc = 'mean K distance (minimize)'
            elif objective_type == 'segments_ring':
                obj_desc = f'ring {ring_index} segment Y distance + spacing penalty (minimize)'
            else:
                obj_desc = 'composite = K_F1*0.3 + position_accuracy*0.7'
            print(f"Objective: {objective_type} (={obj_desc})")
    
    @property
    def global_eval_index(self) -> int:
        """Current global eval index (offset + local count)."""
        return self.eval_offset + self.eval_count
    
    def __call__(self, params: List) -> float:
        """
        Evaluate detection parameters using composite all-segments score.
        
        Args:
            params: List of parameter values in order of param_names
        
        Returns:
            Negative composite score (for minimization)
        """
        self.eval_count += 1
        start_time = time.time()
        
        try:
            param_dict = params_to_detection_json(params, self.param_names, self.fixed_params)
            
            params_file = os.path.join(self.params_dir, 'parameters_detection.json')
            with open(params_file, 'w') as f:
                json.dump(param_dict, f, indent=4)
            
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                detection_result = run_detection(self.tunnel_id, self.data_dir)
            
            # Unpack tuple return from updated run_detection()
            if isinstance(detection_result, tuple):
                k_positions, all_segments = detection_result
            else:
                k_positions = detection_result
                all_segments = k_positions.copy()
                all_segments['Block'] = 'K'

            runtime = time.time() - start_time

            if self.objective_type == 'k_only':
                # Relabel detection rings to GT order so ring identity is consistent
                output_filename = param_dict.get('output_filename', 'all_segments.csv')
                fix_ring_order_to_gt(
                    self.tunnel_dir,
                    segments_file=output_filename,
                    gt_file='all_segments_gt.csv',
                    output_file=output_filename,
                )
                # Use Hungarian mean distance (ring-order agnostic)
                k_results = compute_k_mean_distance_hungarian(
                    k_positions, self.gt_k, self.img_height
                )
                score = k_results['mean_k_distance']  # minimize
                results = {
                    'mean_k_distance': score,
                    'k_score': k_results['k_score'],
                    'num_matched': k_results['num_matched'],
                    'num_gt_segments': len(self.gt_k),
                    'num_pred_segments': len(k_positions),
                    'k_distances': k_results.get('k_distances', []),
                }
                if score < self.best_score:
                    self.best_score = score
                    self.best_params = param_dict.copy()
                    if self.verbose:
                        print(f"  [Eval {self.global_eval_index}] New best: mean_K_dist={score:.1f}px "
                              f"(k_score={k_results['k_score']:.3f})")
                self._log_trial(param_dict, results, len(k_positions), runtime, objective_type='k_only')
                self.history.append({
                    'eval': self.global_eval_index,
                    'params': param_dict,
                    'mean_k_distance': score,
                    'k_score': k_results['k_score'],
                })
                if self.verbose and self.eval_count % 10 == 0:
                    print(f"  [Eval {self.global_eval_index}] mean_K_dist={score:.1f}px")
                return score  # minimize mean K distance
            elif self.objective_type == 'segments_ring':
                ring_results = compute_segments_ring_score(
                    all_segments,
                    self.gt_segments,
                    self.img_height,
                    self.ring_index,
                    min_spacing_px=self.min_spacing_px,
                    spacing_penalty_weight=self.spacing_penalty_weight,
                )
                score = ring_results['total_score']  # minimize
                results = {
                    'mean_distance': ring_results['mean_distance'],
                    'spacing_penalty': ring_results['spacing_penalty'],
                    'total_score': score,
                    'num_matched': ring_results['num_matched'],
                }
                if score < self.best_score:
                    self.best_score = score
                    self.best_params = param_dict.copy()
                    if self.verbose:
                        print(f"  [Eval {self.global_eval_index}] New best: total={score:.1f}px "
                              f"(mean_dist={ring_results['mean_distance']:.1f}, penalty={ring_results['spacing_penalty']:.1f})")
                self._log_trial(param_dict, results, len(all_segments), runtime, objective_type='segments_ring')
                self.history.append({
                    'eval': self.global_eval_index,
                    'params': param_dict,
                    'total_score': score,
                    'mean_distance': ring_results['mean_distance'],
                    'spacing_penalty': ring_results['spacing_penalty'],
                })
                if self.verbose and self.eval_count % 10 == 0:
                    print(f"  [Eval {self.global_eval_index}] total={score:.1f}px, mean_dist={ring_results['mean_distance']:.1f}px")
                return score  # minimize
            else:
                results = compute_all_segments_score(
                    all_segments, self.gt_segments, self.img_height
                )
                score = results['composite_score']
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = param_dict.copy()
                    if self.verbose:
                        print(f"  [Eval {self.global_eval_index}] New best: {score:.4f} "
                              f"(K_F1={results['k_weighted_f1']:.3f}, pos_acc={results['position_accuracy']:.3f}, "
                              f"mean_dist={results['mean_segment_distance']:.1f}px, "
                              f"matched={results['num_matched']}/{results['num_gt_segments']})")
                self._log_trial(param_dict, results, len(all_segments), runtime)
                self.history.append({
                    'eval': self.global_eval_index,
                    'params': param_dict,
                    'composite_score': score,
                    'k_weighted_f1': results['k_weighted_f1'],
                    'position_accuracy': results['position_accuracy'],
                    'mean_segment_distance': results['mean_segment_distance'],
                    'num_matched': results['num_matched'],
                    'num_pred_segments': results['num_pred_segments'],
                })
                if self.verbose and self.eval_count % 10 == 0:
                    print(f"  [Eval {self.global_eval_index}] score={score:.4f}, "
                          f"mean_dist={results['mean_segment_distance']:.1f}px, "
                          f"matched={results['num_matched']}/{results['num_gt_segments']}")
                return -score  # Negative for minimization
            
        except Exception as e:
            runtime = time.time() - start_time
            if self.verbose:
                print(f"  [Eval {self.global_eval_index}] Error: {e}")
            self._log_trial(
                params_to_detection_json(params, self.param_names, self.fixed_params),
                None, 0, runtime, error=str(e),
                objective_type=self.objective_type,
            )
            return 9999.0 if self.objective_type in ('k_only', 'segments_ring') else 0.0
    
    def _log_trial(
        self,
        params: Dict,
        results: Optional[Dict],
        num_detected: int,
        runtime: float,
        error: Optional[str] = None,
        objective_type: str = 'composite',
    ):
        """Log trial to JSON file."""
        global_idx = self.global_eval_index
        trial_id = f"detect_{self.tunnel_id}_{global_idx:03d}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        log_data = {
            'schema_version': 'r4tun.detection.v2',
            'trial': {
                'trial_id': trial_id,
                'timestamp_utc': timestamp,
                'tunnel_id': self.tunnel_id,
                'assembly_type': self.agent_type,
            },
            'params': params,
        }

        if error:
            log_data['trace'] = {'warnings': [f"Error: {error}"]}
            obj_val = 0.0 if objective_type == 'composite' else 9999.0
            log_data['bo'] = {
                'objective_name': objective_type,
                'objective_value': obj_val,
                'eval_index': global_idx,
                'runtime_sec': runtime,
                'is_feasible': False,
            }
        elif objective_type == 'k_only' and results:
            log_data['outputs'] = {
                'num_pred_segments': results['num_pred_segments'],
                'num_gt_segments': results['num_gt_segments'],
                'num_matched': results['num_matched'],
                'k_metrics': {
                    'mean_k_distance_px': results['mean_k_distance'],
                    'k_score': results['k_score'],
                    'k_distances_px': results.get('k_distances', []),
                },
            }
            log_data['bo'] = {
                'objective_name': 'k_centroid_mean_distance',
                'objective_value': float(results['mean_k_distance']),
                'eval_index': global_idx,
                'runtime_sec': float(runtime),
                'is_feasible': True,
            }
        elif objective_type == 'segments_ring' and results:
            log_data['outputs'] = {
                'num_matched': results['num_matched'],
                'ring_metrics': {
                    'mean_distance_px': results['mean_distance'],
                    'spacing_penalty_px': results['spacing_penalty'],
                    'total_score_px': results['total_score'],
                },
            }
            log_data['bo'] = {
                'objective_name': 'segments_ring_total_score',
                'objective_value': float(results['total_score']),
                'eval_index': global_idx,
                'runtime_sec': float(runtime),
                'is_feasible': True,
            }
        else:
            log_data['outputs'] = {
                'num_pred_segments': results['num_pred_segments'],
                'num_gt_segments': results['num_gt_segments'],
                'num_matched': results['num_matched'],
                'num_unmatched_gt': results['num_unmatched_gt'],
                'num_unmatched_pred': results['num_unmatched_pred'],
                'k_metrics': {
                    'weighted_f1': results['k_weighted_f1'],
                    'f1': results['k_f1'],
                    'precision': results['k_precision'],
                    'recall': results['k_recall'],
                    'tp': results['k_tp'],
                    'fp': results['k_fp'],
                    'fn': results['k_fn'],
                    'mean_distance_px': results['k_mean_distance'],
                },
                'segment_metrics': {
                    'composite_score': results['composite_score'],
                    'position_accuracy': results['position_accuracy'],
                    'mean_segment_distance_px': results['mean_segment_distance'],
                },
                'matched_distances_px': results['matched_distances'],
            }
            log_data['bo'] = {
                'objective_name': 'all_segments_composite',
                'objective_value': float(results['composite_score']),
                'eval_index': global_idx,
                'runtime_sec': float(runtime),
                'is_feasible': True,
            }
        
        log_file = os.path.join(self.logs_dir, f"{trial_id}.json")
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def save_best_params(self) -> Optional[str]:
        """Save best parameters to JSON file."""
        if self.best_params is None:
            return None
        
        params_file = os.path.join(self.params_dir, 'parameters_detection.json')
        with open(params_file, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        return params_file


# =============================================================================
# Utilities
# =============================================================================

def find_max_trial_index(logs_dir: str, tunnel_id: str) -> int:
    """Find the highest trial index from existing log files."""
    pattern = os.path.join(logs_dir, f"detect_{tunnel_id}_*.json")
    log_files = glob.glob(pattern)
    max_idx = 0
    for f in log_files:
        basename = os.path.basename(f)
        # e.g. detect_1-4_035.json -> 035
        try:
            idx = int(basename.split('_')[-1].replace('.json', ''))
            max_idx = max(max_idx, idx)
        except ValueError:
            pass
    return max_idx


def load_best_from_logs(
    logs_dir: str,
    tunnel_id: str,
    agent_type: str = DEFAULT_AGENT_TYPE,
    fixed_params: Dict = None,
    objective_type: str = 'composite',
) -> Optional[Tuple[List[float], float]]:
    """
    Load the best trial from existing logs to use as warm-start x0/y0.
    Falls back to current parameters_detection.json if no logs exist.
    For composite: maximize objective_value. For k_only: minimize objective_value.
    """
    if fixed_params is None:
        fixed_params = {}

    pattern = os.path.join(logs_dir, f"detect_{tunnel_id}_*.json")
    log_files = glob.glob(pattern)

    minimize_obj = objective_type == 'k_only'
    target_name = 'k_centroid_mean_distance' if minimize_obj else 'all_segments_composite'
    best_f1 = float('inf') if minimize_obj else -1
    best_params = None

    # First, try to load from previous BO logs (same objective type)
    for log_file in log_files:
        with open(log_file, 'r') as f:
            data = json.load(f)
        if 'bo' not in data or 'objective_value' not in data['bo']:
            continue
        obj_name = data['bo'].get('objective_name', '')
        if target_name and obj_name != target_name:
            continue
        f1 = data['bo']['objective_value']
        if minimize_obj:
            if f1 < best_f1:
                best_f1 = f1
                best_params = data.get('params', {})
        else:
            if f1 > best_f1:
                best_f1 = f1
                best_params = data.get('params', {})
    
    # If no logs found, try to load from current parameters file
    if best_params is None or best_f1 <= 0:
        agent_dir = AGENT_DIR if agent_type == 'complex_staggered' else agent_type
        params_file = os.path.join(
            PROJECT_ROOT,
            'agents', agent_dir, '2_detection',
            'parameters', tunnel_id, 'parameters_detection.json'
        )
        
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                best_params = json.load(f)
            best_f1 = 58.0 if minimize_obj else 0.5  # k_only: current baseline ~58px
            print(f"  No previous BO logs found, using current parameters_detection.json as warm-start")
    if best_params is None:
        return None
    
    # Build param list in dimension order, EXCLUDING fixed params
    # Map from full param names to extraction logic (detection-only, no enhancing)
    param_extractors = {
        # Base detection parameters
        'binary_threshold': lambda p: p.get('binary_threshold', 127),
        'dilation_kernel_size': lambda p: p.get('dilation_kernel_size', 3),
        'dilation_iterations': lambda p: p.get('dilation_iterations', 1),
        'hough_oblique_threshold': lambda p: p.get('hough_oblique_threshold', 50),
        'hough_oblique_min_length': lambda p: p.get('hough_oblique_min_length', 100),
        'hough_oblique_max_gap': lambda p: p.get('hough_oblique_max_gap', 40),
        'angle_min': lambda p: p.get('angle_positive_min', 6.0),
        'angle_max': lambda p: p.get('angle_positive_max', 9.0),
        'hough_vertical_threshold': lambda p: p.get('hough_vertical_threshold', 500),
        'hough_horizontal_threshold': lambda p: p.get('hough_horizontal_threshold', 50),
        'hough_horizontal_min_length': lambda p: p.get('hough_horizontal_min_length', 100),
        'hough_horizontal_max_gap': lambda p: p.get('hough_horizontal_max_gap', 10),
        'horizontal_angle_tolerance': lambda p: p.get('horizontal_angle_tolerance', 1.0),
        'merge_distance_threshold': lambda p: p.get('merge_distance_threshold', 3.0),
        # Complex-specific parameters
        'complex_hough_threshold': lambda p: p.get('complex_hough_threshold', 30),
        'complex_hough_min_length': lambda p: p.get('complex_hough_min_length', 50),
        'complex_hough_max_gap': lambda p: p.get('complex_hough_max_gap', 100),
        'complex_angle_pos_min': lambda p: p.get('complex_angle_pos_min', 4.0),
        'complex_angle_pos_max': lambda p: p.get('complex_angle_pos_max', 12.0),
        'complex_angle_neg_min': lambda p: p.get('complex_angle_neg_min', -12.0),
        'complex_angle_neg_max': lambda p: p.get('complex_angle_neg_max', -4.0),
        'complex_min_y_span': lambda p: p.get('complex_min_y_span', 30),
        'complex_min_x_span': lambda p: p.get('complex_min_x_span', 30),
        'complex_eps_primary': lambda p: p.get('complex_eps_primary', 0.05),
        'complex_eps_secondary': lambda p: p.get('complex_eps_secondary', 0.10),
        'complex_subdivision_threshold': lambda p: p.get('complex_subdivision_threshold', 1.5),
        'complex_max_subdivisions': lambda p: p.get('complex_max_subdivisions', 4),
        'complex_conf_midpoint': lambda p: p.get('complex_conf_midpoint', 0.7),
        'complex_conf_intersection': lambda p: p.get('complex_conf_intersection', 0.9),
        # Fixed expansion geometry
        'k_to_b_px': lambda p: p.get('k_to_b_px', 500.0),
        'ab_step_px': lambda p: p.get('ab_step_px', 500.0),
        # Geometric K
        'ring_offset': lambda p: p.get('ring_offset', 182.0),
        'ring_spacing_px': lambda p: p.get('ring_spacing_px', 364.0),
    }
    
    # Get current search space param names (to know what to include)
    _, param_names, _ = get_detection_dimensions(tunnel_id=tunnel_id, agent_type=agent_type)
    
    # Build param_values list only for params in current search space
    param_values = []
    for param_name in param_names:
        if param_name in param_extractors:
            param_values.append(param_extractors[param_name](best_params))
        else:
            # Fallback: use value from saved params if present (e.g. k_to_b_r0, per-ring offsets)
            param_values.append(best_params.get(param_name, 0.0))
    
    # For composite we minimize -score; for k_only we minimize mean distance
    y0 = best_f1 if minimize_obj else -best_f1
    return param_values, y0


# =============================================================================
# Main Optimization
# =============================================================================

def run_detection_bo(
    tunnel_id: str,
    data_dir: str = 'data',
    n_calls: int = 80,
    n_initial_points: int = 15,
    verbose: bool = True,
    agent_type: str = DEFAULT_AGENT_TYPE,
    objective_type: str = 'composite',
    ring_index: Optional[int] = None,
) -> Dict:
    """Run Bayesian Optimization for detection parameters."""
    if objective_type == 'segments_ring' and ring_index is None:
        raise ValueError("--objective segments_ring requires --ring-index N (0-6)")

    print(f"\n{'='*70}")
    print(f"DETECTION BAYESIAN OPTIMIZATION - Tunnel {tunnel_id} ({agent_type})")
    print(f"{'='*70}")

    logs_dir = os.path.join(
        PROJECT_ROOT,
        'bo', agent_type, 'logs'
    )
    os.makedirs(logs_dir, exist_ok=True)

    # Determine eval offset from existing logs
    eval_offset = find_max_trial_index(logs_dir, tunnel_id)

    # Load config for spacing penalty if segments_ring
    min_spacing_px = 147.0
    spacing_penalty_weight = 2.0
    if tunnel_id and objective_type == 'segments_ring':
        config_file = PROJECT_ROOT / 'bo' / agent_type / 'configs' / f'detect_{tunnel_id}.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            min_spacing_px = float(config.get('min_spacing_px', min_spacing_px))
            spacing_penalty_weight = float(config.get('spacing_penalty_weight', spacing_penalty_weight))

    # Initialize objective
    objective = DetectionObjective(
        tunnel_id=tunnel_id,
        data_dir=data_dir,
        verbose=verbose,
        eval_offset=eval_offset,
        agent_type=agent_type,
        objective_type=objective_type,
        ring_index=ring_index,
        min_spacing_px=min_spacing_px,
        spacing_penalty_weight=spacing_penalty_weight,
    )

    # K-only: minimize mean K distance to GT. If tunnel uses geometric K, tune only ring_offset/ring_spacing_px.
    # If tunnel uses groove_pair/combined, keep it and tune full detection params (line + groove) to get K close to GT first.
    if objective_type == 'k_only':
        agent_dir = AGENT_DIR if agent_type == 'complex_staggered' else agent_type
        params_file = os.path.join(
            PROJECT_ROOT,
            'agents', agent_dir, '2_detection',
            'parameters', tunnel_id, 'parameters_detection.json'
        )
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                current_params = json.load(f)
            objective.fixed_params = {**current_params, **objective.fixed_params}
            k_method = current_params.get('k_detection_method', 'complex_staggered')
            if k_method == 'geometric':
                objective.fixed_params['k_detection_method'] = 'geometric'
                objective.fixed_params['reverse_ring_order'] = True
                objective.param_names = ['ring_offset', 'ring_spacing_px']
                objective.dimensions = [
                    Real(50.0, 400.0, name='ring_offset'),
                    Real(300.0, 500.0, name='ring_spacing_px'),
                ]
                objective.fixed_params.pop('ring_offset', None)
                objective.fixed_params.pop('ring_spacing_px', None)
            else:
                # groove_pair or combined: tune full detection params (do not override to geometric)
                pass  # use dimensions/param_names from get_detection_dimensions already set in objective

    # Segments-ring: 6D search space (per-block Y offsets for one ring); fix all other params from current file
    if objective_type == 'segments_ring':
        agent_dir = AGENT_DIR if agent_type == 'complex_staggered' else agent_type
        params_file = os.path.join(
            PROJECT_ROOT,
            'agents', agent_dir, '2_detection',
            'parameters', tunnel_id, 'parameters_detection.json'
        )
        if not os.path.exists(params_file):
            raise FileNotFoundError(
                f"parameters_detection.json not found: {params_file}. "
                "Run with geometric K and expansion_method=offsets first (e.g. from config)."
            )
        with open(params_file, 'r') as f:
            current_params = json.load(f)
        objective.fixed_params = {**current_params, **objective.fixed_params}
        objective.fixed_params['k_detection_method'] = 'geometric'
        objective.fixed_params['expansion_method'] = 'offsets'
        # Remove the 6 offset keys for this ring so BO tunes them
        for block in SEGMENTS_RING_BLOCKS:
            key = f"{block.lower()}_offset_r{ring_index}"
            objective.fixed_params.pop(key, None)
        objective.param_names = [f"{b.lower()}_offset_r{ring_index}" for b in SEGMENTS_RING_BLOCKS]
        objective.dimensions = [
            Real(-2400.0, 2400.0, name=name) for name in objective.param_names
        ]

    print(f"\nSearch space: {len(objective.param_names)} parameters")
    print(f"N calls: {n_calls}, N initial: {n_initial_points}")
    obj_label = 'mean K distance' if objective_type == 'k_only' else (
        f'ring {ring_index} segment Y + spacing penalty' if objective_type == 'segments_ring' else 'all-segments composite'
    )
    print(f"Objective: {objective_type} (={obj_label})")
    print(f"Algorithm: forest_minimize (Random Forest surrogate)")
    
    # Warm-start from best previous trial or current parameters file
    x0 = None
    y0 = None
    if objective_type == 'segments_ring':
        warm_start = None  # use x0/y0 from GT below if available
        # Warm-start from GT offsets for this ring
        gt_ring = objective.gt_segments[objective.gt_segments['Ring'] == ring_index]
        gt_k_row = gt_ring[gt_ring['Block'] == 'K']
        if len(gt_k_row):
            k_y = float(gt_k_row.iloc[0]['Y'])
            x0_vals = []
            for block in SEGMENTS_RING_BLOCKS:
                br = gt_ring[gt_ring['Block'] == block]
                if len(br):
                    by = float(br.iloc[0]['Y'])
                    offset = (by - k_y) % objective.img_height
                    if offset > objective.img_height / 2:
                        offset -= objective.img_height
                    offset = max(-2400, min(2400, offset))
                    x0_vals.append(offset)
                else:
                    x0_vals.append(0.0)
            if len(x0_vals) == 6:
                x0 = [x0_vals]
                # Initial score unknown; use a moderate value so BO can improve
                y0 = [500.0]
                print(f"\nWarm-starting from GT offsets for ring {ring_index}:")
                for name, val in zip(objective.param_names, x0_vals):
                    print(f"  {name}: {val:.1f}")
    elif objective_type == 'k_only':
        if objective.param_names == ['ring_offset', 'ring_spacing_px']:
            # Geometric K: warm-start from current params
            agent_dir = AGENT_DIR if agent_type == 'complex_staggered' else agent_type
            params_file = os.path.join(
                PROJECT_ROOT, 'agents', agent_dir, '2_detection',
                'parameters', tunnel_id, 'parameters_detection.json'
            )
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    cp = json.load(f)
                x0_vals = [float(cp.get('ring_offset', 182)), float(cp.get('ring_spacing_px', 364))]
                clamped_x0 = [
                    max(50, min(400, x0_vals[0])),
                    max(300, min(500, x0_vals[1])),
                ]
                x0 = [clamped_x0]
                y0 = [58.0]  # baseline mean K distance
                print(f"\nWarm-starting from current parameters (mean_K_dist~58px):")
                print(f"  ring_offset: {x0_vals[0]:.1f}, ring_spacing_px: {x0_vals[1]:.1f}")
        else:
            # Groove_pair/combined: warm-start from best previous K-only trial or current params
            warm_start = load_best_from_logs(
                logs_dir, tunnel_id, agent_type, objective.fixed_params, objective_type='k_only'
            )
            if warm_start is not None:
                x0_vals, y0_val = warm_start
                clamped_x0 = []
                for i, (name, val) in enumerate(zip(objective.param_names, x0_vals)):
                    dim = objective.dimensions[i]
                    if hasattr(dim, 'low') and hasattr(dim, 'high'):
                        clamped_x0.append(max(dim.low, min(dim.high, val)))
                    else:
                        clamped_x0.append(val)
                x0 = [clamped_x0]
                y0 = [y0_val]
                print(f"\nWarm-starting from previous K-only BO (mean_K_dist={y0_val:.1f}px)")
    else:
        # composite
        warm_start = load_best_from_logs(
            logs_dir, tunnel_id, agent_type, objective.fixed_params, objective_type=objective_type
        )
    if warm_start is not None:
        x0_vals, y0_val = warm_start
        clamped_x0 = []
        for i, (name, val) in enumerate(zip(objective.param_names, x0_vals)):
            dim = objective.dimensions[i]
            if hasattr(dim, 'low') and hasattr(dim, 'high'):
                clamped_val = max(dim.low, min(dim.high, val))
                clamped_x0.append(clamped_val)
            else:
                clamped_x0.append(val)
        x0 = [clamped_x0]
        y0 = [y0_val]
        source = "previous BO logs" if eval_offset > 0 else "current parameters_detection.json"
        print(f"\nWarm-starting from {source} (estimated F1={-y0_val:.4f}):")
        for name, val, clamped in zip(objective.param_names, x0_vals, clamped_x0):
            if abs(val - clamped) > 1e-6:
                print(f"  {name}: {val:.4f} -> {clamped:.4f} (clamped to bounds)")
            else:
                print(f"  {name}: {val:.4f}")
    
    # Run optimization
    print(f"\nStarting optimization...")
    result = forest_minimize(
        objective,
        objective.dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        x0=x0,
        y0=y0,
        random_state=42,
        verbose=False,
    )
    
    # Results
    best_params = params_to_detection_json(result.x, objective.param_names, objective.fixed_params)
    minimize_obj = objective_type in ('k_only', 'segments_ring')
    best_score = result.fun if minimize_obj else -result.fun  # k_only and segments_ring: minimize

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    score_label = 'mean K distance (px)' if objective_type == 'k_only' else (
        f'ring {ring_index} total score (px)' if objective_type == 'segments_ring' else 'composite score'
    )
    print(f"Best {score_label}: {best_score:.4f}")
    print(f"\nBest parameters:")
    for name, value in best_params.items():
        if isinstance(value, list):
            print(f"  {name}: {value}")
        elif isinstance(value, float):
            print(f"  {name}: {value:.6f}")
        else:
            print(f"  {name}: {value}")
    
    # Save best parameters
    filepath = objective.save_best_params()
    if filepath:
        print(f"\nSaved parameters to: {filepath}")
    
    return {
        'tunnel_id': tunnel_id,
        'best_score': best_score,
        'best_params': best_params,
        'n_evaluations': objective.eval_count,
        'history': objective.history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detection BO with all-segments composite objective')
    parser.add_argument('tunnel_id', type=str, help='Tunnel identifier (e.g., 1-4)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--n-calls', type=int, default=80, help='Total evaluations')
    parser.add_argument('--n-initial', type=int, default=15, help='Initial random points')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--agent-type', type=str, default='complex_staggered',
                       choices=['complex_staggered', 'continuous', 'complex_staggered'],
                       help='Agent type (default: complex_staggered)')
    parser.add_argument('--objective', type=str, default='composite',
                       choices=['composite', 'k_only', 'segments_ring'],
                       help='Objective: composite, k_only (mean K distance), or segments_ring (per-ring Y offsets)')
    parser.add_argument('--ring-index', type=int, default=None, metavar='N',
                       help='Ring index 0-6 for --objective segments_ring (required when using segments_ring)')
    args = parser.parse_args()

    run_detection_bo(
        tunnel_id=args.tunnel_id,
        data_dir=args.data_dir,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        verbose=args.verbose,
        agent_type=args.agent_type,
        objective_type=args.objective,
        ring_index=args.ring_index,
    )
