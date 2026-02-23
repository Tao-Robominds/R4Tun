"""
SAM-based K Y Position Validator

Uses SAM mask quality as a GT-free signal to validate and correct K Y positions.
Lightweight probing: only processes K blocks (7 calls) for fast validation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import SAM processing function
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'agents' / 'irregular' / '3_segmentation'))
import importlib.util
sam_module_path = PROJECT_ROOT / 'agents' / 'irregular' / '3_segmentation' / '3_sam.py'
spec = importlib.util.spec_from_file_location("sam_module", str(sam_module_path))
sam_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sam_module)
process_segment = sam_module.process_segment
load_sam_parameters = sam_module.load_parameters


def sam_probe_quality(
    k_positions: pd.DataFrame,
    tunnel_id: str,
    base_dir: str = "data",
    group_offsets: Dict[str, float] = None,
    stagger_groups: Dict[str, List[int]] = None,
) -> Dict[int, float]:
    """Probe SAM quality for K positions (K blocks only, fast).
    
    Args:
        k_positions: DataFrame with columns Type, X, Y, Confidence (K-only, one per ring)
        tunnel_id: Tunnel identifier
        base_dir: Base data directory
        group_offsets: Grouped offsets for expanding K to all segments
        stagger_groups: Stagger group mapping (ring_idx -> group)
    
    Returns:
        Dict mapping ring_idx -> sam_quality_score (0.0 to 1.0)
    """
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    # Load SAM parameters
    sam_params = load_sam_parameters(tunnel_id, base_dir)
    
    # Load depth map
    depth_map_path = os.path.join(tunnel_dir, "depth_map.png")
    if not os.path.exists(depth_map_path):
        raise FileNotFoundError(f"Depth map not found: {depth_map_path}")
    image = cv2.imread(depth_map_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read depth map: {depth_map_path}")
    
    # Load SAM model (lightweight - only load once)
    from segment_anything import sam_model_registry, SamPredictor
    SAM_MODEL_TYPE = "vit_h"
    SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "sam_vit_h_4b8939.pth")
    SAM_DEVICE = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
    
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"Warning: SAM checkpoint not found at {SAM_CHECKPOINT}, skipping SAM validation")
        return {i: 0.5 for i in range(len(k_positions))}
    
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=SAM_DEVICE)
    predictor = SamPredictor(sam)
    
    # Prepare SAM config
    resolution = sam_params.get('resolution', 0.001)
    segment_width = sam_params.get('segment_width', 1200.0)
    K_height = sam_params.get('K_height', 1079.92)
    AB_height = sam_params.get('AB_height', 3239.77)
    angle_deg = sam_params.get('angle', 7.5)
    padding = sam_params.get('padding', 150)
    crop_margin = sam_params.get('crop_margin', 50)
    y_bounds = sam_params.get('y_bounds', [4200, 13100])
    
    template_params = {
        'k_mask_width': sam_params.get('k_mask_width', 625.0),
        'k_mask_height_pos': sam_params.get('k_mask_height_pos', 620.0),
        'k_mask_height_neg': sam_params.get('k_mask_height_neg', 460.0),
        'ab_mask_width': sam_params.get('ab_mask_width', 625.0),
        'ab_mask_height': sam_params.get('ab_mask_height', 1620.0),
        'b1_height_top': sam_params.get('b1_height_top', 1620.0),
        'b1_height_bottom_pos': sam_params.get('b1_height_bottom_pos', 1620.0),
        'b1_height_bottom_neg': sam_params.get('b1_height_bottom_neg', 1620.0),
        'b2_height_top_pos': sam_params.get('b2_height_top_pos', 1620.0),
        'b2_height_top_neg': sam_params.get('b2_height_top_neg', 1620.0),
        'b2_height_bottom': sam_params.get('b2_height_bottom', 1620.0),
    }
    
    config = {
        'resolution': resolution,
        'segment_width': segment_width,
        'K_height': K_height,
        'AB_height': AB_height,
        'angle': angle_deg,
        'padding': padding,
        'crop_margin': crop_margin,
        'y_bounds': y_bounds,
        'template_params': template_params,
    }
    
    # Process K blocks only (7 calls, fast)
    ring_qualities = {}
    img_height = image.shape[0]
    
    for ring_idx, (_, k_row) in enumerate(k_positions.iterrows()):
        k_x = k_row['X']
        k_y = k_row['Y']
        k_quality = k_row.get('Confidence', 1.0)
        
        # Create segment row for K block
        segment_row = pd.Series({
            'X': k_x,
            'Y': k_y,
            'Block': 'K',
            'Ring': ring_idx,
            'quality': k_quality,
        })
        
        try:
            result = process_segment(segment_row, image, predictor, config)
            if result is None:
                ring_qualities[ring_idx] = 0.0
                continue
            
            # Extract quality metrics
            sam_score = result.get('score', 0.0)  # SAM's internal confidence
            template_mask = result.get('template_mask')
            mask = result.get('mask', [None])[0]
            
            # Compute template IoU
            template_iou = 0.0
            if template_mask is not None and mask is not None:
                intersection = np.logical_and(template_mask, mask).sum()
                union = np.logical_or(template_mask, mask).sum()
                if union > 0:
                    template_iou = intersection / union
            
            # Compute mask fill rate
            mask_fill_rate = 0.0
            if mask is not None:
                crop_info = result.get('crop_info', {})
                crop_h = crop_info.get('y2', image.shape[0]) - crop_info.get('y1', 0)
                crop_w = crop_info.get('x2', image.shape[1]) - crop_info.get('x1', 0)
                if crop_h > 0 and crop_w > 0:
                    mask_fill_rate = mask.sum() / (crop_h * crop_w)
            
            # Combined quality score
            quality = 0.4 * sam_score + 0.3 * template_iou + 0.3 * mask_fill_rate
            ring_qualities[ring_idx] = float(quality)
            
        except Exception as e:
            print(f"Warning: SAM probe failed for ring {ring_idx}: {e}")
            ring_qualities[ring_idx] = 0.0
    
    return ring_qualities


def select_best_candidate_set(
    candidate_sets: List[pd.DataFrame],
    detection_scores: List[float],
    tunnel_id: str,
    base_dir: str = "data",
    alpha: float = 0.7,
    group_offsets: Dict[str, float] = None,
    stagger_groups: Dict[str, List[int]] = None,
) -> Tuple[pd.DataFrame, float]:
    """Select best candidate set using alpha-weighted detection + SAM quality.
    
    Args:
        candidate_sets: List of K position DataFrames (one per candidate set)
        detection_scores: List of detection scores (one per candidate set)
        tunnel_id: Tunnel identifier
        base_dir: Base data directory
        alpha: Weight for detection score (1-alpha for SAM quality)
        group_offsets: Grouped offsets for SAM probing
        stagger_groups: Stagger group mapping
    
    Returns:
        (best_k_positions, best_final_score)
    """
    if len(candidate_sets) == 0:
        raise ValueError("No candidate sets provided")
    
    if len(candidate_sets) == 1:
        return candidate_sets[0], detection_scores[0]
    
    best_idx = 0
    best_score = -float('inf')
    
    for i, (k_positions, det_score) in enumerate(zip(candidate_sets, detection_scores)):
        # Probe SAM quality
        sam_qualities = sam_probe_quality(
            k_positions, tunnel_id, base_dir, group_offsets, stagger_groups
        )
        sam_score = np.mean(list(sam_qualities.values())) if sam_qualities else 0.0
        
        # Combined score
        final_score = alpha * det_score + (1 - alpha) * sam_score
        
        if final_score > best_score:
            best_score = final_score
            best_idx = i
    
    return candidate_sets[best_idx], best_score


def correct_uncertain_rings(
    k_positions: pd.DataFrame,
    tunnel_id: str,
    base_dir: str = "data",
    sam_quality_threshold: float = 0.5,
    correction_step_px: float = 50.0,
    max_shifts: int = 5,
    group_offsets: Dict[str, float] = None,
    stagger_groups: Dict[str, List[int]] = None,
) -> pd.DataFrame:
    """Correct K Y positions for rings with low SAM quality.
    
    Args:
        k_positions: DataFrame with columns Type, X, Y, Confidence
        tunnel_id: Tunnel identifier
        base_dir: Base data directory
        sam_quality_threshold: Threshold below which rings are considered uncertain
        correction_step_px: Step size for Y shifts
        max_shifts: Maximum number of shifts to try in each direction
        group_offsets: Grouped offsets for validation
        stagger_groups: Stagger group mapping
    
    Returns:
        Corrected K positions DataFrame
    """
    # Probe initial quality
    sam_qualities = sam_probe_quality(
        k_positions, tunnel_id, base_dir, group_offsets, stagger_groups
    )
    
    # Identify uncertain rings
    uncertain_rings = [
        ring_idx for ring_idx, quality in sam_qualities.items()
        if quality < sam_quality_threshold
    ]
    
    if len(uncertain_rings) == 0:
        return k_positions
    
    print(f"  Correcting {len(uncertain_rings)} uncertain rings (SAM quality < {sam_quality_threshold})")
    
    # Load image for SAM probing
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    depth_map_path = os.path.join(tunnel_dir, "depth_map.png")
    image = cv2.imread(depth_map_path)
    img_height = image.shape[0]
    
    # Correct each uncertain ring
    corrected = k_positions.copy()
    
    for ring_idx in uncertain_rings:
        if ring_idx >= len(corrected):
            continue
        
        original_y = corrected.iloc[ring_idx]['Y']
        best_y = original_y
        best_quality = sam_qualities.get(ring_idx, 0.0)
        
        # Try shifts in both directions
        for shift_dir in [-1, 1]:
            for shift_step in range(1, max_shifts + 1):
                test_y = (original_y + shift_dir * shift_step * correction_step_px) % img_height
                
                # Create test K positions with shifted Y
                test_positions = corrected.copy()
                test_positions.iloc[ring_idx, test_positions.columns.get_loc('Y')] = test_y
                
                # Probe quality
                test_qualities = sam_probe_quality(
                    test_positions, tunnel_id, base_dir, group_offsets, stagger_groups
                )
                test_quality = test_qualities.get(ring_idx, 0.0)
                
                if test_quality > best_quality:
                    best_quality = test_quality
                    best_y = test_y
        
        # Apply correction if improvement found
        if best_quality > sam_qualities.get(ring_idx, 0.0):
            corrected.iloc[ring_idx, corrected.columns.get_loc('Y')] = best_y
            print(f"    Ring {ring_idx}: {original_y:.0f} -> {best_y:.0f} (quality: {sam_qualities.get(ring_idx, 0.0):.3f} -> {best_quality:.3f})")
    
    return corrected
