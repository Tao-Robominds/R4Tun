"""
Intrinsic Metrics Extractor for Detection and SAM Stages

Computes intrinsic metrics from pipeline outputs for training the mIoU predictor.
These metrics are computable at runtime without ground truth.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path


# Expected rings per tunnel (from ring_count.txt or known values)
TUNNEL_EXPECTED_RINGS = {
    '1-4': 10,
    '2-2': 10,
    '3-1': 6,
    '4-1': 10,
    '5-1': 7,
}


def load_expected_rings(tunnel_id: str, data_dir: str = 'data') -> int:
    """Load expected ring count from ring_count.txt if available."""
    ring_file = os.path.join(data_dir, tunnel_id, 'ring_count.txt')
    if os.path.exists(ring_file):
        with open(ring_file, 'r') as f:
            return int(f.read().strip())
    return TUNNEL_EXPECTED_RINGS.get(tunnel_id, 10)


def compute_detection_metrics(
    tunnel_id: str,
    detected_csv: str,
    expected_rings: Optional[int] = None,
    data_dir: str = 'data'
) -> Dict[str, float]:
    """
    Compute intrinsic metrics from detection output (detected.csv).
    
    Args:
        tunnel_id: Tunnel identifier
        detected_csv: Path to detected.csv
        expected_rings: Expected K-count (if None, load from ring_count.txt)
        
    Returns:
        Dictionary of intrinsic metrics
    """
    if not os.path.exists(detected_csv):
        return _empty_detection_metrics()
    
    try:
        df = pd.read_csv(detected_csv, comment='#')
    except Exception:
        return _empty_detection_metrics()
    
    if len(df) == 0:
        return _empty_detection_metrics()
    
    if 'X' not in df.columns or 'Y' not in df.columns:
        return _empty_detection_metrics()
    
    valid = df.dropna(subset=['X', 'Y'])
    if len(valid) == 0:
        return _empty_detection_metrics()
    
    k_count = len(valid)
    
    # Type distribution
    type_col = 'Type' if 'Type' in valid.columns else None
    if type_col:
        type_counts = valid[type_col].value_counts()
        
        # Standard detection types
        assume_default = type_counts.reindex(['assume', 'default'], fill_value=0).sum()
        midpoint = type_counts.reindex(['midpoint'], fill_value=0).sum()
        
        # Complex staggered detection types
        intersection_cluster = type_counts.reindex(['intersection_cluster'], fill_value=0).sum()
        midpoint_cluster = type_counts.reindex(['midpoint_cluster'], fill_value=0).sum()
        
        # Calculate ratios
        # For standard detection: midpoint, positive_slope, negative_slope are "real"
        # For complex detection: intersection_cluster is "real", midpoint_cluster is fallback
        real_types_standard = ['midpoint', 'positive_slope', 'negative_slope']
        real_types_complex = ['intersection_cluster']
        
        real_detections_standard = type_counts.reindex(real_types_standard, fill_value=0).sum()
        real_detections_complex = type_counts.reindex(real_types_complex, fill_value=0).sum()
        
        # Use complex types if any complex types found, otherwise use standard
        has_complex_types = intersection_cluster > 0 or midpoint_cluster > 0
        
        if has_complex_types:
            real_detections = real_detections_complex
            midpoint = midpoint_cluster  # For complex, midpoint_cluster is the "midpoint" equivalent
        else:
            real_detections = real_detections_standard
        
        assume_default_ratio = assume_default / len(valid) if len(valid) > 0 else 0.0
        midpoint_ratio = midpoint / len(valid) if len(valid) > 0 else 0.0
        real_detection_ratio = real_detections / len(valid) if len(valid) > 0 else 0.0
    else:
        assume_default_ratio = 0.0
        midpoint_ratio = 0.0
        real_detection_ratio = 1.0
    
    # Y position stats
    y_vals = valid['Y'].values
    y_range = float(np.ptp(y_vals)) if len(y_vals) > 1 else 0.0
    y_std = float(np.std(y_vals)) if len(y_vals) > 1 else 0.0
    
    # X spacing (evenness)
    x_sorted = np.sort(valid['X'].values)
    if len(x_sorted) > 1:
        x_diffs = np.diff(x_sorted)
        x_spacing_mean = np.mean(x_diffs)
        x_spacing_std = np.std(x_diffs)
        x_spacing_cv = x_spacing_std / x_spacing_mean if x_spacing_mean > 0 else 0.0
    else:
        x_spacing_cv = 0.0
    
    if expected_rings is None:
        expected_rings = load_expected_rings(tunnel_id, data_dir)
    
    k_count_match = 1.0 if k_count == expected_rings else 0.0
    
    return {
        'k_count': float(k_count),
        'k_count_match': k_count_match,
        'assume_default_ratio': float(assume_default_ratio),
        'midpoint_ratio': float(midpoint_ratio),
        'real_detection_ratio': float(real_detection_ratio),
        'y_range': float(y_range),
        'y_std': float(y_std),
        'x_spacing_cv': float(x_spacing_cv),
    }


def _empty_detection_metrics() -> Dict[str, float]:
    """Return empty/default metrics when detection output is missing."""
    return {
        'k_count': 0.0,
        'k_count_match': 0.0,
        'assume_default_ratio': 1.0,
        'midpoint_ratio': 0.0,
        'real_detection_ratio': 0.0,
        'y_range': 0.0,
        'y_std': 0.0,
        'x_spacing_cv': 0.0,
    }


def compute_sam_metrics(
    tunnel_id: str,
    final_csv: str,
    detected_csv: Optional[str] = None,
    depth_map_path: Optional[str] = None,
    data_dir: str = 'data'
) -> Dict[str, float]:
    """
    Compute intrinsic metrics from SAM output (final.csv).
    
    Args:
        tunnel_id: Tunnel identifier
        final_csv: Path to final.csv (SAM output)
        detected_csv: Path to detected.csv (for prompt count)
        depth_map_path: Path to depth map for coverage (optional)
        data_dir: Base data directory
        
    Returns:
        Dictionary of intrinsic metrics
    """
    if not os.path.exists(final_csv):
        return _empty_sam_metrics()
    
    try:
        df = pd.read_csv(final_csv, comment='#')
    except Exception:
        return _empty_sam_metrics()
    
    if len(df) == 0:
        return _empty_sam_metrics()
    
    # Prompt count from detected.csv
    prompt_count = 0.0
    if detected_csv and os.path.exists(detected_csv):
        try:
            det_df = pd.read_csv(detected_csv, comment='#')
            prompt_count = float(len(det_df.dropna(subset=['X', 'Y'])))
        except Exception:
            pass
    
    # Segment count from final.csv (pred column)
    if 'pred' in df.columns:
        pred_vals = df['pred'].dropna()
        unique_segments = pred_vals.unique()
        segment_count = len(unique_segments)
        # Exclude background (typically 0)
        segment_count = segment_count - 1 if 0 in unique_segments else segment_count
    else:
        segment_count = 0.0
    
    # Mask fill rate: fraction of points with non-background prediction
    if 'pred' in df.columns:
        pred_vals = df['pred'].dropna()
        non_bg = (pred_vals > 0).sum()
        total = len(pred_vals)
        mask_fill_rate = non_bg / total if total > 0 else 0.0
    else:
        mask_fill_rate = 0.0
    
    # Template coverage: approximate from prompt count and depth map size
    template_coverage = 0.0
    base_dir = os.path.join(data_dir, tunnel_id)
    depth_map_npy = os.path.join(base_dir, 'depth_map_outlier.npy')
    if os.path.exists(depth_map_npy):
        try:
            depth_map = np.load(depth_map_npy)
            depth_h, depth_w = depth_map.shape
            depth_area = depth_h * depth_w
            # Rough estimate: each prompt covers ~1250*3240 pixels, prompts may overlap
            if prompt_count > 0 and depth_area > 0:
                template_area_per_prompt = 1250 * 3240
                total_template_area = prompt_count * template_area_per_prompt
                template_coverage = min(1.0, total_template_area / depth_area)
        except Exception:
            pass
    
    # Segment count match: expected is typically 6 or 7
    expected_segments = 6 if tunnel_id in ['3-1'] else 7 if tunnel_id == '5-1' else 6
    segment_count_match = 1.0 if segment_count >= expected_segments - 1 else 0.0
    
    return {
        'prompt_count': prompt_count,
        'segment_count': float(segment_count),
        'segment_count_match': segment_count_match,
        'mask_fill_rate': float(mask_fill_rate),
        'template_coverage': float(template_coverage),
    }


def _empty_sam_metrics() -> Dict[str, float]:
    """Return empty/default metrics when SAM output is missing."""
    return {
        'prompt_count': 0.0,
        'segment_count': 0.0,
        'segment_count_match': 0.0,
        'mask_fill_rate': 0.0,
        'template_coverage': 0.0,
    }


def compute_all_metrics(
    tunnel_id: str,
    detected_csv: str,
    final_csv: str,
    data_dir: str = 'data'
) -> Dict[str, float]:
    """
    Compute both detection and SAM intrinsic metrics.
    
    Returns combined dict with prefix: det_ for detection, sam_ for SAM.
    """
    expected_rings = load_expected_rings(tunnel_id, data_dir)
    
    det_metrics = compute_detection_metrics(
        tunnel_id, detected_csv, expected_rings
    )
    sam_metrics = compute_sam_metrics(
        tunnel_id, final_csv, detected_csv, data_dir=data_dir
    )
    
    result = {}
    for k, v in det_metrics.items():
        result[f'det_{k}'] = v
    for k, v in sam_metrics.items():
        result[f'sam_{k}'] = v
    
    return result
