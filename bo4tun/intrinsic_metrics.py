"""
Intrinsic Metrics Extractor for Detection, SAM, and Preprocessing Stages

Computes intrinsic metrics from pipeline outputs for training the mIoU predictor.
These metrics are computable at runtime without ground truth.

Preprocessing metrics are implemented as guardrails (fail-fast checks) rather than
predictor features, since preprocessing contributes only +0.1% to mIoU.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
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
    # Simple patterns (1-4, 2-2, 3-1): 6 segments
    # Complex patterns (4-1, 5-1): 7 segments
    expected_segments = 6 if tunnel_id in ['1-4', '2-2', '3-1'] else 7
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


# =============================================================================
# PREPROCESSING GUARDRAILS
# These are fail-fast checks, not predictor features.
# Preprocessing contributes only +0.1% to mIoU.
# =============================================================================

# Default guardrail thresholds
PREPROCESSING_GUARDRAIL_THRESHOLDS = {
    'theta_coverage': {'min': 98.0, 'max': 102.0},  # Should be ~100%
    'interpolation_coverage': {'min': 95.0, 'max': None},
    'point_retention_ratio': {'min': 90.0, 'max': None},
}


def compute_unfolding_guardrails(
    tunnel_id: str,
    data_dir: str = 'data'
) -> Dict[str, float]:
    """
    Compute unfolding stage guardrail metrics.
    
    Args:
        tunnel_id: Tunnel identifier
        data_dir: Base data directory
        
    Returns:
        Dictionary with unfolding guardrail metrics
    """
    metrics = {
        'theta_coverage': 0.0,
        'theta_min': 0.0,
        'theta_max': 0.0,
        'ring_count': 0,
    }
    
    unwrapped_path = os.path.join(data_dir, tunnel_id, 'unwrapped.csv')
    if not os.path.exists(unwrapped_path):
        return metrics
    
    try:
        df = pd.read_csv(unwrapped_path)
        
        if 'theta' in df.columns:
            theta_vals = df['theta'].dropna()
            if len(theta_vals) > 0:
                theta_min = theta_vals.min()
                theta_max = theta_vals.max()
                # Theta coverage as percentage of 2π
                theta_coverage = (theta_max - theta_min) / (2 * np.pi) * 100
                
                metrics['theta_min'] = float(theta_min)
                metrics['theta_max'] = float(theta_max)
                metrics['theta_coverage'] = float(theta_coverage)
        
        if 'ring' in df.columns:
            metrics['ring_count'] = int(df['ring'].nunique())
            
    except Exception as e:
        print(f"Warning: Could not compute unfolding metrics for {tunnel_id}: {e}")
    
    return metrics


def compute_denoising_guardrails(
    tunnel_id: str,
    data_dir: str = 'data'
) -> Dict[str, float]:
    """
    Compute denoising stage guardrail metrics.
    
    Args:
        tunnel_id: Tunnel identifier
        data_dir: Base data directory
        
    Returns:
        Dictionary with denoising guardrail metrics
    """
    metrics = {
        'point_retention_ratio': 0.0,
        'points_before': 0,
        'points_after': 0,
    }
    
    unwrapped_path = os.path.join(data_dir, tunnel_id, 'unwrapped.csv')
    denoised_path = os.path.join(data_dir, tunnel_id, 'denoised.csv')
    
    if not os.path.exists(unwrapped_path) or not os.path.exists(denoised_path):
        return metrics
    
    try:
        df_before = pd.read_csv(unwrapped_path)
        df_after = pd.read_csv(denoised_path)
        
        points_before = len(df_before)
        points_after = len(df_after)
        
        metrics['points_before'] = points_before
        metrics['points_after'] = points_after
        
        if points_before > 0:
            metrics['point_retention_ratio'] = (points_after / points_before) * 100
            
    except Exception as e:
        print(f"Warning: Could not compute denoising metrics for {tunnel_id}: {e}")
    
    return metrics


def compute_enhancing_guardrails(
    tunnel_id: str,
    data_dir: str = 'data'
) -> Dict[str, float]:
    """
    Compute enhancing stage guardrail metrics.
    
    Args:
        tunnel_id: Tunnel identifier
        data_dir: Base data directory
        
    Returns:
        Dictionary with enhancing guardrail metrics
    """
    metrics = {
        'interpolation_coverage': 0.0,
        'depth_map_height': 0,
        'depth_map_width': 0,
        'valid_pixels': 0,
        'total_pixels': 0,
    }
    
    depth_map_path = os.path.join(data_dir, tunnel_id, 'depth_map_outlier.npy')
    if not os.path.exists(depth_map_path):
        return metrics
    
    try:
        depth_map = np.load(depth_map_path)
        
        height, width = depth_map.shape
        total_pixels = height * width
        
        # Count non-NaN and non-zero pixels as valid
        valid_mask = ~np.isnan(depth_map) & (depth_map != 0)
        valid_pixels = int(np.sum(valid_mask))
        
        metrics['depth_map_height'] = height
        metrics['depth_map_width'] = width
        metrics['total_pixels'] = total_pixels
        metrics['valid_pixels'] = valid_pixels
        
        if total_pixels > 0:
            metrics['interpolation_coverage'] = (valid_pixels / total_pixels) * 100
            
    except Exception as e:
        print(f"Warning: Could not compute enhancing metrics for {tunnel_id}: {e}")
    
    return metrics


def compute_preprocessing_guardrails(
    tunnel_id: str,
    data_dir: str = 'data'
) -> Dict[str, float]:
    """
    Compute all preprocessing guardrail metrics.
    
    Args:
        tunnel_id: Tunnel identifier
        data_dir: Base data directory
        
    Returns:
        Combined dictionary with all preprocessing guardrail metrics (prefixed)
    """
    unfolding = compute_unfolding_guardrails(tunnel_id, data_dir)
    denoising = compute_denoising_guardrails(tunnel_id, data_dir)
    enhancing = compute_enhancing_guardrails(tunnel_id, data_dir)
    
    result = {}
    for k, v in unfolding.items():
        result[f'pre_unfold_{k}'] = v
    for k, v in denoising.items():
        result[f'pre_denoise_{k}'] = v
    for k, v in enhancing.items():
        result[f'pre_enhance_{k}'] = v
    
    return result


def check_preprocessing_guardrails(
    tunnel_id: str,
    data_dir: str = 'data',
    thresholds: Optional[Dict] = None
) -> Tuple[bool, List[str], Dict[str, float]]:
    """
    Check if preprocessing guardrails pass.
    
    Args:
        tunnel_id: Tunnel identifier
        data_dir: Base data directory
        thresholds: Optional custom thresholds (uses defaults if None)
        
    Returns:
        Tuple of (passed: bool, violations: list of strings, metrics: dict)
    """
    if thresholds is None:
        thresholds = PREPROCESSING_GUARDRAIL_THRESHOLDS
    
    # Compute metrics
    unfolding = compute_unfolding_guardrails(tunnel_id, data_dir)
    denoising = compute_denoising_guardrails(tunnel_id, data_dir)
    enhancing = compute_enhancing_guardrails(tunnel_id, data_dir)
    
    violations = []
    
    # Check theta_coverage (unfolding)
    theta_coverage = unfolding['theta_coverage']
    thresh = thresholds.get('theta_coverage', {})
    if thresh.get('min') is not None and theta_coverage < thresh['min']:
        violations.append(
            f"theta_coverage={theta_coverage:.1f}% < {thresh['min']}% (possible wraparound)"
        )
    if thresh.get('max') is not None and theta_coverage > thresh['max']:
        violations.append(
            f"theta_coverage={theta_coverage:.1f}% > {thresh['max']}% (over-coverage)"
        )
    
    # Check point_retention_ratio (denoising)
    retention = denoising['point_retention_ratio']
    thresh = thresholds.get('point_retention_ratio', {})
    if thresh.get('min') is not None and retention < thresh['min']:
        violations.append(
            f"point_retention_ratio={retention:.1f}% < {thresh['min']}% (too aggressive denoising)"
        )
    
    # Check interpolation_coverage (enhancing)
    coverage = enhancing['interpolation_coverage']
    thresh = thresholds.get('interpolation_coverage', {})
    if thresh.get('min') is not None and coverage < thresh['min']:
        violations.append(
            f"interpolation_coverage={coverage:.1f}% < {thresh['min']}% (sparse depth map)"
        )
    
    # Combine all metrics
    all_metrics = {
        'theta_coverage': theta_coverage,
        'point_retention_ratio': retention,
        'interpolation_coverage': coverage,
    }
    
    passed = len(violations) == 0
    return passed, violations, all_metrics


def run_preprocessing_check(
    tunnel_id: str,
    data_dir: str = 'data',
    verbose: bool = True
) -> bool:
    """
    Run preprocessing guardrail check with optional verbose output.
    
    Args:
        tunnel_id: Tunnel identifier
        data_dir: Base data directory
        verbose: If True, print results
        
    Returns:
        True if all guardrails pass, False otherwise
    """
    passed, violations, metrics = check_preprocessing_guardrails(tunnel_id, data_dir)
    
    if verbose:
        print(f"\n=== Preprocessing Guardrails for {tunnel_id} ===")
        print(f"  theta_coverage: {metrics['theta_coverage']:.1f}%")
        print(f"  point_retention_ratio: {metrics['point_retention_ratio']:.1f}%")
        print(f"  interpolation_coverage: {metrics['interpolation_coverage']:.1f}%")
        
        if passed:
            print("  Status: ✓ PASSED")
        else:
            print("  Status: ✗ FAILED")
            for v in violations:
                print(f"    - {v}")
    
    return passed


# =============================================================================
# COMPLEX PATTERN METRICS (Phase 1 - High Priority)
# These metrics are designed for complex staggered patterns (4-1, 5-1)
# to enable quality assessment without ground truth.
# =============================================================================

def compute_complex_sam_metrics(
    tunnel_id: str,
    final_csv: str,
    depth_map_path: Optional[str] = None,
    data_dir: str = 'data'
) -> Dict[str, float]:
    """
    Compute SAM intrinsic metrics specific to complex staggered patterns.
    
    These metrics assess segmentation quality from the output alone,
    enabling reflection/rerun decisions without ground truth.
    
    Args:
        tunnel_id: Tunnel identifier
        final_csv: Path to final.csv (SAM output)
        depth_map_path: Optional path to depth map
        data_dir: Base data directory
        
    Returns:
        Dictionary of complex SAM metrics
    """
    metrics = _empty_complex_sam_metrics()
    
    if not os.path.exists(final_csv):
        return metrics
    
    try:
        df = pd.read_csv(final_csv, comment='#')
    except Exception:
        return metrics
    
    if len(df) == 0 or 'pred' not in df.columns:
        return metrics
    
    pred_vals = df['pred'].dropna().values
    
    if len(pred_vals) == 0:
        return metrics
    
    # Get unique segments (excluding background = 0)
    unique_segments = [s for s in np.unique(pred_vals) if s > 0]
    n_segments = len(unique_segments)
    
    if n_segments == 0:
        return metrics
    
    metrics['segment_count'] = float(n_segments)
    
    # =========================================================================
    # Coverage Metrics
    # =========================================================================
    
    # Mask fill rate (already exists, but include here for completeness)
    non_bg = (pred_vals > 0).sum()
    total = len(pred_vals)
    metrics['mask_fill_rate'] = non_bg / total if total > 0 else 0.0
    
    # Gap ratio: estimate from points not assigned to segments
    # (In final.csv, background = 0)
    bg_count = (pred_vals == 0).sum()
    metrics['gap_ratio'] = bg_count / total if total > 0 else 1.0
    
    # =========================================================================
    # Segment Geometry Metrics
    # =========================================================================
    
    # Try different column name conventions
    x_col = 'pixel_x' if 'pixel_x' in df.columns else 'x' if 'x' in df.columns else None
    y_col = 'pixel_y' if 'pixel_y' in df.columns else 'y' if 'y' in df.columns else None
    
    if x_col is not None and y_col is not None:
        segment_widths = []
        segment_heights = []
        segment_areas = []
        
        for seg_id in unique_segments:
            seg_mask = df['pred'] == seg_id
            if seg_mask.sum() < 10:  # Skip tiny segments
                continue
                
            seg_df = df[seg_mask]
            
            # Compute bounding box
            x_min, x_max = seg_df[x_col].min(), seg_df[x_col].max()
            y_min, y_max = seg_df[y_col].min(), seg_df[y_col].max()
            
            width = x_max - x_min
            height = y_max - y_min
            area = seg_mask.sum()
            
            if width > 0 and height > 0:
                segment_widths.append(width)
                segment_heights.append(height)
                segment_areas.append(area)
        
        if len(segment_widths) >= 2:
            # Width consistency (CV)
            width_mean = np.mean(segment_widths)
            width_std = np.std(segment_widths)
            metrics['segment_width_cv'] = width_std / width_mean if width_mean > 0 else 0.0
            
            # Height consistency (CV)
            height_mean = np.mean(segment_heights)
            height_std = np.std(segment_heights)
            metrics['segment_height_cv'] = height_std / height_mean if height_mean > 0 else 0.0
            
            # Area consistency (CV) - indicates overall size uniformity
            area_mean = np.mean(segment_areas)
            area_std = np.std(segment_areas)
            metrics['segment_area_cv'] = area_std / area_mean if area_mean > 0 else 0.0
            
            # Mean aspect ratio
            aspect_ratios = [w/h for w, h in zip(segment_widths, segment_heights) if h > 0]
            if aspect_ratios:
                metrics['aspect_ratio_mean'] = float(np.mean(aspect_ratios))
                metrics['aspect_ratio_cv'] = float(np.std(aspect_ratios) / np.mean(aspect_ratios)) if np.mean(aspect_ratios) > 0 else 0.0
    
    # =========================================================================
    # Ring Consistency Metrics (if ring info available)
    # =========================================================================
    
    if 'ring' in df.columns:
        rings = df['ring'].dropna().unique()
        
        if len(rings) >= 2:
            ring_coverages = []
            for ring_id in rings:
                ring_df = df[df['ring'] == ring_id]
                if len(ring_df) > 0:
                    ring_coverage = (ring_df['pred'] > 0).sum() / len(ring_df)
                    ring_coverages.append(ring_coverage)
            
            if len(ring_coverages) >= 2:
                metrics['ring_coverage_cv'] = float(np.std(ring_coverages) / np.mean(ring_coverages)) if np.mean(ring_coverages) > 0 else 0.0
                metrics['ring_completeness'] = float(np.mean([1 if c > 0.5 else 0 for c in ring_coverages]))
    
    return metrics


def _empty_complex_sam_metrics() -> Dict[str, float]:
    """Return empty/default complex SAM metrics."""
    return {
        'segment_count': 0.0,
        'mask_fill_rate': 0.0,
        'gap_ratio': 1.0,
        'segment_width_cv': 0.0,
        'segment_height_cv': 0.0,
        'segment_area_cv': 0.0,
        'aspect_ratio_mean': 0.0,
        'aspect_ratio_cv': 0.0,
        'ring_coverage_cv': 0.0,
        'ring_completeness': 0.0,
    }


def compute_complex_detection_metrics(
    tunnel_id: str,
    detected_csv: str,
    data_dir: str = 'data'
) -> Dict[str, float]:
    """
    Compute detection metrics specific to complex staggered patterns.
    
    These metrics assess detection quality for complex patterns where
    cluster-based detection is used instead of simple line intersection.
    
    Args:
        tunnel_id: Tunnel identifier
        detected_csv: Path to detected.csv
        data_dir: Base data directory
        
    Returns:
        Dictionary of complex detection metrics
    """
    metrics = _empty_complex_detection_metrics()
    
    if not os.path.exists(detected_csv):
        return metrics
    
    try:
        df = pd.read_csv(detected_csv, comment='#')
    except Exception:
        return metrics
    
    if len(df) == 0:
        return metrics
    
    valid = df.dropna(subset=['X', 'Y'])
    if len(valid) == 0:
        return metrics
    
    k_count = len(valid)
    metrics['k_count'] = float(k_count)
    
    x_vals = valid['X'].values
    y_vals = valid['Y'].values
    
    # =========================================================================
    # Spacing and Distribution Metrics
    # =========================================================================
    
    # X spacing CV (already exists in basic metrics, but include for completeness)
    if len(x_vals) > 1:
        x_sorted = np.sort(x_vals)
        x_diffs = np.diff(x_sorted)
        if len(x_diffs) > 0 and np.mean(x_diffs) > 0:
            metrics['x_spacing_cv'] = float(np.std(x_diffs) / np.mean(x_diffs))
    
    # Y range and distribution
    if len(y_vals) > 1:
        metrics['y_range'] = float(np.ptp(y_vals))
        metrics['y_std'] = float(np.std(y_vals))
        metrics['y_cv'] = float(np.std(y_vals) / np.mean(y_vals)) if np.mean(y_vals) > 0 else 0.0
    
    # =========================================================================
    # Ring Completeness (if ring info available)
    # =========================================================================
    
    if 'Ring' in valid.columns or 'ring' in valid.columns:
        ring_col = 'Ring' if 'Ring' in valid.columns else 'ring'
        rings_detected = valid[ring_col].nunique()
        expected_rings = load_expected_rings(tunnel_id, data_dir)
        metrics['ring_completeness'] = float(rings_detected / expected_rings) if expected_rings > 0 else 0.0
    
    # =========================================================================
    # Detection Type Quality (if Type column available)
    # =========================================================================
    
    if 'Type' in valid.columns:
        type_counts = valid['Type'].value_counts()
        
        # Complex detection types
        intersection_cluster = type_counts.get('intersection_cluster', 0)
        midpoint_cluster = type_counts.get('midpoint_cluster', 0)
        
        total = len(valid)
        metrics['intersection_ratio'] = float(intersection_cluster / total) if total > 0 else 0.0
        metrics['midpoint_cluster_ratio'] = float(midpoint_cluster / total) if total > 0 else 0.0
        
        # High-confidence detection ratio
        high_conf_types = ['intersection_cluster']
        high_conf = sum(type_counts.get(t, 0) for t in high_conf_types)
        metrics['high_confidence_ratio'] = float(high_conf / total) if total > 0 else 0.0
    
    # =========================================================================
    # Cluster Quality (estimate from position distribution)
    # =========================================================================
    
    # Estimate cluster compactness from Y-position grouping
    if len(y_vals) >= 4:
        # Simple clustering by Y position
        y_sorted_idx = np.argsort(y_vals)
        y_sorted = y_vals[y_sorted_idx]
        
        # Find gaps that might indicate cluster boundaries
        y_diffs = np.diff(y_sorted)
        median_diff = np.median(y_diffs)
        
        if median_diff > 0:
            # Large gaps indicate cluster boundaries
            gap_threshold = median_diff * 2
            n_clusters = np.sum(y_diffs > gap_threshold) + 1
            metrics['estimated_clusters'] = float(n_clusters)
            
            # Cluster separation quality
            if n_clusters > 1:
                large_gaps = y_diffs[y_diffs > gap_threshold]
                small_gaps = y_diffs[y_diffs <= gap_threshold]
                if len(small_gaps) > 0 and len(large_gaps) > 0:
                    metrics['cluster_separation'] = float(np.mean(large_gaps) / np.mean(small_gaps))
    
    return metrics


def _empty_complex_detection_metrics() -> Dict[str, float]:
    """Return empty/default complex detection metrics."""
    return {
        'k_count': 0.0,
        'x_spacing_cv': 0.0,
        'y_range': 0.0,
        'y_std': 0.0,
        'y_cv': 0.0,
        'ring_completeness': 0.0,
        'intersection_ratio': 0.0,
        'midpoint_cluster_ratio': 0.0,
        'high_confidence_ratio': 0.0,
        'estimated_clusters': 0.0,
        'cluster_separation': 0.0,
    }


def compute_all_complex_metrics(
    tunnel_id: str,
    detected_csv: str,
    final_csv: str,
    data_dir: str = 'data'
) -> Dict[str, float]:
    """
    Compute all intrinsic metrics for complex staggered patterns.
    
    Combines detection and SAM metrics specific to complex patterns.
    
    Returns combined dict with prefixes: det_ for detection, sam_ for SAM.
    """
    det_metrics = compute_complex_detection_metrics(tunnel_id, detected_csv, data_dir)
    sam_metrics = compute_complex_sam_metrics(tunnel_id, final_csv, data_dir=data_dir)
    
    result = {}
    for k, v in det_metrics.items():
        result[f'det_{k}'] = v
    for k, v in sam_metrics.items():
        result[f'sam_{k}'] = v
    
    return result


# Complex pattern guardrail thresholds
COMPLEX_GUARDRAIL_THRESHOLDS = {
    # SAM metrics
    'sam_gap_ratio': {'min': None, 'max': 0.15},          # Under-segmentation
    'sam_mask_fill_rate': {'min': 0.30, 'max': 0.90},     # Over/under segmentation
    'sam_segment_width_cv': {'min': None, 'max': 0.20},   # Width inconsistency
    'sam_segment_area_cv': {'min': None, 'max': 0.30},    # Size inconsistency
    # Detection metrics
    'det_k_count': {'min': 4, 'max': 12},                 # K-block count
    'det_ring_completeness': {'min': 0.80, 'max': None},  # Ring coverage
    'det_x_spacing_cv': {'min': None, 'max': 0.60},       # Spacing uniformity
}


def check_complex_guardrails(
    tunnel_id: str,
    detected_csv: str,
    final_csv: str,
    data_dir: str = 'data',
    thresholds: Optional[Dict] = None
) -> Tuple[bool, List[str], Dict[str, float]]:
    """
    Check if complex pattern guardrails pass.
    
    Args:
        tunnel_id: Tunnel identifier
        detected_csv: Path to detected.csv
        final_csv: Path to final.csv
        data_dir: Base data directory
        thresholds: Optional custom thresholds
        
    Returns:
        Tuple of (passed: bool, violations: list, metrics: dict)
    """
    if thresholds is None:
        thresholds = COMPLEX_GUARDRAIL_THRESHOLDS
    
    # Compute all complex metrics
    all_metrics = compute_all_complex_metrics(tunnel_id, detected_csv, final_csv, data_dir)
    
    violations = []
    
    for metric_name, bounds in thresholds.items():
        # Handle prefixed metric names
        if metric_name.startswith('sam_') or metric_name.startswith('det_'):
            value = all_metrics.get(metric_name, None)
        else:
            # Try with both prefixes
            value = all_metrics.get(f'sam_{metric_name}', all_metrics.get(f'det_{metric_name}', None))
        
        if value is None:
            continue
        
        if bounds.get('min') is not None and value < bounds['min']:
            violations.append(f"{metric_name}={value:.3f} < {bounds['min']} (below minimum)")
        if bounds.get('max') is not None and value > bounds['max']:
            violations.append(f"{metric_name}={value:.3f} > {bounds['max']} (above maximum)")
    
    passed = len(violations) == 0
    return passed, violations, all_metrics


def suggest_complex_rerun_params(violations: List[str], current_params: Dict) -> Dict[str, str]:
    """
    Suggest parameter adjustments based on guardrail violations.
    
    Args:
        violations: List of violation strings from check_complex_guardrails
        current_params: Current SAM parameters
        
    Returns:
        Dictionary of suggested parameter adjustments
    """
    suggestions = {}
    
    for v in violations:
        if 'sam_gap_ratio' in v and 'above maximum' in v:
            # Under-segmentation: increase segment width
            suggestions['segment_width'] = 'INCREASE by 50-100'
        
        elif 'sam_mask_fill_rate' in v:
            if 'below minimum' in v:
                # Under-segmentation
                suggestions['segment_width'] = 'INCREASE by 50-100'
            elif 'above maximum' in v:
                # Over-segmentation
                suggestions['segment_width'] = 'DECREASE by 50-100'
        
        elif 'sam_segment_width_cv' in v and 'above maximum' in v:
            # Inconsistent widths: adjust k_height
            suggestions['k_height'] = 'ADJUST by ±50'
        
        elif 'det_k_count' in v:
            if 'below minimum' in v:
                suggestions['binary_threshold'] = 'DECREASE by 10-20'
            elif 'above maximum' in v:
                suggestions['binary_threshold'] = 'INCREASE by 10-20'
        
        elif 'det_x_spacing_cv' in v and 'above maximum' in v:
            suggestions['hough_threshold'] = 'INCREASE by 10'
    
    return suggestions
