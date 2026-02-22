"""
Extract Detection Output Intrinsic Metrics (SAM Readiness)

Extracts ONLY critical metrics that can be thresholded to determine if
detection output is suitable for SAM segmentation.

Critical Metrics:
    1. det_k_count_match - Correct number of K positions? (boolean)
    2. det_x_spacing_cv - Regular horizontal spacing? [<=0.15]
    3. det_midpoint_ratio - Detection confidence? [>=0.50]
    4. det_y_pattern_consistency - Y positions follow pattern? [<=3.0%]

Usage:
    python extract_intrinsics.py 1-4 [--data-dir data] [--output ...]
"""

import argparse
import json
import os
import sys
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# =============================================================================
# Guardrail Thresholds
# =============================================================================
K_COUNT_MATCH_REQUIRED = True  # Must match ring_count exactly
X_SPACING_CV_MAX = 0.15  # 15% variation in X spacing
MIDPOINT_RATIO_MIN = 0.50  # At least 50% detected via midpoint
Y_PATTERN_CONSISTENCY_MAX = 3.0  # 3% of image height


# =============================================================================
# Metric Extraction
# =============================================================================

def _load_ring_count(tunnel_dir: str) -> Optional[int]:
    """Load ring_count from ring_count.txt."""
    path = os.path.join(tunnel_dir, "ring_count.txt")
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None


def _load_image_height(tunnel_dir: str) -> Optional[int]:
    """Load image height from depth_map_outlier.npy."""
    path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(path):
        return None
    
    try:
        depth = np.load(path)
        return int(depth.shape[0])
    except Exception:
        return None


def _extract_k_count_match(tunnel_dir: str) -> Optional[bool]:
    """
    Extract K count match: does detected count equal ring_count?
    
    Returns True if len(detected) == ring_count, False otherwise.
    """
    detected_path = os.path.join(tunnel_dir, "detected.csv")
    if not os.path.exists(detected_path):
        return None
    
    try:
        df = pd.read_csv(detected_path)
        k_count = len(df)
        ring_count = _load_ring_count(tunnel_dir)
        
        if ring_count is None:
            return None
        
        return k_count == ring_count
    except Exception:
        return None


def _extract_x_spacing_cv(tunnel_dir: str) -> Optional[float]:
    """
    Extract X spacing coefficient of variation.
    
    Returns CV (std/mean) of consecutive X gaps. Good range: <= 0.15.
    """
    detected_path = os.path.join(tunnel_dir, "detected.csv")
    if not os.path.exists(detected_path):
        return None
    
    try:
        df = pd.read_csv(detected_path)
        if len(df) < 2:
            return None
        
        xs = df["X"].values
        x_diffs = np.diff(np.sort(xs))
        
        if len(x_diffs) == 0 or np.mean(x_diffs) == 0:
            return None
        
        cv = np.std(x_diffs) / np.mean(x_diffs)
        return float(cv)
    except Exception:
        return None


def _extract_midpoint_ratio(tunnel_dir: str) -> Optional[float]:
    """
    Extract midpoint detection ratio.
    
    Returns fraction of K positions detected via "midpoint" method.
    Good range: >= 0.50.
    """
    detected_path = os.path.join(tunnel_dir, "detected.csv")
    if not os.path.exists(detected_path):
        return None
    
    try:
        df = pd.read_csv(detected_path)
        if len(df) == 0:
            return None
        
        if "Type" not in df.columns:
            return None
        
        midpoint_count = int((df["Type"] == "midpoint").sum())
        total_count = len(df)
        
        return float(midpoint_count / total_count)
    except Exception:
        return None


def _extract_y_pattern_consistency(tunnel_dir: str) -> Optional[float]:
    """
    Extract Y pattern consistency.
    
    Splits Y into even/odd groups and computes average intra-group std
    as % of image height. Works for both continuous and staggered patterns.
    Good range: <= 3.0%.
    """
    detected_path = os.path.join(tunnel_dir, "detected.csv")
    if not os.path.exists(detected_path):
        return None
    
    try:
        df = pd.read_csv(detected_path)
        if len(df) < 2:
            return None
        
        if "Y" not in df.columns:
            return None
        
        ys = df["Y"].values
        image_height = _load_image_height(tunnel_dir)
        
        if image_height is None or image_height == 0:
            return None
        
        # Split into even/odd groups (works for both continuous and staggered)
        y_even = ys[0::2]
        y_odd = ys[1::2]
        
        if len(y_even) == 0 or len(y_odd) == 0:
            return None
        
        # Average intra-group std
        std_even = float(np.std(y_even)) if len(y_even) > 1 else 0.0
        std_odd = float(np.std(y_odd)) if len(y_odd) > 1 else 0.0
        intra_std = (std_even + std_odd) / 2.0
        
        # Express as % of image height
        score = intra_std / image_height * 100.0
        
        return float(score)
    except Exception:
        return None


# =============================================================================
# Guardrail Check
# =============================================================================

def _check_guardrails(
    k_count_match: Optional[bool],
    x_spacing_cv: Optional[float],
    midpoint_ratio: Optional[float],
    y_pattern_consistency: Optional[float],
) -> Tuple[bool, List[str]]:
    """Check if metrics pass guardrail thresholds."""
    violations = []
    
    if k_count_match is not None:
        if not k_count_match:
            violations.append(f"k_count_match=False (detected count != ring_count)")
    
    if x_spacing_cv is not None:
        if x_spacing_cv > X_SPACING_CV_MAX:
            violations.append(f"x_spacing_cv={x_spacing_cv:.4f} > {X_SPACING_CV_MAX} (uneven spacing)")
    
    if midpoint_ratio is not None:
        if midpoint_ratio < MIDPOINT_RATIO_MIN:
            violations.append(f"midpoint_ratio={midpoint_ratio:.2f} < {MIDPOINT_RATIO_MIN} (low confidence)")
    
    if y_pattern_consistency is not None:
        if y_pattern_consistency > Y_PATTERN_CONSISTENCY_MAX:
            violations.append(f"y_pattern_consistency={y_pattern_consistency:.2f}% > {Y_PATTERN_CONSISTENCY_MAX}% (inconsistent Y pattern)")
    
    return (len(violations) == 0, violations)


# =============================================================================
# Main Extraction
# =============================================================================

def extract_detection_metrics(tunnel_dir: str) -> dict:
    """
    Extract critical intrinsic metrics from detection outputs.
    
    Returns dict with:
        - det_k_count_match: K count matches ring_count (bool)
        - det_x_spacing_cv: X spacing coefficient of variation (float)
        - det_midpoint_ratio: Fraction detected via midpoint (float)
        - det_y_pattern_consistency: Y pattern consistency as % of image height (float)
        - det_ready_for_sam: Pass/fail verdict (bool)
        - det_guardrail_violations: List of threshold violations
    """
    k_count_match = _extract_k_count_match(tunnel_dir)
    x_spacing_cv = _extract_x_spacing_cv(tunnel_dir)
    midpoint_ratio = _extract_midpoint_ratio(tunnel_dir)
    y_pattern_consistency = _extract_y_pattern_consistency(tunnel_dir)
    
    passed, violations = _check_guardrails(
        k_count_match, x_spacing_cv, midpoint_ratio, y_pattern_consistency
    )
    
    return {
        "det_k_count_match": k_count_match,
        "det_x_spacing_cv": x_spacing_cv,
        "det_midpoint_ratio": midpoint_ratio,
        "det_y_pattern_consistency": y_pattern_consistency,
        "det_ready_for_sam": passed,
        "det_guardrail_violations": violations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract detection intrinsic metrics for SAM readiness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 1-4, 2-2)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    args = parser.parse_args()

    tunnel_dir = os.path.join(args.data_dir, args.tunnel_id)
    if not os.path.isdir(tunnel_dir):
        print(f"Error: {tunnel_dir} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting detection metrics from {tunnel_dir} ...")
    metrics = extract_detection_metrics(tunnel_dir)

    out_path = args.output or os.path.join(tunnel_dir, "detection_characteristics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {out_path}\n")
    print("Metrics:")
    print(f"  det_k_count_match:         {metrics['det_k_count_match']}" if metrics['det_k_count_match'] is not None else "  det_k_count_match:         N/A")
    print(f"  det_x_spacing_cv:         {metrics['det_x_spacing_cv']:.4f}" if metrics['det_x_spacing_cv'] is not None else "  det_x_spacing_cv:         N/A")
    print(f"  det_midpoint_ratio:        {metrics['det_midpoint_ratio']:.2f}" if metrics['det_midpoint_ratio'] is not None else "  det_midpoint_ratio:        N/A")
    print(f"  det_y_pattern_consistency: {metrics['det_y_pattern_consistency']:.2f}%" if metrics['det_y_pattern_consistency'] is not None else "  det_y_pattern_consistency: N/A")
    print(f"\n  det_ready_for_sam:         {metrics['det_ready_for_sam']}")
    
    if metrics["det_guardrail_violations"]:
        print("\n  Violations:")
        for v in metrics["det_guardrail_violations"]:
            print(f"    - {v}")


if __name__ == "__main__":
    main()
