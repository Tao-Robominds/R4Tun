"""
Extract Preprocessing Output Intrinsic Metrics (Detection Readiness)

Extracts ONLY critical metrics that can be thresholded to determine if
preprocessing output is suitable for detection.

Critical Metrics:
    1. pre_theta_coverage_pct  - Is unfolding complete? [98-108%]
    2. pre_depth_map_valid_pixels - Is depth map in sparse regime? [8k-35k]
    3. pre_point_retention_pct - Is denoising reasonable? [65-98%]
    4. pre_depth_map_max_empty_row_run - No big white areas? [<=100 rows]

Usage:
    python extract_preprocessing_characteristics.py 1-4 [--data-dir data] [--output ...]
"""

import argparse
import json
import os
import sys
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image


# =============================================================================
# Guardrail Thresholds
# =============================================================================
THETA_COVERAGE_PCT_MIN = 98.0
THETA_COVERAGE_PCT_MAX = 108.0
POINT_RETENTION_PCT_MIN = 65.0
POINT_RETENTION_PCT_MAX = 98.0
DEPTH_MAP_VALID_PIXELS_MIN = 8_000
DEPTH_MAP_VALID_PIXELS_MAX = 35_000
DEPTH_MAP_MAX_EMPTY_ROW_RUN = 100


# =============================================================================
# Metric Extraction
# =============================================================================

def _load_tunnel_diameter(tunnel_dir: str) -> Optional[float]:
    """Load tunnel_diameter from parameters_preprocessing.json if available."""
    for candidate in [
        os.path.join(tunnel_dir, "parameters_preprocessing.json"),
        os.path.join(os.path.dirname(__file__), "..", "parameters", 
                     os.path.basename(tunnel_dir.rstrip(os.sep)), 
                     "parameters_preprocessing.json"),
    ]:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r") as f:
                    params = json.load(f)
                return float(params.get("tunnel_diameter", 0)) or None
            except Exception:
                pass
    return None


def _extract_theta_coverage(tunnel_dir: str) -> Optional[float]:
    """
    Extract theta coverage percentage from unwrapped.csv.
    
    Returns coverage as % of 360 degrees. Good range: 98-108%.
    """
    path = os.path.join(tunnel_dir, "unwrapped.csv")
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path)
        if "theta" not in df.columns or len(df) == 0:
            return None
        
        theta = df["theta"].dropna()
        if len(theta) == 0:
            return None
        
        t_min, t_max = float(theta.min()), float(theta.max())
        
        # Theta is stored as angle_deg * (pi * tunnel_diameter / 360)
        # Convert back to degrees
        diameter = _load_tunnel_diameter(tunnel_dir)
        if diameter and diameter > 0:
            scale = np.pi * diameter / 360.0
            angle_min_deg = t_min / scale
            angle_max_deg = t_max / scale
            return float((angle_max_deg - angle_min_deg) / 360.0 * 100.0)
        
        return None
    except Exception:
        return None


def _extract_point_retention(tunnel_dir: str) -> Optional[float]:
    """
    Extract point retention percentage (denoised / unwrapped).
    
    Returns retention as %. Good range: 65-98%.
    """
    unwrapped_path = os.path.join(tunnel_dir, "unwrapped.csv")
    denoised_path = os.path.join(tunnel_dir, "denoised.csv")
    
    if not os.path.exists(unwrapped_path) or not os.path.exists(denoised_path):
        return None
    
    try:
        df_unwrapped = pd.read_csv(unwrapped_path)
        df_denoised = pd.read_csv(denoised_path)
        
        n_unwrapped = len(df_unwrapped)
        if n_unwrapped == 0:
            return None
        
        # Valid denoised points: pred != 0
        if "pred" in df_denoised.columns:
            n_denoised = int((df_denoised["pred"] != 0).sum())
        else:
            n_denoised = len(df_denoised)
        
        return float(n_denoised / n_unwrapped * 100.0)
    except Exception:
        return None


def _extract_depth_map_valid_pixels(tunnel_dir: str) -> Optional[int]:
    """
    Extract valid pixel count from depth_map_outlier.npy.
    
    This is the MOST CRITICAL metric. Detection runs on this file.
    Good range: 8k-35k (sparse). >100k indicates over-interpolation.
    """
    path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(path):
        return None
    
    try:
        depth = np.load(path)
        valid_count = int(np.sum(~np.isnan(depth)))
        return valid_count
    except Exception:
        return None


def _extract_depth_map_max_empty_row_run(tunnel_dir: str) -> Optional[int]:
    """
    Extract maximum consecutive row run with >80% white pixels from depth_map.png.
    
    This detects large horizontal white bands that would break line detection.
    Good range: <=100 rows (approximately 4% of typical depth map height).
    """
    path = os.path.join(tunnel_dir, "depth_map.png")
    if not os.path.exists(path):
        return None
    
    try:
        img = Image.open(path)
        # Convert to grayscale numpy array
        if img.mode != 'L':
            img = img.convert('L')
        img_array = np.array(img)
        
        height, width = img_array.shape
        if height == 0 or width == 0:
            return None
        
        # For each row, compute fraction of pixels >= 250 (white)
        white_threshold = 250
        empty_rows = []
        for row_idx in range(height):
            row = img_array[row_idx, :]
            white_pixels = np.sum(row >= white_threshold)
            white_fraction = white_pixels / width
            if white_fraction > 0.80:
                empty_rows.append(row_idx)
        
        if not empty_rows:
            return 0
        
        # Find max consecutive run
        max_run = 1
        current_run = 1
        for i in range(1, len(empty_rows)):
            if empty_rows[i] == empty_rows[i-1] + 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        return max_run
    except Exception:
        return None


# =============================================================================
# Guardrail Check
# =============================================================================

def _check_guardrails(
    theta_coverage: Optional[float],
    point_retention: Optional[float],
    valid_pixels: Optional[int],
    max_empty_row_run: Optional[int],
) -> Tuple[bool, List[str]]:
    """Check if metrics pass guardrail thresholds."""
    violations = []
    
    if theta_coverage is not None:
        if theta_coverage < THETA_COVERAGE_PCT_MIN:
            violations.append(f"theta_coverage={theta_coverage:.1f}% < {THETA_COVERAGE_PCT_MIN}%")
        elif theta_coverage > THETA_COVERAGE_PCT_MAX:
            violations.append(f"theta_coverage={theta_coverage:.1f}% > {THETA_COVERAGE_PCT_MAX}%")
    
    if point_retention is not None:
        if point_retention < POINT_RETENTION_PCT_MIN:
            violations.append(f"point_retention={point_retention:.1f}% < {POINT_RETENTION_PCT_MIN}%")
        elif point_retention > POINT_RETENTION_PCT_MAX:
            violations.append(f"point_retention={point_retention:.1f}% > {POINT_RETENTION_PCT_MAX}%")
    
    if valid_pixels is not None:
        if valid_pixels < DEPTH_MAP_VALID_PIXELS_MIN:
            violations.append(f"depth_map_valid_pixels={valid_pixels} < {DEPTH_MAP_VALID_PIXELS_MIN} (too sparse)")
        elif valid_pixels > DEPTH_MAP_VALID_PIXELS_MAX:
            violations.append(f"depth_map_valid_pixels={valid_pixels} > {DEPTH_MAP_VALID_PIXELS_MAX} (over-filled)")
    
    if max_empty_row_run is not None:
        if max_empty_row_run > DEPTH_MAP_MAX_EMPTY_ROW_RUN:
            violations.append(f"depth_map_max_empty_row_run={max_empty_row_run} > {DEPTH_MAP_MAX_EMPTY_ROW_RUN} (big white areas)")
    
    return (len(violations) == 0, violations)


# =============================================================================
# Main Extraction
# =============================================================================

def extract_preprocessing_metrics(tunnel_dir: str) -> dict:
    """
    Extract critical intrinsic metrics from preprocessing outputs.
    
    Returns dict with:
        - pre_theta_coverage_pct: Unfolding completeness (%)
        - pre_point_retention_pct: Denoising retention (%)
        - pre_depth_map_valid_pixels: Valid pixels in depth map (count)
        - pre_depth_map_max_empty_row_run: Max consecutive empty rows (count)
        - pre_ready_for_detection: Pass/fail verdict (bool)
        - pre_guardrail_violations: List of threshold violations
    """
    theta_coverage = _extract_theta_coverage(tunnel_dir)
    point_retention = _extract_point_retention(tunnel_dir)
    valid_pixels = _extract_depth_map_valid_pixels(tunnel_dir)
    max_empty_row_run = _extract_depth_map_max_empty_row_run(tunnel_dir)
    
    passed, violations = _check_guardrails(theta_coverage, point_retention, valid_pixels, max_empty_row_run)
    
    return {
        "pre_theta_coverage_pct": theta_coverage,
        "pre_point_retention_pct": point_retention,
        "pre_depth_map_valid_pixels": valid_pixels,
        "pre_depth_map_max_empty_row_run": max_empty_row_run,
        "pre_ready_for_detection": passed,
        "pre_guardrail_violations": violations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract preprocessing intrinsic metrics for detection readiness",
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

    print(f"Extracting preprocessing metrics from {tunnel_dir} ...")
    metrics = extract_preprocessing_metrics(tunnel_dir)

    out_path = args.output or os.path.join(tunnel_dir, "preprocessing_characteristics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {out_path}\n")
    print("Metrics:")
    print(f"  pre_theta_coverage_pct:     {metrics['pre_theta_coverage_pct']:.1f}%" if metrics['pre_theta_coverage_pct'] else "  pre_theta_coverage_pct:     N/A")
    print(f"  pre_point_retention_pct:    {metrics['pre_point_retention_pct']:.1f}%" if metrics['pre_point_retention_pct'] else "  pre_point_retention_pct:    N/A")
    print(f"  pre_depth_map_valid_pixels: {metrics['pre_depth_map_valid_pixels']}" if metrics['pre_depth_map_valid_pixels'] else "  pre_depth_map_valid_pixels: N/A")
    print(f"  pre_depth_map_max_empty_row_run: {metrics['pre_depth_map_max_empty_row_run']}" if metrics['pre_depth_map_max_empty_row_run'] is not None else "  pre_depth_map_max_empty_row_run: N/A")
    print(f"\n  pre_ready_for_detection:    {metrics['pre_ready_for_detection']}")
    
    if metrics["pre_guardrail_violations"]:
        print("\n  Violations:")
        for v in metrics["pre_guardrail_violations"]:
            print(f"    - {v}")


if __name__ == "__main__":
    main()
