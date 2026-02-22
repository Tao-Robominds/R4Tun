"""
Extract SAM Segmentation Output Intrinsic Metrics (Quality Assessment)

Extracts ONLY critical metrics that can be thresholded to determine if
SAM segmentation output is of acceptable quality.

Critical Metrics:
    1. sam_segment_type_completeness - All expected block types present? (boolean)
    2. sam_ring_completeness_avg - Avg fraction of expected types per ring [>=0.85]
    3. sam_mask_coverage_pct - Segmented / mappable points [55-90%]
    4. sam_k_size_ratio - K-block proportion of segmented area [3-20%]

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
SEGMENT_TYPE_COMPLETENESS_REQUIRED = True  # Must have all expected types
RING_COMPLETENESS_AVG_MIN = 0.85  # At least 85% of expected types per ring on average
MASK_COVERAGE_PCT_MIN = 55.0  # At least 55% of mappable points segmented
MASK_COVERAGE_PCT_MAX = 90.0  # At most 90% (over-segmentation)
K_SIZE_RATIO_MIN = 3.0  # K-blocks should be at least 3% of segmented area
K_SIZE_RATIO_MAX = 20.0  # K-blocks should be at most 20% of segmented area


# =============================================================================
# Helper Functions
# =============================================================================

def _detect_segment_count(tunnel_dir: str) -> Optional[int]:
    """
    Detect segment count from tunnel geometry (radius → circumference).
    
    Compares circumference to expected values for 6 vs 7 segments.
    """
    enhanced_path = os.path.join(tunnel_dir, 'enhanced.csv')
    
    if os.path.exists(enhanced_path):
        try:
            df = pd.read_csv(enhanced_path)
            if 'r' in df.columns:
                avg_radius = df['r'].mean()
                circumference_mm = 2 * np.pi * avg_radius * 1000
                
                # Default heights for 6-segment tunnels
                DEFAULT_K_HEIGHT_MM = 1079.92
                DEFAULT_AB_HEIGHT_MM = 3239.77
                
                circ_6 = DEFAULT_K_HEIGHT_MM + 5 * DEFAULT_AB_HEIGHT_MM
                circ_7 = DEFAULT_K_HEIGHT_MM + 6 * DEFAULT_AB_HEIGHT_MM
                
                segment_count = 6 if abs(circumference_mm - circ_6) < abs(circumference_mm - circ_7) else 7
                return segment_count
        except Exception:
            pass
    
    # Fallback: try to infer from final.csv
    final_path = os.path.join(tunnel_dir, 'final.csv')
    if os.path.exists(final_path):
        try:
            df = pd.read_csv(final_path)
            if 'pred' in df.columns:
                # Count unique segment types (excluding 0=BG and 8=unmapped)
                seg_types = [v for v in df['pred'].unique() if v > 0 and v < 8]
                if len(seg_types) > 0:
                    return len(seg_types)
        except Exception:
            pass
    
    return None


def _get_expected_segment_types(segment_count: int) -> List[int]:
    """Get expected segment type IDs for given segment count."""
    if segment_count == 7:
        return [1, 2, 3, 4, 5, 6, 7]  # K, B1, A1, A2, A3, A4, B2
    else:
        return [1, 2, 3, 4, 5, 6]  # K, B1, A1, A2, A3, B2


# =============================================================================
# Metric Extraction
# =============================================================================

def _extract_segment_type_completeness(tunnel_dir: str) -> Optional[bool]:
    """
    Extract segment type completeness: are all expected block types present?
    
    Returns True if all expected segment types (1-6 or 1-7) appear in pred column.
    """
    final_path = os.path.join(tunnel_dir, 'final.csv')
    if not os.path.exists(final_path):
        return None
    
    try:
        df = pd.read_csv(final_path)
        if 'pred' not in df.columns:
            return None
        
        # Get expected segment count
        segment_count = _detect_segment_count(tunnel_dir)
        if segment_count is None:
            return None
        
        expected_types = set(_get_expected_segment_types(segment_count))
        
        # Get actual segment types present (excluding 0=BG and 8=unmapped)
        actual_types = set([int(v) for v in df['pred'].unique() if v > 0 and v < 8])
        
        # Check if all expected types are present
        return expected_types.issubset(actual_types)
    except Exception:
        return None


def _extract_ring_completeness_avg(tunnel_dir: str) -> Optional[float]:
    """
    Extract average ring completeness.
    
    For each ring with segments, compute fraction of expected types present.
    Returns average across all rings. Good range: >= 0.85.
    """
    final_path = os.path.join(tunnel_dir, 'final.csv')
    if not os.path.exists(final_path):
        return None
    
    try:
        df = pd.read_csv(final_path)
        if 'pred' not in df.columns or 'pred_ring' not in df.columns:
            return None
        
        # Get expected segment count
        segment_count = _detect_segment_count(tunnel_dir)
        if segment_count is None:
            return None
        
        expected_types = set(_get_expected_segment_types(segment_count))
        expected_count = len(expected_types)
        
        # Get all rings that have segments (pred_ring >= 0, excluding -1)
        rings_with_segments = sorted([r for r in df[df['pred_ring'] >= 0]['pred_ring'].unique()])
        
        if len(rings_with_segments) == 0:
            return None
        
        completeness_scores = []
        for ring in rings_with_segments:
            # Get segment types present in this ring (excluding 0=BG and 8=unmapped)
            ring_data = df[(df['pred_ring'] == ring) & (df['pred'] > 0) & (df['pred'] < 8)]
            types_in_ring = set([int(v) for v in ring_data['pred'].unique()])
            
            # Compute completeness: how many expected types are present?
            completeness = len(types_in_ring.intersection(expected_types)) / expected_count
            completeness_scores.append(completeness)
        
        if len(completeness_scores) == 0:
            return None
        
        return float(np.mean(completeness_scores))
    except Exception:
        return None


def _extract_mask_coverage_pct(tunnel_dir: str) -> Optional[float]:
    """
    Extract mask coverage percentage.
    
    Returns (segmented points / mappable points) * 100.
    Mappable = all points where pred != 8 (unmapped).
    Segmented = points where pred > 0 and pred < 8.
    Good range: [55%, 90%].
    """
    final_path = os.path.join(tunnel_dir, 'final.csv')
    if not os.path.exists(final_path):
        return None
    
    try:
        df = pd.read_csv(final_path)
        if 'pred' not in df.columns:
            return None
        
        # Mappable points: all points where pred != 8
        mappable = (df['pred'] != 8).sum()
        if mappable == 0:
            return None
        
        # Segmented points: pred > 0 and pred < 8
        segmented = ((df['pred'] > 0) & (df['pred'] < 8)).sum()
        
        coverage_pct = segmented / mappable * 100.0
        return float(coverage_pct)
    except Exception:
        return None


def _extract_k_size_ratio(tunnel_dir: str) -> Optional[float]:
    """
    Extract K-block size ratio.
    
    Returns (K-block points / segmented points) * 100.
    K-blocks are segment type 1. Good range: [3%, 20%].
    """
    final_path = os.path.join(tunnel_dir, 'final.csv')
    if not os.path.exists(final_path):
        return None
    
    try:
        df = pd.read_csv(final_path)
        if 'pred' not in df.columns:
            return None
        
        # Segmented points: pred > 0 and pred < 8
        segmented = ((df['pred'] > 0) & (df['pred'] < 8)).sum()
        if segmented == 0:
            return None
        
        # K-block points: pred == 1
        k_count = (df['pred'] == 1).sum()
        
        k_ratio = k_count / segmented * 100.0
        return float(k_ratio)
    except Exception:
        return None


# =============================================================================
# Guardrail Check
# =============================================================================

def _check_guardrails(
    segment_type_completeness: Optional[bool],
    ring_completeness_avg: Optional[float],
    mask_coverage_pct: Optional[float],
    k_size_ratio: Optional[float],
) -> Tuple[bool, List[str]]:
    """Check if metrics pass guardrail thresholds."""
    violations = []
    
    if segment_type_completeness is not None:
        if not segment_type_completeness:
            violations.append("segment_type_completeness=False (missing expected block types)")
    
    if ring_completeness_avg is not None:
        if ring_completeness_avg < RING_COMPLETENESS_AVG_MIN:
            violations.append(f"ring_completeness_avg={ring_completeness_avg:.3f} < {RING_COMPLETENESS_AVG_MIN} (incomplete rings)")
    
    if mask_coverage_pct is not None:
        if mask_coverage_pct < MASK_COVERAGE_PCT_MIN:
            violations.append(f"mask_coverage_pct={mask_coverage_pct:.1f}% < {MASK_COVERAGE_PCT_MIN}% (under-segmentation)")
        elif mask_coverage_pct > MASK_COVERAGE_PCT_MAX:
            violations.append(f"mask_coverage_pct={mask_coverage_pct:.1f}% > {MASK_COVERAGE_PCT_MAX}% (over-segmentation)")
    
    if k_size_ratio is not None:
        if k_size_ratio < K_SIZE_RATIO_MIN:
            violations.append(f"k_size_ratio={k_size_ratio:.1f}% < {K_SIZE_RATIO_MIN}% (K-blocks too small)")
        elif k_size_ratio > K_SIZE_RATIO_MAX:
            violations.append(f"k_size_ratio={k_size_ratio:.1f}% > {K_SIZE_RATIO_MAX}% (K-blocks too large)")
    
    return (len(violations) == 0, violations)


# =============================================================================
# Main Extraction
# =============================================================================

def extract_sam_metrics(tunnel_dir: str) -> dict:
    """
    Extract critical intrinsic metrics from SAM segmentation outputs.
    
    Returns dict with:
        - sam_segment_type_completeness: All expected types present (bool)
        - sam_ring_completeness_avg: Avg ring completeness (float)
        - sam_mask_coverage_pct: Mask coverage percentage (float)
        - sam_k_size_ratio: K-block size ratio (float)
        - sam_ready_for_evaluation: Pass/fail verdict (bool)
        - sam_guardrail_violations: List of threshold violations
    """
    segment_type_completeness = _extract_segment_type_completeness(tunnel_dir)
    ring_completeness_avg = _extract_ring_completeness_avg(tunnel_dir)
    mask_coverage_pct = _extract_mask_coverage_pct(tunnel_dir)
    k_size_ratio = _extract_k_size_ratio(tunnel_dir)
    
    passed, violations = _check_guardrails(
        segment_type_completeness, ring_completeness_avg, mask_coverage_pct, k_size_ratio
    )
    
    return {
        "sam_segment_type_completeness": segment_type_completeness,
        "sam_ring_completeness_avg": ring_completeness_avg,
        "sam_mask_coverage_pct": mask_coverage_pct,
        "sam_k_size_ratio": k_size_ratio,
        "sam_ready_for_evaluation": passed,
        "sam_guardrail_violations": violations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract SAM segmentation intrinsic metrics for quality assessment",
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

    print(f"Extracting SAM metrics from {tunnel_dir} ...")
    metrics = extract_sam_metrics(tunnel_dir)

    out_path = args.output or os.path.join(tunnel_dir, "sam_characteristics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {out_path}\n")
    print("Metrics:")
    print(f"  sam_segment_type_completeness: {metrics['sam_segment_type_completeness']}" if metrics['sam_segment_type_completeness'] is not None else "  sam_segment_type_completeness: N/A")
    print(f"  sam_ring_completeness_avg:    {metrics['sam_ring_completeness_avg']:.3f}" if metrics['sam_ring_completeness_avg'] is not None else "  sam_ring_completeness_avg:    N/A")
    print(f"  sam_mask_coverage_pct:         {metrics['sam_mask_coverage_pct']:.1f}%" if metrics['sam_mask_coverage_pct'] is not None else "  sam_mask_coverage_pct:         N/A")
    print(f"  sam_k_size_ratio:              {metrics['sam_k_size_ratio']:.1f}%" if metrics['sam_k_size_ratio'] is not None else "  sam_k_size_ratio:              N/A")
    print(f"\n  sam_ready_for_evaluation:      {metrics['sam_ready_for_evaluation']}")
    
    if metrics["sam_guardrail_violations"]:
        print("\n  Violations:")
        for v in metrics["sam_guardrail_violations"]:
            print(f"    - {v}")


if __name__ == "__main__":
    main()
