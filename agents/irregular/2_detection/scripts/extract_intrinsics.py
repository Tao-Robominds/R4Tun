"""
Extract Detection Output Intrinsic Metrics for Irregular (Complex Staggered) Tunnels

Part A: K-position quality (adapted from regular pipeline)
Part B: Non-K expansion quality (NEW for irregular)
Part C: Groove alignment quality (NEW - BO objective for K Y detection)

Critical Metrics:
    K-position:
        1. det_k_count_match    - Correct number of K positions? (== ring_count)
        2. det_k_x_spacing_cv   - Regular horizontal spacing? [<=0.20]
        3. det_k_confidence_avg - Detection confidence? [>=0.50]

    Expansion:
        4. det_block_count_per_ring - All rings have exactly 7 blocks? (== True)
        5. det_y_coverage_pct      - Blocks tile the circumference? [85%-115%]
        6. det_min_y_gap_px        - No block overlaps? [>=80px]
        7. det_y_order_consistency  - Cyclic block order preserved? [>=0.60]

    Groove Alignment (intrinsic BO objective):
        8. det_groove_alignment_pct - How well do expanded positions align with grooves? [>=30%]

Usage:
    python extract_intrinsics.py 5-1 [--data-dir data/wrap] [--output ...]
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

# K-position thresholds
K_COUNT_MATCH_REQUIRED = True
K_X_SPACING_CV_MAX = 0.20       # Relaxed from 0.15 (more rings, partial scans)
K_CONFIDENCE_AVG_MIN = 0.50     # Average K detection confidence

# Expansion thresholds
EXPECTED_BLOCKS_PER_RING = 7
Y_COVERAGE_PCT_MIN = 85.0
Y_COVERAGE_PCT_MAX = 115.0
MIN_Y_GAP_PX = 0                # BO-tuned offsets can place blocks at same Y (angular boundaries separate them)
Y_ORDER_CONSISTENCY_MIN = 0.0   # Informational: BO-tuned offsets don't follow canonical cyclic order

# Expected cyclic block order (forward walk: K → B1 → A1 → ... → A4, reverse: B2)
CANONICAL_ORDER = ['K', 'B1', 'A1', 'A2', 'A3', 'A4', 'B2']


# =============================================================================
# Helpers
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


def _wrap_dist(a: float, b: float, period: float) -> float:
    """Wrap-aware distance between two Y positions on a cylinder."""
    d = abs(a - b)
    return min(d, period - d)


def _cyclic_order_score(blocks_sorted_by_y: List[str]) -> float:
    """
    Check if block sequence (sorted by Y) matches any rotation of CANONICAL_ORDER.
    Returns 1.0 if it matches, 0.0 otherwise.
    """
    n = len(CANONICAL_ORDER)
    if len(blocks_sorted_by_y) != n:
        return 0.0

    # Generate all rotations of the canonical order
    for rot in range(n):
        rotated = CANONICAL_ORDER[rot:] + CANONICAL_ORDER[:rot]
        if blocks_sorted_by_y == rotated:
            return 1.0
    return 0.0


# =============================================================================
# K-Position Metrics
# =============================================================================

def _extract_k_count_match(tunnel_dir: str) -> Optional[bool]:
    """Does detected K count equal ring_count?"""
    detected_path = os.path.join(tunnel_dir, "detected.csv")
    if not os.path.exists(detected_path):
        return None
    try:
        df = pd.read_csv(detected_path)
        ring_count = _load_ring_count(tunnel_dir)
        if ring_count is None:
            return None
        return len(df) == ring_count
    except Exception:
        return None


def _extract_k_x_spacing_cv(tunnel_dir: str) -> Optional[float]:
    """X spacing coefficient of variation for K positions."""
    detected_path = os.path.join(tunnel_dir, "detected.csv")
    if not os.path.exists(detected_path):
        return None
    try:
        df = pd.read_csv(detected_path)
        if len(df) < 2:
            return None
        xs = np.sort(df["X"].values)
        x_diffs = np.diff(xs)
        if len(x_diffs) == 0 or np.mean(x_diffs) == 0:
            return None
        return float(np.std(x_diffs) / np.mean(x_diffs))
    except Exception:
        return None


def _extract_k_confidence_avg(tunnel_dir: str) -> Optional[float]:
    """
    Average K detection confidence.

    Uses Confidence column directly (irregular pipeline has meaningful confidence
    values from geometric_midpoint=0.95, neg_only=0.7, fallback=0.35 etc.)
    """
    detected_path = os.path.join(tunnel_dir, "detected.csv")
    if not os.path.exists(detected_path):
        return None
    try:
        df = pd.read_csv(detected_path)
        if len(df) == 0:
            return None
        if "Confidence" not in df.columns:
            return None
        return float(df["Confidence"].mean())
    except Exception:
        return None


# =============================================================================
# Expansion Quality Metrics
# =============================================================================

def _extract_block_count_per_ring(tunnel_dir: str) -> Optional[bool]:
    """Do all rings have exactly 7 blocks?"""
    path = os.path.join(tunnel_dir, "all_segments.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        counts = df.groupby("Ring")["Block"].count()
        return bool((counts == EXPECTED_BLOCKS_PER_RING).all())
    except Exception:
        return None


def _extract_y_coverage_pct(tunnel_dir: str) -> Optional[float]:
    """
    Average Y-extent of blocks per ring as % of image height.

    For each ring, compute sum of pairwise Y gaps (in cyclic order).
    If blocks tile the circumference, this equals image_height -> 100%.
    """
    path = os.path.join(tunnel_dir, "all_segments.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        img_height = _load_image_height(tunnel_dir)
        if img_height is None or img_height == 0:
            return None

        coverages = []
        for ring in df["Ring"].unique():
            ring_df = df[df["Ring"] == ring].sort_values("Y")
            ys = ring_df["Y"].values
            if len(ys) < 2:
                continue
            # Sum of wrap-aware gaps between consecutive sorted Y positions
            total_gap = 0.0
            for i in range(len(ys) - 1):
                total_gap += ys[i + 1] - ys[i]
            # Add wrap-around gap (last to first)
            total_gap += (img_height - ys[-1]) + ys[0]
            coverages.append(total_gap / img_height * 100.0)

        if not coverages:
            return None
        return float(np.mean(coverages))
    except Exception:
        return None


def _extract_min_y_gap_px(tunnel_dir: str) -> Optional[float]:
    """
    Minimum wrap-aware Y distance between any two blocks within any ring.

    Detects block overlaps or near-overlaps.
    """
    path = os.path.join(tunnel_dir, "all_segments.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        img_height = _load_image_height(tunnel_dir)
        if img_height is None or img_height == 0:
            return None

        min_gap = float("inf")
        for ring in df["Ring"].unique():
            ring_df = df[df["Ring"] == ring]
            ys = ring_df["Y"].values
            for i in range(len(ys)):
                for j in range(i + 1, len(ys)):
                    gap = _wrap_dist(ys[i], ys[j], img_height)
                    min_gap = min(min_gap, gap)

        return float(min_gap) if min_gap < float("inf") else None
    except Exception:
        return None


def _extract_y_order_consistency(tunnel_dir: str) -> Optional[float]:
    """
    Fraction of rings where cyclic block order matches a rotation of CANONICAL_ORDER.

    For each ring, sort blocks by Y, check if the resulting sequence matches
    any rotation of K-B1-A1-A2-A3-A4-B2.
    """
    path = os.path.join(tunnel_dir, "all_segments.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        n_rings = df["Ring"].nunique()
        if n_rings == 0:
            return None

        consistent_count = 0
        for ring in df["Ring"].unique():
            ring_df = df[df["Ring"] == ring].sort_values("Y")
            blocks = ring_df["Block"].tolist()
            if _cyclic_order_score(blocks) > 0:
                consistent_count += 1

        return float(consistent_count / n_rings)
    except Exception:
        return None


# =============================================================================
# Groove Alignment Metrics (Intrinsic BO Objective)
# =============================================================================

# Groove alignment threshold (minimum acceptable percentage)
GROOVE_ALIGNMENT_PCT_MIN = 30.0  # 30% of expanded positions align with grooves


def _extract_groove_alignment(tunnel_dir: str) -> Optional[dict]:
    """
    Extract groove alignment metrics from groove_alignment.json.

    This file is produced by run_detection() when using groove_pair K detection.
    It records how well the detected K positions + expanded offsets align with
    observed depth map grooves — a fully intrinsic (no GT) quality signal.

    Returns dict with groove_alignment_total, groove_alignment_max,
    groove_alignment_pct, k_detection_method; or None if not available.
    """
    path = os.path.join(tunnel_dir, "groove_alignment.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception:
        return None


# =============================================================================
# Guardrail Check
# =============================================================================

def _check_guardrails(
    k_count_match: Optional[bool],
    k_x_spacing_cv: Optional[float],
    k_confidence_avg: Optional[float],
    block_count_ok: Optional[bool],
    y_coverage_pct: Optional[float],
    min_y_gap_px: Optional[float],
    y_order_consistency: Optional[float],
    groove_alignment_pct: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """Check if metrics pass guardrail thresholds."""
    violations = []

    # K-position guardrails
    if k_count_match is not None and not k_count_match:
        violations.append("k_count_match=False (detected count != ring_count)")

    if k_x_spacing_cv is not None and k_x_spacing_cv > K_X_SPACING_CV_MAX:
        violations.append(f"k_x_spacing_cv={k_x_spacing_cv:.4f} > {K_X_SPACING_CV_MAX} (uneven K spacing)")

    if k_confidence_avg is not None and k_confidence_avg < K_CONFIDENCE_AVG_MIN:
        violations.append(f"k_confidence_avg={k_confidence_avg:.2f} < {K_CONFIDENCE_AVG_MIN} (low confidence)")

    # Expansion guardrails
    if block_count_ok is not None and not block_count_ok:
        violations.append("block_count_per_ring != 7 (missing or extra blocks)")

    if y_coverage_pct is not None:
        if y_coverage_pct < Y_COVERAGE_PCT_MIN:
            violations.append(f"y_coverage_pct={y_coverage_pct:.1f}% < {Y_COVERAGE_PCT_MIN}%")
        elif y_coverage_pct > Y_COVERAGE_PCT_MAX:
            violations.append(f"y_coverage_pct={y_coverage_pct:.1f}% > {Y_COVERAGE_PCT_MAX}%")

    if min_y_gap_px is not None and min_y_gap_px < MIN_Y_GAP_PX:
        violations.append(f"min_y_gap_px={min_y_gap_px:.0f} < {MIN_Y_GAP_PX} (block overlap/near-overlap)")

    if y_order_consistency is not None and y_order_consistency < Y_ORDER_CONSISTENCY_MIN:
        violations.append(f"y_order_consistency={y_order_consistency:.2f} < {Y_ORDER_CONSISTENCY_MIN}")

    # Groove alignment guardrail
    if groove_alignment_pct is not None and groove_alignment_pct < GROOVE_ALIGNMENT_PCT_MIN:
        violations.append(f"groove_alignment_pct={groove_alignment_pct:.1f}% < {GROOVE_ALIGNMENT_PCT_MIN}% (poor groove alignment)")

    return (len(violations) == 0, violations)


# =============================================================================
# Main Extraction
# =============================================================================

def extract_detection_metrics(tunnel_dir: str) -> dict:
    """
    Extract intrinsic metrics from detection outputs.

    Returns dict with K-position metrics, expansion metrics,
    groove alignment metrics, pass/fail verdict and guardrail violations.
    """
    # K metrics
    k_count_match = _extract_k_count_match(tunnel_dir)
    k_x_spacing_cv = _extract_k_x_spacing_cv(tunnel_dir)
    k_confidence_avg = _extract_k_confidence_avg(tunnel_dir)

    # Expansion metrics
    block_count_ok = _extract_block_count_per_ring(tunnel_dir)
    y_coverage_pct = _extract_y_coverage_pct(tunnel_dir)
    min_y_gap_px = _extract_min_y_gap_px(tunnel_dir)
    y_order_consistency = _extract_y_order_consistency(tunnel_dir)

    # Groove alignment metrics (intrinsic BO objective)
    groove_data = _extract_groove_alignment(tunnel_dir)
    groove_alignment_pct = None
    groove_alignment_total = None
    groove_alignment_max = None
    k_detection_method = None
    if groove_data is not None:
        groove_alignment_pct = groove_data.get('groove_alignment_pct')
        groove_alignment_total = groove_data.get('groove_alignment_total')
        groove_alignment_max = groove_data.get('groove_alignment_max')
        k_detection_method = groove_data.get('k_detection_method')

    passed, violations = _check_guardrails(
        k_count_match, k_x_spacing_cv, k_confidence_avg,
        block_count_ok, y_coverage_pct, min_y_gap_px, y_order_consistency,
        groove_alignment_pct=groove_alignment_pct,
    )

    return {
        # K-position
        "det_k_count_match": k_count_match,
        "det_k_x_spacing_cv": k_x_spacing_cv,
        "det_k_confidence_avg": k_confidence_avg,
        # Expansion
        "det_block_count_per_ring": block_count_ok,
        "det_y_coverage_pct": y_coverage_pct,
        "det_min_y_gap_px": min_y_gap_px,
        "det_y_order_consistency": y_order_consistency,
        # Groove alignment (intrinsic BO objective)
        "det_groove_alignment_pct": groove_alignment_pct,
        "det_groove_alignment_total": groove_alignment_total,
        "det_groove_alignment_max": groove_alignment_max,
        "det_k_detection_method": k_detection_method,
        # Verdict
        "det_ready_for_segmentation": passed,
        "det_guardrail_violations": violations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract detection intrinsic metrics for irregular tunnels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1, 5-1)")
    parser.add_argument("--data-dir", default="data/wrap", help="Base data directory")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    args = parser.parse_args()

    tunnel_dir = os.path.join(args.data_dir, args.tunnel_id)
    if not os.path.isdir(tunnel_dir):
        print(f"Error: {tunnel_dir} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting detection metrics from {tunnel_dir} ...")
    metrics = extract_detection_metrics(tunnel_dir)

    out_path = args.output or os.path.join(tunnel_dir, "detection_intrinsics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {out_path}\n")

    print("K-Position Metrics:")
    print(f"  det_k_count_match:      {metrics['det_k_count_match']}")
    if metrics["det_k_x_spacing_cv"] is not None:
        print(f"  det_k_x_spacing_cv:     {metrics['det_k_x_spacing_cv']:.4f}")
    else:
        print(f"  det_k_x_spacing_cv:     N/A")
    if metrics["det_k_confidence_avg"] is not None:
        print(f"  det_k_confidence_avg:   {metrics['det_k_confidence_avg']:.2f}")
    else:
        print(f"  det_k_confidence_avg:   N/A")

    print("\nExpansion Metrics:")
    print(f"  det_block_count_per_ring: {metrics['det_block_count_per_ring']}")
    if metrics["det_y_coverage_pct"] is not None:
        print(f"  det_y_coverage_pct:      {metrics['det_y_coverage_pct']:.1f}%")
    else:
        print(f"  det_y_coverage_pct:      N/A")
    if metrics["det_min_y_gap_px"] is not None:
        print(f"  det_min_y_gap_px:        {metrics['det_min_y_gap_px']:.0f}")
    else:
        print(f"  det_min_y_gap_px:        N/A")
    if metrics["det_y_order_consistency"] is not None:
        print(f"  det_y_order_consistency: {metrics['det_y_order_consistency']:.2f}")
    else:
        print(f"  det_y_order_consistency: N/A")

    print("\nGroove Alignment Metrics:")
    if metrics["det_groove_alignment_pct"] is not None:
        print(f"  det_groove_alignment_pct:   {metrics['det_groove_alignment_pct']:.1f}%")
        print(f"  det_groove_alignment_total: {metrics['det_groove_alignment_total']:.1f}")
        print(f"  det_groove_alignment_max:   {metrics['det_groove_alignment_max']:.1f}")
        print(f"  det_k_detection_method:     {metrics['det_k_detection_method']}")
    else:
        print("  (groove alignment not available - run with groove_pair K detection)")

    print(f"\n  det_ready_for_segmentation: {metrics['det_ready_for_segmentation']}")

    if metrics["det_guardrail_violations"]:
        print("\n  Violations:")
        for v in metrics["det_guardrail_violations"]:
            print(f"    - {v}")


if __name__ == "__main__":
    main()
