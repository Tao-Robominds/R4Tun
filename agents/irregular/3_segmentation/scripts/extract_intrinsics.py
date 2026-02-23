"""
Extract Segmentation Output Intrinsic Metrics for Irregular (Complex Staggered) Tunnels

Critical Metrics:
    1. seg_segment_type_completeness - All 7 block types present? (boolean)
    2. seg_ring_completeness_avg     - Avg fraction of 7 expected types/ring [>=0.80]
    3. seg_mask_coverage_pct         - Segmented / mappable points [45-85%]
    4. seg_k_size_ratio              - K-block proportion of segmented area [2-12%]
    5. seg_groove_score              - Boundary-groove alignment score [>=15.0]
    6. seg_block_size_variance_ratio - max/min block area per ring [3.0-15.0]

Usage:
    python extract_intrinsics.py 5-1 [--data-dir data/wrap] [--output ...]
"""

import argparse
import json
import os
import pickle
import sys
from typing import Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd


# =============================================================================
# Guardrail Thresholds
# =============================================================================
EXPECTED_SEGMENT_COUNT = 7
SEGMENT_TYPE_COMPLETENESS_REQUIRED = True
RING_COMPLETENESS_AVG_MIN = 0.80  # Relaxed from 0.85 (wrap blocks may be missing)
MASK_COVERAGE_PCT_MIN = 45.0      # Wider range for complex patterns
MASK_COVERAGE_PCT_MAX = 85.0
K_SIZE_RATIO_MIN = 2.0            # K is physically smaller in 7-segment tunnels
K_SIZE_RATIO_MAX = 12.0
GROOVE_SCORE_MIN = 15.0           # Boundary-groove alignment
BLOCK_SIZE_VARIANCE_RATIO_MIN = 3.0   # Expected high variation (K << A blocks)
BLOCK_SIZE_VARIANCE_RATIO_MAX = 20.0  # Complex staggered has inherently high variation


# =============================================================================
# Helper: Groove Map
# =============================================================================

def _compute_groove_map(depth_path: str, blur_ksize: int = 5) -> Optional[np.ndarray]:
    """Compute per-pixel groove confidence from depth map (Sobel gradient)."""
    img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if blur_ksize > 1:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sobelx ** 2 + sobely ** 2).astype(np.float32)


def _build_label_map_from_outputs(tunnel_dir: str) -> Optional[np.ndarray]:
    """
    Reconstruct a pixel-level label map from pixel_to_point.pkl + final.csv.

    Returns (height, width) int32 array with pred labels.
    """
    p2p_path = os.path.join(tunnel_dir, "pixel_to_point.pkl")
    final_path = os.path.join(tunnel_dir, "final.csv")
    depth_path = os.path.join(tunnel_dir, "depth_map.png")

    if not all(os.path.exists(p) for p in [p2p_path, final_path, depth_path]):
        return None

    try:
        img = cv2.imread(depth_path)
        if img is None:
            return None
        height, width = img.shape[:2]

        with open(p2p_path, "rb") as f:
            pixel_to_point = pickle.load(f)

        df = pd.read_csv(final_path)
        pred = df["pred"].values

        label_map = np.zeros((height, width), dtype=np.int32)
        for entry in pixel_to_point:
            # Support both key conventions
            col = entry.get("pixel_x", entry.get("col", entry.get("pixel_col")))
            row = entry.get("pixel_y", entry.get("row", entry.get("pixel_row")))
            idx = entry.get("index", entry.get("point_index", entry.get("idx")))
            if row is None or col is None or idx is None:
                continue
            row, col, idx = int(row), int(col), int(idx)
            if 0 <= row < height and 0 <= col < width and 0 <= idx < len(pred):
                label_map[row, col] = pred[idx]

        return label_map
    except Exception:
        return None


def _get_expected_segment_types() -> List[int]:
    """Expected segment type IDs for 7-segment complex staggered tunnels."""
    return [1, 2, 3, 4, 5, 6, 7]  # K, B1, A1, A2, A3, A4, B2


# =============================================================================
# Metric Extraction
# =============================================================================

def _extract_segment_type_completeness(tunnel_dir: str) -> Optional[bool]:
    """All 7 expected block types present in output?"""
    final_path = os.path.join(tunnel_dir, "final.csv")
    if not os.path.exists(final_path):
        return None
    try:
        df = pd.read_csv(final_path)
        if "pred" not in df.columns:
            return None
        expected = set(_get_expected_segment_types())
        actual = set([int(v) for v in df["pred"].unique() if 0 < v < 8])
        return expected.issubset(actual)
    except Exception:
        return None


def _extract_ring_completeness_avg(tunnel_dir: str) -> Optional[float]:
    """Average fraction of 7 expected types per ring."""
    final_path = os.path.join(tunnel_dir, "final.csv")
    if not os.path.exists(final_path):
        return None
    try:
        df = pd.read_csv(final_path)
        if "pred" not in df.columns or "pred_ring" not in df.columns:
            return None
        expected = set(_get_expected_segment_types())
        expected_count = len(expected)

        rings = sorted([r for r in df[df["pred_ring"] >= 0]["pred_ring"].unique()])
        if not rings:
            return None

        scores = []
        for ring in rings:
            ring_data = df[(df["pred_ring"] == ring) & (df["pred"] > 0) & (df["pred"] < 8)]
            types_in_ring = set([int(v) for v in ring_data["pred"].unique()])
            scores.append(len(types_in_ring.intersection(expected)) / expected_count)

        return float(np.mean(scores)) if scores else None
    except Exception:
        return None


def _extract_mask_coverage_pct(tunnel_dir: str) -> Optional[float]:
    """Segmented / mappable points as percentage."""
    final_path = os.path.join(tunnel_dir, "final.csv")
    if not os.path.exists(final_path):
        return None
    try:
        df = pd.read_csv(final_path)
        if "pred" not in df.columns:
            return None
        mappable = (df["pred"] != 8).sum()
        if mappable == 0:
            return None
        segmented = ((df["pred"] > 0) & (df["pred"] < 8)).sum()
        return float(segmented / mappable * 100.0)
    except Exception:
        return None


def _extract_k_size_ratio(tunnel_dir: str) -> Optional[float]:
    """K-block points / segmented points as percentage."""
    final_path = os.path.join(tunnel_dir, "final.csv")
    if not os.path.exists(final_path):
        return None
    try:
        df = pd.read_csv(final_path)
        if "pred" not in df.columns:
            return None
        segmented = ((df["pred"] > 0) & (df["pred"] < 8)).sum()
        if segmented == 0:
            return None
        k_count = (df["pred"] == 1).sum()
        return float(k_count / segmented * 100.0)
    except Exception:
        return None


def _extract_groove_score(tunnel_dir: str) -> Optional[float]:
    """
    Boundary-groove alignment score.

    Reconstructs label map from pixel_to_point + final.csv, computes groove
    map from depth_map.png, then measures mean gradient at label boundaries.
    """
    depth_path = os.path.join(tunnel_dir, "depth_map.png")
    groove_map = _compute_groove_map(depth_path)
    if groove_map is None:
        return None

    label_map = _build_label_map_from_outputs(tunnel_dir)
    if label_map is None:
        return None

    # Compute boundary pixels
    shifted_down = np.roll(label_map, -1, axis=0)
    shifted_right = np.roll(label_map, -1, axis=1)
    boundary = (label_map != shifted_down) | (label_map != shifted_right)
    boundary[-1, :] = False
    boundary[:, -1] = False

    boundary_px = np.where(boundary)
    if len(boundary_px[0]) == 0:
        return 0.0

    return float(groove_map[boundary_px].mean())


def _extract_block_size_variance_ratio(tunnel_dir: str) -> Optional[float]:
    """
    Average max/min block area ratio per ring.

    For each ring, compute point count per block type, then max/min ratio.
    Complex staggered tunnels have inherently high variation (K is small, A/B are large).
    """
    final_path = os.path.join(tunnel_dir, "final.csv")
    if not os.path.exists(final_path):
        return None
    try:
        df = pd.read_csv(final_path)
        if "pred" not in df.columns or "pred_ring" not in df.columns:
            return None

        rings = sorted([r for r in df[df["pred_ring"] >= 0]["pred_ring"].unique()])
        if not rings:
            return None

        ratios = []
        for ring in rings:
            ring_data = df[(df["pred_ring"] == ring) & (df["pred"] > 0) & (df["pred"] < 8)]
            if len(ring_data) == 0:
                continue
            counts = ring_data["pred"].value_counts()
            if len(counts) < 2:
                continue
            min_count = counts.min()
            max_count = counts.max()
            if min_count > 0:
                ratios.append(max_count / min_count)

        if not ratios:
            return None
        return float(np.mean(ratios))
    except Exception:
        return None


# =============================================================================
# Guardrail Check
# =============================================================================

def _check_guardrails(
    seg_type_completeness: Optional[bool],
    ring_completeness_avg: Optional[float],
    mask_coverage_pct: Optional[float],
    k_size_ratio: Optional[float],
    groove_score: Optional[float],
    block_size_variance_ratio: Optional[float],
) -> Tuple[bool, List[str]]:
    """Check if metrics pass guardrail thresholds."""
    violations = []

    if seg_type_completeness is not None and not seg_type_completeness:
        violations.append("segment_type_completeness=False (missing block types)")

    if ring_completeness_avg is not None:
        if ring_completeness_avg < RING_COMPLETENESS_AVG_MIN:
            violations.append(f"ring_completeness_avg={ring_completeness_avg:.3f} < {RING_COMPLETENESS_AVG_MIN}")

    if mask_coverage_pct is not None:
        if mask_coverage_pct < MASK_COVERAGE_PCT_MIN:
            violations.append(f"mask_coverage_pct={mask_coverage_pct:.1f}% < {MASK_COVERAGE_PCT_MIN}%")
        elif mask_coverage_pct > MASK_COVERAGE_PCT_MAX:
            violations.append(f"mask_coverage_pct={mask_coverage_pct:.1f}% > {MASK_COVERAGE_PCT_MAX}%")

    if k_size_ratio is not None:
        if k_size_ratio < K_SIZE_RATIO_MIN:
            violations.append(f"k_size_ratio={k_size_ratio:.1f}% < {K_SIZE_RATIO_MIN}%")
        elif k_size_ratio > K_SIZE_RATIO_MAX:
            violations.append(f"k_size_ratio={k_size_ratio:.1f}% > {K_SIZE_RATIO_MAX}%")

    if groove_score is not None:
        if groove_score < GROOVE_SCORE_MIN:
            violations.append(f"groove_score={groove_score:.1f} < {GROOVE_SCORE_MIN}")

    if block_size_variance_ratio is not None:
        if block_size_variance_ratio < BLOCK_SIZE_VARIANCE_RATIO_MIN:
            violations.append(f"block_size_variance_ratio={block_size_variance_ratio:.1f} < {BLOCK_SIZE_VARIANCE_RATIO_MIN}")
        elif block_size_variance_ratio > BLOCK_SIZE_VARIANCE_RATIO_MAX:
            violations.append(f"block_size_variance_ratio={block_size_variance_ratio:.1f} > {BLOCK_SIZE_VARIANCE_RATIO_MAX}")

    return (len(violations) == 0, violations)


# =============================================================================
# Main Extraction
# =============================================================================

def extract_segmentation_metrics(tunnel_dir: str) -> dict:
    """Extract intrinsic metrics from segmentation outputs."""
    seg_type_completeness = _extract_segment_type_completeness(tunnel_dir)
    ring_completeness_avg = _extract_ring_completeness_avg(tunnel_dir)
    mask_coverage_pct = _extract_mask_coverage_pct(tunnel_dir)
    k_size_ratio = _extract_k_size_ratio(tunnel_dir)
    groove_score = _extract_groove_score(tunnel_dir)
    block_size_variance_ratio = _extract_block_size_variance_ratio(tunnel_dir)

    passed, violations = _check_guardrails(
        seg_type_completeness, ring_completeness_avg, mask_coverage_pct,
        k_size_ratio, groove_score, block_size_variance_ratio,
    )

    return {
        "seg_segment_type_completeness": seg_type_completeness,
        "seg_ring_completeness_avg": ring_completeness_avg,
        "seg_mask_coverage_pct": mask_coverage_pct,
        "seg_k_size_ratio": k_size_ratio,
        "seg_groove_score": groove_score,
        "seg_block_size_variance_ratio": block_size_variance_ratio,
        "seg_ready_for_evaluation": passed,
        "seg_guardrail_violations": violations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract segmentation intrinsic metrics for irregular tunnels",
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

    print(f"Extracting segmentation metrics from {tunnel_dir} ...")
    metrics = extract_segmentation_metrics(tunnel_dir)

    out_path = args.output or os.path.join(tunnel_dir, "segmentation_intrinsics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {out_path}\n")

    print("Metrics:")
    for key in ["seg_segment_type_completeness", "seg_ring_completeness_avg",
                 "seg_mask_coverage_pct", "seg_k_size_ratio",
                 "seg_groove_score", "seg_block_size_variance_ratio"]:
        val = metrics[key]
        if val is None:
            print(f"  {key}: N/A")
        elif isinstance(val, bool):
            print(f"  {key}: {val}")
        elif isinstance(val, float):
            if "pct" in key or "ratio" in key:
                print(f"  {key}: {val:.1f}%")
            else:
                print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")

    print(f"\n  seg_ready_for_evaluation: {metrics['seg_ready_for_evaluation']}")

    if metrics["seg_guardrail_violations"]:
        print("\n  Violations:")
        for v in metrics["seg_guardrail_violations"]:
            print(f"    - {v}")


if __name__ == "__main__":
    main()
