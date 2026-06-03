"""
Irregular Tunnel Detection Pipeline

Detects oblique groove lines in the depth map, finds K-block positions
via DBSCAN clustering + regulator refinement, and expands to all segment
positions via per-ring offsets.

Outputs:
  - all_segments.csv (Ring, Block, X, Y, quality) — segment centroids
  - boundaries_per_ring.json — boundary positions for downstream segmentation
  - detected_lines.png — visualization

K detection: DBSCAN clustering of pos/neg line intersections, refined by
a regulator that selects the best pos-neg pair midpoint per ring.
"""

import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Set
from sklearn.cluster import DBSCAN, AgglomerativeClustering
EXPECTED_7_BLOCKS = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]
EXPECTED_6_BLOCKS = ["K", "B1", "A1", "A2", "A3", "B2"]
MINUS_DIRECTION_BLOCK_MAP = {
    "K": "K",
    "B1": "B2",
    "A1": "A4",
    "A2": "A3",
    "A3": "A2",
    "A4": "A1",
    "B2": "B1",
}
MINUS_DIRECTION_BLOCK_MAP_6 = {
    "K": "K",
    "B1": "B2",
    "A1": "A3",
    "A2": "A2",
    "A3": "A1",
    "B2": "B1",
}


def _resolve_expected_blocks(
    *,
    segment_count: int | None,
    enabled_blocks: Optional[set],
) -> list[str]:
    if enabled_blocks is not None and len(enabled_blocks) > 0:
        order = EXPECTED_7_BLOCKS if (segment_count or 7) == 7 else EXPECTED_6_BLOCKS
        return [b for b in order if b in enabled_blocks]
    if int(segment_count or 7) == 6:
        return list(EXPECTED_6_BLOCKS)
    return list(EXPECTED_7_BLOCKS)


def _resolve_minus_direction_block_map(expected_blocks: list[str]) -> Dict[str, str]:
    if set(expected_blocks) == set(EXPECTED_6_BLOCKS):
        return dict(MINUS_DIRECTION_BLOCK_MAP_6)
    return dict(MINUS_DIRECTION_BLOCK_MAP)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from methods.ablation.scripts._labelmap_viz import render_labelmap_png
except Exception:  # noqa: BLE001
    render_labelmap_png = None

if render_labelmap_png is None:
    _PALETTE_8 = np.array(
        [
            [0, 0, 0],
            [220, 20, 60],
            [65, 105, 225],
            [50, 205, 50],
            [255, 165, 0],
            [186, 85, 211],
            [255, 215, 0],
            [30, 144, 255],
        ],
        dtype=np.uint8,
    )

    def render_labelmap_png(labelmap: np.ndarray, out_path: str) -> None:
        rgb = _PALETTE_8[np.clip(labelmap, 0, 7).astype(np.int64)]
        cv2.imwrite(out_path, rgb[..., ::-1])


# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(
    tunnel_id: str = None,
    ring_id: int = None,
    regime_label: str = None,
    base_dir: str = "data",
) -> Tuple[Dict, bool]:
    """Load parameters_detection.json for one ring.

    Priority:
        1. agents/.../parameters/<tunnel_id>/r<ring_id>/parameters_detection.json
        2. <base_dir>/<tunnel_id>/r<ring_id>/parameters_detection.json
        3. agents/.../parameters/_warm_start/<regime_label>/parameters_detection.json
        4. agents/.../parameters/_default_irregular/parameters_detection.json
        5. Empty dict.
    """
    script_dir = os.path.dirname(__file__)
    param_file = "parameters_detection.json"

    # When INTRINSIC_PARAMS_BASE_DIR_ONLY=1 (set by the v3 BO driver) we
    # skip the agents/.../parameters/<tunnel>/r<ring>/ and warm-start
    # lookups so per-trial sandbox params are not shadowed by checked-in
    # v1/v2-tuned per-ring overrides.
    base_only = os.environ.get("INTRINSIC_PARAMS_BASE_DIR_ONLY") == "1"

    if tunnel_id is not None and ring_id is not None:
        ring_key = f"r{int(ring_id)}"
        candidates = []
        if not base_only:
            candidates.append(os.path.join(script_dir, "parameters", tunnel_id, ring_key, param_file))
        candidates.append(os.path.join(base_dir, tunnel_id, ring_key, param_file))
        for p in candidates:
            if os.path.exists(p):
                with open(p, "r") as f:
                    return json.load(f), True

    if regime_label and not base_only:
        warm_path = os.path.join(
            script_dir, "parameters", "_warm_start", str(regime_label), param_file
        )
        if os.path.exists(warm_path):
            with open(warm_path, "r") as f:
                return json.load(f), True

    default_path = os.path.join(script_dir, "parameters", "_default_irregular", param_file)
    if os.path.exists(default_path):
        with open(default_path, "r") as f:
            return json.load(f), True

    return {}, False


# =============================================================================
# Defaults (safe-fixed values, overridable via params JSON)
# =============================================================================

DEFAULT_K_EXPECTED_HEIGHT_PX = 300
DEFAULT_DILATION_KERNEL_SIZE = 3
DEFAULT_DILATION_ITERATIONS = 1
DEFAULT_CANNY_LOW = 50
DEFAULT_CANNY_HIGH = 150
DEFAULT_HOUGH_HORIZONTAL_THRESHOLD = 50
DEFAULT_HOUGH_HORIZONTAL_MIN_LENGTH = 100
DEFAULT_HOUGH_HORIZONTAL_MAX_GAP = 10
DEFAULT_HORIZONTAL_ANGLE_TOLERANCE = 1.0
DEFAULT_HOUGH_VERTICAL_THRESHOLD = 500
DEFAULT_MERGE_DISTANCE_THRESHOLD = 3.0
FIXED_MERGE_CLOSE_THRESHOLD = 6.0
# Minimum |angle| (degrees) for a line to count as oblique pos/neg; excludes near-horizontal from K detection
DEFAULT_ANGLE_MIN_OBLIQUE_DEG = 3.0
IRREGULAR_BLOCK_TO_ID = {
    "BG": 0,
    "K": 1,
    "B1": 2,
    "A1": 3,
    "A2": 4,
    "A3": 5,
    "A4": 6,
    "B2": 7,
}

PREPROCESSING_PARAMS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "1_preprocessing", "parameters"
)


# =============================================================================
# Physical Constants (from preprocessing)
# =============================================================================

def load_preprocessing_params(
    tunnel_id: str,
    ring_id: int = None,
    regime_label: str = None,
    base_dir: str = "data",
) -> Dict:
    """Load preprocessing parameters for inherited physical constants.

    Same lookup precedence as the per-ring detection loader.
    """
    base_only = os.environ.get("INTRINSIC_PARAMS_BASE_DIR_ONLY") == "1"
    candidates = []
    if ring_id is not None:
        ring_key = f"r{int(ring_id)}"
        if not base_only:
            candidates.append(os.path.join(PREPROCESSING_PARAMS_DIR, tunnel_id, ring_key, "parameters_preprocessing.json"))
        candidates.append(os.path.join(base_dir, tunnel_id, ring_key, "parameters_preprocessing.json"))
    if regime_label and not base_only:
        candidates.append(
            os.path.join(
                PREPROCESSING_PARAMS_DIR,
                "_warm_start",
                str(regime_label),
                "parameters_preprocessing.json",
            )
        )
    candidates.append(os.path.join(PREPROCESSING_PARAMS_DIR, "_default_irregular", "parameters_preprocessing.json"))
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    return {}


def calculate_segment_heights(tunnel_diameter: float) -> Tuple[float, float]:
    """K and AB block heights (mm) from tunnel diameter.

    K = circumference / 16, AB = 3 * K.
    """
    circumference_mm = np.pi * tunnel_diameter * 1000
    k_height_mm = circumference_mm / 16
    ab_height_mm = 3 * k_height_mm
    return k_height_mm, ab_height_mm


# =============================================================================
# Line Detection
# =============================================================================

def detect_lines(depth_map_outlier: np.ndarray, params: Dict) -> Dict:
    """Unified line detection from depth map.

    BO-tunable (8): binary_threshold, hough_threshold, hough_min_length,
    hough_max_gap, angle_pos_min/max, angle_neg_min/max.

    Remaining params (dilation, canny, horizontal/vertical Hough) read from
    params with DEFAULT_* fallbacks.
    """
    L, W = depth_map_outlier.shape

    binary_threshold = params.get('binary_threshold', 139)
    dilation_kernel_size = params.get('dilation_kernel_size', DEFAULT_DILATION_KERNEL_SIZE)
    dilation_iterations = params.get('dilation_iterations', DEFAULT_DILATION_ITERATIONS)
    canny_low = params.get('canny_low', DEFAULT_CANNY_LOW)
    canny_high = params.get('canny_high', DEFAULT_CANNY_HIGH)

    binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)
    ret, binary_image = cv2.threshold(binary_map, binary_threshold, 255, cv2.THRESH_BINARY)

    depth_valid = depth_map_outlier[~np.isnan(depth_map_outlier)]
    depth_normalized = binary_image.copy()
    if len(depth_valid) > 0:
        depth_min, depth_max = depth_valid.min(), depth_valid.max()
        if depth_max > depth_min:
            out = np.zeros_like(depth_map_outlier, dtype=np.float64)
            valid = ~np.isnan(depth_map_outlier)
            out[valid] = (depth_map_outlier[valid] - depth_min) / (depth_max - depth_min) * 255
            depth_normalized = out.astype(np.uint8)

            canny_edges = cv2.Canny(depth_normalized, canny_low, canny_high)
            combined_edges = cv2.bitwise_or(binary_image, canny_edges)
        else:
            combined_edges = binary_image
    else:
        combined_edges = binary_image

    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_edges = cv2.dilate(combined_edges, kernel, iterations=dilation_iterations)

    hough_threshold = params.get('hough_threshold', 37)
    hough_min_length = params.get('hough_min_length', 31)
    hough_max_gap = params.get('hough_max_gap', 133)

    lines_oblique = cv2.HoughLinesP(
        dilated_edges, 1, np.pi / 180,
        hough_threshold,
        minLineLength=hough_min_length,
        maxLineGap=hough_max_gap
    )

    hough_horizontal_threshold = params.get('hough_horizontal_threshold', DEFAULT_HOUGH_HORIZONTAL_THRESHOLD)
    hough_horizontal_min_length = params.get('hough_horizontal_min_length', DEFAULT_HOUGH_HORIZONTAL_MIN_LENGTH)
    hough_horizontal_max_gap = params.get('hough_horizontal_max_gap', DEFAULT_HOUGH_HORIZONTAL_MAX_GAP)
    horizontal_angle_tolerance = params.get('horizontal_angle_tolerance', DEFAULT_HORIZONTAL_ANGLE_TOLERANCE)

    lines_horizontal = cv2.HoughLinesP(
        dilated_edges, 1, np.pi / 180,
        hough_horizontal_threshold,
        minLineLength=hough_horizontal_min_length,
        maxLineGap=hough_horizontal_max_gap
    )

    hough_vertical_threshold = params.get('hough_vertical_threshold', DEFAULT_HOUGH_VERTICAL_THRESHOLD)
    lines_vertical = cv2.HoughLines(dilated_edges, 1, np.pi / 180, hough_vertical_threshold)
    if lines_vertical is not None:
        lines_vertical = lines_vertical[lines_vertical[:, 0, 0] <= W]

    angle_pos_min = params.get('angle_pos_min', 4.84)
    angle_pos_max = params.get('angle_pos_max', 13.55)
    angle_neg_min = params.get('angle_neg_min', -14.67)
    angle_neg_max = params.get('angle_neg_max', -5.82)
    min_oblique_deg = float(params.get('angle_min_oblique_deg', DEFAULT_ANGLE_MIN_OBLIQUE_DEG))

    positive_lines = []
    negative_lines = []
    horizontal_lines = []

    if lines_oblique is not None:
        for line in lines_oblique:
            x1, y1, x2, y2 = line[0]
            x1, x2, y1, y2 = (x2, x1, y2, y1) if x1 > x2 else (x1, x2, y1, y2)
            angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
            # Exclude near-horizontal: pos must be >= min_oblique_deg, neg must be <= -min_oblique_deg
            if angle_pos_min <= angle <= angle_pos_max and angle >= min_oblique_deg:
                positive_lines.append(line[0])
            elif angle_neg_min <= angle <= angle_neg_max and angle <= -min_oblique_deg:
                negative_lines.append(line[0])

    max_line_length_px = params.get('max_line_length_px')
    if max_line_length_px is not None and max_line_length_px > 0:
        def length_ok(seg):
            x1, y1, x2, y2 = seg
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) <= max_line_length_px
        positive_lines = [s for s in positive_lines if length_ok(s)]
        negative_lines = [s for s in negative_lines if length_ok(s)]

    if lines_horizontal is not None:
        for line in lines_horizontal:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -horizontal_angle_tolerance <= angle <= horizontal_angle_tolerance:
                horizontal_lines.append(line[0])

    merge_distance_threshold = params.get('merge_distance_threshold', DEFAULT_MERGE_DISTANCE_THRESHOLD)
    merged_vertical = []
    if lines_vertical is not None:
        lines_vert_2d = lines_vertical[:, 0]
        for rho, theta in lines_vert_2d:
            if abs(theta) <= 0.5 * np.pi / 180:
                x_pos = rho * np.cos(theta)
                merged = False
                for i, (mrho, mtheta) in enumerate(merged_vertical):
                    mx = mrho * np.cos(mtheta)
                    if abs(x_pos - mx) < merge_distance_threshold:
                        merged_vertical[i] = ((rho + mrho) / 2, (theta + mtheta) / 2)
                        merged = True
                        break
                if not merged:
                    merged_vertical.append((rho, theta))
        merged_vertical.sort(key=lambda l: l[0])

    return {
        'positive_lines': positive_lines,
        'negative_lines': negative_lines,
        'horizontal_lines': horizontal_lines,
        'vertical_lines': merged_vertical,
        'dilated_edges': dilated_edges,
        'depth_image_gray': depth_normalized,
        'image_height': L,
        'image_width': W
    }


# =============================================================================
# Geometric Helpers
# =============================================================================

def extend_line_to_bounds(x1, y1, x2, y2, W, L):
    """Extend a line segment to image boundaries."""
    if x2 == x1:
        return x1, 0, x2, L

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    points = []
    y_at_0 = intercept
    y_at_W = slope * W + intercept

    if 0 <= y_at_0 <= L:
        points.append((0, y_at_0))
    if 0 <= y_at_W <= L:
        points.append((W, y_at_W))

    if slope != 0:
        x_at_0 = -intercept / slope
        if 0 <= x_at_0 <= W:
            points.append((x_at_0, 0))
        x_at_L = (L - intercept) / slope
        if 0 <= x_at_L <= W:
            points.append((x_at_L, L))

    if len(points) >= 2:
        points.sort(key=lambda p: p[0])
        return points[0][0], points[0][1], points[-1][0], points[-1][1]

    return x1, y1, x2, y2


def find_line_intersections(positive_lines, negative_lines, W, L):
    """Find all intersections between positive and negative slope lines."""
    intersections = []

    for pos_line in positive_lines:
        x1, y1, x2, y2 = pos_line
        if x2 == x1:
            continue
        slope1 = (y2 - y1) / (x2 - x1)
        intercept1 = y1 - slope1 * x1

        for neg_line in negative_lines:
            x3, y3, x4, y4 = neg_line
            if x4 == x3:
                continue
            slope2 = (y4 - y3) / (x4 - x3)
            intercept2 = y3 - slope2 * x3

            if abs(slope1 - slope2) < 1e-6:
                continue

            x_int = (intercept2 - intercept1) / (slope1 - slope2)
            y_int = slope1 * x_int + intercept1

            if 0 <= x_int <= W and 0 <= y_int <= L:
                intersections.append((x_int, y_int))

    return intersections


def line_segment_vertical_intersection(vertical_x: float, segment: Tuple) -> Optional[float]:
    """Y coordinate where a line segment crosses vertical x."""
    x1, y1, x2, y2 = segment
    if x1 == x2:
        return None
    if min(x1, x2) <= vertical_x <= max(x1, x2):
        t = (vertical_x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)
    return None


def merge_close_points(points: List[float], threshold: float = None) -> List[float]:
    """Merge Y-values within threshold (px). If threshold is None, use FIXED_MERGE_CLOSE_THRESHOLD."""
    if len(points) == 0:
        return []
    th = threshold if threshold is not None else FIXED_MERGE_CLOSE_THRESHOLD
    pts = np.array(points, dtype=np.float64)
    if len(pts) == 1:
        return [float(pts[0])]

    merged = []
    while len(pts) > 0:
        p = pts[0]
        close_mask = np.abs(pts - p) < th
        merged.append(float(np.mean(pts[close_mask])))
        pts = pts[~close_mask]
    return merged


# =============================================================================
# K-Position Detection: Line Midpoint (per-ring)
# =============================================================================

def _segment_length_and_angle(seg: Tuple) -> Tuple[float, float]:
    """Return (length_px, angle_deg) for segment. Angle: atan2(-(y2-y1), x2-x1) in degrees."""
    x1, y1, x2, y2 = seg
    length = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    angle_deg = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
    return length, float(angle_deg)


def detect_k_line_midpoint(
    line_data: Dict,
    ring_count: int,
    k_height_px: float,
    params: Dict,
) -> pd.DataFrame:
    """K positions from pos/neg line crossings at each ring's vertical slice.

    1. Close enough green and red (pos/neg pair with gap in range) -> K = midpoint.
    2. Only one green or red -> K = line crossing Y +/- (k_height_px/2).
    3. Only one, or pair too far -> pick most consistent line (best angle in 6-9 or -9 to -6, then longer) as base, then K = base Y +/- (k_height_px/2).
    """
    L = line_data['image_height']
    W = line_data['image_width']
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']
    ring_width = W / ring_count
    merge_close_fraction = float(params.get('merge_close_fraction', 0.05))
    merge_threshold = ring_width * merge_close_fraction
    target_gap = float(params.get('reg_target_gap_frac', 0.5)) * k_height_px
    max_k_gap_factor = float(params.get('max_k_gap_factor', 1.2))
    gap_cap = k_height_px * max_k_gap_factor
    half_k = k_height_px / 2.0
    angle_pos_min = float(params.get('angle_pos_min', 6.0))
    angle_pos_max = float(params.get('angle_pos_max', 9.0))
    angle_neg_min = float(params.get('angle_neg_min', -9.0))
    angle_neg_max = float(params.get('angle_neg_max', -6.0))

    def wrap_dy(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, L - d)

    def single_line_candidates(vertical_x: float):
        """List of (y, is_pos, length, angle_deg) for each crossing at vertical_x."""
        cands = []
        for seg in positive_lines:
            y_val = line_segment_vertical_intersection(vertical_x, seg)
            if y_val is not None:
                length, angle_deg = _segment_length_and_angle(seg)
                cands.append((float(y_val), True, length, angle_deg))
        for seg in negative_lines:
            y_val = line_segment_vertical_intersection(vertical_x, seg)
            if y_val is not None:
                length, angle_deg = _segment_length_and_angle(seg)
                cands.append((float(y_val), False, length, angle_deg))
        return cands

    def line_quality(length: float, angle_deg: float, is_pos: bool) -> float:
        """Quality = length; boost if angle in expected range (6-9 pos, -9 to -6 neg)."""
        in_range = (angle_pos_min <= angle_deg <= angle_pos_max) if is_pos else (angle_neg_min <= angle_deg <= angle_neg_max)
        return length * (1.0 if in_range else 0.2)

    def best_single_candidate(candidates: list) -> Optional[Tuple[float, bool]]:
        """Pick (y, is_pos) by: prefer angle in range (6-9 pos, -9 to -6 neg), then longer."""
        if not candidates:
            return None
        def score(c):
            y, is_pos, length, angle = c
            in_range = (angle_pos_min <= angle <= angle_pos_max) if is_pos else (angle_neg_min <= angle <= angle_neg_max)
            return (1 if in_range else 0, length)
        best = max(candidates, key=score)
        return (best[0], best[1])

    rows = []
    for i in range(ring_count):
        vertical_x = (i + 0.5) * ring_width
        # Segment-level crossings: (y, length, angle) per pos/neg so we can score pairs by line quality
        pos_cands = []
        for seg in positive_lines:
            y_val = line_segment_vertical_intersection(vertical_x, seg)
            if y_val is not None:
                length, angle_deg = _segment_length_and_angle(seg)
                pos_cands.append((float(y_val), length, angle_deg))
        neg_cands = []
        for seg in negative_lines:
            y_val = line_segment_vertical_intersection(vertical_x, seg)
            if y_val is not None:
                length, angle_deg = _segment_length_and_angle(seg)
                neg_cands.append((float(y_val), length, angle_deg))

        # Hierarchy: (1) When both red and green with valid pair(s) -> use pair midpoint.
        # (2) When multiple pairs: rings 0-3 use longest+right-angles; ring 4+ use central groove.
        # (3) Fallback (no valid pair): use longest red/green line with right angles.
        if pos_cands and neg_cands:
            pair_candidates = []
            for (py, len_p, angle_p) in pos_cands:
                for (ny, len_n, angle_n) in neg_cands:
                    gap = wrap_dy(py, ny)
                    if gap > gap_cap:
                        continue
                    gap_err = abs(gap - target_gap)
                    mid = (py + ny) / 2.0
                    if i <= 3:
                        # Rings 0-3: prefer pair by longest lines with right angles (no center)
                        quality = line_quality(len_p, angle_p, True) + line_quality(len_n, angle_n, False)
                        score = gap_err - quality  # lower is better
                    else:
                        # Ring 4+: prefer groove closer to image center
                        mid_dist_to_center = wrap_dy(mid, L / 2.0)
                        score = gap_err + 0.1 * mid_dist_to_center
                    pair_candidates.append((score, mid))
            best_mid = None
            if pair_candidates:
                pair_candidates.sort(key=lambda c: c[0])
                best_mid = pair_candidates[0][1]
            if best_mid is not None:
                rows.append(('midpoint', vertical_x, best_mid, 0.9))
            else:
                # Case 3: Both but too far -> pick most consistent line, then K = base Y +/- half_k
                cands = single_line_candidates(vertical_x)
                best = best_single_candidate(cands)
                if best is not None:
                    y_base, is_pos = best
                    y_k = y_base - half_k if is_pos else y_base + half_k
                    y_k = max(0.0, min(L - 1e-6, y_k))
                    rows.append(('midpoint_fallback', vertical_x, y_k, 0.6))
                else:
                    merge_pos = merge_close_points([c[0] for c in pos_cands], merge_threshold)
                    merge_neg = merge_close_points([c[0] for c in neg_cands], merge_threshold)
                    mid_fallback = (merge_pos[0] + merge_neg[0]) / 2.0 if (merge_pos and merge_neg) else L / 2.0
                    rows.append(('midpoint_fallback', vertical_x, mid_fallback, 0.5))
        else:
            # Case 2 & 3: Only one side, or none -> pick most consistent line, K = line Y +/- half_k
            cands = single_line_candidates(vertical_x)
            best = best_single_candidate(cands)
            if best is not None:
                y_base, is_pos = best
                y_k = y_base - half_k if is_pos else y_base + half_k
                y_k = max(0.0, min(L - 1e-6, y_k))
                label = 'positive_slope' if is_pos else 'negative_slope'
                rows.append((label, vertical_x, y_k, 0.6))
            else:
                rows.append(('default', vertical_x, L / 2.0, 0.2))

    df = pd.DataFrame(rows, columns=['Type', 'X', 'Y', 'Confidence'])
    return df.sort_values(by='X').reset_index(drop=True)


# =============================================================================
# K-Position Detection: DBSCAN Clustering
# =============================================================================

def detect_k_dbscan(
    line_data: Dict,
    ring_count: int,
    params: Dict,
) -> pd.DataFrame:
    """Detect K positions via DBSCAN clustering of pos/neg line intersections.

    1. Extend lines and find all pos-neg intersections.
    2. DBSCAN cluster the normalized (x/W, y/L) coordinates.
    3. Subdivide wide clusters spanning multiple ring widths.
    4. Merge intersection clusters with line midpoints via
       AgglomerativeClustering to produce exactly ring_count positions.

    Returns DataFrame with columns Type, X, Y, Confidence (sorted by X).
    """
    L = line_data['image_height']
    W = line_data['image_width']
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']

    eps = params.get('eps', 0.07)
    subdivision_threshold = params.get('complex_subdivision_threshold', 1.5)
    max_subdivisions = params.get('complex_max_subdivisions', 4)

    extended_positive = [extend_line_to_bounds(*line, W, L) for line in positive_lines]
    extended_negative = [extend_line_to_bounds(*line, W, L) for line in negative_lines]
    intersections = find_line_intersections(extended_positive, extended_negative, W, L)

    expected_ring_width = W / ring_count

    if len(intersections) == 0:
        fallback = []
        for x1, y1, x2, y2 in positive_lines + negative_lines:
            fallback.append(('midpoint_fallback', (x1+x2)/2, (y1+y2)/2, 0.3))
        if not fallback:
            for i in range(ring_count):
                fallback.append(('default', (i+0.5)*expected_ring_width, L/2, 0.1))
        df = pd.DataFrame(fallback, columns=['Type', 'X', 'Y', 'Confidence'])
        return df.sort_values(by='X').reset_index(drop=True)

    intersection_array = np.array(intersections)
    x_normalized = intersection_array[:, 0] / W
    y_normalized = intersection_array[:, 1] / L
    features = np.column_stack([x_normalized, y_normalized])

    eps_candidates = [eps, eps * 1.5, eps * 2.0, 0.10, 0.15]
    eps_candidates = sorted(list(set([round(e, 2) for e in eps_candidates])))
    min_clusters = max(3, ring_count // 2)

    for eps in eps_candidates:
        clustering = DBSCAN(eps=eps, min_samples=1).fit(features)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= min(ring_count, min_clusters) or eps == eps_candidates[-1]:
            break

    unique_labels = set(labels) - {-1}

    k_positions = []
    for label in unique_labels:
        cluster_points = intersection_array[labels == label]
        x_range = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
        n_sub = max(1, int(x_range / (expected_ring_width * subdivision_threshold)) + 1)
        n_sub = min(n_sub, max_subdivisions)

        if n_sub > 1:
            sorted_pts = cluster_points[np.argsort(cluster_points[:, 0])]
            sub_size = len(sorted_pts) // n_sub
            for s in range(n_sub):
                start = s * sub_size
                end = (s + 1) * sub_size if s < n_sub - 1 else len(sorted_pts)
                sub = sorted_pts[start:end]
                conf = min(1.0, 0.5 + 0.05 * len(sub))
                k_positions.append(('intersection_sub', np.mean(sub[:, 0]),
                                    np.mean(sub[:, 1]), conf))
        else:
            conf = min(1.0, 0.5 + 0.1 * len(cluster_points))
            k_positions.append(('intersection_cluster', np.mean(cluster_points[:, 0]),
                                np.mean(cluster_points[:, 1]), conf))

    midpoint_confidence = params.get('complex_conf_midpoint', 0.7)
    intersection_confidence = params.get('complex_conf_intersection', 0.9)

    line_midpoints = []
    for x1, y1, x2, y2 in positive_lines:
        line_midpoints.append(('positive_midpoint', (x1+x2)/2, (y1+y2)/2, midpoint_confidence))
    for x1, y1, x2, y2 in negative_lines:
        line_midpoints.append(('negative_midpoint', (x1+x2)/2, (y1+y2)/2, midpoint_confidence))

    all_candidates = k_positions + line_midpoints

    if len(all_candidates) > ring_count and len(all_candidates) >= 2:
        candidate_array = np.array([[p[1], p[2]] for p in all_candidates])
        x_norm = candidate_array[:, 0] / W
        y_norm = candidate_array[:, 1] / L
        feat = np.column_stack([x_norm, y_norm])

        n_clusters = min(ring_count, len(all_candidates))
        agg = AgglomerativeClustering(n_clusters=n_clusters).fit(feat)

        final_positions = []
        for lbl in range(n_clusters):
            mask = agg.labels_ == lbl
            cluster_pts = candidate_array[mask]
            cluster_types = [all_candidates[j][0] for j in range(len(mask)) if mask[j]]
            has_intersection = any('intersection' in t for t in cluster_types)
            det_type = 'intersection_cluster' if has_intersection else 'midpoint_cluster'
            conf = intersection_confidence if has_intersection else midpoint_confidence
            final_positions.append((det_type, np.mean(cluster_pts[:, 0]),
                                    np.mean(cluster_pts[:, 1]), conf))
        k_positions = final_positions

    k_positions.sort(key=lambda p: p[1])
    df = pd.DataFrame(k_positions, columns=['Type', 'X', 'Y', 'Confidence'])
    return df.sort_values(by='X').reset_index(drop=True)


# =============================================================================
# K-Position Regulator (refines Y using oblique line pair geometry)
# =============================================================================

def apply_k_regulator(
    k_df: pd.DataFrame,
    line_data: Dict,
    ring_count: int,
    k_height_px: float,
    params: Dict,
):
    """Refine K Y positions using oblique line pair geometry.

    For each ring, find pos/neg crossing pairs whose gap matches the target;
    pick best pair by gap match or proximity to detection; optionally blend
    with detection Y. Params (with defaults):
      reg_target_gap_frac      expected pos/neg gap as fraction of k_height_px (0.5)
      reg_gap_tolerance        fractional tolerance (0.5)
      reg_blend_weight         blend line vs detection (1.0 = pure line)
      reg_max_det_line_dist_frac  max Y dist for blend as fraction of k_height_px (1.0)
      reg_use_line_when_available  use line Y when valid pair exists (True)
      reg_pick_by_gap          pick pair by best gap match (True)
      reg_relaxed_gap_fallback if no pair in tolerance, try 2x tolerance (True)

    Returns (DataFrame with Ring, Type, X, Y, Confidence; used_pos_indices; used_neg_indices).
    """
    L = line_data['image_height']
    W = line_data['image_width']
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']
    ring_width = W / ring_count
    merge_close_fraction = float(params.get('merge_close_fraction', 0.05))
    merge_threshold = ring_width * merge_close_fraction

    reg_target_gap = float(params.get('reg_target_gap_frac', 0.5)) * k_height_px
    reg_gap_tolerance = float(params.get('reg_gap_tolerance', 0.5))
    reg_blend_weight = float(params.get('reg_blend_weight', 1.0))
    reg_max_det_line_dist = float(params.get('reg_max_det_line_dist_frac', 1.0)) * k_height_px
    reg_use_line_when_available = params.get('reg_use_line_when_available', True)
    reg_pick_by_gap = params.get('reg_pick_by_gap', True)
    gap_allow = reg_gap_tolerance * reg_target_gap
    max_k_gap_factor = float(params.get('max_k_gap_factor', 1.2))
    gap_cap = k_height_px * max_k_gap_factor
    angle_pos_min = float(params.get('angle_pos_min', 6.0))
    angle_pos_max = float(params.get('angle_pos_max', 9.0))
    angle_neg_min = float(params.get('angle_neg_min', -9.0))
    angle_neg_max = float(params.get('angle_neg_max', -6.0))

    def wrap_dy(a, b):
        d = abs(a - b)
        return min(d, L - d)

    def line_quality(length: float, angle_deg: float, is_pos: bool) -> float:
        in_range = (angle_pos_min <= angle_deg <= angle_pos_max) if is_pos else (angle_neg_min <= angle_deg <= angle_neg_max)
        return length * (1.0 if in_range else 0.2)

    det_by_ring = {}
    for idx in range(len(k_df)):
        row = k_df.iloc[idx]
        r = idx
        if 0 <= r < ring_count:
            det_by_ring[r] = {
                'Y': float(row['Y']),
                'Type': str(row.get('Type', 'detected')),
                'Confidence': float(row.get('Confidence', 0.5)),
            }
    for i in range(ring_count):
        if i not in det_by_ring:
            det_by_ring[i] = {'Y': L / 2.0, 'Type': 'fallback', 'Confidence': 0.1}

    # Per-ring: (vertical_x, pos_ys_with_idx, neg_ys_with_idx, det_y) for line-index tracking
    ring_data = []
    for i in range(ring_count):
        vertical_x = (i + 0.5) * ring_width
        pos_ys_with_idx = []
        for j, seg in enumerate(positive_lines):
            y_val = line_segment_vertical_intersection(vertical_x, seg)
            if y_val is not None:
                pos_ys_with_idx.append((y_val, j))
        neg_ys_with_idx = []
        for j, seg in enumerate(negative_lines):
            y_val = line_segment_vertical_intersection(vertical_x, seg)
            if y_val is not None:
                neg_ys_with_idx.append((y_val, j))
        merge_pos = merge_close_points([y for y, _ in pos_ys_with_idx], merge_threshold)
        merge_neg = merge_close_points([y for y, _ in neg_ys_with_idx], merge_threshold)
        ring_data.append((vertical_x, pos_ys_with_idx, neg_ys_with_idx, merge_pos, merge_neg, det_by_ring[i]['Y']))

    used_pos_indices = set()
    used_neg_indices = set()
    rows = []
    for i in range(ring_count):
        vertical_x, pos_ys_with_idx, neg_ys_with_idx, merge_pos, merge_neg, det_y = ring_data[i]
        y_line = None
        best_py, best_ny = None, None

        if merge_pos and merge_neg:
            candidates = []
            for py in merge_pos:
                for ny in merge_neg:
                    gap = wrap_dy(py, ny)
                    if gap > gap_cap or abs(gap - reg_target_gap) > gap_allow:
                        continue
                    mid = (py + ny) / 2.0
                    candidates.append((mid, gap, wrap_dy(mid, det_y), py, ny))
            if not candidates and params.get('reg_relaxed_gap_fallback', True):
                gap_allow_wide = 2.0 * gap_allow
                for py in merge_pos:
                    for ny in merge_neg:
                        gap = wrap_dy(py, ny)
                        if gap > gap_cap or abs(gap - reg_target_gap) > gap_allow_wide:
                            continue
                        mid = (py + ny) / 2.0
                        candidates.append((mid, gap, wrap_dy(mid, det_y), py, ny))
            if candidates:
                # Hierarchy: rings 0-3 pick pair by longest+right-angles; ring 4+ by central groove
                def reg_score(c):
                    mid, gap, dist_to_det = c[0], c[1], c[2]
                    py, ny = c[3], c[4]
                    gap_err = abs(gap - reg_target_gap)
                    mid_dist_to_center = wrap_dy(mid, L / 2.0)
                    if i <= 3:
                        pos_q = max(
                            (line_quality(*_segment_length_and_angle(positive_lines[j]), True)
                             for y_val, j in pos_ys_with_idx if abs(y_val - py) < merge_threshold),
                            default=0.0,
                        )
                        neg_q = max(
                            (line_quality(*_segment_length_and_angle(negative_lines[j]), False)
                             for y_val, j in neg_ys_with_idx if abs(y_val - ny) < merge_threshold),
                            default=0.0,
                        )
                        quality = pos_q + neg_q
                        if reg_pick_by_gap:
                            return gap_err - quality
                        return dist_to_det - quality
                    else:
                        if reg_pick_by_gap:
                            return gap_err + 0.3 * mid_dist_to_center
                        return dist_to_det + 0.3 * mid_dist_to_center
                best = min(candidates, key=reg_score)
                best_mid, _, _, best_py, best_ny = best
                y_line = best_mid

        if y_line is not None and best_py is not None and best_ny is not None:
            for y_val, idx in pos_ys_with_idx:
                if abs(y_val - best_py) < merge_threshold:
                    used_pos_indices.add(idx)
            for y_val, idx in neg_ys_with_idx:
                if abs(y_val - best_ny) < merge_threshold:
                    used_neg_indices.add(idx)

        if y_line is not None:
            use_line = reg_use_line_when_available or wrap_dy(y_line, det_y) <= reg_max_det_line_dist
            if use_line:
                y_reg = reg_blend_weight * y_line + (1.0 - reg_blend_weight) * det_y
                reg_type = 'regulated'
            else:
                y_reg = det_y
                reg_type = 'unregulated'
        else:
            y_reg = det_y
            reg_type = 'unregulated'

        y_reg = max(0.0, min(L - 1e-6, y_reg))
        orig_type = det_by_ring[i]['Type'] + '_' + reg_type
        conf = 0.9 if reg_type == 'regulated' else det_by_ring[i]['Confidence']
        rows.append({
            'Ring': i, 'Type': orig_type, 'X': vertical_x,
            'Y': y_reg, 'Confidence': conf,
        })

    out = pd.DataFrame(rows)
    n_regulated = sum(1 for r in rows if r['Type'].endswith('_regulated'))
    print(f"  Regulator: {n_regulated}/{ring_count} rings regulated "
          f"(target_gap={reg_target_gap:.0f}, tolerance={reg_gap_tolerance}, "
          f"blend={reg_blend_weight})")
    return out, used_pos_indices, used_neg_indices


# =============================================================================
# Single-Ring Local Detection
# =============================================================================

def _cluster_y_values(values: List[float], tol_px: float) -> List[Dict[str, float]]:
    """Cluster nearby Y-values and return cluster centers with support counts."""
    if not values:
        return []
    tol = max(float(tol_px), 1.0)
    sorted_vals = sorted(float(v) for v in values)
    groups: List[List[float]] = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [
        {"y": float(np.mean(g)), "count": int(len(g))}
        for g in groups
    ]


def _circular_midpoint_y(y1: float, y2: float, height: float) -> float:
    """Midpoint on a circular y-axis."""
    a = float(y1) % float(height)
    b = float(y2) % float(height)
    d = b - a
    if d > height / 2.0:
        d -= height
    elif d < -height / 2.0:
        d += height
    return float((a + 0.5 * d) % float(height))


def _horizontal_pattern_k_y(
    y_values: List[float],
    *,
    k_height_px: float,
    image_height: float,
    tolerance_px: float,
) -> float | None:
    """Port r4tun horizontal-pair fallback for single-ring K-y recovery."""
    vals = [float(v) % float(image_height) for v in y_values]
    if len(vals) < 2:
        return None
    # In the original r4tun detector, horizontal candidates are accepted when
    # their spacing follows K + n*AB, n in {2, 4}, with AB ~= 3K.
    targets = [float(k_height_px) * 7.0, float(k_height_px) * 13.0]
    for i in range(len(vals) - 1):
        for j in range(i + 1, len(vals)):
            gap = abs(vals[j] - vals[i])
            gap = min(gap, float(image_height) - gap)
            if any(abs(gap - target) <= float(tolerance_px) for target in targets):
                return _circular_midpoint_y(vals[i], vals[j], float(image_height))
    return None


def _circular_distance(a: float, b: float, height: float) -> float:
    d = abs(float(a) - float(b))
    return min(d, float(height) - d)


def _consensus_circular_y(
    candidates: List[Tuple[float, float]],
    *,
    image_height: float,
    tol_px: float,
) -> float | None:
    """Weighted circular consensus of candidate y values."""
    if not candidates:
        return None
    h = float(image_height)
    tol = max(1.0, float(tol_px))
    ys = [float(y) % h for y, _ in candidates]
    ws = [max(0.01, float(w)) for _, w in candidates]
    best_idxs: List[int] = []
    best_weight = -1.0
    for i, yi in enumerate(ys):
        idxs = [j for j, yj in enumerate(ys) if _circular_distance(yi, yj, h) <= tol]
        wsum = float(sum(ws[j] for j in idxs))
        if wsum > best_weight:
            best_weight = wsum
            best_idxs = idxs
    if not best_idxs:
        return None
    ang = [2.0 * np.pi * ys[j] / h for j in best_idxs]
    w = np.array([ws[j] for j in best_idxs], dtype=float)
    c = float(np.sum(np.cos(ang) * w))
    s = float(np.sum(np.sin(ang) * w))
    theta = float(np.arctan2(s, c))
    if theta < 0.0:
        theta += 2.0 * np.pi
    return float((theta / (2.0 * np.pi)) * h)


def detect_k_single_ring_local(
    line_data: Dict,
    k_height_px: float,
    params: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[int], Set[int], Set[int], Dict]:
    """Detect K for one-ring crops using local oblique/horizontal evidence."""
    L = int(line_data["image_height"])
    W = int(line_data["image_width"])
    positive_lines = line_data["positive_lines"]
    negative_lines = line_data["negative_lines"]
    horizontal_lines = line_data["horizontal_lines"]

    lane_x_fracs = params.get("lane_x_fracs")
    if isinstance(lane_x_fracs, list) and lane_x_fracs:
        lane_fracs = [max(0.05, min(0.95, float(v))) for v in lane_x_fracs]
    else:
        lane_count = int(params.get("single_ring_lane_count", 3))
        lane_count = max(2, min(lane_count, 7))
        lane_fracs = np.linspace(0.25, 0.75, lane_count).tolist()
    lane_xs = [float(f * W) for f in lane_fracs]

    target_gap = float(params.get("reg_target_gap_frac", 0.5)) * float(k_height_px)
    gap_tol_frac = float(params.get("single_ring_gap_tolerance_frac", 0.7))
    gap_tol = max(30.0, gap_tol_frac * target_gap)
    cluster_tol = float(params.get("single_ring_group_y_tol_px", 30.0))
    h_keep_n = int(params.get("single_ring_horizontal_keep_n", 8))
    min_h_len = float(params.get("single_ring_horizontal_min_length_frac", 0.10)) * float(W)

    used_pos_indices: Set[int] = set()
    used_neg_indices: Set[int] = set()
    used_h_indices: Set[int] = set()
    pair_candidates: List[Tuple[float, float, int, int]] = []

    def wrap_dy(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, L - d)

    for vx in lane_xs:
        pos_hits: List[Tuple[float, int, float]] = []
        for j, seg in enumerate(positive_lines):
            y_val = line_segment_vertical_intersection(vx, seg)
            if y_val is not None:
                length, _ = _segment_length_and_angle(seg)
                pos_hits.append((float(y_val), j, float(length)))
        neg_hits: List[Tuple[float, int, float]] = []
        for j, seg in enumerate(negative_lines):
            y_val = line_segment_vertical_intersection(vx, seg)
            if y_val is not None:
                length, _ = _segment_length_and_angle(seg)
                neg_hits.append((float(y_val), j, float(length)))
        for py, pj, plen in pos_hits:
            for ny, nj, nlen in neg_hits:
                gap = wrap_dy(py, ny)
                if abs(gap - target_gap) > gap_tol:
                    continue
                mid = (py + ny) / 2.0
                score = abs(gap - target_gap) - 0.01 * (plen + nlen)
                pair_candidates.append((score, mid, pj, nj))

    pair_candidates.sort(key=lambda x: x[0])
    k_x = float(params.get("single_ring_k_x_frac", 0.5)) * float(W)
    k_y = float(L) / 2.0
    k_type = "local_default"
    k_conf = 0.2

    if pair_candidates:
        _, k_y, best_pj, best_nj = pair_candidates[0]
        used_pos_indices.add(int(best_pj))
        used_neg_indices.add(int(best_nj))
        k_type = "local_oblique_pair"
        k_conf = 0.9
    else:
        oblique_pool: List[Tuple[float, float, bool, int]] = []
        for j, seg in enumerate(positive_lines):
            x1, y1, x2, y2 = seg
            length, _ = _segment_length_and_angle(seg)
            oblique_pool.append((float(length), (float(y1) + float(y2)) / 2.0, True, j))
        for j, seg in enumerate(negative_lines):
            x1, y1, x2, y2 = seg
            length, _ = _segment_length_and_angle(seg)
            oblique_pool.append((float(length), (float(y1) + float(y2)) / 2.0, False, j))
        oblique_pool.sort(key=lambda x: x[0], reverse=True)
        if oblique_pool:
            _, y_mid, is_pos, idx = oblique_pool[0]
            if bool(params.get("single_ring_apply_r4tun_single_oblique_shift", False)):
                # r4tun uses a single oblique intersection as one edge of K and
                # shifts by half K height to estimate the K center.
                k_y = y_mid - 0.5 * float(k_height_px) if is_pos else y_mid + 0.5 * float(k_height_px)
            else:
                k_y = y_mid
            if is_pos:
                used_pos_indices.add(int(idx))
            else:
                used_neg_indices.add(int(idx))
            k_type = "local_oblique_single"
            k_conf = 0.65
        else:
            h_rows: List[Tuple[float, int, float]] = []
            for j, seg in enumerate(horizontal_lines):
                x1, y1, x2, y2 = seg
                length = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                if length >= min_h_len:
                    h_rows.append((length, j, (float(y1) + float(y2)) / 2.0))
            h_rows.sort(key=lambda x: x[0], reverse=True)
            for _, j, ymid in h_rows[: max(1, h_keep_n)]:
                used_h_indices.add(int(j))
            if h_rows:
                kept_h = [float(row[2]) for row in h_rows[: max(1, h_keep_n)]]
                pattern_y = _horizontal_pattern_k_y(
                    kept_h,
                    k_height_px=float(k_height_px),
                    image_height=float(L),
                    tolerance_px=float(params.get("single_ring_horizontal_pattern_tolerance_px", 50.0)),
                )
                if pattern_y is not None:
                    k_y = float(pattern_y)
                    k_type = "local_horizontal_pattern"
                    k_conf = 0.65
                else:
                    k_y = float(np.median(kept_h))
                    k_type = "local_horizontal_anchor"
                    k_conf = 0.55

    h_y_all: List[float] = []
    h_selected = 0
    horizontal_candidates: List[Tuple[float, int, float]] = []
    for j, seg in enumerate(horizontal_lines):
        x1, y1, x2, y2 = seg
        length = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        if length < min_h_len:
            continue
        ymid = (float(y1) + float(y2)) / 2.0
        horizontal_candidates.append((length, int(j), float(ymid)))
    # If Hough horizontal lines are sparse on narrow crops, recover candidates
    # from dense row responses in the edge map as a local fallback.
    if not horizontal_candidates and "dilated_edges" in line_data:
        edge = np.asarray(line_data["dilated_edges"])
        if edge.ndim == 2 and edge.shape[0] == L:
            row_density = np.mean(edge > 0, axis=1)
            min_row_density = float(params.get("single_ring_row_density_min", 0.02))
            row_idxs = np.where(row_density >= min_row_density)[0].tolist()
            clustered = _cluster_y_values([float(i) for i in row_idxs], cluster_tol)
            synth_count = int(params.get("single_ring_horizontal_synth_count", 4))
            for c in sorted(clustered, key=lambda x: x["count"], reverse=True)[: max(1, synth_count)]:
                ymid = float(c["y"])
                seg = (0.0, ymid, float(W - 1), ymid)
                line_data["horizontal_lines"].append(seg)
                idx = len(line_data["horizontal_lines"]) - 1
                score = float(c["count"])
                horizontal_candidates.append((score, int(idx), ymid))
    horizontal_candidates.sort(key=lambda x: x[0], reverse=True)
    for _, j, ymid in horizontal_candidates[: max(1, h_keep_n)]:
        h_y_all.append(float(ymid))
        used_h_indices.add(int(j))
    h_selected = int(len(used_h_indices))
    h_clusters = _cluster_y_values(h_y_all, cluster_tol)

    k_y = max(0.0, min(float(L) - 1e-6, float(k_y)))
    anchor_type = "midpoint"
    if k_type in {"local_horizontal_anchor", "local_horizontal_pattern"}:
        anchor_type = "assume"
    elif k_type == "local_oblique_single":
        if len(used_pos_indices) > 0 and len(used_neg_indices) == 0:
            anchor_type = "positive_slope"
        elif len(used_neg_indices) > 0 and len(used_pos_indices) == 0:
            anchor_type = "negative_slope"
    anchor_row = (anchor_type, float(k_x), float(k_y), float(k_conf))

    detected_rows: List[Tuple[str, float, float, float]] = [anchor_row]
    x_left = max(0.0, min(float(W - 1), float(k_x - 0.12 * W)))
    x_right = max(0.0, min(float(W - 1), float(k_x + 0.12 * W)))
    for idx in sorted(used_pos_indices):
        x1, y1, x2, y2 = positive_lines[idx]
        ymid = max(0.0, min(float(L - 1e-6), (float(y1) + float(y2)) / 2.0))
        detected_rows.append(("positive_slope", x_left, ymid, 0.6))
    for idx in sorted(used_neg_indices):
        x1, y1, x2, y2 = negative_lines[idx]
        ymid = max(0.0, min(float(L - 1e-6), (float(y1) + float(y2)) / 2.0))
        detected_rows.append(("negative_slope", x_right, ymid, 0.6))
    for c in h_clusters[:2]:
        ymid = max(0.0, min(float(L - 1e-6), float(c["y"])))
        detected_rows.append(("assume", float(k_x), ymid, 0.4))

    k_positions = pd.DataFrame(detected_rows, columns=["Type", "X", "Y", "Confidence"])
    anchor_positions = pd.DataFrame([anchor_row], columns=["Type", "X", "Y", "Confidence"])
    meta = {
        "detector_mode": "single_ring_local",
        "image_height": int(L),
        "image_width": int(W),
        "lane_x_fracs": [float(f) for f in lane_fracs],
        "lane_xs": [float(x) for x in lane_xs],
        "target_gap_px": float(target_gap),
        "gap_tolerance_px": float(gap_tol),
        "k_detection_type": str(k_type),
        "k_confidence": float(k_conf),
        "k_y": float(k_y),
        "positive_line_count": int(len(positive_lines)),
        "negative_line_count": int(len(negative_lines)),
        "horizontal_line_count": int(len(horizontal_lines)),
        "selected_positive_count": int(len(used_pos_indices)),
        "selected_negative_count": int(len(used_neg_indices)),
        "selected_horizontal_count": int(h_selected),
        "horizontal_clusters": h_clusters,
        "fallback_only": bool("default" in str(k_type) or "fallback" in str(k_type)),
        "non_fallback_k_count": int(0 if ("default" in str(k_type) or "fallback" in str(k_type)) else 1),
    }
    return k_positions, anchor_positions, used_pos_indices, used_neg_indices, used_h_indices, meta


def detect_k_single_ring_r4tun_contract(
    line_data: Dict,
    k_height_px: float,
    params: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[int], Set[int], Set[int], Dict]:
    """Opt-in r4tun-style candidate consensus for single-ring K detection."""
    L = int(line_data["image_height"])
    W = int(line_data["image_width"])
    positive_lines = line_data["positive_lines"]
    negative_lines = line_data["negative_lines"]
    horizontal_lines = line_data["horizontal_lines"]
    lane_fracs = params.get("lane_x_fracs")
    if isinstance(lane_fracs, list) and lane_fracs:
        lane_xs = [max(0.05, min(0.95, float(v))) * float(W) for v in lane_fracs]
    else:
        lane_xs = [0.25 * float(W), 0.5 * float(W), 0.75 * float(W)]

    used_pos_indices: Set[int] = set()
    used_neg_indices: Set[int] = set()
    used_h_indices: Set[int] = set()
    candidate_list: List[Dict[str, float | str]] = []
    weighted_y: List[Tuple[float, float]] = []
    pair_gap_target = float(params.get("reg_target_gap_frac", 0.5)) * float(k_height_px)
    pair_gap_tol = max(30.0, float(params.get("single_ring_gap_tolerance_frac", 0.7)) * pair_gap_target)
    pattern_tol = float(params.get("single_ring_horizontal_pattern_tolerance_px", 50.0))
    h_keep_n = max(1, int(params.get("single_ring_horizontal_keep_n", 8)))

    for vx in lane_xs:
        pos_hits: List[Tuple[float, int, float]] = []
        neg_hits: List[Tuple[float, int, float]] = []
        h_hits: List[Tuple[float, int, float]] = []
        for j, seg in enumerate(positive_lines):
            yv = line_segment_vertical_intersection(vx, seg)
            if yv is not None:
                length, _ = _segment_length_and_angle(seg)
                pos_hits.append((float(yv), int(j), float(length)))
        for j, seg in enumerate(negative_lines):
            yv = line_segment_vertical_intersection(vx, seg)
            if yv is not None:
                length, _ = _segment_length_and_angle(seg)
                neg_hits.append((float(yv), int(j), float(length)))
        for j, seg in enumerate(horizontal_lines):
            yv = line_segment_vertical_intersection(vx, seg)
            if yv is not None:
                length, _ = _segment_length_and_angle(seg)
                h_hits.append((float(yv), int(j), float(length)))

        pair_cands: List[Tuple[float, float, int, int]] = []
        for py, pj, plen in pos_hits:
            for ny, nj, nlen in neg_hits:
                gap = _circular_distance(py, ny, float(L))
                if abs(gap - pair_gap_target) > pair_gap_tol:
                    continue
                mid = _circular_midpoint_y(py, ny, float(L))
                score = abs(gap - pair_gap_target) - 0.01 * (plen + nlen)
                pair_cands.append((score, float(mid), int(pj), int(nj)))
        pair_cands.sort(key=lambda t: t[0])
        if pair_cands:
            _, y, pj, nj = pair_cands[0]
            used_pos_indices.add(pj)
            used_neg_indices.add(nj)
            weighted_y.append((float(y), 1.0))
            candidate_list.append({"source": "oblique_pair", "y": float(y), "weight": 1.0, "x": float(vx)})
            continue

        if pos_hits or neg_hits:
            merged = sorted(pos_hits + neg_hits, key=lambda t: t[2], reverse=True)
            ymid, idx, _length = merged[0]
            is_pos = any(int(idx) == int(pj) for _, pj, _ in pos_hits)
            if is_pos:
                used_pos_indices.add(int(idx))
                y = float((ymid - 0.5 * float(k_height_px)) % float(L))
            else:
                used_neg_indices.add(int(idx))
                y = float((ymid + 0.5 * float(k_height_px)) % float(L))
            weighted_y.append((y, 0.85))
            candidate_list.append({"source": "single_oblique_shift", "y": y, "weight": 0.85, "x": float(vx)})
            continue

        if h_hits:
            h_rows = sorted(h_hits, key=lambda t: t[2], reverse=True)
            for _y, j, _l in h_rows[:h_keep_n]:
                used_h_indices.add(int(j))
            ys = [float(y) for y, _, _ in h_rows[:h_keep_n]]
            pat_y = _horizontal_pattern_k_y(
                ys,
                k_height_px=float(k_height_px),
                image_height=float(L),
                tolerance_px=float(pattern_tol),
            )
            if pat_y is not None:
                weighted_y.append((float(pat_y), 0.75))
                candidate_list.append({"source": "horizontal_pattern", "y": float(pat_y), "weight": 0.75, "x": float(vx)})
            else:
                h_mid = float(np.median(ys))
                weighted_y.append((h_mid, 0.55))
                candidate_list.append({"source": "horizontal_median", "y": h_mid, "weight": 0.55, "x": float(vx)})

    if not weighted_y:
        edge = np.asarray(line_data.get("dilated_edges"))
        if edge.ndim == 2 and edge.shape[0] == L:
            row_density = np.mean(edge > 0, axis=1)
            peak = float(int(np.argmax(row_density)))
            weighted_y.append((peak, 0.3))
            candidate_list.append({"source": "row_density_peak", "y": peak, "weight": 0.3, "x": 0.5 * float(W)})

    k_y = _consensus_circular_y(
        weighted_y,
        image_height=float(L),
        tol_px=float(params.get("single_ring_consensus_tol_px", max(40.0, 0.8 * float(k_height_px)))),
    )
    if k_y is None:
        k_y = float(L) / 2.0
        k_type = "r4tun_contract_default"
        k_conf = 0.2
    else:
        k_type = "r4tun_contract_consensus"
        k_conf = min(0.95, 0.45 + 0.1 * len(weighted_y))

    k_x = float(params.get("single_ring_k_x_frac", 0.5)) * float(W)
    anchor_row = ("midpoint", float(k_x), float(k_y), float(k_conf))
    detected_rows: List[Tuple[str, float, float, float]] = [anchor_row]
    x_left = max(0.0, min(float(W - 1), float(k_x - 0.12 * W)))
    x_right = max(0.0, min(float(W - 1), float(k_x + 0.12 * W)))
    for idx in sorted(used_pos_indices):
        x1, y1, x2, y2 = positive_lines[idx]
        ymid = max(0.0, min(float(L - 1e-6), (float(y1) + float(y2)) / 2.0))
        detected_rows.append(("positive_slope", x_left, ymid, 0.6))
    for idx in sorted(used_neg_indices):
        x1, y1, x2, y2 = negative_lines[idx]
        ymid = max(0.0, min(float(L - 1e-6), (float(y1) + float(y2)) / 2.0))
        detected_rows.append(("negative_slope", x_right, ymid, 0.6))
    for item in candidate_list:
        if item["source"] in {"horizontal_pattern", "horizontal_median", "row_density_peak"}:
            detected_rows.append(("assume", float(k_x), float(item["y"]), float(item["weight"])))

    k_positions = pd.DataFrame(detected_rows, columns=["Type", "X", "Y", "Confidence"])
    anchor_positions = pd.DataFrame([anchor_row], columns=["Type", "X", "Y", "Confidence"])
    meta = {
        "detector_mode": "single_ring_r4tun_k_contract",
        "image_height": int(L),
        "image_width": int(W),
        "lane_xs": [float(v) for v in lane_xs],
        "k_detection_type": str(k_type),
        "k_confidence": float(k_conf),
        "k_y": float(k_y),
        "k_expected_height_px": float(k_height_px),
        "candidate_count": int(len(candidate_list)),
        "candidates": candidate_list,
        "positive_line_count": int(len(positive_lines)),
        "negative_line_count": int(len(negative_lines)),
        "horizontal_line_count": int(len(horizontal_lines)),
        "selected_positive_count": int(len(used_pos_indices)),
        "selected_negative_count": int(len(used_neg_indices)),
        "selected_horizontal_count": int(len(used_h_indices)),
    }
    return k_positions, anchor_positions, used_pos_indices, used_neg_indices, used_h_indices, meta


def detect_k_single_ring_regular_prior(
    *,
    ring_id: int,
    line_data: Dict,
    params: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[int], Set[int], Set[int], Dict]:
    """r4tun regular-family two-level K prior, exposed as an opt-in mode."""
    L = int(line_data["image_height"])
    W = int(line_data["image_width"])
    low_frac = float(params.get("regular_k_prior_low_frac", 1150.0 / 2777.0))
    high_frac = float(params.get("regular_k_prior_high_frac", 1580.0 / 2777.0))
    low_parity = int(params.get("regular_k_prior_low_ring_parity", 0))
    parity = int(ring_id) % 2
    frac = low_frac if parity == low_parity else high_frac
    k_y = float(frac * float(L)) % float(L)
    k_x = float(params.get("single_ring_k_x_frac", 0.5)) * float(W)
    row = ("regular_prior", float(k_x), float(k_y), 0.7)
    k_positions = pd.DataFrame([row], columns=["Type", "X", "Y", "Confidence"])
    meta = {
        "detector_mode": "single_ring_regular_prior",
        "image_height": int(L),
        "image_width": int(W),
        "ring_id": int(ring_id),
        "ring_parity": int(parity),
        "low_ring_parity": int(low_parity),
        "low_frac": float(low_frac),
        "high_frac": float(high_frac),
        "k_y": float(k_y),
        "k_detection_type": "regular_two_level_prior",
        "k_confidence": 0.7,
        "regular_prior_preferred_branch": str(params.get("regular_prior_preferred_branch", "plus")),
        "positive_line_count": int(len(line_data["positive_lines"])),
        "negative_line_count": int(len(line_data["negative_lines"])),
        "horizontal_line_count": int(len(line_data["horizontal_lines"])),
    }
    return k_positions, k_positions.copy(), set(), set(), set(), meta


# =============================================================================
# Segment Expansion: K → All Segments
# =============================================================================

def expand_k_with_per_ring_offsets(
    k_positions: pd.DataFrame,
    img_height: int,
    per_ring_offsets: Dict[str, Dict[str, float]],
    enabled_blocks: Optional[set] = None,
) -> pd.DataFrame:
    """Expand K positions to all segments using per-ring boundary offsets.

    per_ring_offsets: {"ring_idx": {"block_name": dy, ...}, ...}
    Each offset is a signed circular distance from detected K_Y to the
    block's boundary Y position.  All blocks (including K) must be present.
    """
    rows = []
    for ring_idx, (_, k_row) in enumerate(k_positions.iterrows()):
        k_x = float(k_row['X'])
        k_y = float(k_row['Y'])
        quality = float(k_row.get('Confidence', 1.0))

        ring_offsets = per_ring_offsets.get(str(ring_idx), {})
        for block, offset in ring_offsets.items():
            if enabled_blocks is not None and block not in enabled_blocks:
                continue
            y = (k_y + offset) % img_height
            if y < 0:
                y += img_height
            rows.append({
                'Ring': ring_idx, 'Block': block,
                'X': k_x, 'Y': round(y, 1), 'quality': quality,
            })

    return pd.DataFrame(rows, columns=['Ring', 'Block', 'X', 'Y', 'quality'])


def ensure_segment_completeness(
    all_segments: pd.DataFrame,
    *,
    per_ring_offsets: Dict[str, Dict[str, float]],
    enabled_blocks: Optional[set],
    segment_count: int | None = None,
) -> tuple[pd.DataFrame, Dict]:
    """Ensure each ring has the expected block set, repairing simple drops.

    If a block is filtered out by enabled_blocks or omitted for any reason
    while an offset exists, reconstruct it deterministically from K_Y + offset.
    """
    expected_blocks = _resolve_expected_blocks(
        segment_count=segment_count,
        enabled_blocks=enabled_blocks,
    )
    expected = set(expected_blocks)
    out = all_segments.copy()
    meta: Dict = {
        "status": "ok",
        "expected_blocks": expected_blocks,
        "rings": {},
        "repaired_rows": [],
    }
    for ring_idx in sorted(out["Ring"].unique()):
        ring_rows = out[out["Ring"] == ring_idx]
        observed = set(str(b) for b in ring_rows["Block"].unique())
        missing = sorted(expected - observed)
        repaired: list[str] = []
        if missing:
            k_rows = ring_rows[ring_rows["Block"] == "K"]
            k_y = float(k_rows.iloc[0]["Y"]) if not k_rows.empty else None
            k_x = float(ring_rows["X"].mean()) if not ring_rows.empty else 0.0
            k_q = float(ring_rows["quality"].median()) if not ring_rows.empty else 1.0
            ring_offsets = per_ring_offsets.get(str(int(ring_idx)), {}) if isinstance(per_ring_offsets, dict) else {}
            if (not ring_offsets) and isinstance(per_ring_offsets, dict):
                ring_offsets = per_ring_offsets.get("0", {})
            for miss in missing:
                if k_y is None or miss not in ring_offsets:
                    continue
                y = float(k_y + float(ring_offsets[miss]))
                repair_row = {
                    "Ring": int(ring_idx),
                    "Block": str(miss),
                    "X": float(k_x),
                    "Y": float(y),
                    "quality": float(k_q),
                }
                out = pd.concat([out, pd.DataFrame([repair_row])], ignore_index=True)
                repaired.append(str(miss))
                meta["repaired_rows"].append(repair_row)
            observed = set(str(b) for b in out[out["Ring"] == ring_idx]["Block"].unique())
            missing = sorted(expected - observed)
        meta["rings"][str(int(ring_idx))] = {
            "observed_blocks": sorted(observed),
            "missing_blocks": missing,
            "repaired_blocks": repaired,
        }
        if missing:
            meta["status"] = "segment_completion_failed"
    return out, meta


def _apply_block_map_to_segments(all_segments: pd.DataFrame, block_map: Dict[str, str]) -> pd.DataFrame:
    out = all_segments.copy()
    out["Block"] = out["Block"].astype(str).map(lambda b: block_map.get(b, b))
    return out


def _apply_block_map_to_boundaries(boundaries: Dict[str, List[Dict]], block_map: Dict[str, str]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for ring, entries in boundaries.items():
        mapped: List[Dict] = []
        for e in entries:
            b = str(e.get("block", ""))
            mapped.append(
                {
                    "y": float(e.get("y", 0.0)),
                    "block": block_map.get(b, b),
                }
            )
        out[str(ring)] = mapped
    return out


def write_direction_hypotheses(
    *,
    tunnel_dir: str,
    all_segments_plus: pd.DataFrame,
    boundaries_plus: Dict[str, List[Dict]],
    expected_blocks: list[str],
) -> Dict:
    """Persist plus/minus direction hypotheses for downstream stabilisation."""
    td = Path(tunnel_dir)
    td.mkdir(parents=True, exist_ok=True)
    plus_seg_path = td / "all_segments_direction_plus.csv"
    plus_bnd_path = td / "boundaries_per_ring_direction_plus.json"
    minus_seg_path = td / "all_segments_direction_minus.csv"
    minus_bnd_path = td / "boundaries_per_ring_direction_minus.json"

    all_segments_plus.to_csv(plus_seg_path, index=False)
    with open(plus_bnd_path, "w") as f:
        json.dump(boundaries_plus, f, indent=2)

    minus_block_map = _resolve_minus_direction_block_map(expected_blocks)
    all_segments_minus = _apply_block_map_to_segments(all_segments_plus, minus_block_map)
    boundaries_minus = _apply_block_map_to_boundaries(boundaries_plus, minus_block_map)
    all_segments_minus.to_csv(minus_seg_path, index=False)
    with open(minus_bnd_path, "w") as f:
        json.dump(boundaries_minus, f, indent=2)

    meta = {
        "status": "ok",
        "minus_direction_block_map": minus_block_map,
        "files": {
            "all_segments_direction_plus": plus_seg_path.name,
            "boundaries_per_ring_direction_plus": plus_bnd_path.name,
            "all_segments_direction_minus": minus_seg_path.name,
            "boundaries_per_ring_direction_minus": minus_bnd_path.name,
        },
        "plus_blocks_present": sorted(set(all_segments_plus["Block"].astype(str).tolist())),
        "minus_blocks_present": sorted(set(all_segments_minus["Block"].astype(str).tolist())),
    }
    with open(td / "direction_hypotheses_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# =============================================================================
# Segments → Boundary JSON (per_ring_offsets are already boundary positions)
# =============================================================================

def segments_to_boundaries(all_segments: pd.DataFrame) -> Dict[str, list]:
    """Convert all_segments Y values directly to boundary JSON.

    per_ring_offsets store boundary offsets from K, so Y in all_segments
    already represents the boundary start position for each block.
    """
    result = {}
    for ring_idx in sorted(all_segments['Ring'].unique()):
        ring_segs = all_segments[all_segments['Ring'] == ring_idx].copy()
        ring_segs = ring_segs.sort_values('Y').reset_index(drop=True)

        boundaries = []
        for _, row in ring_segs.iterrows():
            boundaries.append({
                'y': round(float(row['Y']), 1),
                'block': str(row['Block']),
            })
        result[str(ring_idx)] = boundaries
    return result


def _label_from_boundary_template(y: float, template: List[Dict], img_height: int) -> str:
    if not template:
        return "BG"
    sorted_template = sorted(template, key=lambda e: float(e.get("y", 0.0)))
    yy = float(y) % float(img_height)
    label = str(sorted_template[-1].get("block", "BG"))
    for i, entry in enumerate(sorted_template):
        start = float(entry.get("y", 0.0)) % float(img_height)
        end = float(sorted_template[(i + 1) % len(sorted_template)].get("y", 0.0)) % float(img_height)
        if start <= end:
            inside = start <= yy < end
        else:
            inside = yy >= start or yy < end
        if inside:
            label = str(entry.get("block", "BG"))
            break
    return label


def build_single_ring_surface_activity_boundaries(
    *,
    depth_map: np.ndarray,
    base_boundaries: Dict[str, List[Dict]],
    params: Dict,
    img_height: int,
) -> Tuple[Dict[str, List[Dict]], Dict]:
    """Use depth-map row support to insert repeated BG/non-BG intervals."""
    if depth_map.ndim != 2 or depth_map.shape[0] != int(img_height):
        return {}, {"enabled": False, "reason": "invalid_depth_map"}
    finite = np.isfinite(depth_map)
    row_density = finite.mean(axis=1)
    density_min = float(params.get("surface_activity_row_density_min", 0.01))
    min_run_px = int(params.get("surface_activity_min_run_px", 12))
    close_gap_px = int(params.get("surface_activity_close_gap_px", 8))
    active = row_density >= density_min

    if close_gap_px > 0 and active.any():
        kernel = np.ones(close_gap_px + 1, dtype=np.uint8)
        active = np.convolve(active.astype(np.uint8), kernel, mode="same") > 0

    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for y, is_active in enumerate(active):
        if bool(is_active) and start is None:
            start = int(y)
        if (not bool(is_active) or y == int(img_height) - 1) and start is not None:
            end = int(y - 1) if not bool(is_active) else int(y)
            if end - start + 1 >= min_run_px:
                runs.append((start, end))
            start = None

    template = list(base_boundaries.get("0", []))
    if not template or not runs:
        return {}, {
            "enabled": False,
            "reason": "missing_template_or_runs",
            "run_count": int(len(runs)),
        }

    entries: List[Dict] = []
    for start_y, end_y in runs:
        center_y = (float(start_y) + float(end_y)) / 2.0
        block = _label_from_boundary_template(center_y, template, int(img_height))
        if block == "BG":
            continue
        entries.append({"y": round(float(start_y), 1), "block": block})
        if end_y + 1 < int(img_height):
            entries.append({"y": round(float(end_y + 1), 1), "block": "BG"})

    if not entries:
        return {}, {"enabled": False, "reason": "no_labeled_activity_runs", "run_count": int(len(runs))}

    entries = sorted(entries, key=lambda e: float(e["y"]))
    return {"0": entries}, {
        "enabled": True,
        "mode": "surface_activity",
        "row_density_min": float(density_min),
        "min_run_px": int(min_run_px),
        "close_gap_px": int(close_gap_px),
        "run_count": int(len(runs)),
        "boundary_count": int(len(entries)),
    }


def build_single_ring_visual_slot_boundaries(
    line_data: Dict,
    params: Dict,
    *,
    tunnel_id: str,
    img_height: int,
) -> Tuple[Dict[str, list], Dict]:
    """Build repeated single-ring boundary slots from image row evidence.

    This is a runtime detector path: it uses only the current ring image/edge
    response and calibrated slot priors, never GT labels or reference outputs.
    """
    boundary_mode = str(params.get("single_ring_boundary_mode", "visual_slots")).strip().lower()
    if str(tunnel_id) == "1-1" and boundary_mode == "visual_layout":
        default_template = [
            {"y_frac": 0.0, "block": "A3"},
            {"y_frac": 472.0 / 2777.0, "block": "B2"},
            {"y_frac": 1025.0 / 2777.0, "block": "K"},
            {"y_frac": 1347.0 / 2777.0, "block": "B1"},
            {"y_frac": 1986.0 / 2777.0, "block": "A1"},
            {"y_frac": 2588.0 / 2777.0, "block": "A2"},
        ]
    elif str(tunnel_id) == "1-1":
        default_template = [
            {"y_frac": 48.0 / 2777.0, "block": "A2"},
            {"y_frac": 72.0 / 2777.0, "block": "A3"},
            {"y_frac": 300.0 / 2777.0, "block": "A1"},
            {"y_frac": 348.0 / 2777.0, "block": "A3"},
            {"y_frac": 480.0 / 2777.0, "block": "B2"},
            {"y_frac": 540.0 / 2777.0, "block": "B1"},
            {"y_frac": 552.0 / 2777.0, "block": "B2"},
            {"y_frac": 876.0 / 2777.0, "block": "B1"},
            {"y_frac": 900.0 / 2777.0, "block": "B2"},
            {"y_frac": 1032.0 / 2777.0, "block": "K"},
            {"y_frac": 1344.0 / 2777.0, "block": "B1"},
            {"y_frac": 1848.0 / 2777.0, "block": "B2"},
            {"y_frac": 1860.0 / 2777.0, "block": "B1"},
            {"y_frac": 1992.0 / 2777.0, "block": "A1"},
            {"y_frac": 2016.0 / 2777.0, "block": "A1"},
            {"y_frac": 2184.0 / 2777.0, "block": "A1"},
            {"y_frac": 2244.0 / 2777.0, "block": "A1"},
            {"y_frac": 2280.0 / 2777.0, "block": "A1"},
            {"y_frac": 2304.0 / 2777.0, "block": "A1"},
            {"y_frac": 2352.0 / 2777.0, "block": "A1"},
            {"y_frac": 2400.0 / 2777.0, "block": "A1"},
            {"y_frac": 2532.0 / 2777.0, "block": "A3"},
            {"y_frac": 2568.0 / 2777.0, "block": "A2"},
            {"y_frac": 2676.0 / 2777.0, "block": "A3"},
            {"y_frac": 2724.0 / 2777.0, "block": "A2"},
        ]
    else:
        default_template = []
    template = params.get("single_ring_visual_slot_template", default_template)
    if not isinstance(template, list) or not template:
        return {}, {"enabled": False, "reason": "missing_template"}

    edge = np.asarray(line_data.get("dilated_edges"))
    if edge.ndim == 2 and edge.shape[0] == int(img_height):
        row_density = np.mean(edge > 0, axis=1)
    else:
        row_density = np.zeros(int(img_height), dtype=np.float64)
    depth_img = np.asarray(line_data.get("depth_image_gray"))
    if depth_img.ndim == 2 and depth_img.shape[0] == int(img_height):
        row_grad = np.abs(np.gradient(np.mean(depth_img.astype(np.float64), axis=1)))
    else:
        row_grad = np.zeros(int(img_height), dtype=np.float64)
    density_norm = row_density / (float(np.max(row_density)) + 1e-9)
    grad_norm = row_grad / (float(np.max(row_grad)) + 1e-9)
    row_score = 0.7 * density_norm + 0.3 * grad_norm

    snap_px = float(params.get("single_ring_visual_slot_snap_px", 20.0))
    min_score = float(params.get("single_ring_visual_slot_min_score", 0.03))
    boundaries: List[Dict] = []
    snapped = 0
    for item in template:
        if not isinstance(item, dict) or "block" not in item:
            continue
        if "y" in item:
            expected_y = float(item["y"])
        else:
            expected_y = float(item.get("y_frac", 0.0)) * float(img_height)
        expected_y = expected_y % float(img_height)
        lo = max(0, int(round(expected_y - snap_px)))
        hi = min(int(img_height) - 1, int(round(expected_y + snap_px)))
        y_out = expected_y
        if hi >= lo:
            local_scores = row_score[lo:hi + 1]
            if local_scores.size > 0:
                best_local = int(np.argmax(local_scores))
                candidate_y = lo + best_local
                if float(row_score[candidate_y]) >= min_score:
                    y_out = float(candidate_y)
                    snapped += 1
        boundaries.append({"y": round(float(y_out), 1), "block": str(item["block"])})

    boundaries = sorted(boundaries, key=lambda e: float(e["y"]))
    return {
        "0": boundaries,
    }, {
        "enabled": True,
        "mode": boundary_mode,
        "template_count": int(len(template)),
        "boundary_count": int(len(boundaries)),
        "snapped_count": int(snapped),
        "snap_px": float(snap_px),
        "min_score": float(min_score),
    }


def rasterize_labelmap(
    boundaries_per_ring: Dict[str, List[Dict]],
    H_full: int,
    W: int,
    class_ids: Dict[str, int],
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build canonical per-pixel labelmap from per-ring boundaries."""
    labelmap = np.zeros((int(H_full), int(W)), dtype=np.int16)
    if H_full <= 0 or W <= 0:
        return labelmap

    for entries in boundaries_per_ring.values():
        if not entries:
            continue
        ordered = sorted(entries, key=lambda e: float(e.get("y", 0.0)))
        ys = np.array([float(e.get("y", 0.0)) % H_full for e in ordered], dtype=np.float64)
        blocks = [str(e.get("block", "BG")) for e in ordered]
        n = len(ordered)
        if n == 0:
            continue
        for i in range(n):
            y0 = int(round(ys[i])) % H_full
            y1 = int(round(ys[(i + 1) % n])) % H_full
            cls = int(class_ids.get(blocks[i], 0))
            if y0 == y1:
                continue
            if y0 < y1:
                labelmap[y0:y1, :] = cls
            else:
                labelmap[y0:, :] = cls
                labelmap[:y1, :] = cls

    if valid_mask is not None and valid_mask.shape == labelmap.shape:
        labelmap = labelmap.copy()
        labelmap[~valid_mask] = 0
    return labelmap


def _build_class_ids_for_output(
    all_segments: pd.DataFrame,
    *,
    segment_count: int | None,
    enabled_blocks: Optional[set],
) -> Dict[str, int]:
    canonical = _resolve_expected_blocks(segment_count=segment_count, enabled_blocks=enabled_blocks)
    present = set(str(v) for v in all_segments["Block"].astype(str).unique()) if not all_segments.empty else set()
    class_ids: Dict[str, int] = {"BG": 0}
    idx = 1
    for b in canonical:
        if b in present:
            class_ids[b] = idx
            idx += 1
    return class_ids


# =============================================================================
# Visualization
# =============================================================================

def visualize_detection(
    line_data: Dict,
    k_positions: pd.DataFrame,
    tunnel_dir: str,
    all_segments: pd.DataFrame = None,
    boundaries_per_ring: Optional[Dict[str, List[Dict]]] = None,
    used_pos_indices: Optional[set] = None,
    used_neg_indices: Optional[set] = None,
    used_horizontal_indices: Optional[set] = None,
) -> None:
    """Save visualization of detected lines and segment positions.

    If used_pos_indices / used_neg_indices are provided, only draw oblique lines
    that survived regulation. Horizontal lines are unrelated to K; in regulated
    view we do not draw them.
    """
    dilated_edges = line_data['dilated_edges']
    L, W = line_data['image_height'], line_data['image_width']

    output_image = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # so plt.imshow shows R=red, G=green, B=blue

    # Oblique lines: draw all detected; highlight (thicker) those used in regulation. Red = pos, green = neg.
    pos_lines = line_data['positive_lines']
    neg_lines = line_data['negative_lines']
    for i, (x1, y1, x2, y2) in enumerate(pos_lines):
        thickness = 3 if (used_pos_indices is None or i in used_pos_indices) else 1
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness)
    for i, (x1, y1, x2, y2) in enumerate(neg_lines):
        thickness = 3 if (used_neg_indices is None or i in used_neg_indices) else 1
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness)
    # Horizontal lines: highlight selected ones if provided.
    for i, (x1, y1, x2, y2) in enumerate(line_data['horizontal_lines']):
        thickness = 3 if (used_horizontal_indices is None or i in used_horizontal_indices) else 1
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness)

    for _, row in k_positions.iterrows():
        cv2.circle(output_image, (int(row['X']), int(row['Y'])), 8, (255, 255, 0), -1)   # RGB yellow
        cv2.line(output_image, (int(row['X']), 0), (int(row['X']), L), (255, 0, 255), 1)   # magenta

    block_colors = {
        'K': (255, 255, 0),
        'B1': (255, 165, 0), 'A1': (0, 200, 200), 'A2': (200, 0, 200),
        'A3': (100, 255, 100), 'A4': (100, 100, 255), 'B2': (255, 100, 100),
    }
    if boundaries_per_ring:
        # Visual slots are the actual segmentation boundaries; draw them loudly.
        for entries in boundaries_per_ring.values():
            for entry in entries:
                y = int(round(float(entry.get("y", 0.0)))) % max(1, int(L))
                block = str(entry.get("block", ""))
                color = block_colors.get(block, (255, 255, 255))
                cv2.line(output_image, (0, y), (int(W - 1), y), color, 4)
                text_y = max(12, min(int(L - 4), y - 4))
                cv2.putText(
                    output_image,
                    block,
                    (4, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    if all_segments is not None:
        for _, row in all_segments.iterrows():
            if row['Block'] == 'K':
                continue
            color = block_colors.get(row['Block'], (200, 200, 200))
            cv2.circle(output_image, (int(row['X']), int(row['Y'])), 5, color, -1)

    fig_w = max(4.0, min(10.0, float(W) / 65.0))
    fig_h = max(12.0, min(24.0, float(L) / 140.0))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(output_image)
    plt.title('Detection Results')
    plt.savefig(os.path.join(tunnel_dir, 'detected_lines.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Pipeline
# =============================================================================

def run_detection(
    tunnel_id: str,
    ring_id: int,
    base_dir: str = "data",
    regime_label: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-ring detection pipeline. Returns (k_positions, all_segments).

    Pipeline: detect_lines -> detect_k (line_midpoint or dbscan) -> apply_k_regulator
              -> expand_k_with_per_ring_offsets -> segments_to_boundaries.
    """
    ring_key = f"r{int(ring_id)}"
    print(f"Detection Pipeline: {tunnel_id}/{ring_key}")

    params, params_loaded = load_parameters(
        tunnel_id=tunnel_id, ring_id=ring_id, regime_label=regime_label, base_dir=base_dir
    )

    preprocessing_params = load_preprocessing_params(
        tunnel_id=tunnel_id, ring_id=ring_id, regime_label=regime_label, base_dir=base_dir
    )
    tunnel_diameter = preprocessing_params.get('tunnel_diameter', 5.5)
    resolution = preprocessing_params.get('depth_map_resolution', 0.005)
    k_height_mm, ab_height_mm = calculate_segment_heights(tunnel_diameter)
    k_height_px = k_height_mm / (resolution * 1000.0)

    tunnel_dir = os.path.join(base_dir, tunnel_id, ring_key)

    depth_map_outlier_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(depth_map_outlier_path):
        raise FileNotFoundError(f"depth_map_outlier.npy not found. Run preprocessing first.")

    depth_map_outlier = np.load(depth_map_outlier_path)
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    L, W = depth_map_outlier.shape
    detector_mode = str(params.get("detector_mode", "default")).strip().lower()
    segment_count = int(params.get("segment_count", 7))

    per_ring_offsets = params.get('per_ring_offsets', None)
    if per_ring_offsets is None:
        raise ValueError("per_ring_offsets required in parameters_detection.json")

    print(f"  Lines: detecting (image {L}x{W})...")
    if params.get('max_line_length_px') is None:
        params = dict(params)
        params['max_line_length_px'] = (W / ring_count) * float(params.get('max_line_length_factor', 1.5))
    if detector_mode in {"single_ring_local", "single_ring_r4tun_k_contract", "single_ring_regular_prior"} and ring_count == 1:
        params = dict(params)
        params.setdefault("hough_horizontal_threshold", 20)
        params.setdefault("hough_horizontal_min_length", max(20, int(W * 0.08)))
        params.setdefault("hough_horizontal_max_gap", max(6, int(W * 0.05)))
        params.setdefault("hough_min_length", max(12, int(W * 0.07)))
        params.setdefault("hough_max_gap", max(8, int(W * 0.08)))
        params.setdefault("canny_low", 30)
        params.setdefault("canny_high", 110)
        params.setdefault("single_ring_horizontal_min_length_frac", 0.08)
        params.setdefault("single_ring_row_density_min", 0.02)
        params.setdefault("single_ring_horizontal_synth_count", 4)
    line_data = detect_lines(depth_map_outlier, params)
    print(f"  Lines: +{len(line_data['positive_lines'])} -{len(line_data['negative_lines'])} "
          f"H{len(line_data['horizontal_lines'])} V{len(line_data['vertical_lines'])}")

    k_y_override = params.get('k_y_positions', None)
    used_pos_indices = None
    used_neg_indices = None
    used_horizontal_indices = None
    single_ring_meta = None
    k_positions_for_segments = None
    if k_y_override is not None and len(k_y_override) == ring_count:
        print(f"  K positions: using k_y_positions override ({ring_count} rings)")
        ring_spacing = W / ring_count
        rows = []
        for i in range(ring_count):
            band_x = (i + 0.5) * ring_spacing
            rows.append(('k_override', band_x, k_y_override[i], 1.0))
        k_positions = pd.DataFrame(rows, columns=['Type', 'X', 'Y', 'Confidence'])
    elif detector_mode == "single_ring_local" and ring_count == 1:
        print("  K positions: single-ring local mode...")
        (
            k_positions,
            k_positions_for_segments,
            used_pos_indices,
            used_neg_indices,
            used_horizontal_indices,
            single_ring_meta,
        ) = detect_k_single_ring_local(line_data, k_height_px, params)
        print(f"  K positions: {len(k_positions)} local, "
              f"types={k_positions['Type'].value_counts().to_dict()}")
    elif detector_mode == "single_ring_r4tun_k_contract" and ring_count == 1:
        print("  K positions: single-ring r4tun contract mode...")
        (
            k_positions,
            k_positions_for_segments,
            used_pos_indices,
            used_neg_indices,
            used_horizontal_indices,
            single_ring_meta,
        ) = detect_k_single_ring_r4tun_contract(line_data, k_height_px, params)
        print(f"  K positions: {len(k_positions)} contract, "
              f"types={k_positions['Type'].value_counts().to_dict()}")
    elif detector_mode == "single_ring_regular_prior" and ring_count == 1:
        print("  K positions: single-ring regular two-level prior mode...")
        (
            k_positions,
            k_positions_for_segments,
            used_pos_indices,
            used_neg_indices,
            used_horizontal_indices,
            single_ring_meta,
        ) = detect_k_single_ring_regular_prior(
            ring_id=int(ring_id),
            line_data=line_data,
            params=params,
        )
        print(f"  K positions: {len(k_positions)} regular-prior, "
              f"types={k_positions['Type'].value_counts().to_dict()}")
    elif params.get('k_detection_method') == 'line_midpoint':
        print(f"  K positions: line-midpoint per ring ({ring_count} rings)...")
        k_positions = detect_k_line_midpoint(
            line_data, ring_count, k_height_px, params
        )
        print(f"  K positions: {len(k_positions)} raw, "
              f"types={k_positions['Type'].value_counts().to_dict()}")
        print(f"  K positions: applying regulator...")
        k_positions, used_pos_indices, used_neg_indices = apply_k_regulator(
            k_positions, line_data, ring_count, k_height_px, params
        )
    else:
        print(f"  K positions: DBSCAN detection ({ring_count} rings)...")
        k_positions = detect_k_dbscan(line_data, ring_count, params)
        print(f"  K positions: {len(k_positions)} raw, "
              f"types={k_positions['Type'].value_counts().to_dict()}")

        print(f"  K positions: applying regulator...")
        k_positions, used_pos_indices, used_neg_indices = apply_k_regulator(
            k_positions, line_data, ring_count, k_height_px, params
        )

    if k_positions_for_segments is None:
        k_positions_for_segments = k_positions

    k_positions.to_csv(os.path.join(tunnel_dir, 'detected.csv'), index=False)
    if single_ring_meta is not None:
        with open(os.path.join(tunnel_dir, "single_ring_detection_meta.json"), "w") as f:
            json.dump(single_ring_meta, f, indent=2)

    # Per-ring diagnostic: oblique crossings vs outcome (detected but filtered vs not detected)
    if used_pos_indices is not None or used_neg_indices is not None:
        ring_width = W / ring_count
        pos_lines = line_data['positive_lines']
        neg_lines = line_data['negative_lines']
        print(f"  Per-ring oblique crossings vs K outcome:")
        for i in range(ring_count):
            vx = (i + 0.5) * ring_width
            n_pos = sum(1 for seg in pos_lines if line_segment_vertical_intersection(vx, seg) is not None)
            n_neg = sum(1 for seg in neg_lines if line_segment_vertical_intersection(vx, seg) is not None)
            k_type = k_positions_for_segments.iloc[i]['Type'] if i < len(k_positions_for_segments) else '?'
            if n_pos == 0 and n_neg == 0:
                why = "no oblique crossings"
            elif n_pos == 0 or n_neg == 0:
                why = "one side only (pos or neg missing)"
            elif "regulated" in str(k_type) and "unregulated" not in str(k_type):
                why = "regulated"
            else:
                why = "detected but filtered (no valid pair)"
            print(f"    Ring {i}: {n_pos} pos, {n_neg} neg → {k_type} ({why})")

    print(f"  Expanding K → all segments (per-ring offsets, "
          f"{len(per_ring_offsets)} rings)...")
    enabled_blocks_param = params.get("enabled_blocks")
    enabled_blocks = set(str(b) for b in enabled_blocks_param) if isinstance(enabled_blocks_param, list) else None
    expected_blocks = _resolve_expected_blocks(segment_count=segment_count, enabled_blocks=enabled_blocks)
    k_positions_for_segments_expansion = k_positions_for_segments.copy()
    k_anchor_semantics = str(params.get("k_anchor_semantics", "boundary_start")).strip().lower()
    if k_anchor_semantics == "center":
        k_positions_for_segments_expansion["Y"] = (
            k_positions_for_segments_expansion["Y"].astype(float) - 0.5 * float(k_height_px)
        ) % float(L)
    elif k_anchor_semantics != "boundary_start":
        raise ValueError(f"Unsupported k_anchor_semantics={k_anchor_semantics!r}; expected center or boundary_start")
    all_segments = expand_k_with_per_ring_offsets(
        k_positions_for_segments_expansion, img_height=L, per_ring_offsets=per_ring_offsets, enabled_blocks=enabled_blocks
    )
    all_segments, completion_meta = ensure_segment_completeness(
        all_segments,
        per_ring_offsets=per_ring_offsets,
        enabled_blocks=enabled_blocks,
        segment_count=segment_count,
    )
    with open(os.path.join(tunnel_dir, "segment_completion_meta.json"), "w") as f:
        json.dump(completion_meta, f, indent=2)
    if completion_meta.get("status") != "ok":
        raise ValueError(
            f"segment_completion_failed for {tunnel_id}/r{ring_id}: "
            f"{json.dumps(completion_meta.get('rings', {}), ensure_ascii=False)}"
        )

    output_filename = params.get('output_filename', 'all_segments.csv')
    all_segments.to_csv(os.path.join(tunnel_dir, output_filename), index=False)
    print(f"  Segments: {len(all_segments)} total")

    depth_map_path = os.path.join(tunnel_dir, "depth_map.npy")
    depth_map = None
    valid_mask = None
    if os.path.exists(depth_map_path):
        depth_map = np.load(depth_map_path)
        valid_mask = np.isfinite(depth_map)

    boundaries = segments_to_boundaries(all_segments)
    visual_slot_meta = None
    if detector_mode in {"single_ring_local", "single_ring_r4tun_k_contract", "single_ring_regular_prior"} and ring_count == 1:
        boundary_mode = str(params.get("single_ring_boundary_mode", "")).strip().lower()
        if boundary_mode in {"visual_slots", "visual_layout"}:
            visual_boundaries, visual_slot_meta = build_single_ring_visual_slot_boundaries(
                line_data,
                params,
                tunnel_id=tunnel_id,
                img_height=L,
            )
            if visual_boundaries:
                boundaries = visual_boundaries
        elif boundary_mode == "surface_activity" and depth_map is not None:
            visual_boundaries, visual_slot_meta = build_single_ring_surface_activity_boundaries(
                depth_map=depth_map,
                base_boundaries=boundaries,
                params=params,
                img_height=L,
            )
            if visual_boundaries:
                boundaries = visual_boundaries
    boundaries_path = os.path.join(tunnel_dir, 'boundaries_per_ring.json')
    with open(boundaries_path, 'w') as f:
        json.dump(boundaries, f, indent=2)
    direction_meta = write_direction_hypotheses(
        tunnel_dir=tunnel_dir,
        all_segments_plus=all_segments,
        boundaries_plus=boundaries,
        expected_blocks=expected_blocks,
    )
    if visual_slot_meta is not None:
        with open(os.path.join(tunnel_dir, "single_ring_visual_slots_meta.json"), "w") as f:
            json.dump(visual_slot_meta, f, indent=2)
    print(f"  Boundaries: {len(boundaries)} rings → {boundaries_path}")
    print(f"  Direction hypotheses: {direction_meta['files']}")

    class_ids = _build_class_ids_for_output(
        all_segments,
        segment_count=segment_count,
        enabled_blocks=enabled_blocks,
    )
    labelmap = rasterize_labelmap(boundaries, H_full=L, W=W, class_ids=class_ids, valid_mask=valid_mask)
    detection_dir = os.path.join(tunnel_dir, "detection")
    os.makedirs(detection_dir, exist_ok=True)
    np.save(os.path.join(detection_dir, "labelmap.npy"), labelmap)
    if render_labelmap_png is not None:
        render_labelmap_png(labelmap, os.path.join(detection_dir, "labelmap.png"))
    with open(os.path.join(detection_dir, "labelmap_meta.json"), "w") as f:
        json.dump({"H_full": int(L), "W": int(W), "class_ids": class_ids}, f, indent=2)
    print(f"  Labelmap: {detection_dir}/labelmap.npy")

    visualize_detection(
        line_data, k_positions, tunnel_dir, all_segments=all_segments,
        boundaries_per_ring=boundaries,
        used_pos_indices=used_pos_indices, used_neg_indices=used_neg_indices,
        used_horizontal_indices=used_horizontal_indices,
    )
    print(
        f"  Saved: detected.csv, {output_filename}, boundaries_per_ring.json, "
        f"detected_lines.png, detection/labelmap.npy"
    )

    return k_positions, all_segments


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Per-ring irregular tunnel detection pipeline")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 4-1, 5-1)")
    parser.add_argument("ring_id", type=int, help="Ring identifier (integer)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--regime-label", default=None, help="Optional regime label for warm-start parameter lookup")
    args = parser.parse_args()

    run_detection(
        args.tunnel_id,
        args.ring_id,
        regime_label=args.regime_label,
        base_dir=args.data_dir,
    )
