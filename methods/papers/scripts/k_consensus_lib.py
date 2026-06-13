"""K-pattern Y consensus for regular and continuous tunnel families."""
from __future__ import annotations

import numpy as np

_GOOD_Y_DETECTION_TYPES = frozenset({"midpoint", "positive_slope", "negative_slope", "horizontal"})
_FALLBACK_Y_TYPES = frozenset({"default", "assume"})


def tunnel_family_from_id(tunnel_id: str) -> str:
    prefix = tunnel_id.split("-", 1)[0]
    if prefix in ("1", "2"):
        return "regular"
    if prefix == "3":
        return "continuous"
    return "complex"


def _mad_trim_values(ys: np.ndarray, k: float = 2.5) -> np.ndarray:
    ys = np.asarray(ys, dtype=float)
    if len(ys) < 3:
        return ys
    med = float(np.median(ys))
    mad = float(np.median(np.abs(ys - med)))
    if mad < 1e-6:
        return ys
    thresh = k * 1.4826 * mad
    return ys[np.abs(ys - med) <= thresh]


def _indices_for_level_estimation(out: list) -> list[int]:
    mid = [i for i, (t, _) in enumerate(out) if t == "midpoint"]
    if len(mid) >= 2:
        return mid
    return [i for i, (t, _) in enumerate(out) if t in _GOOD_Y_DETECTION_TYPES]


def _two_level_centers(ys: np.ndarray, *, min_gap_px: float = 350.0) -> tuple[float | None, float | None]:
    """Split at largest Y-gap; re-fit if span is too narrow (outlier-contaminated high band)."""
    ys = np.sort(np.asarray(ys, dtype=float))
    if len(ys) < 2:
        return None, None

    def _split(sorted_ys: np.ndarray) -> tuple[float, float]:
        gaps = np.diff(sorted_ys)
        split_at = int(np.argmax(gaps)) + 1 if len(gaps) else 1
        split_at = max(1, min(split_at, len(sorted_ys) - 1))
        low_c = float(np.median(sorted_ys[:split_at]))
        high_c = float(np.median(sorted_ys[split_at:]))
        if low_c > high_c:
            low_c, high_c = high_c, low_c
        return low_c, high_c

    low_center, high_center = _split(ys)
    if high_center - low_center < min_gap_px and len(ys) > 2:
        cutoff = low_center + 0.75 * (high_center - low_center)
        trimmed = ys[ys <= cutoff]
        if len(trimmed) >= 2:
            low_center, high_center = _split(trimmed)
    return low_center, high_center


def correct_fallback_y_positions_v1(adjusted_points, tunnel_id: str) -> list:
    """Original fallback-only correction (v1 behaviour)."""
    n = len(adjusted_points)
    if n == 0:
        return adjusted_points

    out = [(str(t), (float(xy[0]), float(xy[1]))) for t, xy in adjusted_points]
    family = tunnel_family_from_id(tunnel_id)

    def y_of(i):
        return out[i][1][1]

    def set_y(i, new_y):
        t, (x, _) = out[i]
        out[i] = (t, (x, float(new_y)))

    good_idx = [i for i in range(n) if out[i][0] in _GOOD_Y_DETECTION_TYPES]
    if not good_idx:
        return out

    good_ys = np.array([y_of(i) for i in good_idx], dtype=float)

    if family == "continuous":
        med = float(np.median(good_ys))
        for i in range(n):
            if out[i][0] in _FALLBACK_Y_TYPES:
                set_y(i, med)
        return out

    if family == "complex":
        for i in range(n):
            if out[i][0] != "default":
                continue
            prev_y = next_y = None
            for j in range(i - 1, -1, -1):
                if out[j][0] in _GOOD_Y_DETECTION_TYPES:
                    prev_y = y_of(j)
                    break
            for k in range(i + 1, n):
                if out[k][0] in _GOOD_Y_DETECTION_TYPES:
                    next_y = y_of(k)
                    break
            if prev_y is not None and next_y is not None:
                set_y(i, (prev_y + next_y) / 2.0)
            elif prev_y is not None:
                set_y(i, prev_y)
            elif next_y is not None:
                set_y(i, next_y)
        return out

    if len(good_idx) < 2:
        return out

    sorted_ys = np.sort(good_ys)
    split = max(1, len(sorted_ys) // 2)
    low_center = float(np.mean(sorted_ys[:split]))
    high_center = float(np.mean(sorted_ys[split:]))
    if low_center > high_center:
        low_center, high_center = high_center, low_center

    low_votes = {0: 0.0, 1: 0.0}
    high_votes = {0: 0.0, 1: 0.0}
    for gi in good_idx:
        y = y_of(gi)
        p = gi % 2
        if abs(y - low_center) <= abs(y - high_center):
            low_votes[p] += 1.0
        else:
            high_votes[p] += 1.0

    def y_for_parity(parity: int) -> float:
        return low_center if low_votes[parity] >= high_votes[parity] else high_center

    for i in range(n):
        if out[i][0] in _FALLBACK_Y_TYPES:
            set_y(i, y_for_parity(i % 2))
    return out


def correct_k_pattern_y_positions(
    adjusted_points,
    tunnel_id: str,
    *,
    k_pattern_correction: bool = True,
    k_pattern_outlier_tol_px: float = 215.0,
) -> tuple[list, int]:
    """
    v2: robust level estimation + optional snap of wrong confident detections.

    Returns (corrected_points, n_snapped).
    """
    n = len(adjusted_points)
    if n == 0:
        return adjusted_points, 0

    out = [(str(t), (float(xy[0]), float(xy[1]))) for t, xy in adjusted_points]
    family = tunnel_family_from_id(tunnel_id)
    snapped = 0

    def y_of(i):
        return out[i][1][1]

    def set_y(i, new_y):
        nonlocal snapped
        old = y_of(i)
        new_y = float(new_y)
        if abs(old - new_y) > 0.5:
            snapped += 1
        t, (x, _) = out[i]
        out[i] = (t, (x, new_y))

    level_idx = _indices_for_level_estimation(out)
    if not level_idx:
        return out, 0

    level_ys = np.array([y_of(i) for i in level_idx], dtype=float)

    if family == "continuous":
        med = float(np.median(_mad_trim_values(level_ys)))
        for i in range(n):
            expected = med
            is_fallback = out[i][0] in _FALLBACK_Y_TYPES
            is_outlier = k_pattern_correction and abs(y_of(i) - expected) > k_pattern_outlier_tol_px
            if is_fallback or is_outlier:
                set_y(i, expected)
        return out, snapped

    if family == "complex":
        return correct_fallback_y_positions_v1(adjusted_points, tunnel_id), 0

    low_center, high_center = _two_level_centers(level_ys)
    if low_center is None or high_center is None:
        return out, 0

    good_idx = [i for i in range(n) if out[i][0] in _GOOD_Y_DETECTION_TYPES]
    low_votes = {0: 0.0, 1: 0.0}
    high_votes = {0: 0.0, 1: 0.0}
    for gi in good_idx:
        y = y_of(gi)
        p = gi % 2
        if abs(y - low_center) <= abs(y - high_center):
            low_votes[p] += 1.0
        else:
            high_votes[p] += 1.0

    def y_for_parity(parity: int) -> float:
        return low_center if low_votes[parity] >= high_votes[parity] else high_center

    for i in range(n):
        y = y_of(i)
        expected = y_for_parity(i % 2)
        dist_nearest_level = min(abs(y - low_center), abs(y - high_center))
        is_fallback = out[i][0] in _FALLBACK_Y_TYPES
        # Snap confident rows only when Y is far from BOTH valid stagger levels (true outlier).
        is_outlier = (
            k_pattern_correction
            and dist_nearest_level > k_pattern_outlier_tol_px
        )
        if is_fallback or is_outlier:
            set_y(i, expected)

    return out, snapped
