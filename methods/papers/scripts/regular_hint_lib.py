"""Graded hint modes for regular-tunnel (1-*, 2-*) K-anchor detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from k_consensus_lib import (
    _GOOD_Y_DETECTION_TYPES,
    _two_level_centers,
    tunnel_family_from_id,
)

DEFAULT_HINT_Y_LEVELS = (1210.0, 1640.0)
DEFAULT_K_PATTERN_STEP_PX = 431.87
REFERENCE_TUNNEL_ID = "2-2"

_TYPE_PRIORITY = {
    "midpoint": 4,
    "positive_slope": 3,
    "negative_slope": 3,
    "horizontal": 2,
    "assume": 1,
    "default": 0,
}


def uniform_ring_x_positions(width: float, ring_count: int) -> list[float]:
    """Evenly spaced ring-column X at block centres."""
    block_width = width / ring_count
    return [(i + 0.5) * block_width for i in range(ring_count)]


def infer_zigzag_levels(
    ys: np.ndarray,
    *,
    step_px: float = DEFAULT_K_PATTERN_STEP_PX,
    min_gap_px: float = 350.0,
) -> tuple[float, float]:
    """Split Y values into low/high stagger bands."""
    ys = np.asarray(ys, dtype=float)
    if len(ys) == 0:
        low, high = DEFAULT_HINT_Y_LEVELS
        return float(low), float(high)
    if len(ys) == 1:
        y = float(ys[0])
        return y - step_px / 2, y + step_px / 2
    low, high = _two_level_centers(ys, min_gap_px=min_gap_px)
    if low is None or high is None:
        low, high = DEFAULT_HINT_Y_LEVELS
    return float(low), float(high)


def y_for_ring_parity(
    ring_index: int,
    low_y: float,
    high_y: float,
    *,
    low_parity: int = 0,
) -> float:
    parity = ring_index % 2
    return low_y if parity == low_parity else high_y


def propagate_zigzag_y(
    n: int,
    low_y: float,
    high_y: float,
    *,
    low_parity: int = 0,
) -> list[float]:
    return [y_for_ring_parity(i, low_y, high_y, low_parity=low_parity) for i in range(n)]


def infer_low_parity_from_anchors(
    anchor_indices: list[int],
    anchor_ys: list[float],
    low_y: float,
    high_y: float,
) -> int:
    votes = {0: 0.0, 1: 0.0}
    for idx, y in zip(anchor_indices, anchor_ys):
        parity = idx % 2
        if abs(y - low_y) <= abs(y - high_y):
            votes[parity] += 1.0
        else:
            votes[1 - parity] += 1.0
    return 0 if votes[0] >= votes[1] else 1


def pick_best_anchor_rings(
    hough_points: list[tuple[str, tuple[float, float]]],
    k: int = 2,
) -> list[int]:
    scored: list[tuple[float, int]] = []
    for i, (typ, _) in enumerate(hough_points):
        scored.append((_TYPE_PRIORITY.get(typ, 0), i))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [idx for _, idx in scored[:k]]


def gt_k_ring_table(tunnel_dir: str | Path) -> pd.DataFrame:
    """Per-ring GT K theta and h from final.csv (sorted by h descending)."""
    path = Path(tunnel_dir) / "final.csv"
    if not path.is_file():
        raise FileNotFoundError(f"final.csv not found in {tunnel_dir}")
    df = pd.read_csv(path, usecols=["segment", "ring", "theta", "h"])
    k = df[df["segment"] == 1].groupby("ring").agg(
        theta=("theta", "median"),
        h=("h", "median"),
    )
    k = k.sort_values("h", ascending=False).reset_index()
    k["ring_index"] = range(len(k))
    return k


def _fit_h_theta_to_pixel(
    hs: np.ndarray,
    thetas: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coef_h = np.polyfit(hs, xs, 1)
    coef_t = np.polyfit(thetas, ys, 1)
    return coef_h, coef_t


def gt_k_pixel_positions(
    tunnel_dir: str | Path,
    detected_x: list[float] | None = None,
    detected_y: list[float] | None = None,
) -> tuple[list[float], list[float], pd.DataFrame]:
    """Map GT K per ring to depth-map pixel (X, Y) via h->X and theta->Y linear fits."""
    tunnel_dir = Path(tunnel_dir)
    ring_tbl = gt_k_ring_table(tunnel_dir)

    if (
        detected_x is not None
        and detected_y is not None
        and len(detected_x) == len(ring_tbl)
        and len(detected_y) == len(ring_tbl)
    ):
        coef_h, coef_t = _fit_h_theta_to_pixel(
            ring_tbl["h"].to_numpy(),
            ring_tbl["theta"].to_numpy(),
            np.asarray(detected_x, dtype=float),
            np.asarray(detected_y, dtype=float),
        )
        xs = np.polyval(coef_h, ring_tbl["h"].to_numpy())
        # Piecewise-linear theta->Y on calibration rings (exact at calib points)
        thetas = ring_tbl["theta"].to_numpy()
        y_cal = np.asarray(detected_y, dtype=float)
        order = np.argsort(thetas)
        ys = np.interp(thetas, thetas[order], y_cal[order])
    else:
        ref_dir = tunnel_dir.parent / REFERENCE_TUNNEL_ID
        ref_det = ref_dir / "detected.csv"
        if ref_det.is_file() and (ref_dir / "final.csv").is_file():
            det = pd.read_csv(ref_det)
            ref_rings = gt_k_ring_table(ref_dir)
            n = min(len(det), len(ref_rings))
            coef_h, coef_t = _fit_h_theta_to_pixel(
                ref_rings["h"].to_numpy()[:n],
                ref_rings["theta"].to_numpy()[:n],
                det["X"].to_numpy()[:n],
                det["Y"].to_numpy()[:n],
            )
        else:
            coef_h = np.array([-205.92, 7347.93])
            coef_t = np.array([-206.65, 3213.34])

        xs = np.polyval(coef_h, ring_tbl["h"].to_numpy())
        ys = np.polyval(coef_t, ring_tbl["theta"].to_numpy())

    ring_tbl = ring_tbl.copy()
    ring_tbl["pixel_x"] = xs
    ring_tbl["pixel_y"] = ys
    return list(xs), list(ys), ring_tbl


def cross_tunnel_y_levels(
    tunnel_dir: str | Path,
    *,
    ref_tunnel_id: str = REFERENCE_TUNNEL_ID,
) -> tuple[float, float, int]:
    """Y levels from reference tunnel stagger + one GT K anchor on target."""
    tunnel_dir = Path(tunnel_dir)
    ref_dir = tunnel_dir.parent / ref_tunnel_id

    def _read_ref_y() -> np.ndarray:
        for name in ("detected_calib.csv", "detected.csv"):
            p = ref_dir / name
            if p.is_file():
                return pd.read_csv(p)["Y"].to_numpy()
        anthropic = Path(tunnel_dir).resolve().parents[3] / "data" / "ablation_anthropic" / "memory+state+knowledge" / ref_tunnel_id / "detected.csv"
        if anthropic.is_file():
            return pd.read_csv(anthropic)["Y"].to_numpy()
        return np.array(DEFAULT_HINT_Y_LEVELS)

    ref_ys = _read_ref_y()
    low_ref, high_ref = infer_zigzag_levels(ref_ys)
    step = high_ref - low_ref

    ring_tbl = gt_k_ring_table(tunnel_dir)
    _, ys_gt, _ = gt_k_pixel_positions(tunnel_dir)
    anchor_y = float(ys_gt[0])

    if abs(anchor_y - low_ref) <= abs(anchor_y - high_ref):
        low_y = anchor_y
        high_y = anchor_y + step
        low_parity = 0
    else:
        high_y = anchor_y
        low_y = anchor_y - step
        low_parity = infer_low_parity_from_anchors([0], [anchor_y], low_y, high_y)
    return low_y, high_y, low_parity


def build_hint_points(
    xs: list[float],
    ys: list[float],
    *,
    typ: str = "hint",
) -> list[tuple[str, tuple[float, float]]]:
    return [(typ, (float(x), float(y))) for x, y in zip(xs, ys)]


def apply_hint_mode(
    hough_points: list[tuple[str, tuple[float, float]]],
    tunnel_id: str,
    tunnel_dir: str | Path,
    *,
    hint_mode: str = "off",
    ring_count: int,
    image_width: float,
    image_height: float,
    hint_y_levels: tuple[float, float] = DEFAULT_HINT_Y_LEVELS,
    k_pattern_step_px: float = DEFAULT_K_PATTERN_STEP_PX,
    hint_gt_k_rings: int = 2,
) -> list[tuple[str, tuple[float, float]]]:
    hint_mode = (hint_mode or "off").lower().strip()
    if hint_mode == "off" or tunnel_family_from_id(tunnel_id) != "regular":
        return hough_points

    tunnel_dir = Path(tunnel_dir)
    uniform_x = uniform_ring_x_positions(image_width, ring_count)
    n = ring_count

    def _from_levels(
        low_y: float,
        high_y: float,
        low_parity: int = 0,
        use_hough_x: bool = False,
    ):
        ys_out = propagate_zigzag_y(n, low_y, high_y, low_parity=low_parity)
        if use_hough_x and len(hough_points) == n:
            xs_out = [float(hough_points[i][1][0]) for i in range(n)]
        else:
            xs_out = uniform_x
        return build_hint_points(xs_out, ys_out, typ="hint_zigzag")

    if hint_mode == "zigzag_prior":
        low_y, high_y = float(hint_y_levels[0]), float(hint_y_levels[1])
        if low_y > high_y:
            low_y, high_y = high_y, low_y
        return _from_levels(low_y, high_y)

    if hint_mode == "zigzag_fit":
        mid_ys = [
            hough_points[i][1][1]
            for i, (t, _) in enumerate(hough_points)
            if t == "midpoint"
        ]
        if not mid_ys:
            mid_ys = [
                hough_points[i][1][1]
                for i, (t, _) in enumerate(hough_points)
                if t in _GOOD_Y_DETECTION_TYPES
            ]
        low_y, high_y = infer_zigzag_levels(np.array(mid_ys), step_px=k_pattern_step_px)
        mid_idx = [i for i, (t, _) in enumerate(hough_points) if t == "midpoint"]
        low_parity = (
            infer_low_parity_from_anchors(
                mid_idx,
                [hough_points[i][1][1] for i in mid_idx],
                low_y,
                high_y,
            )
            if mid_idx
            else 0
        )
        return _from_levels(low_y, high_y, low_parity=low_parity)

    if hint_mode == "two_best":
        anchor_idx = pick_best_anchor_rings(hough_points, k=2)
        anchor_ys = [hough_points[i][1][1] for i in anchor_idx]
        low_y, high_y = infer_zigzag_levels(np.array(anchor_ys), step_px=k_pattern_step_px)
        low_parity = infer_low_parity_from_anchors(anchor_idx, anchor_ys, low_y, high_y)
        return _from_levels(low_y, high_y, low_parity=low_parity)

    if hint_mode == "two_gt_k":
        k = min(hint_gt_k_rings, ring_count)
        anchor_idx = list(range(k))
        calib = tunnel_dir / "detected_calib.csv"
        det_x, det_y = None, None
        if calib.is_file():
            det = pd.read_csv(calib).sort_values("X").reset_index(drop=True)
            det_x, det_y = det["X"].tolist(), det["Y"].tolist()
        _, ys_gt, _ = gt_k_pixel_positions(tunnel_dir, detected_x=det_x, detected_y=det_y)
        anchor_ys = ys_gt[:k]
        low_y, high_y = infer_zigzag_levels(np.array(anchor_ys), step_px=k_pattern_step_px)
        low_parity = infer_low_parity_from_anchors(anchor_idx, anchor_ys, low_y, high_y)
        return _from_levels(low_y, high_y, low_parity=low_parity)

    if hint_mode == "gt_k_all":
        _, ys_gt, _ = gt_k_pixel_positions(tunnel_dir)
        return build_hint_points(uniform_x[:n], ys_gt[:n], typ="hint_gt_k")

    if hint_mode in ("oracle", "gt_k_hough_x"):
        _, ys_gt, _ = gt_k_pixel_positions(tunnel_dir)
        if len(hough_points) == n:
            xs_out = [float(hough_points[i][1][0]) for i in range(n)]
        else:
            xs_out, _, _ = gt_k_pixel_positions(tunnel_dir)
            xs_out = xs_out[:n]
        return build_hint_points(xs_out, ys_gt[:n], typ="hint_oracle")

    if hint_mode == "cross_tunnel":
        low_y, high_y, low_parity = cross_tunnel_y_levels(tunnel_dir)
        return _from_levels(low_y, high_y, low_parity=low_parity)

    return hough_points


def hint_level_to_mode(level: str) -> str:
    mapping = {
        "L0": "off",
        "L1": "zigzag_prior",
        "L2": "zigzag_fit",
        "L3": "two_best",
        "L4": "two_gt_k",
        "L5": "gt_k_all",
        "L6": "oracle",
        "L7": "cross_tunnel",
    }
    return mapping.get(level, level)
