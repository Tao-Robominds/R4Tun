"""
Ring-native unfolding: single-plane ellipse RANSAC (ported from r4tun/agents/unfolding.py).

Tunnel-wide slicing / centerline polynomial fitting is intentionally omitted.

Gravity-bottom anchoring (v3 default): after the per-ring PCA + RANSAC ellipse
basis produces ``theta`` (arc length around the ring), the bin with the
lowest median ``z`` is rolled to ``theta = 0`` so the world-frame physical
bottom of the tunnel is always anchored at the start of the unfolded image.
This collapses the rotational degree of freedom that previously drifted per
ring and made calibration templates non-transferable.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd


class RANSAC:
    """Copied from r4tun/agents/unfolding.py (ellipse fitting)."""

    _MAX_ITERATIONS = 5000

    def __init__(
        self,
        data,
        threshold,
        P,
        S,
        N,
        initial_iterations=999,
        inlier_threshold_multiplier=0.8,
        seed=None,
    ):
        self.point_data = np.asarray(data, dtype=np.float64)
        self.error_threshold = threshold
        self.N = N
        self.S = S
        self.P = P
        self.max_inliers = len(data) * S
        self.items = initial_iterations
        self.inlier_threshold_multiplier = inlier_threshold_multiplier
        self.count = 0
        self.best_model = ((0, 0), (1e-6, 1e-6), 0)
        self._rng = np.random.default_rng(seed)

    def random_sampling(self, n):
        n_pts = len(self.point_data)
        if n_pts == 0:
            raise ValueError("RANSAC: empty point_data")
        if n_pts < n:
            idx = self._rng.choice(n_pts, size=n, replace=True)
        else:
            idx = self._rng.choice(n_pts, size=n, replace=False)
        return self.point_data[idx]

    def Geometric2Conic(self, ellipse):
        (x0, y0), (bb, aa), phi_b_deg = ellipse
        a, b = aa / 2, bb / 2
        phi_b_rad = np.radians(phi_b_deg)
        ax, ay = -np.sin(phi_b_rad), np.cos(phi_b_rad)

        a2, b2 = a * a, b * b
        if a2 > 0 and b2 > 0:
            A = ax * ax / a2 + ay * ay / b2
            B = 2 * ax * ay / a2 - 2 * ax * ay / b2
            C = ay * ay / a2 + ax * ax / b2
            D = (-2 * ax * ay * y0 - 2 * ax * ax * x0) / a2 + (2 * ax * ay * y0 - 2 * ay * ay * x0) / b2
            E = (-2 * ax * ay * x0 - 2 * ay * ay * y0) / a2 + (2 * ax * ay * x0 - 2 * ax * ax * y0) / b2
            F = (2 * ax * ay * x0 * y0 + ax * ax * x0 * x0 + ay * ay * y0 * y0) / a2 + (
                -2 * ax * ay * x0 * y0 + ay * ay * x0 * x0 + ax * ax * y0 * y0
            ) / b2 - 1
        else:
            A, B, C, D, E, F = 1, 0, 1, 0, 0, -1e-6

        return np.array([A, B, C, D, E, F])

    def eval_model(self, ellipse):
        a, b, c, d, e, f = self.Geometric2Conic(ellipse)
        E = 4 * a * c - b * b
        if E <= 0:
            return 0, np.array([])

        (x, y), (LAxis, SAxis), Angle = ellipse
        LAxis, SAxis = LAxis / 2, SAxis / 2
        if SAxis > LAxis:
            SAxis, LAxis = LAxis, SAxis

        Axis = math.sqrt(LAxis**2 - SAxis**2)
        f1_x = x - Axis * math.cos(math.radians(Angle))
        f1_y = y - Axis * math.sin(math.radians(Angle))
        f2_x = x + Axis * math.cos(math.radians(Angle))
        f2_y = y + Axis * math.sin(math.radians(Angle))

        f1, f2 = np.array([f1_x, f1_y]), np.array([f2_x, f2_y])
        f1_distance = np.sum((self.point_data - f1) ** 2, axis=1)
        f2_distance = np.sum((self.point_data - f2) ** 2, axis=1)
        all_distance = np.sqrt(f1_distance) + np.sqrt(f2_distance)

        Z = np.abs(2 * LAxis - all_distance)
        delta = np.sqrt(np.mean((Z - np.mean(Z)) ** 2))
        inliers = np.where(Z < self.inlier_threshold_multiplier * delta)[0]
        inlier_points = self.point_data[inliers]

        return len(inlier_points), inlier_points

    def execute_ransac(self):
        n_pts = len(self.point_data)
        inliers_set = np.ascontiguousarray(np.zeros((0, 2), dtype=np.float32))
        if n_pts < self.N:
            return self.best_model, inliers_set

        budget = min(max(int(math.ceil(self.items)), 1), self._MAX_ITERATIONS)
        executed = 0
        while budget > 0 and executed < self._MAX_ITERATIONS:
            budget -= 1
            executed += 1

            select_points = self.random_sampling(self.N)
            pts_f = np.ascontiguousarray(select_points, dtype=np.float32)
            try:
                ellipse = cv2.fitEllipse(pts_f)
            except cv2.error:
                continue

            inliers_count, inliers_set = self.eval_model(ellipse)
            if inliers_count == 0:
                continue
            inliers_set = np.ascontiguousarray(inliers_set, dtype=np.float32)

            if inliers_count > self.count:
                self.count = inliers_count
                try:
                    self.best_model = cv2.fitEllipse(inliers_set)
                except cv2.error:
                    continue

                if self.count > self.max_inliers:
                    break

                w = inliers_count / n_pts
                if 0 < w < 1:
                    try:
                        denom = math.log(1 - w**self.N)
                        if denom < -1e-12:
                            extra = int(math.ceil(math.log(1 - self.P) / denom))
                            extra = max(0, min(extra, 2000))
                            budget = min(budget + extra, self._MAX_ITERATIONS - executed)
                    except (ValueError, ZeroDivisionError, OverflowError):
                        pass

        return self.best_model, inliers_set


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (e1, e2) spanning the plane orthogonal to unit normal."""
    n = normal / np.linalg.norm(normal)
    if abs(n[2]) < 0.9:
        aux = np.array([0.0, 0.0, 1.0])
    else:
        aux = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, aux)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2, n


def _ring_plane_axes(points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smallest PCA eigenvector ≈ ring-plane normal; largest spread in-plane."""
    X = points_xyz - points_xyz.mean(axis=0)
    cov = np.cov(X.T)
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(w)
    normal = v[:, idx[0]]
    return _plane_basis(normal)


def _project_to_plane_2d(points_xyz: np.ndarray, center: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    shifted = points_xyz - center
    x = np.dot(shifted, e1)
    y = np.dot(shifted, e2)
    return np.column_stack((x, y))


def _gravity_align_theta(
    theta: np.ndarray,
    z: np.ndarray,
    t_range: float,
    n_bins: int = 360,
) -> Tuple[np.ndarray, dict]:
    """Roll ``theta`` so the bin with minimum median ``z`` lands at ``theta=0``.

    Operates on the already-computed ``theta`` (arc length) array and the
    world-frame ``z`` of each point. Returns the shifted ``theta`` plus a
    metadata dict for logging.

    Direction is **not** flipped: that requires a per-tunnel reference
    z-profile which we do not yet have in v3 calibration. The bottom-shift
    alone is enough to collapse the absolute rotational phase difference
    between rings of the same tunnel; any residual direction inconsistency
    becomes a per-ring failure mode that BO can see (and the unanchored
    ablation will quantify against the anchored baseline).
    """
    if t_range <= 0 or theta.size == 0:
        return theta.copy(), {"theta_shift": 0.0, "theta_range": float(t_range), "bottom_bin_z": float("nan"), "n_bins": int(n_bins)}

    t_min = float(theta.min())
    bins = np.linspace(t_min, t_min + t_range, n_bins + 1)
    idx = np.clip(((theta - t_min) / t_range * n_bins).astype(np.int64), 0, n_bins - 1)
    z_by_bin = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            z_by_bin[b] = float(np.median(z[m]))
    finite = np.isfinite(z_by_bin)
    if not finite.any():
        return theta.copy(), {"theta_shift": 0.0, "theta_range": float(t_range), "bottom_bin_z": float("nan"), "n_bins": int(n_bins)}
    bot_bin = int(np.argmin(np.where(finite, z_by_bin, np.inf)))
    t_shift = float(bins[bot_bin])
    shifted = ((theta - t_shift) % t_range) + t_min
    meta = {
        "theta_shift": float(t_shift - t_min),
        "theta_range": float(t_range),
        "bottom_bin_z": float(z_by_bin[bot_bin]),
        "top_bin_z": float(np.max(z_by_bin[finite])),
        "n_bins": int(n_bins),
    }
    return shifted.astype(np.float64), meta


def unfold_single_ring(
    df: pd.DataFrame,
    tunnel_diameter: float,
    vertical_filter_window: float,
    ransac_threshold: float,
    ransac_probability: float,
    ransac_inlier_ratio: float,
    ransac_sample_size: int,
    ransac_initial_iterations: int,
    ransac_inlier_threshold_multiplier: float,
    gravity_anchor_enabled: bool = True,
    gravity_anchor_n_bins: int = 360,
    gravity_meta_out: Optional[dict] = None,
) -> Tuple[pd.DataFrame, int]:
    """Compute r, theta (arc length), h along ring normal; ring_count fixed to 1.

    With ``gravity_anchor_enabled=True`` (v3 default), the produced ``theta``
    is rolled so the world-frame physical bottom of the tunnel sits at
    ``theta = 0``. ``gravity_meta_out`` (if provided) is mutated in place
    with metadata describing the shift; pass ``None`` to discard.
    """
    pts = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    center = pts.mean(axis=0)
    e1, e2, normal = _ring_plane_axes(pts)

    pts2d = _project_to_plane_2d(pts, center, e1, e2)

    # Upper-band filter (same convention as r4tun: keep points near max "vertical" in 2D)
    y2 = pts2d[:, 1]
    y_max = float(y2.max())
    m = np.abs(y2 - y_max) <= float(vertical_filter_window)
    filtered = pts2d[m]
    if filtered.shape[0] < max(ransac_sample_size, 5):
        filtered = pts2d

    points_data = np.reshape(filtered, (-1, 2))
    ransac = RANSAC(
        data=points_data,
        threshold=ransac_threshold,
        P=ransac_probability,
        S=ransac_inlier_ratio,
        N=ransac_sample_size,
        initial_iterations=ransac_initial_iterations,
        inlier_threshold_multiplier=ransac_inlier_threshold_multiplier,
        seed=42,
    )
    _, inliers_set = ransac.execute_ransac()
    points_data = np.reshape(inliers_set, (-1, 2))
    ransac2 = RANSAC(
        data=points_data,
        threshold=ransac_threshold,
        P=ransac_probability,
        S=ransac_inlier_ratio,
        N=ransac_sample_size,
        initial_iterations=ransac_initial_iterations,
        inlier_threshold_multiplier=ransac_inlier_threshold_multiplier,
        seed=1042,
    )
    ellipse_params, _ = ransac2.execute_ransac()
    (cx, cy), _, _ = ellipse_params

    xp = pts2d[:, 0] - cx
    yp = pts2d[:, 1] - cy
    r = np.hypot(xp, yp)
    theta_deg = (np.degrees(np.arctan2(yp, xp)) + 90.0) % 360.0
    theta = theta_deg * (np.pi * float(tunnel_diameter) / 360.0)

    if gravity_anchor_enabled:
        z_world = pts[:, 2]
        t_range = float(np.pi * float(tunnel_diameter))
        theta_g, gmeta = _gravity_align_theta(
            theta=theta, z=z_world, t_range=t_range, n_bins=int(gravity_anchor_n_bins),
        )
        theta = theta_g
        gmeta["enabled"] = True
    else:
        gmeta = {"enabled": False}
    if gravity_meta_out is not None:
        gravity_meta_out.clear()
        gravity_meta_out.update(gmeta)

    # h: thickness along ring normal (tunnel-axis proxy for a thin ring)
    h_raw = np.dot(pts - center, normal)
    h = h_raw - h_raw.min()

    out = df.copy()
    out["r"] = r.astype(np.float64)
    out["theta"] = theta.astype(np.float64)
    out["h"] = h.astype(np.float64)
    return out, 1
