"""
Raw characteristic extractor for bo4tun preprocessing.

Produces accurate cross_section_radius_m, median_nn_distance_m, and density_cv
from the raw point cloud so analyst/coder get correct radius_min/max and other
parameters. Uses PCA + circle fit in the cross-section plane for radius (no
percentile guesswork). Output schema matches knowledge/raw.md.

Usage:
  python extract_characteristics.py 1-4 [--data-dir data] [--output path]
  python extract_characteristics.py 1-4 --output parameters/1-4/characteristics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------

def load_point_cloud(filepath: str) -> np.ndarray:
    """Load x,y,z from a text file. Shape (N, 3), float64."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Need at least 3 columns (x,y,z), got {data.shape[1]}")
    return data[:, :3].astype(np.float64)


def subsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Subsample to max_points for speed; no-op if already smaller."""
    n = len(points)
    if n <= max_points:
        return points
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx]


# -----------------------------------------------------------------------------
# Tunnel axis (PCA)
# -----------------------------------------------------------------------------

def fit_tunnel_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Principal axis of the point cloud (tunnel direction).
    Returns (center, unit_axis). Axis is the eigenvector of largest eigenvalue.
    """
    center = np.mean(points, axis=0)
    centered = points - center
    cov = (centered.T @ centered) / (len(centered) - 1) if len(centered) > 1 else np.zeros((3, 3))
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    axis = eigenvectors[:, np.argmax(eigenvalues)]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return center, axis


def project_to_cross_section_plane(
    points: np.ndarray, center: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    """
    Project 3D points onto the plane perpendicular to axis through center.
    Returns (N, 2) array: 2D coordinates in the cross-section plane.
    """
    d = points - center
    # Along-axis component
    along = (d @ axis).reshape(-1, 1) * axis
    # In-plane component
    in_plane = d - along
    # Build an orthonormal basis in the plane (two vectors perpendicular to axis)
    if abs(axis[0]) < 0.9:
        u = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    else:
        u = np.cross(axis, np.array([0.0, 1.0, 0.0]))
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(axis, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    # Coordinates in plane
    u_coord = in_plane @ u
    v_coord = in_plane @ v
    return np.column_stack([u_coord, v_coord])


# -----------------------------------------------------------------------------
# Circle fit (cross-section radius)
# -----------------------------------------------------------------------------

def fit_circle_2d(points_2d: np.ndarray) -> tuple[float, float, float]:
    """
    Least-squares circle fit to 2D points: (x - cx)^2 + (y - cy)^2 = R^2.
    Uses algebraic (Kása) fit: minimize sum (x^2 + y^2 - 2*cx*x - 2*cy*y + c).
    Returns (cx, cy, R). R is non-negative.
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    n = len(x)
    if n < 3:
        return float(np.mean(x)), float(np.mean(y)), 0.0

    # Build linear system: 2*cx*x + 2*cy*y + c = x^2 + y^2  with c = R^2 - cx^2 - cy^2
    # So R^2 = c + cx^2 + cy^2 => R = sqrt(c + cx^2 + cy^2)
    A = np.column_stack([2 * x, 2 * y, np.ones(n)])
    b = x * x + y * y
    # Least squares
    cx2, cy2, c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx = float(cx2)
    cy = float(cy2)
    c = float(c)
    r_sq = c + cx * cx + cy * cy
    R = np.sqrt(max(0.0, r_sq))
    return cx, cy, float(R)


def extract_cross_section_radius(
    points: np.ndarray, 
    slice_thickness: float = 0.5,
    min_points_per_slice: int = 50,
) -> float:
    """
    Tunnel radius from geometry using slice-and-fit method (robust to tunnel curvature).
    
    Method:
    1. PCA to find tunnel axis
    2. Divide points into thin slices along the axis
    3. For each slice, project to local cross-section and fit circle
    4. Return median of per-slice radii
    
    This avoids the bias from projecting all points onto a single plane, which
    inflates radius for curved tunnels.
    
    Args:
        points: (N, 3) point cloud
        slice_thickness: Thickness of each slice in meters (default 0.5)
        min_points_per_slice: Minimum points required to fit a circle (default 50)
    
    Returns:
        Median radius across all valid slices
    """
    center, axis = fit_tunnel_axis(points)
    
    # Project points onto axis to get along-axis positions
    d = points - center
    along_axis = d @ axis  # (N,) array of signed distances along axis
    
    # Create slices
    min_pos = float(np.min(along_axis))
    max_pos = float(np.max(along_axis))
    num_slices = max(1, int(np.ceil((max_pos - min_pos) / slice_thickness)))
    
    slice_radii = []
    
    for i in range(num_slices):
        slice_start = min_pos + i * slice_thickness
        slice_end = min_pos + (i + 1) * slice_thickness
        
        # Find points in this slice
        mask = (along_axis >= slice_start) & (along_axis < slice_end)
        if i == num_slices - 1:  # Include last point in final slice
            mask |= (along_axis == max_pos)
        
        slice_points = points[mask]
        
        if len(slice_points) < min_points_per_slice:
            continue
        
        # Project slice points to local cross-section plane
        # Use slice center as the plane origin
        slice_center = np.mean(slice_points, axis=0)
        plane_2d = project_to_cross_section_plane(slice_points, slice_center, axis)
        
        # Fit circle to this slice
        cx, cy, R = fit_circle_2d(plane_2d)
        
        if R > 0.1:  # Filter out degenerate fits (radius too small)
            slice_radii.append(R)
    
    if len(slice_radii) == 0:
        # Fallback: use old method if no valid slices
        plane_2d = project_to_cross_section_plane(points, center, axis)
        cx, cy, R = fit_circle_2d(plane_2d)
        return float(R)
    
    # Return median radius (robust to outlier slices)
    return float(np.median(slice_radii))


# -----------------------------------------------------------------------------
# Point spacing (median k-NN distance)
# -----------------------------------------------------------------------------

def extract_median_nn_distance(points: np.ndarray, k: int = 5) -> float:
    """
    Median nearest-neighbor distance (k=5). Used for depth_map_resolution and target_distances.
    """
    n = len(points)
    if n < k + 1:
        return np.nan
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1, workers=-1)
    nn_dists = dists[:, 1:]
    per_point_median = np.median(nn_dists, axis=1)
    return float(np.median(per_point_median))


# -----------------------------------------------------------------------------
# Density variation (CV)
# -----------------------------------------------------------------------------

def extract_density_cv(
    points: np.ndarray,
    k: int = 20,
    sample_size: int = 5000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Coefficient of variation of local density (1 / mean_k_distance). Used for gradient_threshold.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(points)
    if n < k + 1:
        return np.nan
    sample_n = min(sample_size, n)
    idx = rng.choice(n, size=sample_n, replace=False)
    sample = points[idx]
    tree = cKDTree(points)
    dists, _ = tree.query(sample, k=k + 1, workers=-1)
    mean_r = np.mean(dists[:, 1:], axis=1)
    density_proxy = 1.0 / (mean_r + 1e-12)
    mean_d = np.mean(density_proxy)
    std_d = np.std(density_proxy)
    if mean_d <= 0:
        return 0.0
    return float(std_d / mean_d)


# -----------------------------------------------------------------------------
# Main extraction
# -----------------------------------------------------------------------------

def extract_raw_characteristics(
    filepath: str,
    max_points: int = 200_000,
    rng_seed: int = 42,
    slice_thickness: float = 0.5,
) -> dict:
    """
    Extract cross_section_radius_m, median_nn_distance_m, density_cv from geometry only.
    Uses slice-and-fit method for robust radius estimation.
    """
    rng = np.random.default_rng(rng_seed)
    points = load_point_cloud(filepath)
    points_sub = subsample(points, max_points, rng)

    radius = extract_cross_section_radius(points_sub, slice_thickness=slice_thickness)
    nn_dist = extract_median_nn_distance(points_sub, k=5)
    density_cv = extract_density_cv(points_sub, k=20, sample_size=5000, rng=rng)

    return {
        "cross_section_radius_m": radius,
        "median_nn_distance_m": nn_dist,
        "density_cv": density_cv,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract raw point cloud characteristics for preprocessing (bo4tun)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 1-4, 2-2)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    parser.add_argument("--max-points", type=int, default=200_000, help="Max points for computation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--slice-thickness",
        type=float,
        default=0.5,
        help="Thickness of slices for radius estimation in meters (default 0.5)",
    )
    args = parser.parse_args()

    filepath = os.path.join(args.data_dir, f"{args.tunnel_id}.txt")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found", file=sys.stderr)
        sys.exit(1)

    chars = extract_raw_characteristics(
        filepath,
        max_points=args.max_points,
        rng_seed=args.seed,
        slice_thickness=args.slice_thickness,
    )

    out_path = args.output or os.path.join(args.data_dir, args.tunnel_id, "raw_characteristics.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(chars, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  cross_section_radius_m: {chars['cross_section_radius_m']:.4f}  → radius_min, radius_max")
    print(f"  median_nn_distance_m:   {chars['median_nn_distance_m']:.6f}  → depth_map_resolution, target_distances")
    print(f"  density_cv:            {chars['density_cv']:.4f}  → gradient_threshold")


if __name__ == "__main__":
    main()
