"""
Extract Raw Point Cloud Characteristics for Preprocessing Parameter Adaptation

Extracts ONLY characteristics that differentiate tunnels and inform
preprocessing parameter choices.

Critical Characteristics:
    1. cross_section_radius_m  → radius_min, radius_max
    2. median_nn_distance_m    → depth_map_resolution, target_distances
    3. density_cv              → gradient_threshold, curvature_neighbors

Usage:
    python extract_raw_characteristics.py 1-4 [--data-dir data] [--output ...]
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree


# =============================================================================
# Loading
# =============================================================================

def load_point_cloud(filepath: str) -> np.ndarray:
    """Load point cloud from text file. Returns Nx3 array (x, y, z)."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Point cloud must have at least 3 columns (x,y,z)")
    return data[:, :3].astype(np.float64)


def subsample(points: np.ndarray, max_points: int, rng) -> np.ndarray:
    """Subsample to at most max_points for faster computation."""
    n = len(points)
    if n <= max_points:
        return points
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx]


# =============================================================================
# Characteristic Extraction
# =============================================================================

def _extract_cross_section_radius(points: np.ndarray) -> float:
    """
    Extract cross-section radius (median distance from principal axis).
    
    Used for: radius_min, radius_max parameter bounds.
    """
    # PCA to find principal axis (tunnel direction)
    center = np.mean(points, axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    axis = eigenvectors[:, np.argmax(eigenvalues)]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    
    # Distance from axis (radial distance)
    d = points - center
    along = (d @ axis).reshape(-1, 1) * axis
    radial = d - along
    radius = np.linalg.norm(radial, axis=1)
    
    return float(np.median(radius))


def _extract_median_nn_distance(points: np.ndarray, k: int = 5) -> float:
    """
    Extract median nearest-neighbor distance.
    
    Used for: depth_map_resolution, target_distances.
    """
    n = len(points)
    if n < k + 1:
        return np.nan
    
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1, workers=-1)
    nn_dists = dists[:, 1:]  # Exclude self
    per_point_median = np.median(nn_dists, axis=1)
    
    return float(np.median(per_point_median))


def _extract_density_cv(points: np.ndarray, k: int = 20, sample_size: int = 5000, rng=None) -> float:
    """
    Extract coefficient of variation of local density.
    
    Higher CV = more variable density → need lower gradient_threshold.
    Used for: gradient_threshold, curvature_neighbors.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(points)
    if n < k + 1:
        return np.nan
    
    # Sample for speed
    sample_n = min(sample_size, n)
    idx = rng.choice(n, size=sample_n, replace=False)
    sample = points[idx]
    
    tree = cKDTree(points)
    dists, _ = tree.query(sample, k=k + 1, workers=-1)
    mean_r = np.mean(dists[:, 1:], axis=1)
    
    # Density proxy = 1 / mean_r
    density_proxy = 1.0 / (mean_r + 1e-12)
    cv = float(np.std(density_proxy) / (np.mean(density_proxy) + 1e-12))
    
    return cv


# =============================================================================
# Main Extraction
# =============================================================================

def extract_raw_characteristics(
    filepath: str,
    max_points: int = 200_000,
    rng_seed: int = 42,
) -> dict:
    """
    Extract critical raw point cloud characteristics.
    
    Returns dict with:
        - cross_section_radius_m: Median tunnel radius (for radius_min/max)
        - median_nn_distance_m: Point spacing (for resolution)
        - density_cv: Density variation (for gradient_threshold)
    """
    rng = np.random.default_rng(rng_seed)
    points = load_point_cloud(filepath)
    
    # Subsample for speed
    points_sub = subsample(points, max_points, rng)
    
    return {
        "cross_section_radius_m": _extract_cross_section_radius(points_sub),
        "median_nn_distance_m": _extract_median_nn_distance(points_sub, k=5),
        "density_cv": _extract_density_cv(points_sub, k=20, sample_size=5000, rng=rng),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw point cloud characteristics for preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 1-4, 2-2)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    parser.add_argument("--max-points", type=int, default=200_000, help="Max points for computation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    filepath = os.path.join(args.data_dir, f"{args.tunnel_id}.txt")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting raw characteristics from {filepath} ...")
    chars = extract_raw_characteristics(filepath, args.max_points, args.seed)

    out_path = args.output or os.path.join(args.data_dir, args.tunnel_id, "raw_characteristics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(chars, f, indent=2)

    print(f"Wrote {out_path}\n")
    print("Characteristics → Parameter Mapping:")
    print(f"  cross_section_radius_m: {chars['cross_section_radius_m']:.3f}  → radius_min, radius_max")
    print(f"  median_nn_distance_m:   {chars['median_nn_distance_m']:.4f}  → depth_map_resolution")
    print(f"  density_cv:             {chars['density_cv']:.3f}  → gradient_threshold")


if __name__ == "__main__":
    main()
