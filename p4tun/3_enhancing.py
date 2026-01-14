"""
Geometry-Guided Point Cloud Enhancement

This module enhances the denoised tunnel point cloud by:
1. Upsampling the surface via midpoint interpolation
2. Detecting and interpolating around outlier (boundary) points
3. Projecting to a depth map for subsequent processing

Algorithm Overview:
    1. Compute local curvature for all valid points
    2. Iteratively upsample by inserting midpoints between neighboring points
    3. Detect outlier points with significant local depth variation
    4. Interpolate new points around outliers to enhance boundaries
    5. Generate depth map with pixel-to-point mapping
"""

import os
import sys
import time
import json
import pickle
from typing import Tuple, List, Dict, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.spatial import KDTree, cKDTree
from scipy.interpolate import griddata
from tqdm.notebook import tqdm

# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(tunnel_id: str = None, base_dir: str = "data") -> Tuple[Dict[str, Any], bool]:
    """
    Load parameters from JSON file with fallback to defaults.
    
    Priority:
        1. Centralized: p4tun/parameters/<tunnel_id>/parameters_enhancing.json
        2. Tunnel-specific: data/<tunnel_id>/parameters_enhancing.json
        3. Default: p4tun/parameters_enhancing.json (if exists)
        4. Hardcoded defaults (if no file found)
    
    Returns:
        Tuple of (params_dict, was_loaded_from_file)
    """
    script_dir = os.path.dirname(__file__)
    param_file = "parameters_enhancing.json"
    
    if tunnel_id:
        params_path = os.path.join(script_dir, "parameters", tunnel_id, param_file)
        if os.path.exists(params_path):
            print(f"Loading parameters from {params_path}")
            with open(params_path, 'r') as f:
                return json.load(f), True
        
        tunnel_path = os.path.join(base_dir, tunnel_id, param_file)
        if os.path.exists(tunnel_path):
            print(f"Loading parameters from {tunnel_path}")
            with open(tunnel_path, 'r') as f:
                return json.load(f), True
    
    default_path = os.path.join(script_dir, param_file)
    if os.path.exists(default_path):
        print(f"Loading default parameters from {default_path}")
        with open(default_path, 'r') as f:
            return json.load(f), True
    
    print("Warning: No parameter file found, using hardcoded defaults")
    return {}, False


def get_param(params: Dict, *keys, default=None, allow_default: bool = True):
    """
    Get nested parameter value with optional default fallback.
    
    Args:
        params: Parameter dictionary
        keys: Nested keys to traverse
        default: Default value if key not found
        allow_default: If False, raise KeyError instead of using default
    
    Returns:
        Parameter value or default (if allow_default=True)
    """
    value = params
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            if allow_default:
                return default
            else:
                raise KeyError(f"Parameter not found: {' -> '.join(keys)}")
    return value


# =============================================================================
# Default Constants
# =============================================================================

# --- Physical Constants ---
DEFAULT_RING_SPACING = 1.2

# --- Curvature ---
DEFAULT_CURVATURE_NEIGHBORS = 20

# --- Upsampling ---
DEFAULT_UPSAMPLING_TARGET_DISTANCES = [0.08, 0.04, 0.02]
DEFAULT_CURVATURE_THRESHOLD = 0.0005
DEFAULT_UPSAMPLING_NEIGHBORS = 20
DEFAULT_DISTANCE_TOLERANCE_LOW = 0.9
DEFAULT_DISTANCE_TOLERANCE_HIGH = 2.0
DEFAULT_RADIUS_FILTER_FACTOR = 0.15
DEFAULT_MIN_NEW_POINT_DISTANCE_FACTOR = 0.2

# --- Outlier Detection ---
DEFAULT_DEPTH_THRESHOLD_LOW = 0.003
DEFAULT_DEPTH_THRESHOLD_HIGH = 0.008
DEFAULT_HIGH_DENSITY_RING_START = 0
DEFAULT_HIGH_DENSITY_RING_END = 5
DEFAULT_OUTLIER_NEIGHBORS = 20

# --- Outlier Interpolation ---
DEFAULT_INTERPOLATION_RADIUS = 0.06
DEFAULT_NUM_INTERPOLATIONS = 2
DEFAULT_DUPLICATE_THRESHOLD = 0.02
DEFAULT_MAX_OUTLIER_POINTS = 5000

# --- Depth Map ---
DEFAULT_DEPTH_MAP_RESOLUTION = 0.005
DEFAULT_INTERPOLATION_WINDOW = 9


# =============================================================================
# Curvature Computation
# =============================================================================

@njit(parallel=True)
def compute_curvatures_numba(
    points: np.ndarray,
    neighbor_indices: np.ndarray
) -> np.ndarray:
    """
    Compute surface curvature for each point using PCA of local neighborhood.
    
    Args:
        points: (N, 3) array of 3D points.
        neighbor_indices: (N, K) array of neighbor indices.
        
    Returns:
        Array of curvature values.
    """
    n_points = len(points)
    curvatures = np.zeros(n_points)
    
    for i in prange(n_points):
        # Get neighbor coordinates (excluding self)
        neighbors = points[neighbor_indices[i, 1:]]
        
        # Compute covariance matrix
        cov_matrix = np.cov(neighbors.T)
        
        # Curvature = smallest eigenvalue / sum of eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)
    
    return curvatures


def add_curvature_column(
    df: pd.DataFrame,
    k: int = DEFAULT_CURVATURE_NEIGHBORS
) -> pd.DataFrame:
    """
    Add curvature values to a DataFrame.
    
    Args:
        df: DataFrame with 'x', 'y', 'z' columns.
        k: Number of neighbors for curvature estimation.
        
    Returns:
        DataFrame with added 'curvature' column.
    """
    points = df[['x', 'y', 'z']].values
    tree = KDTree(points)
    _, indices = tree.query(points, k=k + 1)
    
    curvatures = compute_curvatures_numba(points, indices)
    
    df = df.copy()
    df['curvature'] = curvatures
    return df


# =============================================================================
# Surface Upsampling
# =============================================================================

@njit(parallel=False)
def compute_midpoints(
    points: np.ndarray,
    neighbor_indices: np.ndarray,
    distances: np.ndarray,
    target_distance: float,
    curvature_threshold: float
) -> np.ndarray:
    """
    Compute midpoints between neighboring point pairs.
    
    Points array format: [h, theta, r, curvature, intensity]
    
    Args:
        points: (N, 5) array of point data.
        neighbor_indices: (N, K) array of neighbor indices.
        distances: (N, K) array of distances to neighbors.
        target_distance: Target spacing between points.
        curvature_threshold: Maximum curvature difference for interpolation.
        
    Returns:
        Array of new midpoint coordinates.
    """
    n_points = len(points)
    max_new = n_points * (neighbor_indices.shape[1] - 1)
    new_points = np.zeros((max_new, 5), dtype=np.float64)
    count = 0
    
    dist_min = 0.9 * target_distance  # DEFAULT_DISTANCE_TOLERANCE_LOW
    dist_max = 2.0 * target_distance  # DEFAULT_DISTANCE_TOLERANCE_HIGH
    
    for i in prange(n_points):
        for j in range(1, neighbor_indices.shape[1]):
            dist = distances[i, j]
            idx = neighbor_indices[i, j]
            
            # Check distance and curvature criteria
            curvature_diff = abs(points[i, 3] - points[idx, 3])
            if dist_min <= dist <= dist_max and curvature_diff <= curvature_threshold:
                # Compute midpoint
                mid_h = (points[i, 0] + points[idx, 0]) / 2
                mid_theta = (points[i, 1] + points[idx, 1]) / 2
                mid_r = (points[i, 2] + points[idx, 2]) / 2
                mid_curv = (points[i, 3] + points[idx, 3]) / 2
                mid_intensity = (points[i, 4] + points[idx, 4]) / 2
                
                new_points[count] = np.array([mid_h, mid_theta, mid_r, mid_curv, mid_intensity])
                count += 1
    
    return new_points[:count]


@njit(parallel=False)
def filter_clustered_points(
    neighbors_array: np.ndarray,
    valid_mask: np.ndarray,
    num_points: int
) -> np.ndarray:
    """
    Filter out clustered points, keeping only one per cluster.
    
    Args:
        neighbors_array: (N, K) array of neighbor indices.
        valid_mask: (N, K) boolean mask of valid neighbors.
        num_points: Total number of points.
        
    Returns:
        Array of indices to keep.
    """
    keep_indices = np.zeros(num_points, dtype=np.int32)
    removed = np.zeros(num_points, dtype=np.int32)
    count = 0
    
    for i in prange(num_points):
        if removed[i] == 0:
            keep_indices[count] = i
            count += 1
            # Mark neighbors for removal
            for j in range(neighbors_array.shape[1]):
                neighbor_idx = neighbors_array[i, j]
                if valid_mask[i, j] and removed[neighbor_idx] == 0:
                    removed[neighbor_idx] = 1
    
    return keep_indices[:count]


def upsample_surface(
    df: pd.DataFrame,
    target_distance: float,
    curvature_threshold: float = DEFAULT_CURVATURE_THRESHOLD,
    num_neighbors: int = DEFAULT_UPSAMPLING_NEIGHBORS,
    min_new_point_distance_factor: float = DEFAULT_MIN_NEW_POINT_DISTANCE_FACTOR,
    radius_filter_factor: float = DEFAULT_RADIUS_FILTER_FACTOR
) -> pd.DataFrame:
    """
    Upsample surface by inserting midpoints between neighboring points.
    
    Args:
        df: DataFrame with columns ['h', 'theta', 'r', 'curvature', 'intensity'].
        target_distance: Target spacing between points.
        curvature_threshold: Maximum curvature difference for interpolation.
        num_neighbors: Number of neighbors to consider.
        
    Returns:
        DataFrame of new upsampled points.
    """
    start_time = time.time()
    
    print('Building spatial index...')
    points = df[['h', 'theta', 'r', 'curvature', 'intensity']].values
    coords_2d = points[:, :2]
    tree = cKDTree(coords_2d)
    
    distances, indices = tree.query(coords_2d, k=min(num_neighbors + 1, len(points)))
    
    print('Computing midpoints...')
    new_points = compute_midpoints(points, indices, distances, target_distance, curvature_threshold)
    
    print('Filtering excess points...')
    # Remove points too close to existing points
    if len(new_points) > 0:
        distances_to_existing, _ = tree.query(new_points[:, :2], k=1)
        min_dist = min_new_point_distance_factor * target_distance
        valid_new = new_points[distances_to_existing >= min_dist]
    else:
        valid_new = new_points
    
    # Create DataFrame
    new_df = pd.DataFrame(valid_new, columns=['h', 'theta', 'r', 'curvature', 'intensity'])
    new_df = new_df[(new_df != 0).any(axis=1)]
    new_df['pred'] = 8  # Mark as interpolated point
    
    # Remove clustered new points
    if len(new_df) > 0:
        new_coords = new_df[['h', 'theta']].values
        new_tree = cKDTree(new_coords)
        r_dist = radius_filter_factor * target_distance
        
        neighbors_list = new_tree.query_ball_point(new_coords, r=r_dist)
        max_neighbors = max(len(n) for n in neighbors_list)
        neighbors_array = np.full((len(new_coords), max_neighbors), -1, dtype=np.int32)
        valid_mask = np.zeros((len(new_coords), max_neighbors), dtype=np.bool_)
        
        for i, neighbors in enumerate(neighbors_list):
            neighbors_array[i, :len(neighbors)] = neighbors
            valid_mask[i, :len(neighbors)] = True
        
        keep_indices = filter_clustered_points(neighbors_array, valid_mask, len(new_coords))
        new_df = new_df.iloc[keep_indices].reset_index(drop=True)
    
    elapsed = time.time() - start_time
    print(f"Upsampling completed in {elapsed:.2f}s, added {len(new_df)} points (target: {target_distance})")
    
    return new_df


def progressive_upsample(
    df: pd.DataFrame,
    target_distances: List[float] = None,
    curvature_threshold: float = DEFAULT_CURVATURE_THRESHOLD,
    upsampling_neighbors: int = DEFAULT_UPSAMPLING_NEIGHBORS,
    min_new_point_distance_factor: float = DEFAULT_MIN_NEW_POINT_DISTANCE_FACTOR,
    radius_filter_factor: float = DEFAULT_RADIUS_FILTER_FACTOR
) -> pd.DataFrame:
    """
    Progressively upsample surface at multiple resolutions.
    
    Args:
        df: Input DataFrame with curvature.
        target_distances: List of target distances for each pass.
        curvature_threshold: Max curvature difference for midpoint insertion.
        upsampling_neighbors: Neighbors to consider for midpoint candidates.
        min_new_point_distance_factor: Minimum distance from existing points.
        radius_filter_factor: Factor for removing clustered new points.
        
    Returns:
        DataFrame with original and all upsampled points.
    """
    if target_distances is None:
        target_distances = DEFAULT_UPSAMPLING_TARGET_DISTANCES
    
    result_df = df.copy()
    
    for target_dist in target_distances:
        new_points = upsample_surface(
            result_df, target_distance=target_dist,
            curvature_threshold=curvature_threshold,
            num_neighbors=upsampling_neighbors,
            min_new_point_distance_factor=min_new_point_distance_factor,
            radius_filter_factor=radius_filter_factor
        )
        result_df = pd.concat([result_df, new_points], ignore_index=False)
    
    return result_df


# =============================================================================
# Outlier Detection and Enhancement
# =============================================================================

@njit(parallel=True)
def detect_outlier_points(
    points: np.ndarray,
    radial_values: np.ndarray,
    neighbor_indices: np.ndarray,
    depth_threshold_low: float,
    depth_threshold_high: float,
    h_min: float,
    high_density_start: float,
    high_density_end: float,
    ring_spacing: float
) -> np.ndarray:
    """
    Detect outlier points with significant local depth variation.
    
    Args:
        points: (N, 3) array of (h, theta, r) coordinates.
        radial_values: Radial (depth) values.
        neighbor_indices: (N, K) neighbor indices.
        depth_threshold_low: Threshold for low-density regions.
        depth_threshold_high: Threshold for high-density regions.
        h_min: Minimum h coordinate.
        high_density_start: Start of high-density region (ring units).
        high_density_end: End of high-density region (ring units).
        ring_spacing: Nominal ring spacing.
        
    Returns:
        Boolean mask of outlier points.
    """
    n_points = len(points)
    outlier_mask = np.zeros(n_points, dtype=np.bool_)
    
    for i in prange(n_points):
        neighbors = neighbor_indices[i, 1:]
        if len(neighbors) < 20:
            continue
        
        # Compute average depth difference
        neighbor_depths = radial_values[neighbors]
        avg_diff = points[i, 2] - np.mean(neighbor_depths)
        
        # Determine threshold based on region
        h_coord = points[i, 0]
        in_high_density = (h_min + ring_spacing * high_density_start <= h_coord <= 
                          h_min + ring_spacing * high_density_end)
        
        threshold = depth_threshold_high if in_high_density else depth_threshold_low
        
        if avg_diff > threshold:
            outlier_mask[i] = True
    
    return outlier_mask


@njit(parallel=False)
def interpolate_between_outliers(
    outlier_indices: np.ndarray,
    points: np.ndarray,
    inter_radius: float,
    num_interpolations: int,
    duplicate_threshold: float,
    resolution: float
) -> np.ndarray:
    """
    Interpolate new points between pairs of outlier points.
    
    Args:
        outlier_indices: Indices of outlier points.
        points: (N, 4) array of [h, theta, r, intensity].
        inter_radius: Maximum distance for interpolation.
        num_interpolations: Number of points per pair.
        duplicate_threshold: Minimum distance between new points.
        resolution: Minimum distance for interpolation (image resolution).
        
    Returns:
        Array of new interpolated points.
    """
    n_outliers = len(outlier_indices)
    max_new = n_outliers * n_outliers * num_interpolations
    new_points = np.zeros((max_new, 4))
    count = 0
    
    for i in prange(n_outliers):
        idx1 = outlier_indices[i]
        p1 = points[idx1]
        
        for j in range(i + 1, n_outliers):
            idx2 = outlier_indices[j]
            p2 = points[idx2]
            
            # Check distance
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if not (resolution < dist < inter_radius):
                continue
            
            # Generate interpolation points
            for k in range(1, num_interpolations + 1):
                t = k / (num_interpolations + 1)
                new_h = (1 - t) * p1[0] + t * p2[0]
                new_theta = (1 - t) * p1[1] + t * p2[1]
                new_r = (1 - t) * p1[2] + t * p2[2]
                new_intensity = (1 - t) * p1[3] + t * p2[3]
                
                # Check for duplicates
                is_duplicate = False
                if count > 0:
                    for m in range(count):
                        d = np.sqrt((new_points[m, 0] - new_h)**2 + 
                                   (new_points[m, 1] - new_theta)**2)
                        if d < duplicate_threshold:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    new_points[count] = np.array([new_h, new_theta, new_r, new_intensity])
                    count += 1
    
    return new_points[:count]


def enhance_outlier_boundaries(
    df: pd.DataFrame,
    depth_threshold_low: float = DEFAULT_DEPTH_THRESHOLD_LOW,
    depth_threshold_high: float = DEFAULT_DEPTH_THRESHOLD_HIGH,
    inter_radius: float = DEFAULT_INTERPOLATION_RADIUS,
    num_interpolations: int = DEFAULT_NUM_INTERPOLATIONS,
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    high_density_range: Tuple[int, int] = None,
    resolution: float = DEFAULT_DEPTH_MAP_RESOLUTION,
    outlier_neighbors: int = DEFAULT_OUTLIER_NEIGHBORS,
    max_outlier_points: int = DEFAULT_MAX_OUTLIER_POINTS,
    ring_spacing: float = DEFAULT_RING_SPACING
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outlier points and interpolate around them.
    
    Args:
        df: Input DataFrame.
        depth_threshold_low: Threshold for low-density regions.
        depth_threshold_high: Threshold for high-density regions.
        inter_radius: Maximum interpolation distance.
        num_interpolations: Points per pair.
        duplicate_threshold: Minimum new point spacing.
        high_density_range: Ring range for high-density region.
        resolution: Image resolution.
        outlier_neighbors: Neighbors for outlier detection.
        max_outlier_points: Maximum outliers to process.
        ring_spacing: Nominal ring spacing.
        
    Returns:
        Tuple of (outlier_df, new_points_df).
    """
    if high_density_range is None:
        high_density_range = (DEFAULT_HIGH_DENSITY_RING_START, DEFAULT_HIGH_DENSITY_RING_END)
    
    start_time = time.time()
    
    print('Building spatial index...')
    points = df[['h', 'theta', 'r', 'intensity']].values
    coords_2d = points[:, :2]
    tree = cKDTree(coords_2d)
    
    _, indices = tree.query(coords_2d, k=outlier_neighbors + 1)
    h_min = np.min(points[:, 0])
    
    print('Detecting outlier points...')
    outlier_mask = detect_outlier_points(
        points[:, :3], points[:, 2], indices,
        depth_threshold_low, depth_threshold_high,
        h_min, high_density_range[0], high_density_range[1],
        ring_spacing
    )
    
    outlier_indices = np.where(outlier_mask)[0]
    print(f"Found {len(outlier_indices)} outlier points")
    
    outlier_df = df.iloc[outlier_indices].copy()
    
    # Filter out high-density region for interpolation
    print("Filtering high-density region...")
    h_low = h_min + ring_spacing * high_density_range[0]
    h_high = h_min + ring_spacing * high_density_range[1]
    
    filtered_indices = []
    for idx in outlier_indices:
        h = points[idx, 0]
        if not (h_low <= h <= h_high):
            filtered_indices.append(idx)
    
    filtered_indices = np.array(filtered_indices, dtype=np.int64)
    
    # Limit for memory
    if len(filtered_indices) > max_outlier_points:
        print(f"Warning: Limiting to {max_outlier_points} outlier points")
        np.random.seed(42)
        filtered_indices = np.random.choice(filtered_indices, max_outlier_points, replace=False)
    
    print(f"Interpolating around {len(filtered_indices)} outlier points...")
    new_points = interpolate_between_outliers(
        filtered_indices, points,
        inter_radius, num_interpolations, duplicate_threshold, resolution
    )
    
    new_df = pd.DataFrame(new_points, columns=['h', 'theta', 'r', 'intensity'])
    new_df['pred'] = 8
    
    elapsed = time.time() - start_time
    print(f"Outlier enhancement completed in {elapsed:.2f}s, added {len(new_df)} points")
    
    return outlier_df, new_df


# =============================================================================
# Depth Map Generation
# =============================================================================

def generate_depth_map(
    data_surface: Dict,
    data_boundary: Dict,
    resolution: float = DEFAULT_DEPTH_MAP_RESOLUTION,
    window_size: int = DEFAULT_INTERPOLATION_WINDOW,
    record_mapping: bool = True
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Project point cloud data to a 2D depth map.
    
    Args:
        data_surface: Dict with keys 'index', 'x', 'y', 'z', 'pred'.
        data_boundary: Dict with keys 'x', 'y', 'z', 'pred'.
        resolution: Depth map resolution (meters/pixel).
        window_size: Window size for gap interpolation.
        record_mapping: Whether to record pixel-to-point mapping.
        
    Returns:
        Tuple of (depth_map, pixel_to_point_mapping).
    """
    # Convert to arrays
    def to_arrays(data):
        if isinstance(data, pd.DataFrame):
            return data[['x', 'y', 'z', 'pred']].values.T
        return np.array([data['x'], data['y'], data['z'], data['pred']])
    
    surface_index = data_surface.get('index')
    surface = to_arrays(data_surface)
    boundary = to_arrays(data_boundary)
    
    # Compute bounds
    x_min = min(surface[0].min(), boundary[0].min())
    x_max = max(surface[0].max(), boundary[0].max())
    y_min = min(surface[1].min(), boundary[1].min())
    y_max = max(surface[1].max(), boundary[1].max())
    
    # Grid dimensions
    height = int((y_max - y_min) / resolution)
    width = int((x_max - x_min) / resolution)
    print(f'Depth map dimensions: {height} x {width}')
    
    depth_map = np.full((height, width), np.nan, dtype=np.float32)
    
    def process_points(data, index, record=False):
        """Process points and update depth map."""
        grid_x = np.clip(((data[0] - x_min) / resolution).astype(int), 0, width - 1)
        grid_y = np.clip(((data[1] - y_min) / resolution).astype(int), 0, height - 1)
        
        pixel_values = defaultdict(list)
        mapping = []
        
        if index is None:
            index = range(len(data[0]))
        
        for idx, (gx, gy, z, pred) in zip(index, zip(grid_x, grid_y, data[2], data[3])):
            pixel_values[(gy, gx)].append(z)
            if record and pred != 8:
                mapping.append({'pixel_x': gx, 'pixel_y': gy, 'index': idx})
        
        for (gy, gx), z_vals in pixel_values.items():
            depth_map[gy, gx] = np.mean(z_vals)
        
        return mapping if record else None
    
    # Process both point sets
    with tqdm(total=2, desc="Projecting to depth map") as pbar:
        mapping = process_points(surface, surface_index, record=record_mapping)
        pbar.update(1)
        process_points(boundary, None, record=False)
        pbar.update(1)
    
    if record_mapping:
        print(f"Mapped {len(mapping)} points to pixels")
    
    # Interpolate gaps
    if window_size > 1:
        valid_points = []
        half_w = window_size // 2
        
        for i in tqdm(range(half_w, height - half_w), desc="Finding gaps to fill"):
            for j in range(half_w, width - half_w):
                if np.isnan(depth_map[i, j]):
                    window = depth_map[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1]
                    if np.any(~np.isnan(window)):
                        valid_points.append((i, j))
        
        if valid_points:
            interp_coords = np.array(valid_points)
            known_coords = np.argwhere(~np.isnan(depth_map))
            known_values = depth_map[~np.isnan(depth_map)]
            
            with tqdm(total=1, desc="Interpolating gaps") as pbar:
                interp_values = griddata(known_coords, known_values, interp_coords, method='nearest')
                pbar.update(1)
            
            depth_map[interp_coords[:, 0], interp_coords[:, 1]] = interp_values
    
    return depth_map, mapping if record_mapping else []


def save_depth_map_image(
    depth_map: np.ndarray,
    filepath: str,
    resolution: float = DEFAULT_DEPTH_MAP_RESOLUTION
) -> None:
    """
    Save depth map as an image with exact pixel dimensions.
    
    Args:
        depth_map: 2D depth map array.
        filepath: Output file path.
        resolution: Resolution used (for DPI calculation).
    """
    height, width = depth_map.shape
    dpi = 1.0 / resolution
    
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(depth_map, cmap='viridis')
    
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


# =============================================================================
# Main Pipeline
# =============================================================================

def enhance_point_cloud(tunnel_id: str, base_dir: str = "data") -> None:
    """
    Execute the complete enhancement pipeline.
    
    Args:
        tunnel_id: Tunnel identifier.
        base_dir: Base data directory.
    """
    print(f"Processing tunnel: {tunnel_id}")
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    # Load parameters
    params, params_loaded = load_parameters(tunnel_id, base_dir)
    
    # Extract parameters - use defaults ONLY if no file was loaded
    allow_defaults = not params_loaded
    ring_spacing = get_param(params, 'physical_constants', 'ring_spacing', default=DEFAULT_RING_SPACING, allow_default=allow_defaults)
    curvature_neighbors = get_param(params, 'curvature', 'curvature_neighbors', default=DEFAULT_CURVATURE_NEIGHBORS, allow_default=allow_defaults)
    target_distances = get_param(params, 'upsampling', 'target_distances', default=DEFAULT_UPSAMPLING_TARGET_DISTANCES, allow_default=allow_defaults)
    curvature_threshold = get_param(params, 'upsampling', 'curvature_threshold', default=DEFAULT_CURVATURE_THRESHOLD, allow_default=allow_defaults)
    upsampling_neighbors = get_param(params, 'upsampling', 'upsampling_neighbors', default=DEFAULT_UPSAMPLING_NEIGHBORS, allow_default=allow_defaults)
    min_new_point_distance_factor = get_param(params, 'upsampling', 'min_new_point_distance_factor', default=DEFAULT_MIN_NEW_POINT_DISTANCE_FACTOR, allow_default=allow_defaults)
    radius_filter_factor = get_param(params, 'upsampling', 'radius_filter_factor', default=DEFAULT_RADIUS_FILTER_FACTOR, allow_default=allow_defaults)
    depth_threshold_low = get_param(params, 'outlier_detection', 'depth_threshold_low', default=DEFAULT_DEPTH_THRESHOLD_LOW, allow_default=allow_defaults)
    depth_threshold_high = get_param(params, 'outlier_detection', 'depth_threshold_high', default=DEFAULT_DEPTH_THRESHOLD_HIGH, allow_default=allow_defaults)
    high_density_start = get_param(params, 'outlier_detection', 'high_density_ring_start', default=DEFAULT_HIGH_DENSITY_RING_START, allow_default=allow_defaults)
    high_density_end = get_param(params, 'outlier_detection', 'high_density_ring_end', default=DEFAULT_HIGH_DENSITY_RING_END, allow_default=allow_defaults)
    outlier_neighbors = get_param(params, 'outlier_detection', 'outlier_neighbors', default=DEFAULT_OUTLIER_NEIGHBORS, allow_default=allow_defaults)
    interpolation_radius = get_param(params, 'outlier_interpolation', 'interpolation_radius', default=DEFAULT_INTERPOLATION_RADIUS, allow_default=allow_defaults)
    num_interpolations = get_param(params, 'outlier_interpolation', 'num_interpolations', default=DEFAULT_NUM_INTERPOLATIONS, allow_default=allow_defaults)
    duplicate_threshold = get_param(params, 'outlier_interpolation', 'duplicate_threshold', default=DEFAULT_DUPLICATE_THRESHOLD, allow_default=allow_defaults)
    max_outlier_points = get_param(params, 'outlier_interpolation', 'max_outlier_points', default=DEFAULT_MAX_OUTLIER_POINTS, allow_default=allow_defaults)
    depth_map_resolution = get_param(params, 'depth_map', 'resolution', default=DEFAULT_DEPTH_MAP_RESOLUTION, allow_default=allow_defaults)
    interpolation_window = get_param(params, 'depth_map', 'interpolation_window', default=DEFAULT_INTERPOLATION_WINDOW, allow_default=allow_defaults)
    
    # Load denoised data
    df = pd.read_csv(os.path.join(tunnel_dir, "denoised.csv"))
    df_valid = df[df['pred'] != 0].copy()
    
    # Step 1: Compute curvature
    print("\n=== Step 1: Computing curvature ===")
    df_with_curvature = add_curvature_column(df_valid, k=curvature_neighbors)
    
    # Step 2: Progressive upsampling
    print("\n=== Step 2: Surface upsampling ===")
    df_upsampled = progressive_upsample(
        df_with_curvature,
        target_distances=target_distances,
        curvature_threshold=curvature_threshold,
        upsampling_neighbors=upsampling_neighbors,
        min_new_point_distance_factor=min_new_point_distance_factor,
        radius_filter_factor=radius_filter_factor
    )
    
    # Step 3: Outlier detection and enhancement
    print("\n=== Step 3: Boundary enhancement ===")
    outlier_df, boundary_points = enhance_outlier_boundaries(
        df_with_curvature,
        depth_threshold_low=depth_threshold_low,
        depth_threshold_high=depth_threshold_high,
        inter_radius=interpolation_radius,
        num_interpolations=num_interpolations,
        duplicate_threshold=duplicate_threshold,
        high_density_range=(high_density_start, high_density_end),
        resolution=depth_map_resolution,
        outlier_neighbors=outlier_neighbors,
        max_outlier_points=max_outlier_points,
        ring_spacing=ring_spacing
    )
    df_boundary = pd.concat([outlier_df, boundary_points], ignore_index=False)
    
    # Update original predictions for outlier points
    df.loc[outlier_df.index, 'pred'] = 0
    
    # Step 4: Generate depth maps
    print("\n=== Step 4: Generating depth maps ===")
    
    surface_data = {
        'index': df_upsampled.index,
        'x': df_upsampled['h'],
        'y': df_upsampled['theta'],
        'z': df_upsampled['r'],
        'pred': df_upsampled['pred']
    }
    
    boundary_data = {
        'x': df_boundary['h'],
        'y': df_boundary['theta'],
        'z': df_boundary['r'],
        'pred': df_boundary['pred']
    }
    
    depth_map, pixel_mapping = generate_depth_map(
        surface_data, boundary_data,
        resolution=depth_map_resolution, window_size=interpolation_window
    )
    
    # Save pixel mapping
    with open(os.path.join(tunnel_dir, "pixel_to_point.pkl"), 'wb') as f:
        pickle.dump(pixel_mapping, f)
    
    # Save depth map image
    save_depth_map_image(depth_map, os.path.join(tunnel_dir, "depth_map.png"), resolution=depth_map_resolution)
    
    # Generate outlier-only depth map
    outlier_data = {
        'x': df_boundary['h'],
        'y': df_boundary['theta'],
        'z': df_boundary['r'],
        'pred': df_boundary['pred'],
        'intensity': df_boundary.get('intensity', pd.Series([0] * len(df_boundary)))
    }
    
    depth_map_outlier, _ = generate_depth_map(
        surface_data, outlier_data,
        resolution=depth_map_resolution, window_size=1, record_mapping=False
    )
    np.save(os.path.join(tunnel_dir, "depth_map_outlier.npy"), depth_map_outlier)
    
    # Step 5: Merge enhanced points and save
    print("\n=== Step 5: Saving results ===")
    
    new_surface_points = df_upsampled[df_upsampled['pred'] == 8].copy()
    new_boundary_points = df_boundary[df_boundary['pred'] == 8].copy()
    
    # Ensure all columns exist
    for col in df.columns:
        if col not in new_surface_points.columns:
            new_surface_points[col] = np.nan if col in ['x', 'y', 'z'] else None
        if col not in new_boundary_points.columns:
            new_boundary_points[col] = np.nan if col in ['x', 'y', 'z'] else None
    
    all_new = pd.concat([new_surface_points, new_boundary_points], ignore_index=True)
    df_enhanced = pd.concat([df, all_new], ignore_index=True)
    
    df_enhanced.to_csv(os.path.join(tunnel_dir, "enhanced.csv"), index=False)
    
    print(f"Added {len(all_new)} new enhanced points")
    print(f"Total points in enhanced.csv: {len(df_enhanced)}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 3_enhancing_clean.py <tunnel_id>")
        print("Example: python 3_enhancing_clean.py 1-4")
        sys.exit(1)
    
    enhance_point_cloud(sys.argv[1])

