"""
Detection Pipeline: Enhancing + Line Detection and K-Position Calculation

This module enhances point clouds, detects oblique lines, finds intersections, and 
calculates K-block positions with **only critical parameters** exposed for Bayesian Optimization.

Based on P4TUN optimization reports:
- Detection provided +6.3% mIoU improvement - the LARGEST single-stage gain
- binary_threshold, hough_oblique_threshold, angle parameters are HIGH sensitivity
- Post-detection tweaks cannot compensate for poor detection

Critical Parameters:
- Enhancing: target_distances, curvature_neighbors, depth_map_resolution, interpolation_window
- Detection: binary_threshold, hough_oblique_threshold, angle parameters, hough_vertical_threshold
- Physical constants: k_height_mm, ab_height_mm (tunnel-specific)

All non-critical parameters use fixed defaults that performed well across tunnels.
"""

import os
import sys
import json
import time
import pickle
from typing import Tuple, List, Dict, Optional, Any
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.spatial import KDTree, cKDTree
from scipy.interpolate import griddata
from scipy.cluster.vq import kmeans2
from tqdm.notebook import tqdm


# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(tunnel_id: str = None, base_dir: str = "data") -> Tuple[Dict, bool]:
    """
    Load parameters from parameters_detection.json.
    
    Priority:
        1. bo4tun_agents/simple_staggered/2_detection/parameters/<tunnel_id>/parameters_detection.json
        2. data/<tunnel_id>/parameters_detection.json
        3. bo4tun_agents/simple_staggered/2_detection/parameters/sample/parameters_detection.json
        4. Hardcoded defaults
    """
    script_dir = os.path.dirname(__file__)
    param_file = "parameters_detection.json"
    
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
    
    sample_path = os.path.join(script_dir, "parameters", "sample", param_file)
    if os.path.exists(sample_path):
        print(f"Loading sample parameters from {sample_path}")
        with open(sample_path, 'r') as f:
            return json.load(f), True
    
    print("Warning: No parameter file found, using hardcoded defaults")
    return {}, False


def get_param(params: Dict, key: str, default=None):
    """Get parameter value with default fallback."""
    return params.get(key, default)


# =============================================================================
# CRITICAL PARAMETERS (tunable via JSON)
# =============================================================================

# Preprocessing - HIGH sensitivity
DEFAULT_BINARY_THRESHOLD = 127

# Hough Oblique - HIGH sensitivity
DEFAULT_HOUGH_OBLIQUE_THRESHOLD = 50
DEFAULT_ANGLE_POSITIVE_MIN = 6.0
DEFAULT_ANGLE_POSITIVE_MAX = 9.0
DEFAULT_ANGLE_NEGATIVE_MIN = -9.0
DEFAULT_ANGLE_NEGATIVE_MAX = -6.0

# Hough Vertical - MEDIUM-HIGH sensitivity
DEFAULT_HOUGH_VERTICAL_THRESHOLD = 500

# Enhancing - Critical for downstream detection
DEFAULT_TARGET_DISTANCES = [0.08, 0.04, 0.02]
DEFAULT_CURVATURE_NEIGHBORS = 20
DEFAULT_DEPTH_MAP_RESOLUTION = 0.005
DEFAULT_INTERPOLATION_WINDOW = 9

# Enhancing fixed parameters
FIXED_CURVATURE_THRESHOLD = 0.0005
FIXED_UPSAMPLING_NEIGHBORS = 20
FIXED_DISTANCE_TOLERANCE_LOW = 0.9
FIXED_DISTANCE_TOLERANCE_HIGH = 2.0
FIXED_RADIUS_FILTER_FACTOR = 0.15
FIXED_MIN_NEW_POINT_DISTANCE_FACTOR = 0.2
FIXED_DEPTH_THRESHOLD_LOW = 0.003
FIXED_DEPTH_THRESHOLD_HIGH = 0.008
FIXED_HIGH_DENSITY_RING_START = 0
FIXED_HIGH_DENSITY_RING_END = 5
FIXED_OUTLIER_NEIGHBORS = 20
FIXED_INTERPOLATION_RADIUS = 0.06
FIXED_NUM_INTERPOLATIONS = 2
FIXED_DUPLICATE_THRESHOLD = 0.02
FIXED_MAX_OUTLIER_POINTS = 5000

# Physical Constants - READ FROM PREPROCESSING STAGE
# k_height_mm = π * tunnel_diameter * 1000 / 16 (for 6-segment ring)
# ab_height_mm = 3 * k_height_mm
# tunnel_diameter and ring_spacing come from preprocessing params

PREPROCESSING_PARAMS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "1_preprocessing", "parameters"
)


def load_preprocessing_params(tunnel_id: str) -> Dict:
    """
    Load preprocessing parameters for the tunnel.
    
    Physical constants (tunnel_diameter, ring_spacing) are defined
    in preprocessing and must be read from there - not duplicated.
    """
    # Try tunnel-specific first, then sample
    for subdir in [tunnel_id, "sample"]:
        params_path = os.path.join(PREPROCESSING_PARAMS_DIR, subdir, "parameters_preprocessing.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                return json.load(f)
    return {}


def calculate_segment_heights(tunnel_diameter: float) -> Tuple[float, float]:
    """
    Calculate K-block and AB-block heights from tunnel diameter.
    
    For 6-segment simple staggered pattern:
    - K-block spans 1/16 of circumference
    - AB-block spans 3/16 of circumference (3x K-block)
    
    Args:
        tunnel_diameter: Tunnel diameter in meters
        
    Returns:
        (k_height_mm, ab_height_mm)
    """
    circumference_mm = np.pi * tunnel_diameter * 1000
    k_height_mm = circumference_mm / 16
    ab_height_mm = 3 * k_height_mm
    return k_height_mm, ab_height_mm


# =============================================================================
# ENHANCING FUNCTIONS (moved from preprocessing)
# =============================================================================

@njit(parallel=True)
def compute_curvatures_numba(
    points: np.ndarray,
    neighbor_indices: np.ndarray
) -> np.ndarray:
    """Compute surface curvature for each point using PCA of local neighborhood."""
    n_points = len(points)
    curvatures = np.zeros(n_points)
    
    for i in prange(n_points):
        neighbors = points[neighbor_indices[i, 1:]]
        cov_matrix = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)
    
    return curvatures


def add_curvature_column(df: pd.DataFrame, curvature_neighbors: int) -> pd.DataFrame:
    """Add curvature values to DataFrame."""
    points = df[['x', 'y', 'z']].values
    tree = KDTree(points)
    _, indices = tree.query(points, k=curvature_neighbors + 1)
    
    curvatures = compute_curvatures_numba(points, indices)
    
    df = df.copy()
    df['curvature'] = curvatures
    return df


@njit(parallel=False)
def compute_midpoints(
    points: np.ndarray,
    neighbor_indices: np.ndarray,
    distances: np.ndarray,
    target_distance: float
) -> np.ndarray:
    """Compute midpoints between neighboring point pairs."""
    n_points = len(points)
    max_new = n_points * (neighbor_indices.shape[1] - 1)
    new_points = np.zeros((max_new, 5), dtype=np.float64)
    count = 0
    
    dist_min = FIXED_DISTANCE_TOLERANCE_LOW * target_distance
    dist_max = FIXED_DISTANCE_TOLERANCE_HIGH * target_distance
    
    for i in range(n_points):
        for j in range(1, neighbor_indices.shape[1]):
            dist = distances[i, j]
            idx = neighbor_indices[i, j]

            curvature_diff = abs(points[i, 3] - points[idx, 3])
            if dist_min <= dist <= dist_max and curvature_diff <= FIXED_CURVATURE_THRESHOLD:
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
    """Filter out clustered points, keeping only one per cluster."""
    keep_indices = np.zeros(num_points, dtype=np.int32)
    removed = np.zeros(num_points, dtype=np.int32)
    count = 0
    
    for i in range(num_points):
        if removed[i] == 0:
            keep_indices[count] = i
            count += 1
            for j in range(neighbors_array.shape[1]):
                neighbor_idx = neighbors_array[i, j]
                if valid_mask[i, j] and removed[neighbor_idx] == 0:
                    removed[neighbor_idx] = 1

    return keep_indices[:count]


def upsample_surface(df: pd.DataFrame, target_distance: float) -> pd.DataFrame:
    """Upsample surface by inserting midpoints between neighboring points."""
    start_time = time.time()
    
    print('Building spatial index...')
    points = df[['h', 'theta', 'r', 'curvature', 'intensity']].values
    coords_2d = points[:, :2]
    tree = cKDTree(coords_2d)
    
    distances, indices = tree.query(coords_2d, k=min(FIXED_UPSAMPLING_NEIGHBORS + 1, len(points)))
    
    print('Computing midpoints...')
    new_points = compute_midpoints(points, indices, distances, target_distance)
    
    print('Filtering excess points...')
    if len(new_points) > 0:
        distances_to_existing, _ = tree.query(new_points[:, :2], k=1)
        min_dist = FIXED_MIN_NEW_POINT_DISTANCE_FACTOR * target_distance
        valid_new = new_points[distances_to_existing >= min_dist]
    else:
        valid_new = new_points
    
    new_df = pd.DataFrame(valid_new, columns=['h', 'theta', 'r', 'curvature', 'intensity'])
    new_df = new_df[(new_df != 0).any(axis=1)]
    new_df['pred'] = 8
    
    if len(new_df) > 0:
        new_coords = new_df[['h', 'theta']].values
        new_tree = cKDTree(new_coords)
        r_dist = FIXED_RADIUS_FILTER_FACTOR * target_distance
        
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


def progressive_upsample(df: pd.DataFrame, target_distances: List[float]) -> pd.DataFrame:
    """
    Progressively upsample surface at multiple resolutions.
    
    CRITICAL PARAMETER:
    - target_distances: Controls upsampling density (HIGH impact)
    """
    result_df = df.copy()
    
    for target_dist in target_distances:
        new_points = upsample_surface(result_df, target_distance=target_dist)
        result_df = pd.concat([result_df, new_points], ignore_index=False)
    
    return result_df


@njit(parallel=True)
def detect_outlier_points(
    points: np.ndarray,
    radial_values: np.ndarray,
    neighbor_indices: np.ndarray,
    h_min: float,
    ring_spacing: float
) -> np.ndarray:
    """Detect outlier points with significant local depth variation."""
    n_points = len(points)
    outlier_mask = np.zeros(n_points, dtype=np.bool_)

    for i in prange(n_points):
        neighbors = neighbor_indices[i, 1:]
        if len(neighbors) < FIXED_OUTLIER_NEIGHBORS:
            continue
        
        neighbor_depths = radial_values[neighbors]
        avg_diff = points[i, 2] - np.mean(neighbor_depths)
        
        h_coord = points[i, 0]
        in_high_density = (h_min + ring_spacing * FIXED_HIGH_DENSITY_RING_START <= h_coord <= 
                          h_min + ring_spacing * FIXED_HIGH_DENSITY_RING_END)
        
        threshold = FIXED_DEPTH_THRESHOLD_HIGH if in_high_density else FIXED_DEPTH_THRESHOLD_LOW
        
        if avg_diff > threshold:
            outlier_mask[i] = True
    
    return outlier_mask


@njit(parallel=False)
def interpolate_between_outliers(
    outlier_indices: np.ndarray,
    points: np.ndarray,
    resolution: float
) -> np.ndarray:
    """Interpolate new points between pairs of outlier points."""
    n_outliers = len(outlier_indices)
    max_new = n_outliers * n_outliers * FIXED_NUM_INTERPOLATIONS
    new_points = np.zeros((max_new, 4))
    count = 0
    
    for i in range(n_outliers):
        idx1 = outlier_indices[i]
        p1 = points[idx1]

        for j in range(i + 1, n_outliers):
            idx2 = outlier_indices[j]
            p2 = points[idx2]
            
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if not (resolution < dist < FIXED_INTERPOLATION_RADIUS):
                continue
            
            for k in range(1, FIXED_NUM_INTERPOLATIONS + 1):
                t = k / (FIXED_NUM_INTERPOLATIONS + 1)
                new_h = (1 - t) * p1[0] + t * p2[0]
                new_theta = (1 - t) * p1[1] + t * p2[1]
                new_r = (1 - t) * p1[2] + t * p2[2]
                new_intensity = (1 - t) * p1[3] + t * p2[3]
                
                is_duplicate = False
                if count > 0:
                    for m in range(count):
                        d = np.sqrt((new_points[m, 0] - new_h)**2 + 
                                   (new_points[m, 1] - new_theta)**2)
                        if d < FIXED_DUPLICATE_THRESHOLD:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    new_points[count] = np.array([new_h, new_theta, new_r, new_intensity])
                    count += 1
    
    return new_points[:count]


def enhance_outlier_boundaries(
    df: pd.DataFrame,
    depth_map_resolution: float,
    ring_spacing: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect outlier points and interpolate around them."""
    start_time = time.time()
    
    print('Building spatial index...')
    points = df[['h', 'theta', 'r', 'intensity']].values
    coords_2d = points[:, :2]
    tree = cKDTree(coords_2d)
    
    _, indices = tree.query(coords_2d, k=FIXED_OUTLIER_NEIGHBORS + 1)
    h_min = np.min(points[:, 0])
    
    print('Detecting outlier points...')
    outlier_mask = detect_outlier_points(
        points[:, :3], points[:, 2], indices, h_min, ring_spacing
    )
    
    outlier_indices = np.where(outlier_mask)[0]
    print(f"Found {len(outlier_indices)} outlier points")
    
    outlier_df = df.iloc[outlier_indices].copy()
    
    # Filter out high-density region for interpolation
    print("Filtering high-density region...")
    h_low = h_min + ring_spacing * FIXED_HIGH_DENSITY_RING_START
    h_high = h_min + ring_spacing * FIXED_HIGH_DENSITY_RING_END
    
    filtered_indices = []
    for idx in outlier_indices:
        h = points[idx, 0]
        if not (h_low <= h <= h_high):
            filtered_indices.append(idx)
    
    filtered_indices = np.array(filtered_indices, dtype=np.int64)
    
    if len(filtered_indices) > FIXED_MAX_OUTLIER_POINTS:
        print(f"Warning: Limiting to {FIXED_MAX_OUTLIER_POINTS} outlier points")
        np.random.seed(42)
        filtered_indices = np.random.choice(filtered_indices, FIXED_MAX_OUTLIER_POINTS, replace=False)
    
    print(f"Interpolating around {len(filtered_indices)} outlier points...")
    new_points = interpolate_between_outliers(
        filtered_indices, points, depth_map_resolution
    )
    
    new_df = pd.DataFrame(new_points, columns=['h', 'theta', 'r', 'intensity'])
    new_df['pred'] = 8
    
    elapsed = time.time() - start_time
    print(f"Outlier enhancement completed in {elapsed:.2f}s, added {len(new_df)} points")
    
    return outlier_df, new_df


def generate_depth_map(
    data_surface: Dict,
    data_boundary: Dict,
    resolution: float,
    record_mapping: bool = True,
    outlier_mode: bool = False,
    window_size: int = None
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Project point cloud data to 2D depth map.
    
    CRITICAL PARAMETER:
    - resolution: Affects all downstream detection/SAM (HIGH impact)
    """
    def to_arrays(data):
        if isinstance(data, pd.DataFrame):
            return data[['x', 'y', 'z', 'pred']].values.T
        return np.array([data['x'], data['y'], data['z'], data['pred']])
    
    surface_index = data_surface.get('index')
    surface = to_arrays(data_surface)
    boundary = to_arrays(data_boundary)
    
    x_min = min(surface[0].min(), boundary[0].min())
    x_max = max(surface[0].max(), boundary[0].max())
    y_min = min(surface[1].min(), boundary[1].min())
    y_max = max(surface[1].max(), boundary[1].max())
    
    height = int((y_max - y_min) / resolution)
    width = int((x_max - x_min) / resolution)
    print(f'Depth map dimensions: {height} x {width}')
    
    depth_map = np.full((height, width), np.nan, dtype=np.float32)
    
    def process_points(data, index, record=False):
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
    
    mapping = []
    total_steps = 1 if outlier_mode else 2
    with tqdm(total=total_steps, desc="Projecting to depth map") as pbar:
        if not outlier_mode:
            mapping = process_points(surface, surface_index, record=record_mapping)
            pbar.update(1)
        process_points(boundary, None, record=False)
        pbar.update(1)
    
    if record_mapping and not outlier_mode:
        print(f"Mapped {len(mapping)} points to pixels")
    
    # Interpolate gaps (use provided window_size or default)
    effective_window = window_size if window_size is not None else DEFAULT_INTERPOLATION_WINDOW
    if effective_window > 1:
        valid_points = []
        half_w = effective_window // 2
        
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


def save_depth_map_image(depth_map: np.ndarray, filepath: str, resolution: float) -> None:
    """Save depth map as an image with exact pixel dimensions."""
    height, width = depth_map.shape
    dpi = 1.0 / resolution
    
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(depth_map, cmap='viridis')
    
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


def classify_tunnel_pattern(depth_map_outlier: np.ndarray, tunnel_dir: str) -> Dict[str, Any]:
    """Classify tunnel joint pattern from enhanced depth map."""
    L, W = depth_map_outlier.shape
    
    binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)
    ret, binary = cv2.threshold(binary_map, 120, 255, cv2.THRESH_BINARY)
    
    lines_oblique = cv2.HoughLinesP(binary, 1, np.pi/180, 30, minLineLength=50, maxLineGap=20)
    
    angles = []
    y_positions = []
    
    if lines_oblique is not None:
        for line in lines_oblique[:200]:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
            
            if 5 <= abs(angle) <= 10:
                angles.append(angle)
                mid_y = (y1 + y2) / 2
                y_positions.append(mid_y)
    
    if len(angles) > 0 and len(y_positions) > 0:
        angle_std = np.std(angles)
        angle_mean = np.mean(np.abs(angles))
        y_std = np.std(y_positions)
        y_mean = np.mean(y_positions)
        
        is_alternating = False
        if len(y_positions) >= 4:
            try:
                y_array = np.array(y_positions).reshape(-1, 1)
                centroids, labels = kmeans2(y_array, 2, iter=10, minit="++")
                cluster_stds = [float(np.std(y_array[labels == i])) for i in range(2)]
                unique_labels = len(set(labels))
                c0 = float(np.asarray(centroids[0]).flat[0])
                c1 = float(np.asarray(centroids[1]).flat[0])

                is_alternating = (unique_labels == 2 and
                                 all(std < 50 for std in cluster_stds) and
                                 abs(c0 - c1) > 200)
            except Exception:
                pass
        
        if y_std < 100:
            pattern_type = "continuous"
            confidence = min(1.0, (100 - y_std) / 100)
            description = "Continuous joints (T3-like): K-blocks horizontally aligned"
        elif is_alternating:
            pattern_type = "simple_staggered"
            confidence = 0.8
            description = "Simple staggered joints (T1/T2-like): Regular alternating pattern"
        elif y_std < 250 and angle_std < 8.0:
            pattern_type = "simple_staggered"
            confidence = 0.7
            description = "Simple staggered joints (T1/T2-like): Regular pattern"
        else:
            pattern_type = "complex_staggered"
            confidence = 0.7
            description = "Complex staggered joints (T4/T5-like): Irregular/variable offset"
    else:
        pattern_type = "unknown"
        confidence = 0.0
        angle_std = 0.0
        angle_mean = 0.0
        y_std = 0.0
        y_mean = 0.0
        description = "No oblique lines detected - pattern classification unavailable"
    
    pattern_info = {
        "pattern_type": pattern_type,
        "confidence": float(confidence),
        "description": description,
        "statistics": {
            "oblique_lines_detected": len(angles),
            "angle_mean_deg": float(angle_mean) if len(angles) > 0 else 0.0,
            "angle_std_deg": float(angle_std) if len(angles) > 0 else 0.0,
            "y_position_mean": float(y_mean) if len(y_positions) > 0 else 0.0,
            "y_position_std": float(y_std) if len(y_positions) > 0 else 0.0,
            "is_alternating": is_alternating if len(y_positions) >= 4 else False
        }
    }
    
    return pattern_info


def enhance_point_cloud(
    df: pd.DataFrame,
    tunnel_dir: str,
    ring_spacing: float,
    target_distances: List[float],
    curvature_neighbors: int,
    depth_map_resolution: float,
    interpolation_window: int = DEFAULT_INTERPOLATION_WINDOW
) -> pd.DataFrame:
    """
    Execute enhancing stage: curvature, upsampling, outlier enhancement, depth map generation.
    
    CRITICAL PARAMETERS:
    - target_distances: Controls upsampling density (HIGH impact)
    - curvature_neighbors: Affects surface smoothness (MEDIUM impact)
    - depth_map_resolution: Affects all downstream stages (HIGH impact)
    - interpolation_window: Gap filling window for main depth_map (LOW impact)
    """
    df_valid = df[df['pred'] != 0].copy()
    
    # Add curvature
    df_with_curvature = add_curvature_column(df_valid, curvature_neighbors)
    
    # Progressive upsampling
    df_upsampled = progressive_upsample(df_with_curvature, target_distances)
    
    # Outlier boundary enhancement
    outlier_df, boundary_points = enhance_outlier_boundaries(
        df_with_curvature, depth_map_resolution, ring_spacing
    )
    df_boundary = pd.concat([outlier_df, boundary_points], ignore_index=False)
    
    # Mark outliers as removed
    df.loc[outlier_df.index, 'pred'] = 0
    
    # Generate depth maps
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
        surface_data, boundary_data, resolution=depth_map_resolution,
        window_size=interpolation_window  # Tunable for main depth_map
    )
    
    with open(os.path.join(tunnel_dir, "pixel_to_point.pkl"), 'wb') as f:
        pickle.dump(pixel_mapping, f)
    
    save_depth_map_image(depth_map, os.path.join(tunnel_dir, "depth_map.png"), resolution=depth_map_resolution)
    
    # Generate outlier depth map
    outlier_data = {
        'x': df_boundary['h'],
        'y': df_boundary['theta'],
        'z': df_boundary['r'],
        'pred': df_boundary['pred'],
        'intensity': df_boundary.get('intensity', pd.Series([0] * len(df_boundary)))
    }
    
    depth_map_outlier, _ = generate_depth_map(
        surface_data, outlier_data,
        resolution=depth_map_resolution, record_mapping=False, outlier_mode=True,
        window_size=1  # No gap interpolation for outlier depth map (matches original p4tun)
    )
    np.save(os.path.join(tunnel_dir, "depth_map_outlier.npy"), depth_map_outlier)
    
    # Combine original and new points
    new_surface_points = df_upsampled[df_upsampled['pred'] == 8].copy()
    new_boundary_points = df_boundary[df_boundary['pred'] == 8].copy()
    
    for col in df.columns:
        if col not in new_surface_points.columns:
            new_surface_points[col] = np.nan if col in ['x', 'y', 'z'] else None
        if col not in new_boundary_points.columns:
            new_boundary_points[col] = np.nan if col in ['x', 'y', 'z'] else None
    
    all_new = pd.concat([new_surface_points, new_boundary_points], ignore_index=True)
    df_enhanced = pd.concat([df, all_new], ignore_index=True)
    
    df_enhanced.to_csv(os.path.join(tunnel_dir, "enhanced.csv"), index=False)
    
    # Pattern classification
    pattern_info = classify_tunnel_pattern(depth_map_outlier, tunnel_dir)
    with open(os.path.join(tunnel_dir, "pattern_type.json"), 'w') as f:
        json.dump(pattern_info, f, indent=2)
    
    return df_enhanced


# =============================================================================
# MEDIUM SENSITIVITY PARAMETERS (tunable but less critical)
# =============================================================================

DEFAULT_DILATION_KERNEL_SIZE = 3
DEFAULT_DILATION_ITERATIONS = 1
DEFAULT_HOUGH_OBLIQUE_MIN_LENGTH = 100
DEFAULT_HOUGH_OBLIQUE_MAX_GAP = 40

# Hough Horizontal (fixed - MEDIUM sensitivity, not critical)
FIXED_HOUGH_HORIZONTAL_THRESHOLD = 50
FIXED_HOUGH_HORIZONTAL_MIN_LENGTH = 100
FIXED_HOUGH_HORIZONTAL_MAX_GAP = 10
FIXED_HORIZONTAL_ANGLE_TOLERANCE = 1

# Line Processing (fixed - LOW sensitivity)
FIXED_MERGE_DISTANCE_THRESHOLD = 3
FIXED_MERGE_CLOSE_THRESHOLD = 6


# =============================================================================
# Utility Functions
# =============================================================================

def mm_to_px(mm: float, resolution: float) -> float:
    """Convert millimeters to pixels."""
    return mm / (resolution * 1000)


# =============================================================================
# Line Detection
# =============================================================================

def detect_lines(
    depth_map_outlier: np.ndarray,
    binary_threshold: int,
    hough_oblique_threshold: int,
    angle_positive_min: float,
    angle_positive_max: float,
    angle_negative_min: float,
    angle_negative_max: float,
    hough_vertical_threshold: int,
    dilation_kernel_size: int,
    dilation_iterations: int,
    hough_oblique_min_length: int,
    hough_oblique_max_gap: int
) -> Dict:
    """
    Detect oblique, horizontal, and vertical lines from depth map.
    
    CRITICAL PARAMETERS (HIGH sensitivity):
    - binary_threshold: Edge detection sensitivity
    - hough_oblique_threshold: Line detection sensitivity
    - angle_positive/negative_min/max: Oblique line filtering
    - hough_vertical_threshold: Ring boundary detection
    
    MEDIUM SENSITIVITY PARAMETERS:
    - dilation_kernel_size, dilation_iterations: Morphological operations
    - hough_oblique_min_length, hough_oblique_max_gap: Line filtering
    """
    L, W = depth_map_outlier.shape
    
    # Pre-processing - Binary on NaN/non-NaN
    binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)
    ret, binary_image = cv2.threshold(binary_map, binary_threshold, 255, cv2.THRESH_BINARY)
    
    # Enhanced edge detection using depth values
    depth_valid = depth_map_outlier[~np.isnan(depth_map_outlier)]
    if len(depth_valid) > 0:
        depth_min, depth_max = depth_valid.min(), depth_valid.max()
        if depth_max > depth_min:
            out = np.zeros_like(depth_map_outlier, dtype=np.float64)
            valid = ~np.isnan(depth_map_outlier)
            out[valid] = (depth_map_outlier[valid] - depth_min) / (depth_max - depth_min) * 255
            depth_normalized = out.astype(np.uint8)
            
            canny_edges = cv2.Canny(depth_normalized, 50, 150)
            combined_edges = cv2.bitwise_or(binary_image, canny_edges)
        else:
            combined_edges = binary_image
    else:
        combined_edges = binary_image
    
    # Dilation to connect broken line segments
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_edges = cv2.dilate(combined_edges, kernel, iterations=dilation_iterations)
    
    # Detect oblique lines
    lines_oblique = cv2.HoughLinesP(
        dilated_edges, 1, np.pi / 180,
        hough_oblique_threshold,
        minLineLength=hough_oblique_min_length,
        maxLineGap=hough_oblique_max_gap
    )
    
    # Detect horizontal lines (FIXED parameters)
    lines_horizontal = cv2.HoughLinesP(
        dilated_edges, 1, np.pi / 180,
        FIXED_HOUGH_HORIZONTAL_THRESHOLD,
        minLineLength=FIXED_HOUGH_HORIZONTAL_MIN_LENGTH,
        maxLineGap=FIXED_HOUGH_HORIZONTAL_MAX_GAP
    )
    
    # Detect vertical lines
    lines_vertical = cv2.HoughLines(dilated_edges, 1, np.pi / 180, hough_vertical_threshold)
    if lines_vertical is not None:
        max_rho = W
        lines_vertical = lines_vertical[lines_vertical[:, 0, 0] <= max_rho]
    
    # Separate positive and negative slope lines
    positive_lines = []
    negative_lines = []
    horizontal_lines = []
    
    if lines_oblique is not None:
        for line in lines_oblique:
            x1, y1, x2, y2 = line[0]
            x1, x2, y1, y2 = (x2, x1, y2, y1) if x1 > x2 else (x1, x2, y1, y2)
            angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
            
            if angle_positive_min <= angle <= angle_positive_max:
                positive_lines.append(line[0])
            elif angle_negative_min <= angle <= angle_negative_max:
                negative_lines.append(line[0])
    
    if lines_horizontal is not None:
        for line in lines_horizontal:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -FIXED_HORIZONTAL_ANGLE_TOLERANCE <= angle <= FIXED_HORIZONTAL_ANGLE_TOLERANCE:
                horizontal_lines.append(line[0])
    
    # Process vertical lines - merge close ones (FIXED threshold)
    merged_vertical = []
    if lines_vertical is not None:
        lines_vert_2d = lines_vertical[:, 0]
        for rho, theta in lines_vert_2d:
            if abs(theta) <= 0.5 * np.pi / 180:
                x_pos = rho * np.cos(theta)
                merged = False
                for i, (mrho, mtheta) in enumerate(merged_vertical):
                    mx = mrho * np.cos(mtheta)
                    if abs(x_pos - mx) < FIXED_MERGE_DISTANCE_THRESHOLD:
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
        'image_height': L,
        'image_width': W
    }


# =============================================================================
# Ring Center Calculation
# =============================================================================

def compute_ring_centers(
    line_data: Dict, 
    ring_count: int,
    ring_spacing: float,
    resolution: float
) -> List[float]:
    """Compute ring center X positions from vertical lines."""
    L, W = line_data['image_height'], line_data['image_width']
    expected_ring_width_px = ring_spacing / resolution  # Expected ring width in pixels
    vertical_lines = line_data['vertical_lines']
    
    if not vertical_lines:
        print("No vertical lines detected. Using fallback method.")
        block_width = W / ring_count
        return [(i + 0.5) * block_width for i in range(ring_count)]
    
    # Calculate midpoints between adjacent vertical lines
    mid_lines = []
    for i in range(len(vertical_lines) - 1):
        rho1, theta1 = vertical_lines[i]
        rho2, theta2 = vertical_lines[i + 1]
        new_rho = (rho1 + rho2) / 2
        new_theta = (theta1 + theta2) / 2
        a = np.cos(new_theta)
        x_pos = a * new_rho
        mid_lines.append((x_pos, new_theta))
    
    if len(mid_lines) == 0:
        block_width = W / ring_count
        return [(i + 0.5) * block_width for i in range(ring_count)]
    
    # Calculate average distance
    x_positions = [x for x, _ in mid_lines]
    distances = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
    avg_distance_detected = np.mean(distances) if distances else 0
    avg_distance_designed = W / ring_count
    
    if abs(avg_distance_detected - expected_ring_width_px) <= abs(avg_distance_designed - expected_ring_width_px):
        avg_distance = avg_distance_detected
    else:
        avg_distance = avg_distance_designed
    
    # Extend to cover all rings
    all_mid_lines = list(mid_lines)
    
    if mid_lines:
        # Extend left
        leftmost_x, leftmost_theta = mid_lines[0]
        x = leftmost_x - avg_distance
        while x >= 0:
            all_mid_lines.insert(0, (x, leftmost_theta))
            x -= avg_distance
        
        # Extend right
        rightmost_x, rightmost_theta = mid_lines[-1]
        x = rightmost_x + avg_distance
        while x <= W:
            all_mid_lines.append((x, rightmost_theta))
            x += avg_distance
    
    all_mid_lines = sorted(list(set(all_mid_lines)), key=lambda line: line[0])
    x_positions = [x for x, _ in all_mid_lines]
    
    return x_positions


# =============================================================================
# K-Position Calculation
# =============================================================================

def line_segment_vertical_intersection(vertical_x: float, segment: Tuple) -> Optional[float]:
    """Compute intersection of vertical line with line segment."""
    x1, y1, x2, y2 = segment
    if x1 == x2:
        return None
    if min(x1, x2) <= vertical_x <= max(x1, x2):
        t = (vertical_x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)
    return None


def merge_close_points(points: List[float]) -> List[float]:
    """Merge Y-values that are within threshold distance (FIXED)."""
    if len(points) == 0:
        return []
    pts = np.array(points, dtype=np.float64)
    if len(pts) == 1:
        return [float(pts[0])]

    merged = []
    while len(pts) > 0:
        p = pts[0]
        close_mask = np.abs(pts - p) < FIXED_MERGE_CLOSE_THRESHOLD
        merged.append(float(np.mean(pts[close_mask])))
        pts = pts[~close_mask]
    return merged


def calculate_k_positions(
    line_data: Dict,
    ring_centers: List[float],
    k_height_mm: float,
    ab_height_mm: float,
    resolution: float
) -> pd.DataFrame:
    """
    Calculate K positions using midpoint logic.
    
    CRITICAL PARAMETERS:
    - k_height_mm: K-block height for offset calculation
    - ab_height_mm: AB-block height for alternation pattern
    """
    K_HEIGHT_PX = mm_to_px(k_height_mm, resolution)
    AB_HEIGHT_PX = mm_to_px(ab_height_mm, resolution)
    L = line_data['image_height']

    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']

    adjusted_points = []

    for vertical_x in ring_centers:
        # Find intersections with positive slope lines
        pos_intersections = []
        for x1, y1, x2, y2 in positive_lines:
            y_int = line_segment_vertical_intersection(vertical_x, (x1, y1, x2, y2))
            if y_int is not None:
                pos_intersections.append(y_int)

        # Find intersections with negative slope lines
        neg_intersections = []
        for x1, y1, x2, y2 in negative_lines:
            y_int = line_segment_vertical_intersection(vertical_x, (x1, y1, x2, y2))
            if y_int is not None:
                neg_intersections.append(y_int)

        merge_positive = merge_close_points(pos_intersections)
        merge_negative = merge_close_points(neg_intersections)
        
        # Case 1: Both positive and negative slope intersections → midpoint
        if len(merge_positive) > 0 and len(merge_negative) > 0:
            midpoint_y = (merge_positive[0] + merge_negative[0]) / 2
            adjusted_points.append(('midpoint', vertical_x, midpoint_y))
        
        # Case 2: Only positive slope → adjust by -0.5*K_height
        elif len(merge_positive) > 0:
            y = merge_positive[0] - 0.5 * K_HEIGHT_PX
            adjusted_points.append(('positive_slope', vertical_x, y))
        
        # Case 3: Only negative slope → adjust by +0.5*K_height
        elif len(merge_negative) > 0:
            y = merge_negative[0] + 0.5 * K_HEIGHT_PX
            adjusted_points.append(('negative_slope', vertical_x, y))
        
        # Case 4: No line intersections — use alternation pattern
        else:
            if adjusted_points:
                last_y = adjusted_points[-1][2]
                alternation_offset = (2.0 / 3.0) * AB_HEIGHT_PX

                low_center, low_hw = 0.25 * L, 0.10 * L
                high_center, high_hw = 0.65 * L, 0.10 * L
                low_lo, low_hi = low_center - low_hw, low_center + low_hw
                high_lo, high_hi = high_center - high_hw, high_center + high_hw

                if low_lo <= last_y <= low_hi:
                    assumed_y = last_y + alternation_offset
                elif high_lo <= last_y <= high_hi:
                    assumed_y = last_y - alternation_offset
                else:
                    if len(adjusted_points) > 1:
                        second_last_y = adjusted_points[-2][2]
                        if low_lo <= second_last_y <= low_hi:
                            assumed_y = second_last_y
                        elif high_lo <= second_last_y <= high_hi:
                            assumed_y = second_last_y
                        else:
                            assumed_y = L / 2
                    else:
                        assumed_y = L / 2

                assumed_y = max(0.0, min(L, assumed_y))
                adjusted_points.append(('assume', vertical_x, assumed_y))
            else:
                adjusted_points.append(('default', vertical_x, L / 2))
    
    df = pd.DataFrame(adjusted_points, columns=['Type', 'X', 'Y'])
    df = df.sort_values(by='X').reset_index(drop=True)
    
    return df


# =============================================================================
# Visualization
# =============================================================================

def visualize_detection(
    line_data: Dict,
    ring_centers: List[float],
    k_positions: pd.DataFrame,
    tunnel_dir: str
) -> None:
    """Generate visualization of detected lines and K positions."""
    dilated_edges = line_data['dilated_edges']
    L, W = line_data['image_height'], line_data['image_width']
    
    output_image = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)
    
    # Colors
    color_positive = (255, 0, 0)    # Red
    color_negative = (0, 255, 0)    # Green
    color_horizontal = (0, 0, 255)  # Blue
    color_vertical = (255, 0, 255)  # Magenta
    line_thickness = 3
    
    # Draw positive slope lines (red)
    for x1, y1, x2, y2 in line_data['positive_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color_positive, line_thickness)
    
    # Draw negative slope lines (green)
    for x1, y1, x2, y2 in line_data['negative_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color_negative, line_thickness)
    
    # Draw horizontal lines (blue)
    for x1, y1, x2, y2 in line_data['horizontal_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color_horizontal, line_thickness)
    
    # Draw ring centers (magenta vertical lines)
    for x in ring_centers:
        cv2.line(output_image, (int(x), 0), (int(x), L), color_vertical, 1)
    
    # Draw K positions (yellow circles)
    for _, row in k_positions.iterrows():
        cv2.circle(output_image, (int(row['X']), int(row['Y'])), 8, (0, 255, 255), -1)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(output_image)
    plt.title('Detection Results')
    plt.savefig(os.path.join(tunnel_dir, 'detected_lines.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Detection Pipeline
# =============================================================================

def run_detection(tunnel_id: str, base_dir: str = "data") -> pd.DataFrame:
    """
    Execute the complete detection pipeline: Enhancing + Detection.
    
    CRITICAL PARAMETERS:
    - Enhancing: target_distances, curvature_neighbors, depth_map_resolution, interpolation_window
    - Detection: binary_threshold, hough_oblique_threshold, angle parameters, hough_vertical_threshold
    - Physical constants: k_height_mm, ab_height_mm (tunnel-specific)
    
    Args:
        tunnel_id: Identifier for the tunnel (e.g., "1-4", "2-2")
        base_dir: Base directory for data files
    """
    print(f"{'=' * 60}")
    print(f"Detection Pipeline: {tunnel_id}")
    print(f"{'=' * 60}")
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    # Load parameters
    params, params_loaded = load_parameters(tunnel_id, base_dir)
    
    # Physical constants - READ FROM PREPROCESSING STAGE
    preprocessing_params = load_preprocessing_params(tunnel_id)
    tunnel_diameter = preprocessing_params.get('tunnel_diameter', 5.5)
    ring_spacing = preprocessing_params.get('ring_spacing', 1.2)
    k_height_mm, ab_height_mm = calculate_segment_heights(tunnel_diameter)
    
    # Extract ENHANCING parameters (from detection params now)
    target_distances = get_param(params, 'target_distances', DEFAULT_TARGET_DISTANCES)
    curvature_neighbors = get_param(params, 'curvature_neighbors', DEFAULT_CURVATURE_NEIGHBORS)
    depth_map_resolution = get_param(params, 'depth_map_resolution', DEFAULT_DEPTH_MAP_RESOLUTION)
    interpolation_window = get_param(params, 'interpolation_window', DEFAULT_INTERPOLATION_WINDOW)
    
    # Extract DETECTION parameters (HIGH sensitivity)
    binary_threshold = get_param(params, 'binary_threshold', DEFAULT_BINARY_THRESHOLD)
    hough_oblique_threshold = get_param(params, 'hough_oblique_threshold', DEFAULT_HOUGH_OBLIQUE_THRESHOLD)
    angle_positive_min = get_param(params, 'angle_positive_min', DEFAULT_ANGLE_POSITIVE_MIN)
    angle_positive_max = get_param(params, 'angle_positive_max', DEFAULT_ANGLE_POSITIVE_MAX)
    angle_negative_min = get_param(params, 'angle_negative_min', DEFAULT_ANGLE_NEGATIVE_MIN)
    angle_negative_max = get_param(params, 'angle_negative_max', DEFAULT_ANGLE_NEGATIVE_MAX)
    hough_vertical_threshold = get_param(params, 'hough_vertical_threshold', DEFAULT_HOUGH_VERTICAL_THRESHOLD)
    
    # Extract MEDIUM sensitivity parameters
    dilation_kernel_size = get_param(params, 'dilation_kernel_size', DEFAULT_DILATION_KERNEL_SIZE)
    dilation_iterations = get_param(params, 'dilation_iterations', DEFAULT_DILATION_ITERATIONS)
    hough_oblique_min_length = get_param(params, 'hough_oblique_min_length', DEFAULT_HOUGH_OBLIQUE_MIN_LENGTH)
    hough_oblique_max_gap = get_param(params, 'hough_oblique_max_gap', DEFAULT_HOUGH_OBLIQUE_MAX_GAP)
    
    print("\nEnhancing parameters:")
    print(f"  target_distances:       {target_distances}")
    print(f"  curvature_neighbors:    {curvature_neighbors}")
    print(f"  depth_map_resolution:   {depth_map_resolution}")
    print(f"  interpolation_window:  {interpolation_window}")
    print("\nDetection parameters (HIGH sensitivity):")
    print(f"  binary_threshold:        {binary_threshold}")
    print(f"  hough_oblique_threshold: {hough_oblique_threshold}")
    print(f"  angle_positive_min:      {angle_positive_min}°")
    print(f"  angle_positive_max:      {angle_positive_max}°")
    print(f"  angle_negative_min:      {angle_negative_min}°")
    print(f"  angle_negative_max:      {angle_negative_max}°")
    print(f"  hough_vertical_threshold: {hough_vertical_threshold}")
    print(f"\nPhysical constants (from preprocessing stage):")
    print(f"  tunnel_diameter:         {tunnel_diameter}m")
    print(f"  ring_spacing:            {ring_spacing}m")
    print(f"  k_height_mm (calculated): {k_height_mm:.2f}")
    print(f"  ab_height_mm (calculated): {ab_height_mm:.2f}")
    print("\nMedium sensitivity parameters:")
    print(f"  dilation_kernel_size:    {dilation_kernel_size}")
    print(f"  dilation_iterations:     {dilation_iterations}")
    print(f"  hough_oblique_min_length: {hough_oblique_min_length}")
    print(f"  hough_oblique_max_gap:   {hough_oblique_max_gap}")
    
    # Load denoised point cloud
    print(f"\n[Step 0] Loading denoised point cloud...")
    df_denoised = pd.read_csv(os.path.join(tunnel_dir, "denoised.csv"))
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    print(f"  Loaded {len(df_denoised)} points, {ring_count} rings")
    
    # Run enhancing
    print(f"\n[Step 1] Enhancing point cloud...")
    df_enhanced = enhance_point_cloud(
        df_denoised, tunnel_dir,
        ring_spacing=ring_spacing,
        target_distances=target_distances,
        curvature_neighbors=curvature_neighbors,
        depth_map_resolution=depth_map_resolution,
        interpolation_window=interpolation_window
    )
    
    # Load depth map for detection
    depth_map_outlier = np.load(os.path.join(tunnel_dir, "depth_map_outlier.npy"))
    L, W = depth_map_outlier.shape
    
    print(f"\n[Step 2] Detecting lines...")
    line_data = detect_lines(
        depth_map_outlier,
        binary_threshold=binary_threshold,
        hough_oblique_threshold=hough_oblique_threshold,
        angle_positive_min=angle_positive_min,
        angle_positive_max=angle_positive_max,
        angle_negative_min=angle_negative_min,
        angle_negative_max=angle_negative_max,
        hough_vertical_threshold=hough_vertical_threshold,
        dilation_kernel_size=dilation_kernel_size,
        dilation_iterations=dilation_iterations,
        hough_oblique_min_length=hough_oblique_min_length,
        hough_oblique_max_gap=hough_oblique_max_gap
    )
    print(f"  Positive slope lines: {len(line_data['positive_lines'])}")
    print(f"  Negative slope lines: {len(line_data['negative_lines'])}")
    print(f"  Horizontal lines: {len(line_data['horizontal_lines'])}")
    print(f"  Vertical lines: {len(line_data['vertical_lines'])}")
    
    print(f"\n[Step 3] Computing ring centers...")
    ring_centers = compute_ring_centers(line_data, ring_count, ring_spacing, depth_map_resolution)
    print(f"  Found {len(ring_centers)} ring centers")
    
    print(f"\n[Step 4] Calculating K positions...")
    k_positions = calculate_k_positions(
        line_data, ring_centers, k_height_mm, ab_height_mm, depth_map_resolution
    )
    print(f"  Calculated {len(k_positions)} K positions")
    print(f"  Detection types: {k_positions['Type'].value_counts().to_dict()}")
    
    # Save results
    k_positions.to_csv(os.path.join(tunnel_dir, 'detected.csv'), index=False)
    print(f"\n  Saved: {os.path.join(tunnel_dir, 'detected.csv')}")
    
    # Generate visualization
    visualize_detection(line_data, ring_centers, k_positions, tunnel_dir)
    print(f"  Saved: {os.path.join(tunnel_dir, 'detected_lines.png')}")
    
    print(f"\n{'=' * 60}")
    print(f"Detection complete!")
    print(f"{'=' * 60}")
    
    print("\nK Position Summary:")
    print(k_positions.to_string(index=False))
    
    return k_positions


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simplified detection pipeline")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 1-4)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    run_detection(args.tunnel_id, base_dir=args.data_dir)
