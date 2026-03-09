"""
Consolidated Preprocessing Pipeline: Unfolding + Denoising + Enhancing

This module combines all preprocessing stages into a single script with only
CRITICAL parameters exposed for Bayesian Optimization experiments.

Based on P4TUN optimization reports:
- Unfolding: +0.0% improvement from BO (defaults already optimal)
- Denoising: gradient_threshold is highly sensitive (0.1 best for 2-2)
- Enhancing: Combined preprocessing yielded only +0.1% improvement

Critical Parameters (8 total):
- Unfolding: ring_spacing, tunnel_diameter (physical constants)
- Denoising: radius_min, radius_max, gradient_threshold
- Enhancing: depth_map_resolution, target_distances, curvature_neighbors, interpolation_window

All non-critical parameters use fixed defaults that performed well across tunnels.
"""

import os
import sys
import json
import math
import random
import time
import pickle
from typing import Tuple, List, Dict, Any
from collections import defaultdict

import cv2
import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numba import njit, prange
from scipy.spatial import ConvexHull, KDTree, cKDTree
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import uniform_filter1d

from shapely.geometry import Polygon
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from tqdm.notebook import tqdm

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(tunnel_id: str = None, base_dir: str = "data") -> Tuple[Dict[str, Any], bool]:
    """
    Load parameters from parameters_preprocessing.json.
    
    Priority:
        1. agents/.../parameters/<tunnel_id>/parameters_preprocessing.json
        2. data/<tunnel_id>/parameters_preprocessing.json
        3. Hardcoded defaults (if no file found)
    
    Returns:
        Tuple of (params_dict, was_loaded_from_file)
    """
    script_dir = os.path.dirname(__file__)
    param_file = "parameters_preprocessing.json"
    
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
    
    print("Warning: No parameter file found, using hardcoded defaults")
    return {}, False


def get_param(params: Dict, *keys, default=None, allow_default: bool = True):
    """Get nested parameter value with optional default fallback."""
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


# Label assigned to valid surface points before segmentation.
# Segmentation will overwrite [0, SURFACE_PRED] with block labels.
SURFACE_PRED = 7

# =============================================================================
# A. TUNNEL-PHYSICAL — measured once per tunnel, not BO-tuned
# =============================================================================

DEFAULT_RING_SPACING = 1.2          # metres between ring centres
DEFAULT_TUNNEL_DIAMETER = 5.5       # metres
DEFAULT_DEPTH_MAP_RESOLUTION = 0.005  # metres per pixel

# =============================================================================
# B. BO-CRITICAL — tunable per tunnel via JSON, candidates for BO
# =============================================================================

# Denoising (HIGH sensitivity)
DEFAULT_RADIUS_MIN = 2.7            # inner radial band (metres)
DEFAULT_RADIUS_MAX = 2.8            # outer radial band (metres)
DEFAULT_GRADIENT_THRESHOLD = 0.2    # surface cutoff steepness

# Denoising (MEDIUM sensitivity)
DEFAULT_DOUBLE_ZERO_CUTOFF = True   # consecutive-empty-bin cutoff
DEFAULT_SMOOTHING_OFFSET = -0.003   # additive shift to smoothed boundary

# Enhancing (HIGH sensitivity)
DEFAULT_TARGET_DISTANCES = [0.08, 0.04, 0.02]  # progressive upsample
DEFAULT_CURVATURE_NEIGHBORS = 20    # surface smoothness
DEFAULT_INTERPOLATION_WINDOW = 9    # depth map gap filling

# Slicing (MEDIUM sensitivity — decoupled from ring_spacing)
DEFAULT_NUM_SLICING_PLANES = None   # None = auto from ring_spacing
DEFAULT_SAMPLES_PER_RING = 1210

# Outlier enhancement (MEDIUM sensitivity)
DEFAULT_OUTLIER_DEPTH_LOW = 0.003
DEFAULT_OUTLIER_DEPTH_HIGH = 0.008
DEFAULT_OUTLIER_HD_RING_START = 0   # -1 = disabled
DEFAULT_OUTLIER_HD_RING_END = 5
DEFAULT_OUTLIER_NEIGHBORS = 20
DEFAULT_OUTLIER_MAX_POINTS = 5000
DEFAULT_OUTLIER_INTERP_RADIUS = 0.06
DEFAULT_OUTLIER_NUM_INTERPS = 2
DEFAULT_OUTLIER_DUP_THRESHOLD = 0.02
DEFAULT_OUTLIER_BIDIRECTIONAL = False
DEFAULT_OUTLIER_DEPTH_MAP_WINDOW = 1

# =============================================================================
# C. SAFE-FIXED — proven defaults, +0.0% to +0.1% from BO, never tuned
# =============================================================================

# Unfolding
FIXED_SLICE_HALF_THICKNESS = 0.005
FIXED_MAX_DISTANCE_FROM_TOP = 4.5
FIXED_POLYNOMIAL_DEGREE = 3
FIXED_RANSAC_INLIER_RATIO = 0.75
FIXED_RANSAC_CONFIDENCE = 0.9
FIXED_RANSAC_MIN_SAMPLES = 5
FIXED_RANSAC_INLIER_THRESHOLD = 0.8
FIXED_BATCH_SIZE = 1_000_000
FIXED_NUM_JOBS = 12

# Denoising
FIXED_THETA_STEP = 0.5
FIXED_RADIAL_STEP = 0.001
FIXED_GRADIENT_EPSILON = 1e-6
FIXED_SMOOTHING_WINDOW = 3

# Enhancing
FIXED_CURVATURE_THRESHOLD = 0.0005
FIXED_UPSAMPLING_NEIGHBORS = 20
FIXED_DISTANCE_TOLERANCE_LOW = 0.9
FIXED_DISTANCE_TOLERANCE_HIGH = 2.0
FIXED_RADIUS_FILTER_FACTOR = 0.15
FIXED_MIN_NEW_POINT_DISTANCE_FACTOR = 0.2


# =============================================================================
# STAGE 1: UNFOLDING
# =============================================================================

def load_point_cloud(filepath: str) -> pd.DataFrame:
    """
    Load point cloud data from a text file.
    
    Supported formats:
        - 3 columns: x, y, z
        - 4 columns: x, y, z, intensity
        - 6 columns: x, y, z, intensity, segment, ring
    """
    data = np.loadtxt(filepath)
    ncols = data.shape[1] if data.ndim == 2 else 0
    if ncols < 3:
        raise ValueError(f"Point cloud must have at least 3 columns (x,y,z), got {ncols}")

    out = {
        'x': data[:, 0],
        'y': data[:, 1],
        'z': data[:, 2],
    }
    if ncols >= 4:
        out['intensity'] = data[:, 3]
    if ncols >= 6:
        out['segment'] = data[:, 4].astype(int)
        out['ring'] = data[:, 5].astype(int)
    return pd.DataFrame(out)


def compute_tunnel_direction(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute tunnel direction from XY projection using minimum bounding rectangle."""
    hull = ConvexHull(points_xy)
    hull_points = points_xy[hull.vertices]
    polygon = Polygon(hull_points)
    min_rect = polygon.minimum_rotated_rectangle
    
    vertices = np.array(min_rect.exterior.coords)[:-1]
    edge_lengths = [np.linalg.norm(vertices[i] - vertices[(i + 1) % 4]) for i in range(4)]
    short_edge_idx = np.argmin(edge_lengths)
    
    center1 = (vertices[short_edge_idx] + vertices[(short_edge_idx + 1) % 4]) / 2
    center2 = (vertices[(short_edge_idx + 2) % 4] + vertices[(short_edge_idx + 3) % 4]) / 2
    
    return center1, center2


def generate_slicing_planes(
    center1: np.ndarray,
    center2: np.ndarray,
    points_xyz: np.ndarray,
    num_planes: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Generate slicing planes perpendicular to tunnel axis."""
    total_distance = np.linalg.norm(center2 - center1)
    direction_2d = (center2 - center1) / total_distance
    
    first_distance = total_distance / (2 * num_planes)
    last_distance = total_distance - first_distance
    
    origins = []
    planes = []
    
    for i in range(num_planes):
        if i == 0:
            segment_length = first_distance
        elif i == num_planes - 1:
            segment_length = last_distance
        else:
            segment_length = first_distance + i * (last_distance - first_distance) / (num_planes - 1)
        
        point_on_plane = center1 + (segment_length / total_distance) * (center2 - center1)
        origins.append(np.array([point_on_plane[0], point_on_plane[1], 0.0]))
        
        normal = np.array([direction_2d[0], direction_2d[1], 0.0])
        d = -np.dot(normal[:2], point_on_plane)
        planes.append(np.array([normal[0], normal[1], normal[2], d]))
    
    # Slice point cloud
    points_xyz = np.asarray(points_xyz)
    sliced_clouds = []
    delta = FIXED_SLICE_HALF_THICKNESS
    
    for plane in tqdm(planes, desc="Slicing point cloud"):
        a, b, c, d = plane
        dist = a * points_xyz[:, 0] + b * points_xyz[:, 1] + c * points_xyz[:, 2] + d
        mask = np.abs(dist) <= delta
        sliced_clouds.append(points_xyz[mask])
    
    return origins, planes, sliced_clouds


def project_to_plane(points_3d: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project 3D points onto a plane and convert to 2D coordinates."""
    shifted = np.array(points_3d) - np.array(origin)
    projection = np.dot(shifted, normal)
    projected = shifted - np.outer(projection, normal)
    
    x_axis = np.array([-normal[1], normal[0], 0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    
    x_coords = np.dot(projected, x_axis)
    y_coords = np.dot(projected, y_axis)
    
    return np.column_stack((x_coords, y_coords))


def filter_upper_tunnel_points(points_2d: np.ndarray) -> np.ndarray:
    """Filter points to keep only those near the top of tunnel cross-section."""
    y_max = np.max(points_2d[:, 1])
    mask = np.abs(points_2d[:, 1] - y_max) <= FIXED_MAX_DISTANCE_FROM_TOP
    return points_2d[mask]


def convert_2d_to_3d(point_2d: np.ndarray, plane_params: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Convert 2D plane coordinates back to 3D world coordinates."""
    xp, yp = point_2d
    A, B, C, D = plane_params
    x0, y0, z0 = origin
    
    N = np.array([A, B, C])
    N = N / np.linalg.norm(N)
    
    V = np.array([-B, A, 0.0])
    V = V / np.linalg.norm(V)
    
    U = np.cross(N, V)
    U = U / np.linalg.norm(U)
    
    return np.array([
        x0 + xp * V[0] + yp * U[0],
        y0 + xp * V[1] + yp * U[1],
        z0 + xp * V[2] + yp * U[2]
    ])


class EllipseRANSAC:
    """RANSAC-based ellipse fitting for 2D point data."""
    
    def __init__(self, points: np.ndarray):
        self.points = points
        self.inlier_ratio = FIXED_RANSAC_INLIER_RATIO
        self.confidence = FIXED_RANSAC_CONFIDENCE
        self.min_samples = FIXED_RANSAC_MIN_SAMPLES
        self.inlier_threshold = FIXED_RANSAC_INLIER_THRESHOLD
        self.max_inliers = len(points) * self.inlier_ratio
        
        self.best_model = ((0, 0), (1e-6, 1e-6), 0)
        self.best_inliers = np.array([])
        self.best_count = 0
    
    def _geometric_to_conic(self, ellipse: tuple) -> np.ndarray:
        """Convert ellipse geometric parameters to conic coefficients."""
        (x0, y0), (bb, aa), phi_deg = ellipse
        a, b = aa / 2, bb / 2
        phi_rad = np.radians(phi_deg)
        ax, ay = -np.sin(phi_rad), np.cos(phi_rad)
        
        a2, b2 = a * a, b * b
        if a2 <= 0 or b2 <= 0:
            return np.array([1, 0, 1, 0, 0, -1e-6])
        
        A = ax * ax / a2 + ay * ay / b2
        B = 2 * ax * ay / a2 - 2 * ax * ay / b2
        C = ay * ay / a2 + ax * ax / b2
        D = (-2 * ax * ay * y0 - 2 * ax * ax * x0) / a2 + \
            (2 * ax * ay * y0 - 2 * ay * ay * x0) / b2
        E = (-2 * ax * ay * x0 - 2 * ay * ay * y0) / a2 + \
            (2 * ax * ay * x0 - 2 * ax * ax * y0) / b2
        F = (2 * ax * ay * x0 * y0 + ax * ax * x0 * x0 + ay * ay * y0 * y0) / a2 + \
            (-2 * ax * ay * x0 * y0 + ay * ay * x0 * x0 + ax * ax * y0 * y0) / b2 - 1
        
        return np.array([A, B, C, D, E, F])
    
    def _evaluate_model(self, ellipse: tuple) -> Tuple[int, np.ndarray]:
        """Evaluate ellipse model and count inliers."""
        conic = self._geometric_to_conic(ellipse)
        
        if 4 * conic[0] * conic[2] - conic[1] ** 2 <= 0:
            return 0, np.array([])
        
        (x, y), (axis1, axis2), angle = ellipse
        semi_major = max(axis1, axis2) / 2
        semi_minor = min(axis1, axis2) / 2
        
        focal_dist = math.sqrt(semi_major ** 2 - semi_minor ** 2)
        angle_rad = math.radians(angle)
        
        focus1 = np.array([
            x - focal_dist * math.cos(angle_rad),
            y - focal_dist * math.sin(angle_rad)
        ])
        focus2 = np.array([
            x + focal_dist * math.cos(angle_rad),
            y + focal_dist * math.sin(angle_rad)
        ])
        
        dist1 = np.linalg.norm(self.points - focus1, axis=1)
        dist2 = np.linalg.norm(self.points - focus2, axis=1)
        focal_sum = dist1 + dist2
        
        residuals = np.abs(2 * semi_major - focal_sum)
        threshold = self.inlier_threshold * np.std(residuals)
        inlier_mask = residuals < threshold
        
        return np.sum(inlier_mask), self.points[inlier_mask]
    
    def fit(self) -> Tuple[tuple, np.ndarray]:
        """Execute RANSAC to find best ellipse fit."""
        max_iterations = 999
        iteration = 0
        
        while iteration < max_iterations:
            sample_idx = random.sample(range(len(self.points)), self.min_samples)
            sample = self.points[sample_idx].astype(np.float32)
            
            try:
                ellipse = cv2.fitEllipse(sample)
            except cv2.error:
                iteration += 1
                continue
            
            count, inliers = self._evaluate_model(ellipse)
            
            if count > self.best_count:
                self.best_count = count
                inliers_float = np.array(inliers, dtype=np.float32)
                if len(inliers_float) >= 5:
                    self.best_model = cv2.fitEllipse(inliers_float)
                    self.best_inliers = inliers_float
                
                if count > self.max_inliers:
                    break
                
                ratio = count / len(self.points)
                if ratio > 0:
                    max_iterations = math.log(1 - self.confidence) / \
                                    math.log(1 - ratio ** self.min_samples)
            
            iteration += 1
        
        return self.best_model, self.best_inliers


def fit_ellipse_centers(
    sliced_clouds: List[np.ndarray],
    origins: List[np.ndarray],
    planes: List[np.ndarray]
) -> np.ndarray:
    """Fit ellipses to each slice and return 3D center points."""
    normal = np.array([planes[0][0], planes[0][1], 0])
    centers_3d = []
    
    for i in range(len(sliced_clouds)):
        points_2d = project_to_plane(sliced_clouds[i], origins[i], normal)
        filtered = filter_upper_tunnel_points(points_2d)
        points_data = np.reshape(filtered, (-1, 2))
        
        ransac1 = EllipseRANSAC(points_data)
        _, inliers = ransac1.fit()
        
        ransac2 = EllipseRANSAC(inliers)
        ellipse, _ = ransac2.fit()
        
        (cx, cy), _, _ = ellipse
        center_3d = convert_2d_to_3d(np.array([cx, cy]), planes[i], origins[i])
        centers_3d.append(center_3d)
    
    return np.array(centers_3d)


def fit_centerline_curve(centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit polynomial curve through ellipse centers using RANSAC."""
    n = len(centers)
    t = np.arange(n)
    
    poly = PolynomialFeatures(FIXED_POLYNOMIAL_DEGREE)
    t_poly = poly.fit_transform(t.reshape(-1, 1))
    
    ransac_x = RANSACRegressor(random_state=RANDOM_SEED).fit(t_poly, centers[:, 0])
    ransac_y = RANSACRegressor(random_state=RANDOM_SEED).fit(t_poly, centers[:, 1])
    ransac_z = RANSACRegressor(random_state=RANDOM_SEED).fit(t_poly, centers[:, 2])
    
    def get_params(ransac):
        coef = ransac.estimator_.coef_.copy()
        coef[0] = ransac.estimator_.intercept_
        return coef
    
    return get_params(ransac_x), get_params(ransac_y), get_params(ransac_z)


@njit
def poly_eval(coeffs: np.ndarray, x: float) -> float:
    """Evaluate polynomial using Horner's method."""
    result = 0.0
    for coeff in coeffs:
        result = result * x + coeff
    return result


@njit(parallel=True)
def evaluate_curve(
    t: np.ndarray,
    x_params: np.ndarray,
    y_params: np.ndarray,
    z_params: np.ndarray
) -> np.ndarray:
    """Evaluate the 3D curve at parameter values t."""
    result = np.empty((len(t), 3))
    x_rev, y_rev, z_rev = x_params[::-1], y_params[::-1], z_params[::-1]
    for i in prange(len(t)):
        result[i, 0] = poly_eval(x_rev, t[i])
        result[i, 1] = poly_eval(y_rev, t[i])
        result[i, 2] = poly_eval(z_rev, t[i])
    return result


@njit
def poly_derivative(coeffs: np.ndarray) -> np.ndarray:
    """Compute derivative of polynomial coefficients."""
    return np.array([i * c for i, c in enumerate(coeffs[:0:-1])][::-1])


@njit(parallel=True)
def evaluate_curve_derivative(
    t: np.ndarray,
    x_params: np.ndarray,
    y_params: np.ndarray,
    z_params: np.ndarray
) -> np.ndarray:
    """Evaluate tangent vectors of the 3D curve."""
    result = np.empty((len(t), 3))
    dx = poly_derivative(x_params[::-1])
    dy = poly_derivative(y_params[::-1])
    dz = poly_derivative(z_params[::-1])
    for i in prange(len(t)):
        result[i, 0] = poly_eval(dx, t[i])
        result[i, 1] = poly_eval(dy, t[i])
        result[i, 2] = poly_eval(dz, t[i])
    return result


@njit
def compute_angle_and_distance(
    point: np.ndarray,
    curve_point: np.ndarray,
    reference_point: np.ndarray
) -> Tuple[float, float]:
    """Compute angle and distance for cylindrical coordinate transformation."""
    vec_to_point = curve_point - point
    vec_along_curve = reference_point - curve_point
    
    norm1 = np.sqrt(np.dot(vec_to_point, vec_to_point))
    norm2 = np.sqrt(np.dot(vec_along_curve, vec_along_curve))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0, norm1
    
    cos_angle = np.dot(vec_to_point, vec_along_curve) / (norm1 * norm2)
    if cos_angle > 1.0:
        cos_angle = 1.0
    elif cos_angle < -1.0:
        cos_angle = -1.0
    angle_rad = np.arccos(cos_angle)
    angle_deg = angle_rad * (180.0 / np.pi)
    
    cross = np.cross(vec_to_point, vec_along_curve)
    if cross[2] < 0:
        angle_deg = 360 - angle_deg
    
    return angle_deg, norm1


@njit
def compute_reference_points_and_arc_lengths(
    curve_points: np.ndarray,
    tangent_vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reference points (for angle calculation) and arc lengths."""
    n = curve_points.shape[0]
    ref_points = np.empty_like(curve_points)
    arc_lengths = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        B = curve_points[i]
        T = tangent_vectors[i]
        
        denom = T[0] ** 2 + T[1] ** 2
        if denom > 0:
            lambda_ = -T[2] / denom
            ref_points[i] = B + lambda_ * np.array([T[0], T[1], 0]) + np.array([0, 0, 1])
        else:
            ref_points[i] = B + np.array([0, 0, 1])
        
        if i > 0:
            arc_lengths[i] = arc_lengths[i - 1] + np.linalg.norm(B - curve_points[i - 1])
    
    return ref_points, arc_lengths


def transform_to_cylindrical(
    points_xyz: np.ndarray,
    x_params: np.ndarray,
    y_params: np.ndarray,
    z_params: np.ndarray,
    n_planes: int,
    tunnel_diameter: float,
    samples_per_ring: int = DEFAULT_SAMPLES_PER_RING
) -> np.ndarray:
    """Transform points to cylindrical coordinates (r, θ, h)."""
    num_samples = n_planes * samples_per_ring
    t_samples = np.linspace(-20, n_planes + 20, num_samples)
    
    curve_points = evaluate_curve(t_samples, x_params, y_params, z_params)
    tangent_vectors = evaluate_curve_derivative(t_samples, x_params, y_params, z_params)
    ref_points, arc_lengths = compute_reference_points_and_arc_lengths(
        curve_points, tangent_vectors
    )
    
    index = faiss.IndexFlatL2(3)
    index.add(curve_points.astype(np.float32))
    
    def process_batch(batch: np.ndarray) -> List[Tuple[float, float, float]]:
        _, indices = index.search(batch.astype(np.float32), 1)
        results = []
        
        for i, idx in enumerate(indices.flatten()):
            angle, distance = compute_angle_and_distance(
                batch[i], curve_points[idx], ref_points[idx]
            )
            arc_length = arc_lengths[idx]
            results.append((distance, angle, arc_length))
        
        return results
    
    num_batches = (len(points_xyz) + FIXED_BATCH_SIZE - 1) // FIXED_BATCH_SIZE
    batches = np.array_split(points_xyz, num_batches)
    
    results = Parallel(n_jobs=FIXED_NUM_JOBS)(
        delayed(process_batch)(batch) 
        for batch in tqdm(batches, desc="Computing cylindrical coordinates")
    )
    
    cylindrical = []
    for batch_result in results:
        cylindrical.extend(batch_result)
    
    cylindrical = np.array(cylindrical)
    cylindrical[:, 1] *= (np.pi * tunnel_diameter / 360)
    
    return cylindrical


def unfold_point_cloud(
    df: pd.DataFrame,
    ring_spacing: float,
    tunnel_diameter: float,
    num_slicing_planes: int = None,
    samples_per_ring: int = DEFAULT_SAMPLES_PER_RING
) -> Tuple[pd.DataFrame, int, int]:
    """
    Execute Stage 1: Unfolding.
    
    Returns (df_out, ring_count, n_planes) where:
    - ring_count = physical rings (from ring_spacing)
    - n_planes = slicing planes used for curve fitting
    """
    points_xyz = df[['x', 'y', 'z']].values
    
    center1, center2 = compute_tunnel_direction(points_xyz[:, :2])
    total_distance = np.linalg.norm(center2 - center1)
    ring_count = round(total_distance / ring_spacing)
    n_planes = num_slicing_planes if num_slicing_planes else ring_count
    
    print(f"  total_distance={total_distance:.3f}m, ring_count={ring_count}, num_slicing_planes={n_planes}")
    
    origins, planes, sliced_clouds = generate_slicing_planes(
        center1, center2, points_xyz, n_planes
    )
    
    centers_3d = fit_ellipse_centers(sliced_clouds, origins, planes)
    x_params, y_params, z_params = fit_centerline_curve(centers_3d)
    
    cylindrical = transform_to_cylindrical(
        points_xyz, x_params, y_params, z_params,
        len(sliced_clouds), tunnel_diameter, samples_per_ring
    )
    
    df_out = df.copy()
    df_out['r'] = cylindrical[:, 0]
    df_out['theta'] = cylindrical[:, 1]
    df_out['h'] = cylindrical[:, 2]
    
    return df_out, ring_count, n_planes


# =============================================================================
# STAGE 2: DENOISING
# =============================================================================

@njit(parallel=True)
def calculate_density_matrix(
    theta_points: np.ndarray,
    radial_points: np.ndarray,
    theta_bins: np.ndarray,
    radial_bins: np.ndarray
) -> np.ndarray:
    """Calculate 2D histogram of point density in theta-radial space."""
    num_theta_bins = len(theta_bins) - 1
    num_radial_bins = len(radial_bins) - 1
    counts = np.zeros((num_theta_bins, num_radial_bins))
    
    for i in prange(num_theta_bins):
        theta_min, theta_max = theta_bins[i], theta_bins[i + 1]
        for j in range(num_radial_bins):
            radial_min, radial_max = radial_bins[j], radial_bins[j + 1]
            mask = ((theta_points >= theta_min) & (theta_points < theta_max) &
                    (radial_points >= radial_min) & (radial_points < radial_max))
            counts[i, j] = np.sum(mask)
    
    return counts


@njit(parallel=True)
def compute_surface_cutoffs(
    density_matrix: np.ndarray,
    radial_bins: np.ndarray,
    gradient_threshold: float,
    radius_min: float,
    double_zero_cutoff: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial cutoff values for surface boundary detection."""
    num_theta_bins = density_matrix.shape[0]
    cutoff_values = np.full(num_theta_bins, radius_min)
    peak_radial_values = np.zeros(num_theta_bins)
    
    for i in prange(num_theta_bins):
        counts = density_matrix[i, :]
        
        if np.all(counts == 0):
            continue
        
        peak_idx = np.argmax(counts)
        peak_radial_values[i] = radial_bins[peak_idx]
        
        gradient = np.diff(counts) / (counts[:-1] + FIXED_GRADIENT_EPSILON)
        
        last_valid_idx = peak_idx
        for j in range(peak_idx, 0, -1):
            if counts[j] != 0:
                last_valid_idx = j
            
            grad_trigger = gradient[j - 1] < -gradient_threshold
            dz_trigger = double_zero_cutoff and (counts[j] == 0 and counts[j - 1] == 0)
            if grad_trigger or dz_trigger:
                cutoff_values[i] = radial_bins[last_valid_idx]
                break
    
    return cutoff_values, peak_radial_values


def smooth_cutoff_values(cutoff_values: np.ndarray, smoothing_offset: float = DEFAULT_SMOOTHING_OFFSET) -> np.ndarray:
    """Smooth and interpolate cutoff values for robust boundary detection."""
    nan_mask = np.isnan(cutoff_values)
    if np.any(nan_mask):
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) > 0:
            interp_func = interp1d(
                valid_indices,
                cutoff_values[~nan_mask],
                kind='linear',
                fill_value='extrapolate'
            )
            cutoff_values[nan_mask] = interp_func(np.where(nan_mask)[0])
    
    smoothed = uniform_filter1d(cutoff_values, size=FIXED_SMOOTHING_WINDOW, mode='nearest')
    return smoothed - smoothing_offset


def denoise_point_cloud(
    df: pd.DataFrame,
    ring_count: int,
    radius_min: float,
    radius_max: float,
    gradient_threshold: float,
    double_zero_cutoff: bool = True,
    smoothing_offset: float = DEFAULT_SMOOTHING_OFFSET
) -> pd.DataFrame:
    """
    Execute Stage 2: Denoising.
    
    CRITICAL PARAMETERS:
    - radius_min/radius_max: Must match actual tunnel radius (VERY HIGH impact)
    - gradient_threshold: Controls noise detection aggressiveness (HIGH impact)
    - double_zero_cutoff: Whether consecutive empty bins trigger cutoff (MEDIUM impact)
    - smoothing_offset: Additive shift to smoothed cutoff boundary (MEDIUM impact)
    """
    df = df.copy()
    df['pred'] = SURFACE_PRED
    
    # Step 1: Initial radius filtering
    radius_mask = (df['r'] < radius_min) | (df['r'] > radius_max)
    df.loc[radius_mask, 'pred'] = 0
    
    valid_df = df[~radius_mask].copy()
    
    h_coords = valid_df['h'].values
    theta_coords = valid_df['theta'].values
    radial_coords = valid_df['r'].values
    
    h_min, h_max = np.min(h_coords), np.max(h_coords)
    theta_min, theta_max = np.min(theta_coords), np.max(theta_coords)
    radial_min, radial_max = np.min(radial_coords), np.max(radial_coords)
    
    h_step = (h_max - h_min) / ring_count
    h_bins = np.arange(h_min, h_max + h_step, h_step)
    theta_bins = np.arange(theta_min, theta_max + FIXED_THETA_STEP, FIXED_THETA_STEP)
    radial_bins = np.arange(radial_min, radial_max + FIXED_RADIAL_STEP, FIXED_RADIAL_STEP)
    
    # Step 2: Process each axial slice
    for h_idx in range(len(h_bins) - 1):
        h_low, h_high = h_bins[h_idx], h_bins[h_idx] + h_step
        slice_mask = (h_coords >= h_low) & (h_coords < h_high)
        
        theta_slice = theta_coords[slice_mask]
        radial_slice = radial_coords[slice_mask]
        
        if len(theta_slice) == 0:
            continue
        
        density_matrix = calculate_density_matrix(
            theta_slice, radial_slice, theta_bins, radial_bins
        )
        
        cutoffs, _ = compute_surface_cutoffs(
            density_matrix, radial_bins, gradient_threshold, radius_min,
            double_zero_cutoff=double_zero_cutoff
        )
        
        smoothed_cutoffs = smooth_cutoff_values(cutoffs, smoothing_offset=smoothing_offset)
        
        # Step 3: Filter points below cutoff
        theta_bin_indices = np.digitize(theta_slice, theta_bins) - 1
        theta_bin_indices = np.clip(theta_bin_indices, 0, len(smoothed_cutoffs) - 1)
        
        below_surface = radial_slice < smoothed_cutoffs[theta_bin_indices]
        
        slice_indices = valid_df.index[slice_mask]
        filtered_indices = slice_indices[below_surface]
        df.loc[filtered_indices, 'pred'] = 0
    
    return df


# =============================================================================
# STAGE 3: ENHANCING
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
    ring_spacing: float,
    depth_threshold_low: float = DEFAULT_OUTLIER_DEPTH_LOW,
    depth_threshold_high: float = DEFAULT_OUTLIER_DEPTH_HIGH,
    high_density_ring_start: int = DEFAULT_OUTLIER_HD_RING_START,
    high_density_ring_end: int = DEFAULT_OUTLIER_HD_RING_END,
    outlier_neighbors: int = DEFAULT_OUTLIER_NEIGHBORS,
    bidirectional: int = 0,
) -> np.ndarray:
    """Detect outlier points with significant local depth variation."""
    n_points = len(points)
    outlier_mask = np.zeros(n_points, dtype=np.bool_)

    for i in prange(n_points):
        neighbors = neighbor_indices[i, 1:]
        if len(neighbors) < outlier_neighbors:
            continue
        
        neighbor_depths = radial_values[neighbors]
        avg_diff = points[i, 2] - np.mean(neighbor_depths)
        
        h_coord = points[i, 0]
        in_high_density = (h_min + ring_spacing * high_density_ring_start <= h_coord <= 
                          h_min + ring_spacing * high_density_ring_end)
        
        threshold = depth_threshold_high if in_high_density else depth_threshold_low
        
        if bidirectional == 1:
            if abs(avg_diff) > threshold:
                outlier_mask[i] = True
        else:
            if avg_diff > threshold:
                outlier_mask[i] = True
    
    return outlier_mask


@njit(parallel=False)
def interpolate_between_outliers(
    outlier_indices: np.ndarray,
    points: np.ndarray,
    resolution: float,
    interpolation_radius: float = DEFAULT_OUTLIER_INTERP_RADIUS,
    num_interpolations: int = DEFAULT_OUTLIER_NUM_INTERPS,
    duplicate_threshold: float = DEFAULT_OUTLIER_DUP_THRESHOLD,
) -> np.ndarray:
    """Interpolate new points between pairs of outlier points."""
    n_outliers = len(outlier_indices)
    # Cap pre-allocation to avoid OOM; in practice, only nearby pairs produce points
    max_new = min(n_outliers * n_outliers * num_interpolations, 10_000_000)
    new_points = np.zeros((max_new, 4))
    count = 0
    
    for i in range(n_outliers):
        idx1 = outlier_indices[i]
        p1 = points[idx1]

        for j in range(i + 1, n_outliers):
            idx2 = outlier_indices[j]
            p2 = points[idx2]
            
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if not (resolution < dist < interpolation_radius):
                continue
            
            for k in range(1, num_interpolations + 1):
                t = k / (num_interpolations + 1)
                new_h = (1 - t) * p1[0] + t * p2[0]
                new_theta = (1 - t) * p1[1] + t * p2[1]
                new_r = (1 - t) * p1[2] + t * p2[2]
                new_intensity = (1 - t) * p1[3] + t * p2[3]
                
                new_points[count] = np.array([new_h, new_theta, new_r, new_intensity])
                count += 1
                if count >= max_new:
                    return new_points[:count]
    
    return new_points[:count]


def enhance_outlier_boundaries(
    df: pd.DataFrame,
    depth_map_resolution: float,
    ring_spacing: float,
    depth_threshold_low: float = DEFAULT_OUTLIER_DEPTH_LOW,
    depth_threshold_high: float = DEFAULT_OUTLIER_DEPTH_HIGH,
    high_density_ring_start: int = DEFAULT_OUTLIER_HD_RING_START,
    high_density_ring_end: int = DEFAULT_OUTLIER_HD_RING_END,
    outlier_neighbors: int = DEFAULT_OUTLIER_NEIGHBORS,
    max_outlier_points: int = DEFAULT_OUTLIER_MAX_POINTS,
    interpolation_radius: float = DEFAULT_OUTLIER_INTERP_RADIUS,
    num_interpolations: int = DEFAULT_OUTLIER_NUM_INTERPS,
    duplicate_threshold: float = DEFAULT_OUTLIER_DUP_THRESHOLD,
    bidirectional: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect outlier points and interpolate around them."""
    start_time = time.time()
    
    print('Building spatial index...')
    points = df[['h', 'theta', 'r', 'intensity']].values
    coords_2d = points[:, :2]
    tree = cKDTree(coords_2d)
    
    _, indices = tree.query(coords_2d, k=outlier_neighbors + 1)
    h_min = np.min(points[:, 0])
    
    print('Detecting outlier points...')
    outlier_mask = detect_outlier_points(
        points[:, :3], points[:, 2], indices, h_min, ring_spacing,
        depth_threshold_low=depth_threshold_low,
        depth_threshold_high=depth_threshold_high,
        high_density_ring_start=high_density_ring_start,
        high_density_ring_end=high_density_ring_end,
        outlier_neighbors=outlier_neighbors,
        bidirectional=1 if bidirectional else 0,
    )
    
    outlier_indices = np.where(outlier_mask)[0]
    print(f"Found {len(outlier_indices)} outlier points")
    
    outlier_df = df.iloc[outlier_indices].copy()
    
    # Filter out high-density region for interpolation (skip if disabled via start=-1)
    if high_density_ring_start >= 0:
        print("Filtering high-density region...")
        h_low = h_min + ring_spacing * high_density_ring_start
        h_high = h_min + ring_spacing * high_density_ring_end
        
        filtered_indices = []
        for idx in outlier_indices:
            h = points[idx, 0]
            if not (h_low <= h <= h_high):
                filtered_indices.append(idx)
        
        filtered_indices = np.array(filtered_indices, dtype=np.int64)
    else:
        print("High-density filter disabled — using ALL outlier points for interpolation")
        filtered_indices = outlier_indices.copy()
    
    if len(filtered_indices) > max_outlier_points:
        print(f"Warning: Limiting to {max_outlier_points} outlier points (from {len(filtered_indices)})")
        np.random.seed(42)
        filtered_indices = np.random.choice(filtered_indices, max_outlier_points, replace=False)
    
    print(f"Interpolating around {len(filtered_indices)} outlier points...")
    new_points = interpolate_between_outliers(
        filtered_indices, points, depth_map_resolution,
        interpolation_radius=interpolation_radius,
        num_interpolations=num_interpolations,
        duplicate_threshold=duplicate_threshold,
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


def enhance_point_cloud(
    df: pd.DataFrame,
    tunnel_dir: str,
    ring_spacing: float,
    target_distances: List[float],
    curvature_neighbors: int,
    depth_map_resolution: float,
    interpolation_window: int = DEFAULT_INTERPOLATION_WINDOW,
    outlier_depth_threshold_low: float = DEFAULT_OUTLIER_DEPTH_LOW,
    outlier_depth_threshold_high: float = DEFAULT_OUTLIER_DEPTH_HIGH,
    outlier_high_density_ring_start: int = DEFAULT_OUTLIER_HD_RING_START,
    outlier_high_density_ring_end: int = DEFAULT_OUTLIER_HD_RING_END,
    outlier_neighbors: int = DEFAULT_OUTLIER_NEIGHBORS,
    max_outlier_points: int = DEFAULT_OUTLIER_MAX_POINTS,
    outlier_interpolation_radius: float = DEFAULT_OUTLIER_INTERP_RADIUS,
    outlier_num_interpolations: int = DEFAULT_OUTLIER_NUM_INTERPS,
    outlier_duplicate_threshold: float = DEFAULT_OUTLIER_DUP_THRESHOLD,
    outlier_bidirectional: bool = False,
    outlier_depth_map_window: int = 1,
) -> pd.DataFrame:
    """
    Execute Stage 3: Enhancing.
    
    CRITICAL PARAMETERS:
    - target_distances: Controls upsampling density (HIGH impact)
    - curvature_neighbors: Affects surface smoothness (MEDIUM impact)
    - depth_map_resolution: Affects all downstream stages (HIGH impact)
    - interpolation_window: Gap filling window for main depth_map (LOW impact)
    - outlier_*: Outlier boundary detection and interpolation (HIGH impact on depth_map_outlier density)
    """
    df_valid = df[df['pred'] != 0].copy()
    
    # Add curvature
    df_with_curvature = add_curvature_column(df_valid, curvature_neighbors)
    
    # Progressive upsampling
    df_upsampled = progressive_upsample(df_with_curvature, target_distances)
    
    # Outlier boundary enhancement
    outlier_df, boundary_points = enhance_outlier_boundaries(
        df_with_curvature, depth_map_resolution, ring_spacing,
        depth_threshold_low=outlier_depth_threshold_low,
        depth_threshold_high=outlier_depth_threshold_high,
        high_density_ring_start=outlier_high_density_ring_start,
        high_density_ring_end=outlier_high_density_ring_end,
        outlier_neighbors=outlier_neighbors,
        max_outlier_points=max_outlier_points,
        interpolation_radius=outlier_interpolation_radius,
        num_interpolations=outlier_num_interpolations,
        duplicate_threshold=outlier_duplicate_threshold,
        bidirectional=outlier_bidirectional,
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
        window_size=interpolation_window
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
        window_size=outlier_depth_map_window
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
    
    return df_enhanced


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

def run_preprocessing(tunnel_id: str, base_dir: str = "data") -> None:
    """
    Execute the complete preprocessing pipeline: Unfolding + Denoising + Enhancing.
    
    Data flows in memory between stages. Outputs are saved for downstream use.
    
    CRITICAL PARAMETERS (8 total):
    - Unfolding: ring_spacing, tunnel_diameter (physical constants)
    - Denoising: radius_min, radius_max, gradient_threshold
    - Enhancing: depth_map_resolution, target_distances, curvature_neighbors, interpolation_window
    
    Args:
        tunnel_id: Identifier for the tunnel (e.g., "1-4", "5-1").
        base_dir: Base directory for data files.
    """
    print(f"{'=' * 60}")
    print(f"Preprocessing Pipeline: {tunnel_id}")
    print(f"{'=' * 60}")
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    os.makedirs(tunnel_dir, exist_ok=True)
    
    # Load parameters
    params, params_loaded = load_parameters(tunnel_id, base_dir)
    allow_defaults = not params_loaded
    
    # Extract CRITICAL parameters
    ring_spacing = get_param(params, 'ring_spacing', default=DEFAULT_RING_SPACING, allow_default=allow_defaults)
    tunnel_diameter = get_param(params, 'tunnel_diameter', default=DEFAULT_TUNNEL_DIAMETER, allow_default=allow_defaults)
    radius_min = get_param(params, 'radius_min', default=DEFAULT_RADIUS_MIN, allow_default=allow_defaults)
    radius_max = get_param(params, 'radius_max', default=DEFAULT_RADIUS_MAX, allow_default=allow_defaults)
    gradient_threshold = get_param(params, 'gradient_threshold', default=DEFAULT_GRADIENT_THRESHOLD, allow_default=allow_defaults)
    target_distances = get_param(params, 'target_distances', default=DEFAULT_TARGET_DISTANCES, allow_default=allow_defaults)
    curvature_neighbors = get_param(params, 'curvature_neighbors', default=DEFAULT_CURVATURE_NEIGHBORS, allow_default=allow_defaults)
    depth_map_resolution = get_param(params, 'depth_map_resolution', default=DEFAULT_DEPTH_MAP_RESOLUTION, allow_default=allow_defaults)
    interpolation_window = get_param(params, 'interpolation_window', default=DEFAULT_INTERPOLATION_WINDOW, allow_default=True)  # Always allow default (LOW impact)
    
    # Slicing / sampling parameters (decoupled from ring_spacing)
    num_slicing_planes = params.get('num_slicing_planes', DEFAULT_NUM_SLICING_PLANES)
    if num_slicing_planes is not None:
        num_slicing_planes = int(num_slicing_planes)
    samples_per_ring = int(params.get('samples_per_ring', DEFAULT_SAMPLES_PER_RING))

    # Outlier enhancement parameters (BO-tunable)
    outlier_depth_threshold_low = params.get('outlier_depth_threshold_low', DEFAULT_OUTLIER_DEPTH_LOW)
    outlier_depth_threshold_high = params.get('outlier_depth_threshold_high', DEFAULT_OUTLIER_DEPTH_HIGH)
    outlier_high_density_ring_start = int(params.get('outlier_high_density_ring_start', DEFAULT_OUTLIER_HD_RING_START))
    outlier_high_density_ring_end = int(params.get('outlier_high_density_ring_end', DEFAULT_OUTLIER_HD_RING_END))
    outlier_neighbors = int(params.get('outlier_neighbors', DEFAULT_OUTLIER_NEIGHBORS))
    max_outlier_points = int(params.get('max_outlier_points', DEFAULT_OUTLIER_MAX_POINTS))
    outlier_interpolation_radius = float(params.get('outlier_interpolation_radius', DEFAULT_OUTLIER_INTERP_RADIUS))
    outlier_num_interpolations = int(params.get('outlier_num_interpolations', DEFAULT_OUTLIER_NUM_INTERPS))
    outlier_duplicate_threshold = float(params.get('outlier_duplicate_threshold', DEFAULT_OUTLIER_DUP_THRESHOLD))
    outlier_bidirectional = bool(params.get('outlier_bidirectional', DEFAULT_OUTLIER_BIDIRECTIONAL))
    outlier_depth_map_window = int(params.get('outlier_depth_map_window', DEFAULT_OUTLIER_DEPTH_MAP_WINDOW))

    # Denoising surface cutoff parameters (BO-tunable)
    double_zero_cutoff = bool(params.get('double_zero_cutoff', DEFAULT_DOUBLE_ZERO_CUTOFF))
    smoothing_offset = float(params.get('smoothing_offset', DEFAULT_SMOOTHING_OFFSET))
    
    print("\nCritical parameters:")
    print(f"  ring_spacing:       {ring_spacing}")
    print(f"  num_slicing_planes: {num_slicing_planes} (None=auto from ring_spacing)")
    print(f"  samples_per_ring:   {samples_per_ring}")
    print(f"  tunnel_diameter:    {tunnel_diameter}")
    print(f"  radius_min:         {radius_min}")
    print(f"  radius_max:         {radius_max}")
    print(f"  gradient_threshold: {gradient_threshold}")
    print(f"  double_zero_cutoff: {double_zero_cutoff}")
    print(f"  smoothing_offset:   {smoothing_offset}")
    print(f"  target_distances:   {target_distances}")
    print(f"  curvature_neighbors: {curvature_neighbors}")
    print(f"  depth_map_resolution: {depth_map_resolution}")
    print(f"  interpolation_window: {interpolation_window} (LOW impact)")
    print(f"\nOutlier enhancement parameters:")
    print(f"  outlier_depth_threshold_low:  {outlier_depth_threshold_low}")
    print(f"  outlier_depth_threshold_high: {outlier_depth_threshold_high}")
    print(f"  outlier_high_density_ring_start: {outlier_high_density_ring_start} (-1=disabled)")
    print(f"  outlier_high_density_ring_end:   {outlier_high_density_ring_end}")
    print(f"  outlier_neighbors:       {outlier_neighbors}")
    print(f"  max_outlier_points:      {max_outlier_points}")
    print(f"  outlier_interpolation_radius: {outlier_interpolation_radius}")
    print(f"  outlier_num_interpolations:   {outlier_num_interpolations}")
    print(f"  outlier_bidirectional:   {outlier_bidirectional}")
    print(f"  outlier_depth_map_window: {outlier_depth_map_window}")
    
    # ---- Stage 1: Unfolding ----
    print("\n[Stage 1] Unfolding...")
    filepath = os.path.join(base_dir, f"{tunnel_id}.txt")
    df_raw = load_point_cloud(filepath)
    df_unwrapped, ring_count, n_planes = unfold_point_cloud(
        df_raw, ring_spacing, tunnel_diameter,
        num_slicing_planes=num_slicing_planes,
        samples_per_ring=samples_per_ring
    )
    
    df_unwrapped.to_csv(os.path.join(tunnel_dir, "unwrapped.csv"), index=False)
    with open(os.path.join(tunnel_dir, "ring_count.txt"), 'w') as f:
        f.write(str(ring_count))
    print(f"  Saved unwrapped.csv, ring_count={ring_count}")
    
    # Effective spacing for denoising/enhancement (based on slicing planes, not ring_spacing)
    h_vals = df_unwrapped['h'].values
    effective_spacing = (h_vals.max() - h_vals.min()) / n_planes
    print(f"  effective_spacing={effective_spacing:.4f} (from n_planes={n_planes})")
    
    # ---- Stage 2: Denoising ----
    print("\n[Stage 2] Denoising...")
    df_denoised = denoise_point_cloud(
        df_unwrapped, n_planes,
        radius_min=radius_min, radius_max=radius_max,
        gradient_threshold=gradient_threshold,
        double_zero_cutoff=double_zero_cutoff,
        smoothing_offset=smoothing_offset
    )
    df_denoised.to_csv(os.path.join(tunnel_dir, "denoised.csv"), index=False)
    valid_count = (df_denoised['pred'] != 0).sum()
    print(f"  Saved denoised.csv, valid points: {valid_count}/{len(df_denoised)}")
    
    # ---- Stage 3: Enhancing ----
    print("\n[Stage 3] Enhancing...")
    df_enhanced = enhance_point_cloud(
        df_denoised, tunnel_dir,
        ring_spacing=effective_spacing,
        target_distances=target_distances,
        curvature_neighbors=curvature_neighbors,
        depth_map_resolution=depth_map_resolution,
        interpolation_window=interpolation_window,
        outlier_depth_threshold_low=outlier_depth_threshold_low,
        outlier_depth_threshold_high=outlier_depth_threshold_high,
        outlier_high_density_ring_start=outlier_high_density_ring_start,
        outlier_high_density_ring_end=outlier_high_density_ring_end,
        outlier_neighbors=outlier_neighbors,
        max_outlier_points=max_outlier_points,
        outlier_interpolation_radius=outlier_interpolation_radius,
        outlier_num_interpolations=outlier_num_interpolations,
        outlier_duplicate_threshold=outlier_duplicate_threshold,
        outlier_bidirectional=outlier_bidirectional,
        outlier_depth_map_window=outlier_depth_map_window,
    )
    
    print(f"\n{'=' * 60}")
    print(f"Preprocessing complete: {len(df_enhanced)} points in enhanced.csv")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 1_preprocessing.py <tunnel_id>")
        print("Example: python 1_preprocessing.py 1-4")
        sys.exit(1)

    run_preprocessing(sys.argv[1])
