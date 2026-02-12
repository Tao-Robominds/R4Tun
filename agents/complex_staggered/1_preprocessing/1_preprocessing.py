"""
Consolidated Preprocessing Pipeline: Unfolding + Denoising

This module combines unfolding and denoising stages into a single script with only
CRITICAL parameters exposed for Bayesian Optimization experiments.

Based on P4TUN optimization reports:
- Unfolding: +0.0% improvement from BO (defaults already optimal)
- Denoising: gradient_threshold is highly sensitive (0.1 best for 2-2)

Critical Parameters (4 total):
- Unfolding: ring_spacing, tunnel_diameter (physical constants)
- Denoising: radius_min, radius_max, gradient_threshold

All non-critical parameters use fixed defaults that performed well across tunnels.
"""

import os
import sys
import json
import math
import random
from typing import Tuple, List, Dict, Any
from collections import defaultdict

import cv2
import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numba import njit, prange
from scipy.spatial import ConvexHull, KDTree
from scipy.interpolate import interp1d
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
        1. bo4tun_agents/simple_staggered/1_preprocessing/parameters/<tunnel_id>/parameters_preprocessing.json
        2. data/<tunnel_id>/parameters_preprocessing.json
        3. bo4tun_agents/simple_staggered/1_preprocessing/parameters/sample/parameters_preprocessing.json
        4. Hardcoded defaults (if no file found)
    
    Returns:
        Tuple of (params_dict, was_loaded_from_file)
    """
    script_dir = os.path.dirname(__file__)
    param_file = "parameters_preprocessing.json"
    
    # Try centralized parameters folder first
    if tunnel_id:
        params_path = os.path.join(script_dir, "parameters", tunnel_id, param_file)
        if os.path.exists(params_path):
            print(f"Loading parameters from {params_path}")
            with open(params_path, 'r') as f:
                return json.load(f), True
        
        # Try data folder
        tunnel_path = os.path.join(base_dir, tunnel_id, param_file)
        if os.path.exists(tunnel_path):
            print(f"Loading parameters from {tunnel_path}")
            with open(tunnel_path, 'r') as f:
                return json.load(f), True
    
    # Try sample parameters
    sample_path = os.path.join(script_dir, "parameters", "sample", param_file)
    if os.path.exists(sample_path):
        print(f"Loading sample parameters from {sample_path}")
        with open(sample_path, 'r') as f:
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


# =============================================================================
# CRITICAL PARAMETERS (tunable via JSON)
# =============================================================================

# Unfolding - Physical constants (tunnel-specific)
DEFAULT_RING_SPACING = 1.2
DEFAULT_TUNNEL_DIAMETER = 5.5

# Denoising - Critical for noise removal quality
DEFAULT_RADIUS_MIN = 2.7
DEFAULT_RADIUS_MAX = 2.8
DEFAULT_GRADIENT_THRESHOLD = 0.2


# =============================================================================
# FIXED PARAMETERS (non-critical, use proven defaults)
# Based on BO experiments showing +0.0% to +0.1% improvement when tuned
# =============================================================================

# Unfolding fixed parameters
FIXED_SLICE_HALF_THICKNESS = 0.005
FIXED_MAX_DISTANCE_FROM_TOP = 4.5
FIXED_POLYNOMIAL_DEGREE = 3
FIXED_RANSAC_INLIER_RATIO = 0.75
FIXED_RANSAC_CONFIDENCE = 0.9
FIXED_RANSAC_MIN_SAMPLES = 5
FIXED_RANSAC_INLIER_THRESHOLD = 0.8
FIXED_SAMPLES_PER_RING = 1210
FIXED_BATCH_SIZE = 1_000_000
FIXED_NUM_JOBS = 12

# Denoising fixed parameters
FIXED_THETA_STEP = 0.5
FIXED_RADIAL_STEP = 0.001
FIXED_GRADIENT_EPSILON = 1e-6
FIXED_SMOOTHING_WINDOW = 3
FIXED_SMOOTHING_OFFSET = -0.003


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
    ring_spacing: float
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Generate slicing planes perpendicular to tunnel axis."""
    total_distance = np.linalg.norm(center2 - center1)
    num_rings = round(total_distance / ring_spacing)
    direction_2d = (center2 - center1) / total_distance
    
    first_distance = total_distance / (2 * num_rings)
    last_distance = total_distance - first_distance
    
    origins = []
    planes = []
    
    for i in range(num_rings):
        if i == 0:
            segment_length = first_distance
        elif i == num_rings - 1:
            segment_length = last_distance
        else:
            segment_length = first_distance + i * (last_distance - first_distance) / (num_rings - 1)
        
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
    ring_count: int,
    tunnel_diameter: float
) -> np.ndarray:
    """Transform points to cylindrical coordinates (r, θ, h)."""
    num_samples = ring_count * FIXED_SAMPLES_PER_RING
    t_samples = np.linspace(-20, ring_count + 20, num_samples)
    
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
    tunnel_diameter: float
) -> Tuple[pd.DataFrame, int]:
    """
    Execute Stage 1: Unfolding.
    
    CRITICAL PARAMETERS:
    - ring_spacing: Controls number of rings detected
    - tunnel_diameter: Used for θ calculation
    """
    points_xyz = df[['x', 'y', 'z']].values
    
    center1, center2 = compute_tunnel_direction(points_xyz[:, :2])
    origins, planes, sliced_clouds = generate_slicing_planes(
        center1, center2, points_xyz, ring_spacing
    )
    ring_count = len(sliced_clouds)
    
    centers_3d = fit_ellipse_centers(sliced_clouds, origins, planes)
    x_params, y_params, z_params = fit_centerline_curve(centers_3d)
    
    cylindrical = transform_to_cylindrical(
        points_xyz, x_params, y_params, z_params, ring_count, tunnel_diameter
    )
    
    df_out = df.copy()
    df_out['r'] = cylindrical[:, 0]
    df_out['theta'] = cylindrical[:, 1]
    df_out['h'] = cylindrical[:, 2]
    
    return df_out, ring_count


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
    radius_min: float
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
            
            if gradient[j - 1] < -gradient_threshold or (counts[j] == 0 and counts[j - 1] == 0):
                cutoff_values[i] = radial_bins[last_valid_idx]
                break
    
    return cutoff_values, peak_radial_values


def smooth_cutoff_values(cutoff_values: np.ndarray) -> np.ndarray:
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
    return smoothed - FIXED_SMOOTHING_OFFSET


def denoise_point_cloud(
    df: pd.DataFrame,
    ring_count: int,
    radius_min: float,
    radius_max: float,
    gradient_threshold: float
) -> pd.DataFrame:
    """
    Execute Stage 2: Denoising.
    
    CRITICAL PARAMETERS:
    - radius_min/radius_max: Must match actual tunnel radius (VERY HIGH impact)
    - gradient_threshold: Controls noise detection aggressiveness (HIGH impact)
    """
    df = df.copy()
    df['pred'] = 7  # Default: valid point
    
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
            density_matrix, radial_bins, gradient_threshold, radius_min
        )
        
        smoothed_cutoffs = smooth_cutoff_values(cutoffs)
        
        # Step 3: Filter points below cutoff
        theta_bin_indices = np.digitize(theta_slice, theta_bins) - 1
        theta_bin_indices = np.clip(theta_bin_indices, 0, len(smoothed_cutoffs) - 1)
        
        below_surface = radial_slice < smoothed_cutoffs[theta_bin_indices]
        
        slice_indices = valid_df.index[slice_mask]
        filtered_indices = slice_indices[below_surface]
        df.loc[filtered_indices, 'pred'] = 0
    
    return df


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

def run_preprocessing(tunnel_id: str, base_dir: str = "data") -> None:
    """
    Execute the complete preprocessing pipeline: Unfolding + Denoising.
    
    Data flows in memory between stages. Outputs are saved for downstream use.
    
    CRITICAL PARAMETERS (4 total):
    - Unfolding: ring_spacing, tunnel_diameter (physical constants)
    - Denoising: radius_min, radius_max, gradient_threshold
    
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
    
    print("\nCritical parameters:")
    print(f"  ring_spacing:       {ring_spacing}")
    print(f"  tunnel_diameter:    {tunnel_diameter}")
    print(f"  radius_min:         {radius_min}")
    print(f"  radius_max:         {radius_max}")
    print(f"  gradient_threshold: {gradient_threshold}")
    
    # ---- Stage 1: Unfolding ----
    print("\n[Stage 1] Unfolding...")
    filepath = os.path.join(base_dir, f"{tunnel_id}.txt")
    df_raw = load_point_cloud(filepath)
    df_unwrapped, ring_count = unfold_point_cloud(df_raw, ring_spacing, tunnel_diameter)
    
    df_unwrapped.to_csv(os.path.join(tunnel_dir, "unwrapped.csv"), index=False)
    with open(os.path.join(tunnel_dir, "ring_count.txt"), 'w') as f:
        f.write(str(ring_count))
    print(f"  Saved unwrapped.csv, ring_count={ring_count}")
    
    # ---- Stage 2: Denoising ----
    print("\n[Stage 2] Denoising...")
    df_denoised = denoise_point_cloud(
        df_unwrapped, ring_count,
        radius_min=radius_min, radius_max=radius_max,
        gradient_threshold=gradient_threshold
    )
    df_denoised.to_csv(os.path.join(tunnel_dir, "denoised.csv"), index=False)
    valid_count = (df_denoised['pred'] != 0).sum()
    print(f"  Saved denoised.csv, valid points: {valid_count}/{len(df_denoised)}")
    
    print(f"\n{'=' * 60}")
    print(f"Preprocessing complete: {valid_count} valid points in denoised.csv")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 1_preprocessing.py <tunnel_id>")
        print("Example: python 1_preprocessing.py 1-4")
        sys.exit(1)

    run_preprocessing(sys.argv[1])
