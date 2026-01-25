"""
Tunnel Centre Line Extraction and Point Cloud Unfolding

This module extracts the tunnel centerline from 3D point cloud data and transforms
the points into cylindrical coordinates for subsequent analysis.

Algorithm Overview:
    1. Determine tunnel direction via minimum bounding rectangle of XY projection
    2. Generate slicing planes perpendicular to the tunnel axis
    3. Fit ellipses to each cross-sectional slice using RANSAC
    4. Fit a 3D polynomial curve through the ellipse centers
    5. Transform all points to cylindrical coordinates (r, θ, h)
"""

import os
import sys
import json
import math
import random
from typing import Tuple, List, Dict, Any

import cv2
import faiss
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit, prange
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from tqdm.notebook import tqdm

# Set random seed for reproducibility (RANSAC uses random sampling)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(tunnel_id: str = None, base_dir: str = "data") -> Tuple[Dict[str, Any], bool]:
    """
    Load parameters from JSON file with fallback to defaults.
    
    Priority:
        1. Centralized: p4tun/parameters/<tunnel_id>/parameters_unfolding.json
        2. Tunnel-specific: data/<tunnel_id>/parameters_unfolding.json
        3. Default: p4tun/parameters_unfolding.json (if exists)
        4. Hardcoded defaults (if no file found)
    
    Returns:
        Tuple of (params_dict, was_loaded_from_file)
    """
    script_dir = os.path.dirname(__file__)
    param_file = "parameters_unfolding.json"
    
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
    
    # Fall back to default (if exists)
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
# Default Constants (used if no parameter file found)
# =============================================================================

# --- Physical Constants ---
DEFAULT_RING_SPACING = 1.2
DEFAULT_TUNNEL_DIAMETER = 5.5

# --- Quality Parameters ---
DEFAULT_SLICE_HALF_THICKNESS = 0.005
DEFAULT_MAX_DISTANCE_FROM_TOP = 4.5
DEFAULT_POLYNOMIAL_DEGREE = 3
DEFAULT_RANSAC_INLIER_RATIO = 0.75
DEFAULT_RANSAC_CONFIDENCE = 0.9
DEFAULT_RANSAC_MIN_SAMPLES = 5
DEFAULT_RANSAC_INLIER_THRESHOLD = 0.8
DEFAULT_SAMPLES_PER_RING = 1210

# --- Performance Parameters ---
DEFAULT_BATCH_SIZE = 1_000_000
DEFAULT_NUM_JOBS = 12


# =============================================================================
# Data Loading
# =============================================================================

def load_point_cloud(filepath: str) -> pd.DataFrame:
    """
    Load point cloud data from a text file.

    No ground-truth columns required. Algorithm uses only x, y, z.

    Supported formats:
        - 3 columns: x, y, z (unfolding only).
        - 4 columns: x, y, z, intensity (required for denoising/enhancing).
        - 6 columns: x, y, z, intensity, segment, ring.
    segment/ring are scanner metadata only; never used for pipeline logic.

    Args:
        filepath: Path to the point cloud text file.

    Returns:
        DataFrame with at least x, y, z; optionally intensity, segment, ring.
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


# =============================================================================
# Direction Vector Determination
# =============================================================================

def compute_tunnel_direction(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the tunnel direction vector from the XY projection.
    
    Uses the minimum bounding rectangle of the convex hull to find
    the tunnel's longitudinal axis.
    
    Args:
        points_xy: (N, 2) array of XY coordinates.
        
    Returns:
        Tuple of (center1, center2) representing the midpoints of the short edges.
    """
    # Compute convex hull and minimum bounding rectangle
    hull = ConvexHull(points_xy)
    hull_points = points_xy[hull.vertices]
    polygon = Polygon(hull_points)
    min_rect = polygon.minimum_rotated_rectangle
    
    # Extract rectangle vertices (drop redundant closing point)
    vertices = np.array(min_rect.exterior.coords)[:-1]
    
    # Find the short edge
    edge_lengths = [np.linalg.norm(vertices[i] - vertices[(i + 1) % 4]) 
                    for i in range(4)]
    short_edge_idx = np.argmin(edge_lengths)
    
    # Compute centers of opposite short edges
    center1 = (vertices[short_edge_idx] + vertices[(short_edge_idx + 1) % 4]) / 2
    center2 = (vertices[(short_edge_idx + 2) % 4] + vertices[(short_edge_idx + 3) % 4]) / 2
    
    return center1, center2


# =============================================================================
# Slicing Plane Generation
# =============================================================================


def generate_slicing_planes(
    center1: np.ndarray,
    center2: np.ndarray,
    points_xyz: np.ndarray,
    delta: float = DEFAULT_SLICE_HALF_THICKNESS,
    ring_spacing: float = DEFAULT_RING_SPACING
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Generate slicing planes perpendicular to the tunnel axis.
    
    Args:
        center1: Starting point of tunnel axis (2D).
        center2: Ending point of tunnel axis (2D).
        points_xyz: (N, 3) array of point cloud coordinates.
        delta: Half-thickness of slicing planes.
        
    Returns:
        Tuple of (origins, planes, sliced_clouds):
            - origins: 3D origin points for each plane
            - planes: Plane equations [A, B, C, D]
            - sliced_clouds: Points within each slice
    """
    # Compute tunnel length and ring count
    total_distance = np.linalg.norm(center2 - center1)
    num_rings = round(total_distance / ring_spacing)
    
    # Direction vector (normalized, 2D)
    direction_2d = (center2 - center1) / total_distance
    
    # Generate plane positions
    first_distance = total_distance / (2 * num_rings)
    last_distance = total_distance - first_distance
    
    origins = []
    planes = []
    
    for i in range(num_rings):
        # Compute distance along axis for this plane
        if i == 0:
            segment_length = first_distance
        elif i == num_rings - 1:
            segment_length = last_distance
        else:
            segment_length = first_distance + i * (last_distance - first_distance) / (num_rings - 1)
        
        # Plane origin (3D with z=0)
        point_on_plane = center1 + (segment_length / total_distance) * (center2 - center1)
        origins.append(np.array([point_on_plane[0], point_on_plane[1], 0.0]))
        
        # Plane equation: Ax + By + Cz + D = 0
        normal = np.array([direction_2d[0], direction_2d[1], 0.0])
        d = -np.dot(normal[:2], point_on_plane)
        planes.append(np.array([normal[0], normal[1], normal[2], d]))
    
    # Slice the point cloud
    points_xyz = np.asarray(points_xyz)
    sliced_clouds = []
    
    for plane in tqdm(planes, desc="Slicing point cloud"):
        a, b, c, d = plane
        # Compute signed distances to both boundaries
        dist = a * points_xyz[:, 0] + b * points_xyz[:, 1] + c * points_xyz[:, 2] + d
        mask = np.abs(dist) <= delta
        sliced_clouds.append(points_xyz[mask])
    
    return origins, planes, sliced_clouds


# =============================================================================
# 2D Projection
# =============================================================================

def project_to_plane(
    points_3d: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray
) -> np.ndarray:
    """
    Project 3D points onto a plane and convert to 2D coordinates.
    
    The origin in 3D becomes (0, 0) in the 2D projection.
    
    Args:
        points_3d: (N, 3) array of 3D points.
        origin: 3D origin point on the plane.
        normal: Normal vector of the plane.
        
    Returns:
        (N, 2) array of 2D coordinates on the plane.
    """
    # Shift to origin
    shifted = np.array(points_3d) - np.array(origin)
    
    # Project onto plane
    projection = np.dot(shifted, normal)
    projected = shifted - np.outer(projection, normal)
    
    # Define 2D coordinate system
    x_axis = np.array([-normal[1], normal[0], 0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    
    # Convert to 2D coordinates
    x_coords = np.dot(projected, x_axis)
    y_coords = np.dot(projected, y_axis)
    
    return np.column_stack((x_coords, y_coords))


def filter_upper_tunnel_points(
    points_2d: np.ndarray,
    max_distance: float = DEFAULT_MAX_DISTANCE_FROM_TOP
) -> np.ndarray:
    """
    Filter points to keep only those near the top of the tunnel cross-section.
    
    Args:
        points_2d: (N, 2) array of 2D points.
        max_distance: Maximum distance from the highest point.
        
    Returns:
        Filtered array of 2D points.
    """
    y_max = np.max(points_2d[:, 1])
    mask = np.abs(points_2d[:, 1] - y_max) <= max_distance
    return points_2d[mask]


def convert_2d_to_3d(
    point_2d: np.ndarray,
    plane_params: np.ndarray,
    origin: np.ndarray
) -> np.ndarray:
    """
    Convert 2D plane coordinates back to 3D world coordinates.
    
    Args:
        point_2d: 2D point (x, y) on the plane.
        plane_params: Plane equation [A, B, C, D].
        origin: 3D origin of the plane coordinate system.
        
    Returns:
        3D coordinates as array [x, y, z].
    """
    xp, yp = point_2d
    A, B, C, D = plane_params
    x0, y0, z0 = origin
    
    # Plane normal (normalized)
    N = np.array([A, B, C])
    N = N / np.linalg.norm(N)
    
    # X-axis of 2D coordinate system
    V = np.array([-B, A, 0.0])
    V = V / np.linalg.norm(V)
    
    # Y-axis of 2D coordinate system
    U = np.cross(N, V)
    U = U / np.linalg.norm(U)
    
    # Transform to 3D
    return np.array([
        x0 + xp * V[0] + yp * U[0],
        y0 + xp * V[1] + yp * U[1],
        z0 + xp * V[2] + yp * U[2]
    ])


# =============================================================================
# RANSAC Ellipse Fitting
# =============================================================================

class EllipseRANSAC:
    """
    RANSAC-based ellipse fitting for 2D point data.
    
    Uses geometric constraints to identify inliers based on the
    ellipse definition (sum of distances to foci equals 2a).
    """
    
    def __init__(
        self,
        points: np.ndarray,
        inlier_ratio: float = DEFAULT_RANSAC_INLIER_RATIO,
        confidence: float = DEFAULT_RANSAC_CONFIDENCE,
        min_samples: int = DEFAULT_RANSAC_MIN_SAMPLES,
        inlier_threshold: float = DEFAULT_RANSAC_INLIER_THRESHOLD
    ):
        """
        Initialize the RANSAC ellipse fitter.
        
        Args:
            points: (N, 2) array of 2D points.
            inlier_ratio: Expected ratio of inliers.
            confidence: Desired probability of finding a good model.
            min_samples: Minimum points to sample for fitting.
            inlier_threshold: Threshold multiplier for inlier detection (times std).
        """
        self.points = points
        self.inlier_ratio = inlier_ratio
        self.confidence = confidence
        self.min_samples = min_samples
        self.inlier_threshold = inlier_threshold
        self.max_inliers = len(points) * inlier_ratio
        
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
        
        # Check if it's a valid ellipse (discriminant > 0)
        if 4 * conic[0] * conic[2] - conic[1] ** 2 <= 0:
            return 0, np.array([])
        
        (x, y), (axis1, axis2), angle = ellipse
        semi_major = max(axis1, axis2) / 2
        semi_minor = min(axis1, axis2) / 2
        
        # Compute focal distance
        focal_dist = math.sqrt(semi_major ** 2 - semi_minor ** 2)
        angle_rad = math.radians(angle)
        
        # Compute foci positions
        focus1 = np.array([
            x - focal_dist * math.cos(angle_rad),
            y - focal_dist * math.sin(angle_rad)
        ])
        focus2 = np.array([
            x + focal_dist * math.cos(angle_rad),
            y + focal_dist * math.sin(angle_rad)
        ])
        
        # Sum of distances to foci for each point
        dist1 = np.linalg.norm(self.points - focus1, axis=1)
        dist2 = np.linalg.norm(self.points - focus2, axis=1)
        focal_sum = dist1 + dist2
        
        # Inliers: points where focal sum ≈ 2 * semi_major
        residuals = np.abs(2 * semi_major - focal_sum)
        threshold = self.inlier_threshold * np.std(residuals)
        inlier_mask = residuals < threshold
        
        return np.sum(inlier_mask), self.points[inlier_mask]
    
    def fit(self) -> Tuple[tuple, np.ndarray]:
        """
        Execute RANSAC to find the best ellipse fit.
        
        Returns:
            Tuple of (ellipse_params, inlier_points).
        """
        max_iterations = 999
        iteration = 0
        
        while iteration < max_iterations:
            # Random sample
            sample_idx = random.sample(range(len(self.points)), self.min_samples)
            sample = self.points[sample_idx].astype(np.float32)
            
            # Fit ellipse
            try:
                ellipse = cv2.fitEllipse(sample)
            except cv2.error:
                iteration += 1
                continue
            
            # Evaluate model
            count, inliers = self._evaluate_model(ellipse)
            
            if count > self.best_count:
                self.best_count = count
                inliers_float = np.array(inliers, dtype=np.float32)
                if len(inliers_float) >= 5:
                    self.best_model = cv2.fitEllipse(inliers_float)
                    self.best_inliers = inliers_float
                
                if count > self.max_inliers:
                    break
                
                # Update iteration count
                ratio = count / len(self.points)
                if ratio > 0:
                    max_iterations = math.log(1 - self.confidence) / \
                                    math.log(1 - ratio ** self.min_samples)
            
            iteration += 1
        
        return self.best_model, self.best_inliers


def fit_ellipse_centers(
    sliced_clouds: List[np.ndarray],
    origins: List[np.ndarray],
    planes: List[np.ndarray],
    max_distance: float = DEFAULT_MAX_DISTANCE_FROM_TOP,
    inlier_ratio: float = DEFAULT_RANSAC_INLIER_RATIO,
    confidence: float = DEFAULT_RANSAC_CONFIDENCE,
    min_samples: int = DEFAULT_RANSAC_MIN_SAMPLES,
    inlier_threshold: float = DEFAULT_RANSAC_INLIER_THRESHOLD
) -> np.ndarray:
    """
    Fit ellipses to each slice and return the 3D center points.
    
    Args:
        sliced_clouds: List of point clouds for each slice.
        origins: 3D origins for each slicing plane.
        planes: Plane equations for each slice.
        max_distance: Maximum distance from top for filtering.
        inlier_ratio: RANSAC inlier ratio.
        confidence: RANSAC confidence level.
        min_samples: RANSAC minimum samples.
        inlier_threshold: RANSAC inlier threshold.
        
    Returns:
        (N, 3) array of 3D ellipse center points.
    """
    # Normal vector for projection (from first plane)
    normal = np.array([planes[0][0], planes[0][1], 0])
    
    centers_3d = []
    
    for i in range(len(sliced_clouds)):
        # Project to 2D
        points_2d = project_to_plane(sliced_clouds[i], origins[i], normal)
        
        # Filter to upper tunnel
        filtered = filter_upper_tunnel_points(points_2d, max_distance=max_distance)
        points_data = np.reshape(filtered, (-1, 2))
        
        # First RANSAC pass
        ransac1 = EllipseRANSAC(
            points_data, inlier_ratio=inlier_ratio, confidence=confidence,
            min_samples=min_samples, inlier_threshold=inlier_threshold
        )
        _, inliers = ransac1.fit()
        
        # Refined RANSAC pass
        ransac2 = EllipseRANSAC(
            inliers, inlier_ratio=inlier_ratio, confidence=confidence,
            min_samples=min_samples, inlier_threshold=inlier_threshold
        )
        ellipse, _ = ransac2.fit()
        
        # Extract center and convert to 3D
        (cx, cy), _, _ = ellipse
        center_3d = convert_2d_to_3d(np.array([cx, cy]), planes[i], origins[i])
        centers_3d.append(center_3d)
    
    return np.array(centers_3d)


# =============================================================================
# Centerline Curve Fitting
# =============================================================================

def fit_centerline_curve(
    centers: np.ndarray,
    degree: int = DEFAULT_POLYNOMIAL_DEGREE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a polynomial curve through the ellipse centers using RANSAC.
    
    Args:
        centers: (N, 3) array of 3D center points.
        degree: Polynomial degree.
        
    Returns:
        Tuple of (x_params, y_params, z_params) polynomial coefficients.
    """
    n = len(centers)
    t = np.arange(n)
    
    # Polynomial features
    poly = PolynomialFeatures(degree)
    t_poly = poly.fit_transform(t.reshape(-1, 1))
    
    # Fit RANSAC regressors for each coordinate
    ransac_x = RANSACRegressor(random_state=RANDOM_SEED).fit(t_poly, centers[:, 0])
    ransac_y = RANSACRegressor(random_state=RANDOM_SEED).fit(t_poly, centers[:, 1])
    ransac_z = RANSACRegressor(random_state=RANDOM_SEED).fit(t_poly, centers[:, 2])
    
    # Extract coefficients (with intercept as first term)
    def get_params(ransac):
        coef = ransac.estimator_.coef_.copy()
        coef[0] = ransac.estimator_.intercept_
        return coef
    
    return get_params(ransac_x), get_params(ransac_y), get_params(ransac_z)


# =============================================================================
# Numba-Accelerated Functions
# =============================================================================

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
    """Evaluate the tangent vectors of the 3D curve."""
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
    """
    Compute the angle and distance for cylindrical coordinate transformation.
    
    Args:
        point: Original point in 3D space.
        curve_point: Closest point on the centerline curve.
        reference_point: Reference point for angle calculation.
        
    Returns:
        Tuple of (angle_degrees, distance).
    """
    vec_to_point = curve_point - point
    vec_along_curve = reference_point - curve_point
    
    norm1 = np.sqrt(np.dot(vec_to_point, vec_to_point))
    norm2 = np.sqrt(np.dot(vec_along_curve, vec_along_curve))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0, norm1
    
    cos_angle = np.dot(vec_to_point, vec_along_curve) / (norm1 * norm2)
    # Clamp cos_angle to [-1, 1] to avoid numerical issues with arccos
    if cos_angle > 1.0:
        cos_angle = 1.0
    elif cos_angle < -1.0:
        cos_angle = -1.0
    angle_rad = np.arccos(cos_angle)
    angle_deg = angle_rad * (180.0 / np.pi)
    
    # Determine sign using cross product
    cross = np.cross(vec_to_point, vec_along_curve)
    if cross[2] < 0:
        angle_deg = 360 - angle_deg
    
    return angle_deg, norm1


@njit
def compute_reference_points_and_arc_lengths(
    curve_points: np.ndarray,
    tangent_vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reference points (for angle calculation) and arc lengths.
    
    Args:
        curve_points: Points along the centerline curve.
        tangent_vectors: Tangent vectors at each point.
        
    Returns:
        Tuple of (reference_points, arc_lengths).
    """
    n = curve_points.shape[0]
    ref_points = np.empty_like(curve_points)
    arc_lengths = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        B = curve_points[i]
        T = tangent_vectors[i]
        
        # Compute reference point (perpendicular in horizontal plane, offset by 1 in z)
        denom = T[0] ** 2 + T[1] ** 2
        if denom > 0:
            lambda_ = -T[2] / denom
            ref_points[i] = B + lambda_ * np.array([T[0], T[1], 0]) + np.array([0, 0, 1])
        else:
            ref_points[i] = B + np.array([0, 0, 1])
        
        # Compute cumulative arc length
        if i > 0:
            arc_lengths[i] = arc_lengths[i - 1] + np.linalg.norm(B - curve_points[i - 1])
    
    return ref_points, arc_lengths


# =============================================================================
# Cylindrical Coordinate Transformation
# =============================================================================

def transform_to_cylindrical(
    points_xyz: np.ndarray,
    x_params: np.ndarray,
    y_params: np.ndarray,
    z_params: np.ndarray,
    ring_count: int,
    diameter: float = DEFAULT_TUNNEL_DIAMETER,
    samples_per_ring: int = DEFAULT_SAMPLES_PER_RING,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_jobs: int = DEFAULT_NUM_JOBS
) -> np.ndarray:
    """
    Transform points to cylindrical coordinates (r, θ, h).
    
    Args:
        points_xyz: (N, 3) array of 3D points.
        x_params, y_params, z_params: Polynomial coefficients for centerline.
        ring_count: Number of rings in the tunnel.
        diameter: Tunnel diameter for angle scaling.
        samples_per_ring: Samples per ring for arc length calculation.
        batch_size: Batch size for parallel processing.
        num_jobs: Number of parallel jobs.
        
    Returns:
        (N, 3) array of cylindrical coordinates [r, theta, h].
    """
    # Sample the curve densely
    num_samples = ring_count * samples_per_ring
    t_samples = np.linspace(-20, ring_count + 20, num_samples)
    
    # Precompute curve geometry
    curve_points = evaluate_curve(t_samples, x_params, y_params, z_params)
    tangent_vectors = evaluate_curve_derivative(t_samples, x_params, y_params, z_params)
    ref_points, arc_lengths = compute_reference_points_and_arc_lengths(
        curve_points, tangent_vectors
    )
    
    # Build spatial index for nearest neighbor search
    index = faiss.IndexFlatL2(3)
    index.add(curve_points.astype(np.float32))
    
    def process_batch(batch: np.ndarray) -> List[Tuple[float, float, float]]:
        """Process a batch of points."""
        _, indices = index.search(batch.astype(np.float32), 1)
        results = []
        
        for i, idx in enumerate(indices.flatten()):
            angle, distance = compute_angle_and_distance(
                batch[i], curve_points[idx], ref_points[idx]
            )
            arc_length = arc_lengths[idx]
            results.append((distance, angle, arc_length))
        
        return results
    
    # Process in batches
    num_batches = (len(points_xyz) + batch_size - 1) // batch_size
    batches = np.array_split(points_xyz, num_batches)
    
    results = Parallel(n_jobs=num_jobs)(
        delayed(process_batch)(batch) 
        for batch in tqdm(batches, desc="Computing cylindrical coordinates")
    )
    
    # Flatten results
    cylindrical = []
    for batch_result in results:
        cylindrical.extend(batch_result)
    
    cylindrical = np.array(cylindrical)
    
    # Convert angle to arc length on circumference
    cylindrical[:, 1] *= (np.pi * diameter / 360)
    
    return cylindrical


# =============================================================================
# Main Pipeline
# =============================================================================

def unfold_tunnel(tunnel_id: str, base_dir: str = "data") -> None:
    """
    Execute the complete tunnel unfolding pipeline.
    
    Args:
        tunnel_id: Identifier for the tunnel (e.g., "1-4", "5-1").
        base_dir: Base directory for data files.
    """
    print(f"Processing tunnel: {tunnel_id}")
    
    # Load parameters
    params, params_loaded = load_parameters(tunnel_id, base_dir)
    
    # Extract parameters - use defaults ONLY if no file was loaded
    allow_defaults = not params_loaded
    ring_spacing = get_param(params, 'physical_constants', 'ring_spacing', default=DEFAULT_RING_SPACING, allow_default=allow_defaults)
    tunnel_diameter = get_param(params, 'physical_constants', 'tunnel_diameter', default=DEFAULT_TUNNEL_DIAMETER, allow_default=allow_defaults)
    slice_half_thickness = get_param(params, 'slicing', 'slice_half_thickness', default=DEFAULT_SLICE_HALF_THICKNESS, allow_default=allow_defaults)
    max_distance_from_top = get_param(params, 'slicing', 'max_distance_from_top', default=DEFAULT_MAX_DISTANCE_FROM_TOP, allow_default=allow_defaults)
    polynomial_degree = get_param(params, 'curve_fitting', 'polynomial_degree', default=DEFAULT_POLYNOMIAL_DEGREE, allow_default=allow_defaults)
    ransac_inlier_ratio = get_param(params, 'ransac_ellipse', 'inlier_ratio', default=DEFAULT_RANSAC_INLIER_RATIO, allow_default=allow_defaults)
    ransac_confidence = get_param(params, 'ransac_ellipse', 'confidence', default=DEFAULT_RANSAC_CONFIDENCE, allow_default=allow_defaults)
    ransac_min_samples = get_param(params, 'ransac_ellipse', 'min_samples', default=DEFAULT_RANSAC_MIN_SAMPLES, allow_default=allow_defaults)
    ransac_inlier_threshold = get_param(params, 'ransac_ellipse', 'inlier_threshold', default=DEFAULT_RANSAC_INLIER_THRESHOLD, allow_default=allow_defaults)
    samples_per_ring = get_param(params, 'arc_length', 'samples_per_ring', default=DEFAULT_SAMPLES_PER_RING, allow_default=allow_defaults)
    batch_size = get_param(params, 'performance', 'batch_size', default=DEFAULT_BATCH_SIZE, allow_default=allow_defaults)
    num_jobs = get_param(params, 'performance', 'num_jobs', default=DEFAULT_NUM_JOBS, allow_default=allow_defaults)
    
    # Load point cloud
    filepath = os.path.join(base_dir, f"{tunnel_id}.txt")
    df = load_point_cloud(filepath)
    points_xyz = df[['x', 'y', 'z']].values
    
    # Step 1: Determine tunnel direction
    print("Step 1: Computing tunnel direction...")
    center1, center2 = compute_tunnel_direction(points_xyz[:, :2])
    
    # Step 2: Generate slicing planes
    print("Step 2: Generating slicing planes...")
    origins, planes, sliced_clouds = generate_slicing_planes(
        center1, center2, points_xyz, delta=slice_half_thickness,
        ring_spacing=ring_spacing
    )
    ring_count = len(sliced_clouds)
    print(f"Generated {ring_count} slices (ring_spacing={ring_spacing})")
    
    # Step 3: Fit ellipse centers
    print("Step 3: Fitting ellipse centers...")
    centers_3d = fit_ellipse_centers(
        sliced_clouds, origins, planes,
        max_distance=max_distance_from_top,
        inlier_ratio=ransac_inlier_ratio,
        confidence=ransac_confidence,
        min_samples=ransac_min_samples,
        inlier_threshold=ransac_inlier_threshold
    )
    
    # Step 4: Fit centerline curve
    print("Step 4: Fitting centerline curve...")
    x_params, y_params, z_params = fit_centerline_curve(centers_3d, degree=polynomial_degree)
    
    # Step 5: Transform to cylindrical coordinates
    print("Step 5: Transforming to cylindrical coordinates...")
    cylindrical = transform_to_cylindrical(
        points_xyz, x_params, y_params, z_params, ring_count,
        diameter=tunnel_diameter, samples_per_ring=samples_per_ring,
        batch_size=batch_size, num_jobs=num_jobs
    )
    
    # Add cylindrical coordinates to dataframe
    df['r'] = cylindrical[:, 0]
    df['theta'] = cylindrical[:, 1]
    df['h'] = cylindrical[:, 2]
    
    # Save results
    output_dir = os.path.join(base_dir, tunnel_id)
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(os.path.join(output_dir, 'unwrapped.csv'), index=False)
    with open(os.path.join(output_dir, 'ring_count.txt'), 'w') as f:
        f.write(str(ring_count))
    
    print(f"Results saved to {output_dir}/")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 1_unfolding.py <tunnel_id>")
        print("Example: python 1_unfolding.py 1-4")
        sys.exit(1)

    unfold_tunnel(sys.argv[1])

