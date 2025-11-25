# Algorithm 1 - Tunnel Centre Line Extraction extracted from notebook

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import cv2
import random
import time
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from numba import njit, prange
import faiss
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import os
import math
import sys
import json

# Check if tunnel_id is provided
if len(sys.argv) != 2:
    print("Usage: python 1_upfolding.py <tunnel_id>")
    print("Example: python 1_upfolding.py 1-4")
    sys.exit(1)

tunnel_id = sys.argv[1]

# Load parameters
def load_parameters(tunnel_id):
    """Load parameters from configurable directory where analyst saves parameters"""
    
    # Determine script directory to handle both project root and configurable execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.join(script_dir, tunnel_id, 'parameters_unfolding.json')
    
    if os.path.exists(param_file):
        try:
            with open(param_file, 'r') as f:
                params = json.load(f)
            print(f"✅ Loaded parameters from configurable/{tunnel_id}/parameters_unfolding.json")
            return params, param_file
        except Exception as e:
            print(f"❌ Error loading parameters: {e}")
            sys.exit(1)
    else:
        print(f"❌ Error: Parameter file not found at configurable/{tunnel_id}/parameters_unfolding.json")
        print("Please run the analyst to generate parameters first.")
        sys.exit(1)

# Load configuration
params, param_file = load_parameters(tunnel_id)
expected_keys = [
    "delta",
    "slice_spacing_factor",
    "vertical_filter_window",
    "ransac_threshold",
    "ransac_probability",
    "ransac_inlier_ratio",
    "ransac_sample_size",
    "polynomial_degree",
    "num_samples_factor",
    "diameter",
]

for key in expected_keys:
    if key not in params:
        print(f"❌ Error: Missing required parameter '{key}' in {param_file}")
        sys.exit(1)

delta = params["delta"]
slice_spacing_factor = params["slice_spacing_factor"]
vertical_filter_window = params["vertical_filter_window"]
ransac_threshold = params["ransac_threshold"]
ransac_probability = params["ransac_probability"]
ransac_inlier_ratio = params["ransac_inlier_ratio"]
ransac_sample_size = params["ransac_sample_size"]
polynomial_degree = params["polynomial_degree"]
num_samples_factor = params["num_samples_factor"]
diameter = params["diameter"]

print(f"Using parameters: delta={delta}, slice_spacing_factor={slice_spacing_factor}, vertical_filter_window={vertical_filter_window}, diameter={diameter}")
# Determine if we're running from project root or configurable/
if os.path.exists(f"data/{tunnel_id}.txt"):
    base_dir = "data/"  # Running from project root
else:
    base_dir = "../data/"  # Running from configurable/
point_cloud_data = np.loadtxt(os.path.join(base_dir, f"{tunnel_id}.txt")) # file name

print(f"Processing tunnel: {tunnel_id}")

# Check the size of the point cloud data
# The sample data should consist of six columns 
# including x, y, z, intensity, block type, and ring number
points_xyz = point_cloud_data[:, :3]
intensity = point_cloud_data[:, 3]
segment = point_cloud_data[:, 4].astype(int)
ring = point_cloud_data[:, 5].astype(int)

df_point_cloud = pd.DataFrame({
    'x': points_xyz[:, 0],
    'y': points_xyz[:, 1],
    'z': points_xyz[:, 2],
    'intensity': intensity,
    'segment': segment,
    'ring': ring
})

# ## 1. Determine direction vector

# Obtain minimum bounding rectangle for the 2D XOY projection
points_2d_xoy = points_xyz[:, :2]
convex_hull = ConvexHull(points_2d_xoy)
convex_hull_points = points_2d_xoy[convex_hull.vertices]
convex_polygon = Polygon(convex_hull_points)
min_bounding_rect = convex_polygon.minimum_rotated_rectangle

# Get the four vertices of the minimum bounding rectangle
rect_vertices = np.array(min_bounding_rect.exterior.coords)[:-1]  # Drop the redundant last point

# Calculate the lengths of edges
edges = [np.linalg.norm(rect_vertices[i] - rect_vertices[(i + 1) % 4]) for i in range(4)]
short_edge_index = np.argmin(edges)

# Determine the centers of the two short sides
center1 = (rect_vertices[short_edge_index] + rect_vertices[(short_edge_index + 1) % 4]) / 2
center2 = (rect_vertices[(short_edge_index + 2) % 4] + rect_vertices[(short_edge_index + 3) % 4]) / 2

vector = center2 - center1

# ## 2. Generate point cloud slices

def generate_slicing_planes_point_cloud(center1, center2, points_xyz, delta):
    """
    Generate slicing planes and point cloud slices along the line segment between two points.

    Parameters:
    center1 (array-like): Starting point of the line segment.
    center2 (array-like): Ending point of the line segment.
    points_xyz (numpy array): The point cloud data.
    delta (float): Half the thickness of slices.

    Returns:
    origin (list of numpy arrays): List of 3D coordinates for each plane.
    planes (list of numpy arrays): List of plane equations [A, B, C, D].
    slicing_cloud (list of lists of numpy arrays): List of sliced point clouds for each plane.
    """
    # Calculate the distance between center1 and center2 in the XY plane
    l = np.linalg.norm(center2[:2] - center1[:2])
    
    # Find the optimal integer n such that slice_spacing_factor * n is closest to l
    n = round(l / slice_spacing_factor)
    min_diff = abs(l - slice_spacing_factor * n)
    optimal_n = n

    # Check nearby integer values for better match
    for candidate_n in [n - 1, n + 1]:
        diff = abs(l - slice_spacing_factor * candidate_n)
        if diff < min_diff:
            optimal_n = candidate_n
            min_diff = diff

    # Initialize lists to hold points and planes
    points_on_plane = []
    planes = []
    origin = []

    # Calculate direction vector and total distance
    direction = (center2 - center1) / np.linalg.norm(center2 - center1)
    total_distance = np.linalg.norm(center2 - center1)
    first_distance = total_distance / (2 * optimal_n)
    last_distance = total_distance - first_distance

    # Generate planes
    for i in range(optimal_n):
        if i == 0:
            segment_length = first_distance
        elif i == optimal_n - 1:
            segment_length = last_distance
        else:
            segment_length = first_distance + (i * (last_distance - first_distance)) / (optimal_n - 1)

        point_on_plane = center1 + (segment_length / total_distance) * (center2 - center1)
        points_on_plane.append(point_on_plane)
        origin.append(np.append(point_on_plane, 0))  # Convert to 3D by adding a zero z-component

        normal_vector = np.append(direction, 0)
        d = -np.dot(normal_vector[:2], point_on_plane)
        plane = np.append(normal_vector, d)
        planes.append(plane)

    # Initialize slicing cloud list
    slicing_cloud = []
    Delta = delta  # Half the thickness of slices

    points_xyz = np.asarray(points_xyz)  # Ensure points_xyz is a numpy array

    # Iterate over each plane with progress bar
    for plane in tqdm(planes, desc="Processing planes"):
        a, b, c, d = plane
        Wr = a * points_xyz[:, 0] + b * points_xyz[:, 1] + c * points_xyz[:, 2] + d - Delta
        Wl = a * points_xyz[:, 0] + b * points_xyz[:, 1] + c * points_xyz[:, 2] + d + Delta
        mask = (Wr * Wl <= 0)
        slicing_cloud.append(points_xyz[mask])

    return origin, planes, slicing_cloud
origin, planes, slicing_cloud = generate_slicing_planes_point_cloud(center1, center2, points_xyz, delta)
ring_count = len(slicing_cloud)

# ## 3. Ellipse centre fitting of Cloud<sub>Slices</sub>

def project_to_plane(point_cloud, center, normal):
    '''
    Project a 3D point cloud onto a known plane, and convert the projected points to 2D coordinates.
    The origin (0, 0) in 2D corresponds to the `center` point in 3D.

    Args:
        point_cloud: numpy array, shape (N, 3). Represents a 3D point cloud.
        center: numpy array, shape (3,). Represents a point on the plane, which will be the origin in the 2D projection.
        normal: numpy array, shape (3,). Represents the normal vector of the plane.

    Returns:
        numpy array, shape (N, 2). Represents the 2D coordinates of the projected points on the plane.
    '''
    
    # Move the center of the point cloud to the origin.
    shifted_point_cloud = np.array(point_cloud) - np.array(center)
    
    # Compute the projection of the point cloud onto the plane's normal vector.
    projection = np.dot(shifted_point_cloud, normal)
    
    # Compute the coordinates of the projected points onto the plane.
    projected_points = shifted_point_cloud - np.outer(projection, normal)
    
    # Define the 2D coordinate system on the plane:
    x_axis = np.array([-normal[1], normal[0], 0])
    x_axis /= np.linalg.norm(x_axis)

    # y-axis is orthogonal to both the normal and x_axis.
    y_axis = np.cross(normal, x_axis)

    # Calculate the 2D coordinates by projecting the 3D points onto the x and y axes.
    x_coords = np.dot(projected_points, x_axis)
    y_coords = np.dot(projected_points, y_axis)
    
    return np.vstack((x_coords, y_coords)).T

# Define the normal vector and center for projection
normal = np.array([planes[0][0], planes[0][1], 0])

# Project each point in slicing_cloud onto the plane
point2ds = []
for i in range(len(origin)):
    point2ds_temp = project_to_plane(slicing_cloud[i], origin[i], normal)
    point2ds.append(point2ds_temp)

# Process each set of 2D points
filtered_point2ds = []
for points in point2ds:
    # Find the maximum y-coordinate
    y_max = max(point[1] for point in points)
    # Filter points where y-coordinate is within vertical_filter_window units of y_max
    filtered_points = [point for point in points if abs(point[1] - y_max) <= vertical_filter_window]
    filtered_point2ds.append(filtered_points)

class RANSAC:
    def __init__(self, data, threshold, P, S, N):
        self.point_data = data  # Ellipse contour points
        self.error_threshold = threshold  # Error tolerance threshold
        self.N = N  # Number of points to sample
        self.S = S  # Inlier ratio
        self.P = P  # Probability of finding a correct model
        self.max_inliers = len(data) * S  # Maximum number of inliers
        self.items = 999  # Number of iterations
        self.count = 0  # Number of inliers
        self.best_model = ((0, 0), (1e-6, 1e-6), 0)  # Best ellipse model

    def random_sampling(self, n):
        """Randomly select n data points."""
        return np.asarray(random.sample(list(self.point_data), n))

    def Geometric2Conic(self, ellipse):
        """Convert ellipse parameters to conic coefficients."""
        (x0, y0), (bb, aa), phi_b_deg = ellipse
        a, b = aa / 2, bb / 2  # Semi-major and semi-minor axes
        phi_b_rad = np.radians(phi_b_deg)  # Convert angle to radians
        ax, ay = -np.sin(phi_b_rad), np.cos(phi_b_rad)  # Major axis unit vector

        # Conic parameters
        a2, b2 = a * a, b * b
        if a2 > 0 and b2 > 0:
            A = ax * ax / a2 + ay * ay / b2
            B = 2 * ax * ay / a2 - 2 * ax * ay / b2
            C = ay * ay / a2 + ax * ax / b2
            D = (-2 * ax * ay * y0 - 2 * ax * ax * x0) / a2 + (2 * ax * ay * y0 - 2 * ay * ay * x0) / b2
            E = (-2 * ax * ay * x0 - 2 * ay * ay * y0) / a2 + (2 * ax * ay * x0 - 2 * ax * ax * y0) / b2
            F = (2 * ax * ay * x0 * y0 + ax * ax * x0 * x0 + ay * ay * y0 * y0) / a2 + \
                (-2 * ax * ay * x0 * y0 + ay * ay * x0 * x0 + ax * ax * y0 * y0) / b2 - 1
        else:
            A, B, C, D, E, F = 1, 0, 1, 0, 0, -1e-6  # Default for degenerate cases

        return np.array([A, B, C, D, E, F])

    def eval_model(self, ellipse):
        """Evaluate the ellipse model and count inliers."""
        a, b, c, d, e, f = self.Geometric2Conic(ellipse)
        E = 4 * a * c - b * b
        if E <= 0:
            return 0, np.array([])  # Not an ellipse

        (x, y), (LAxis, SAxis), Angle = ellipse
        LAxis, SAxis = LAxis / 2, SAxis / 2
        if SAxis > LAxis:
            SAxis, LAxis = LAxis, SAxis  # Ensure LAxis is the longer one

        # Calculate foci
        Axis = math.sqrt(LAxis**2 - SAxis**2)
        f1_x = x - Axis * math.cos(math.radians(Angle))
        f1_y = y - Axis * math.sin(math.radians(Angle))
        f2_x = x + Axis * math.cos(math.radians(Angle))
        f2_y = y + Axis * math.sin(math.radians(Angle))

        # Compute distances to foci
        f1, f2 = np.array([f1_x, f1_y]), np.array([f2_x, f2_y])
        f1_distance = np.sum((self.point_data - f1)**2, axis=1)
        f2_distance = np.sum((self.point_data - f2)**2, axis=1)
        all_distance = np.sqrt(f1_distance) + np.sqrt(f2_distance)

        # Identify inliers
        Z = np.abs(2 * LAxis - all_distance)
        delta = np.sqrt(np.mean((Z - np.mean(Z))**2))
        inliers = np.where(Z < 0.8 * delta)[0]
        inlier_points = self.point_data[inliers]

        return len(inlier_points), inlier_points

    def execute_ransac(self):
        """Run RANSAC algorithm to fit an ellipse."""
        while math.ceil(self.items):
            # Randomly sample N points
            select_points = self.random_sampling(self.N)
            select_points_list = [(point[0], point[1]) for point in select_points]

            # Fit an ellipse to the selected points
            ellipse = cv2.fitEllipse(np.array(select_points_list, dtype=np.float32))

            # Evaluate the model and find inliers
            inliers_count, inliers_set = self.eval_model(ellipse)
            inliers_set = np.array([tuple(point) for point in inliers_set], dtype=np.float32)

            # Update the best model if current inliers are better
            if inliers_count > self.count:
                self.count = inliers_count
                self.best_model = cv2.fitEllipse(inliers_set)  # Fit ellipse on inliers

                # Check if we have reached the expected number of inliers
                if self.count > self.max_inliers:
                    break

                # Update number of iterations
                self.items = math.log(1 - self.P) / math.log(1 - (inliers_count / len(self.point_data))**self.N)

        return self.best_model, inliers_set

# Initialize lists to store ellipse centers
X_center = []
Y_center = []
LAxis_sets = []
SAxis_sets = []
Angle_sets = []
in_sets = []

for i in range(len(slicing_cloud)):
    # Prepare point data for RANSAC
    points_data = np.reshape(filtered_point2ds[i], (-1, 2))  # Ellipse edge points

    # First RANSAC fit to find initial inliers
    ransac = RANSAC(data=points_data, threshold=ransac_threshold, P=ransac_probability, S=ransac_inlier_ratio, N=ransac_sample_size)
    _, inliers_set = ransac.execute_ransac()

    # Refine fit using inliers from the first RANSAC
    points_data = np.reshape(inliers_set, (-1, 2))
    ransac = RANSAC(data=points_data, threshold=ransac_threshold, P=ransac_probability, S=ransac_inlier_ratio, N=ransac_sample_size)
    ellipse_params, _ = ransac.execute_ransac()

    # Extract center coordinates
    ((X, Y), (LAxis, SAxis), Angle) = ellipse_params

    X_center.append(X)
    Y_center.append(Y)
    LAxis_sets.append(LAxis)
    SAxis_sets.append(SAxis)
    Angle_sets.append(Angle)
    in_sets.append(inliers_set)

def get_3dcoordinates_from_plane(point2d,plane_params,origin):
    """
    Computes the coordinates of a point in 3D space given its coordinates in the plane coordinate system.
    
    Args:
        point2d: numpy array, shape (2,). Represents 2D points in the plane.
        plane_params: numpy array, shape (4,). Represents the parameters of the known plane.
        origin: numpy array, shape (3,). Represents the 3D coordinates of the origin of the known plane.

    Returns:
        numpy array, shape (N, 3). Represents the 3D coordinates of the 2D points.
    """
    xp,yp = point2d
    A,B,C,D = plane_params
    x0,y0,z0 = origin
    
    # normal vector of the plane
    N = np.array([A, B, C])
    N = N / np.linalg.norm(N)
    
    # calculate vector V, which is the x-axis of the 2D coordinate system
    Vx = -B
    Vy = A
    Vz = 0
    V = np.array([Vx, Vy, Vz])
    V = V / np.linalg.norm(V)
    
    # calculate vector U, which is the y-axis of the 2D coordinate system
    U = np.cross(N, V)
    U = U / np.linalg.norm(U)
    
    # calculate 3D coordinates
    x = x0+xp*V[0]+yp*U[0]
    y = y0+xp*V[1]+yp*U[1]
    z = z0+xp*V[2]+yp*U[2]
    
    return [x,y,z]

# Initialize list to store 3D coordinates
cps = []

# Compute 3D coordinates for each center point
for i in range(len(slicing_cloud)):
    point2d_cp = np.array([X_center[i], Y_center[i]])
    cp = get_3dcoordinates_from_plane(point2d_cp, planes[i], origin[i])
    cps.append(cp)

# Construct final list of coordinates
cps_arr= np.array(cps)

# ## 4. 3D Curve Curve<sub>centre</sub> fitting

# Generate parameter t for each point (using indices as parameter t)
t = np.arange(ring_count)

# Polynomial feature expansion
degree = polynomial_degree
poly = PolynomialFeatures(degree)

# Polynomial feature transformation for x(t), y(t), z(t)
t_poly = poly.fit_transform(t.reshape(-1, 1))
x_poly = t_poly
y_poly = t_poly
z_poly = t_poly

# Initialize RANSAC Regressor for x, y, z
ransac_x = RANSACRegressor()
ransac_y = RANSACRegressor()
ransac_z = RANSACRegressor()

# Fit the RANSAC model to x, y, z coordinates
ransac_x.fit(x_poly, cps_arr[:, 0])
ransac_y.fit(y_poly, cps_arr[:, 1])
ransac_z.fit(z_poly, cps_arr[:, 2])

# Get polynomial coefficients and intercepts
x_coef = ransac_x.estimator_.coef_
y_coef = ransac_y.estimator_.coef_
z_coef = ransac_z.estimator_.coef_
x_intercept = ransac_x.estimator_.intercept_
y_intercept = ransac_y.estimator_.intercept_
z_intercept = ransac_z.estimator_.intercept_

# Adjust coefficients to include the intercept term
x_params = x_coef.copy()
x_params[0] = x_intercept

y_params = y_coef.copy()
y_params[0] = y_intercept

z_params = z_coef.copy()
z_params[0] = z_intercept

@njit
def poly_eval(coeffs, x):
    result = 0.0
    for coeff in coeffs:
        result = result * x + coeff
    return result

@njit(parallel=True)
def curve_func(t, x_params, y_params, z_params):
    result = np.empty((len(t), 3))
    for i in prange(len(t)):
        result[i, 0] = poly_eval(x_params[::-1], t[i])
        result[i, 1] = poly_eval(y_params[::-1], t[i])
        result[i, 2] = poly_eval(z_params[::-1], t[i])
    return result

@njit
def poly_deriv(coeffs):
    return np.array([i * c for i, c in enumerate(coeffs[:0:-1])][::-1])

@njit(parallel=True)
def curve_deriv(t, x_params, y_params, z_params):
    result = np.empty((len(t), 3))
    dx_params = poly_deriv(x_params[::-1])
    dy_params = poly_deriv(y_params[::-1])
    dz_params = poly_deriv(z_params[::-1])
    for i in prange(len(t)):
        result[i, 0] = poly_eval(dx_params, t[i])
        result[i, 1] = poly_eval(dy_params, t[i])
        result[i, 2] = poly_eval(dz_params, t[i])
    return result
    
@njit
def calculate_angle_with_direction(A, B, C):
    ''' 
    A is point from point cloud, 
    B is the closest point of A on the curve, 
    Angle ABC is angle value of point A in cylindrical coordinates
    '''
    AB = B - A
    BC = C - B
    
    dot_product = np.dot(AB, BC)
    norm_AB = np.sqrt(np.dot(AB, AB))
    norm_BC = np.sqrt(np.dot(BC, BC))
    
    if norm_AB == 0 or norm_BC == 0:
        return 0.0, np.linalg.norm(AB)
    
    cos_angle = dot_product / (norm_AB * norm_BC)
    angle_radians = np.arccos(cos_angle)
    angle_degrees = angle_radians * (180.0 / np.pi)
    cross_product = np.cross(AB, BC)
    if cross_product[2] < 0:
        angle_degrees = 360 - angle_degrees

    return angle_degrees, np.linalg.norm(AB)

@njit
def compute_C_points_and_arc_length(B_points, T_vectors, arc_lengths):
    C_points = np.empty_like(B_points)
    for i in range(B_points.shape[0]):
        B = B_points[i]
        T = T_vectors[i]
        lambda_ = -T[2] / (T[0]**2 + T[1]**2)
        C = B + lambda_ * np.array([T[0], T[1], 0]) + np.array([0, 0, 1])
        C_points[i] = C

        # Compute arc length
        if i > 0:
            prev_B = B_points[i-1]
            arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(B - prev_B)
        else:
            arc_lengths[i] = 0.0
    
    return C_points, arc_lengths

# Precompute the curve points, derivatives, C_points, and arc lengths based on B_points
num_samples = ring_count * num_samples_factor # around 1mm accuracy
t_samples = np.linspace(-20, ring_count + 20, num_samples)
B_points = curve_func(t_samples, x_params, y_params, z_params)
T_vectors = curve_deriv(t_samples, x_params, y_params, z_params)
arc_lengths = np.zeros(num_samples, dtype=np.float32)

C_points, arc_lengths = compute_C_points_and_arc_length(B_points, T_vectors, arc_lengths)

# Build Faiss index
index = faiss.IndexFlatL2(3)
index.add(B_points)

# Define batch size for Faiss search to improve performance
batch_size = 1000000  # Adjust batch size based on memory constraints

def process_batch(points_batch):
    ''' Process a batch of points to find nearest neighbors, angles, and distances '''
    _, idx_batch = index.search(points_batch, 1)
    results = []
    
    for i in range(points_batch.shape[0]):
        A = points_batch[i]
        idx = idx_batch[i][0]
        B = B_points[idx]
        C = C_points[idx]
        angle_ABC, distance_AB = calculate_angle_with_direction(A, B, C)
        arc_length_B = arc_lengths[idx]
        results.append((distance_AB, angle_ABC, arc_length_B))
    
    return results

# Split points into batches for parallel processing
num_batches = (len(points_xyz) + batch_size - 1) // batch_size
points_batches = np.array_split(points_xyz, num_batches)

# Using Joblib for parallel batch processing
cylindrical_coords_batches = Parallel(n_jobs=12)(
    delayed(process_batch)(batch) for batch in tqdm(points_batches, desc="Calculating cylindrical coordinates", total=len(points_batches))
)

cylindrical_coords = []
for batch_result in cylindrical_coords_batches:
    cylindrical_coords.extend(batch_result)

# recording data
df_point_cloud['r'] = np.array(cylindrical_coords)[:,0]
df_point_cloud['theta'] = np.array(cylindrical_coords)[:,1]* (np.pi*diameter / 360)
df_point_cloud['h'] = np.array(cylindrical_coords)[:,2]

# save to data tunnel 5-1
os.makedirs(f'data/{tunnel_id}', exist_ok=True)
df_point_cloud.to_csv(f'data/{tunnel_id}/unwrapped.csv',index=False)
# save ring count
with open(f'data/{tunnel_id}/ring_count.txt', 'w') as f:
    f.write(str(ring_count))

