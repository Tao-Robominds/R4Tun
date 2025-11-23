# Algorithm 3 - Geometry Guided Enhancing extracted from notebook

# # Algorithm 3: geometry guided enhancing

import os
import pandas as pd
import numpy as np
from scipy.spatial import KDTree, cKDTree
import numba as nb
from numba import njit, prange
from scipy.interpolate import griddata
from tqdm.notebook import tqdm
from collections import defaultdict
import pickle
import sys
import json

# Check if tunnel_id is provided
if len(sys.argv) != 2:
    print("Usage: python 3_enhancing.py <tunnel_id>")
    print("Example: python 3_enhancing.py 1-4")
    sys.exit(1)

tunnel_id = sys.argv[1]

# Load parameters
def load_parameters(tunnel_id):
    """Load parameters from data directory where analyst saves parameters"""
    
    data_param_file = os.path.join('data', tunnel_id, 'parameters', 'parameters_enhancing.json')
    if os.path.exists(data_param_file):
        try:
            with open(data_param_file, 'r') as f:
                params = json.load(f)
            print(f"✅ Loaded parameters from data/{tunnel_id}/parameters/parameters_enhancing.json")
            return params
        except Exception as e:
            print(f"❌ Error loading parameters: {e}")
            sys.exit(1)
    else:
        print(f"❌ Error: Parameter file not found at data/{tunnel_id}/parameters/parameters_enhancing.json")
        print("Please run the analyst to generate parameters first.")
        sys.exit(1)

# Load configuration
params = load_parameters(tunnel_id)
upsampling_stage1_target_distance = params["upsampling_stage1_target_distance"]
upsampling_stage2_target_distance = params["upsampling_stage2_target_distance"]
upsampling_stage3_target_distance = params["upsampling_stage3_target_distance"]
curvature_threshold = params["curvature_threshold"]
depth_threshold_low = params["depth_threshold_low"]
depth_threshold_high = params["depth_threshold_high"]
inter_radius = params["inter_radius"]
duplicate_threshold = params["duplicate_threshold"]
n_segment_start = params["n_segment_start"]
n_segment_end = params["n_segment_end"]
num_neighbors = params["num_neighbors"]
num_interpolations = params["num_interpolations"]
resolution = params["resolution"]
window_size = params["window_size"]

print(f"Using parameters: stage1_distance={upsampling_stage1_target_distance}, stage2_distance={upsampling_stage2_target_distance}, stage3_distance={upsampling_stage3_target_distance}")
# Determine if we're running from project root or agents/configurable/
if os.path.exists(f"data/{tunnel_id}/denoised.csv"):
    base_dir = f"data/{tunnel_id}/"  # Running from project root
else:
    base_dir = f"../../data/{tunnel_id}/"  # Running from agents/configurable/
denoised_file = os.path.join(base_dir, "denoised.csv")
df_point_cloud = pd.read_csv(denoised_file)

print(f"Processing tunnel: {tunnel_id}")

# Cell 1
df_support_filtered = df_point_cloud[df_point_cloud['pred'] != 0]
df_support_filtered.tail()

# Cell 2
# curvature calculation or you can use cloudcompare
import numpy as np
from scipy.spatial import KDTree
import numba as nb
from numba import njit, prange

@njit(parallel=True)
def calculate_curvatures(points, indices, k):
    curvatures = np.zeros(len(points))
    for i in prange(len(points)):
        neighbors = points[indices[i, 1:]]
        cov_matrix = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)
    return curvatures

def compute_curvature(df, k=20):
    points = df[['x', 'y', 'z']].values
    tree = KDTree(points)
    
    _, indices = tree.query(points, k=k+1)
    
    # curvature calculation
    curvatures = calculate_curvatures(points, indices, k)
    
    df = df.copy()  # Create a copy to ensure we're working with a new DataFrame
    df.loc[:, 'curvature'] = curvatures
    return df

df_support_filtered_curva = compute_curvature(df_support_filtered)
df_support_filtered_curva.head()

# ## 1. enhance the surface of segment

# Cell 5
import time
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import numba as nb
from numba import njit, prange

@njit(parallel=False)
def compute_midpoints_and_filter(points, indices, distances, target_distance, curvature_threshold):
    num_points = len(points)
    max_new_points = num_points * (len(indices[0]) - 1)
    new_points = np.zeros((max_new_points, points.shape[1]), dtype=np.float64)
    new_points_count = 0
    
    for i in nb.prange(len(points)):
        for j in range(1, len(indices[i])):
            dist = distances[i, j]
            idx = indices[i, j]
            curvature_diff = abs(points[i, 3] - points[idx, 3])
            if 0.9 * target_distance <= dist <= 2 * target_distance and curvature_diff <= curvature_threshold:
                mid_point = (points[i, :2] + points[idx, :2]) / 2
                mid_r = (points[i, 2] + points[idx, 2]) / 2
                mid_curvature = (points[i, 3] + points[idx, 3]) / 2
                mid_intensity = (points[i, 4] + points[idx, 4]) / 2
                new_point = np.array([mid_point[0], mid_point[1], mid_r, mid_curvature, mid_intensity])

                new_points[new_points_count] = new_point
                new_points_count += 1
    return new_points[:new_points_count]

@njit(parallel=False)
def _filter_points_to_keep(neighbors_array, valid_mask, num_points):

    keep_indices = np.zeros(num_points, dtype=np.int32)
    count = 0
    removed_indices = np.zeros(num_points, dtype=np.int32)

    for i in prange(num_points):
        if removed_indices[i] == 0:
            keep_indices[count] = i
            count += 1
            # Mark all neighbors as needing removal
            for j in range(neighbors_array.shape[1]):
                neighbor_idx = neighbors_array[i, j]
                if valid_mask[i, j] and removed_indices[neighbor_idx] == 0:
                    removed_indices[neighbor_idx] = 1

    return keep_indices[:count]

def optimized_radius_filter(df, target_distance):
    points = df[['h', 'theta']].values
    r_dist = 0.15 * target_distance
    num_points = len(points)
    tree = cKDTree(points)
    
    neighbors_list = tree.query_ball_point(points, r=r_dist)
    max_neighbors = max(len(neighbors) for neighbors in neighbors_list)
    neighbors_array = np.full((len(points), max_neighbors), -1, dtype=np.int32)
    valid_mask = np.zeros((len(points), max_neighbors), dtype=np.bool_)
    
    for i in range(len(points)):
        length = len(neighbors_list[i])
        neighbors_array[i, :length] = neighbors_list[i]
        valid_mask[i, :length] = True
    
    keep_indices = _filter_points_to_keep(neighbors_array, valid_mask, num_points)
    filtered_df = df.iloc[keep_indices].reset_index(drop=True)
    
    return filtered_df

# -----main function-----
def enhance_segment_surface(df, target_distance=0.08, curvature_threshold_param=0.0005, num_neighbors_param=20):
    start_time = time.time()
    
    print('reading points ...')
    points = df[['h', 'theta', 'r', 'curvature', 'intensity']].values
    original_points = points[:, :2]

    print('KDTree generation ...')
    original_tree = cKDTree(original_points)
    
    distances, indices = original_tree.query(original_points, k=min(num_neighbors_param + 1, len(points)))

    print('midpoint calculation ...')
    all_new_points = compute_midpoints_and_filter(points, indices, distances, target_distance, curvature_threshold_param)

    print('filter out excess points ...')
    distances, _ = original_tree.query(all_new_points[:, :2], k=1)
        
    distances_flat = distances.flatten()
    valid_new_points = all_new_points[distances_flat >= 0.2 * target_distance]
        
    add_point_df = pd.DataFrame(valid_new_points, columns=['h', 'theta', 'r', 'curvature', 'intensity'])
    add_point_df = add_point_df[(add_point_df != 0).any(axis=1)]
            
    add_point_df['pred'] = 8
            
    add_point_df_rf = optimized_radius_filter(add_point_df, target_distance)

    new_df = add_point_df_rf.reset_index(drop=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"insert_midpoints function took {elapsed_time:.2f} seconds with target distance is", target_distance)
    print('The number of newly added interpolation points is', len(new_df))

    return new_df


# Cell 6
# Define the parameters for each upsampling step
upsampling_params = [
    {'target_distance': upsampling_stage1_target_distance},  # First upsampling
    {'target_distance': upsampling_stage2_target_distance},  # Second upsampling
    {'target_distance': upsampling_stage3_target_distance}   # Third upsampling
]

# Initialize the DataFrame for upsampling
df_upsampling_all = df_support_filtered_curva

# Loop through the parameters and perform upsampling
for params in upsampling_params:
    df_upsampling = enhance_segment_surface(df_upsampling_all, 
                                            target_distance=params.get('target_distance'),
                                            curvature_threshold_param=curvature_threshold,
                                            num_neighbors_param=num_neighbors)
    df_upsampling_all = pd.concat([df_upsampling_all, df_upsampling], ignore_index=False)

df_enhance_segment = df_upsampling_all

# ## 2. enhance the outlier points

# Cell 9
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from numba import njit, prange
import time

def enhance_outlier_points(df, depth_threshold_low=0.003, depth_threshold_high=0.008,
                           inter_radius=0.06, num_interpolations=2, duplicate_threshold=0.02, n_segment=[10,21], resolution=0.005):
    """
    Process point cloud data to find points with significant local depth changes,
    and interpolate new points between them to enhance boundaries.

    Parameters:
    - df: DataFrame containing the point cloud data.
    - depth_threshold_low/high: Threshold for depth variation to determine significant points in low/high density area.
    - inter_radius: Distance range between interpolation points. This is mainly decided by the distance from bolt to edge, avoiding unless interpolation.
    - num_interpolations: Number of interpolation points between each pair of points, default is 2.
    - duplicate_threshold: Distance threshold for determining duplicate points.
    - n_segment: Range of high-density area, total 11 rings, 5 before to 5 rings after plus the ring put scanner.
    - resolution: image resolution, like a lower bound to ensure that interpolated points are applied to every pixel.

    Returns:
    - df_upsample: The processed DataFrame including the original points, new interpolated points, and their attributes.
    - meaningful_df: DataFrame containing only the outlier points.
    - new_df: DataFrame containing only the interpolated points
    """
    start_time = time.time()
    
    # Extract relevant columns and values
    print('reading points ...')
    points = df[['h', 'theta', 'r', 'intensity']].values
    points_array = points[:, :3]  # (h, theta, r) coordinates
    z_values = df['r'].values  # z values for depth
    
    # Construct KDTree using (h, theta) coordinates
    print('KDTree generation ...')
    tree = cKDTree(points_array[:, :2])
    
    # Query each point's 20 nearest neighbors (excluding the point itself)
    distances, indices = tree.query(points_array[:, :2], k=21)
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    
    # Define a function to find meaningful points in parallel
    @njit(parallel=True)
    def find_meaningful_indices(points_array, z_values, indices, depth_threshold_low, depth_threshold_high, n_segment):
        meaningful_mask = np.zeros(len(points_array), dtype=np.bool_)
        
        for i in prange(len(points_array)):
            neighbors_indices = indices[i, 1:]  # Exclude the point itself
            
            if len(neighbors_indices) < 20:
                continue
            
            neighbors_z = z_values[neighbors_indices]
            
            # Compute the average local depth difference
            average_diff = points_array[i, 2] - np.mean(neighbors_z)
            
            # If the average depth difference exceeds the threshold, mark as meaningful
            if (x_min + 1.2 * n_segment[0]) <= points_array[i, 0] <= (x_min + 1.2 * n_segment[1]):
                if average_diff > depth_threshold_high:
                    meaningful_mask[i] = True
            else:
                if average_diff > depth_threshold_low:
                    meaningful_mask[i] = True
        
        return meaningful_mask
    
    # Execute the function in numba
    print('searching outlier points ...')
    meaningful_mask = find_meaningful_indices(points_array, z_values, indices, depth_threshold_low, depth_threshold_high, n_segment)
    meaningful_indices = np.where(meaningful_mask)[0]
    print(f"Number of outlier points: {len(meaningful_indices)}")
    
    # Extract a DataFrame of meaningful points
    meaningful_df = df.iloc[meaningful_indices]

    # Define the interpolation function
    @njit(parallel=False)
    def interpolate_points(filtered_indices, points, inter_radius, num_interpolations, duplicate_threshold, resolution):
        num_indices = len(filtered_indices)
        max_new_points = num_indices * num_indices * num_interpolations
        new_points = np.zeros((max_new_points, 4))
        count = 0
    
        for i in prange(num_indices):
            index1 = filtered_indices[i]
            point1 = points[index1]
            x1, y1, z1, i1 = point1
            
            for j in range(i + 1, num_indices):
                index2 = filtered_indices[j]
                point2 = points[index2]
                x2, y2, z2, i2 = point2
                
                # distance filter
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if not (resolution < dist < inter_radius):
                    continue
                
                # interpolation
                for t in np.linspace(0, 1, num=num_interpolations + 2)[1:-1]:
                    new_x = (1 - t) * x1 + t * x2
                    new_y = (1 - t) * y1 + t * y2
                    new_z = (1 - t) * z1 + t * z2
                    new_i = (1 - t) * i1 + t * i2
    
                    # delete too close point
                    if count > 0:
                        dists = np.sqrt((new_points[:count, 0] - new_x) ** 2 + (new_points[:count, 1] - new_y) ** 2)
                        if np.any(dists < duplicate_threshold):
                            continue
    
                    new_points[count] = np.array([new_x, new_y, new_z, new_i])
                    count += 1
    
        return new_points[:count]

    # Generate interpolated points
    print("filter out high density part ...")

    # Get boundary values of the points and filter out high density part
    filtered_high_density_indices = []
    for idx in meaningful_indices:
        x = points[idx, 0]
        if not ((x_min + 1.2 * n_segment[0]) <= x <= (x_min + 1.2 * n_segment[1])):
            filtered_high_density_indices.append(idx)
    
    filtered_indices = np.array(filtered_high_density_indices, dtype=np.int64)
    
    print("Generating interpolated points ...")
    
    new_points_array = interpolate_points(filtered_indices, points, inter_radius, num_interpolations, duplicate_threshold, resolution)
    
    # Add new points to DataFrame
    new_df = pd.DataFrame(new_points_array, columns=['h', 'theta', 'r', 'intensity'])
    new_df['pred'] = 8
    print(f"Number of new added points: {len(new_df)}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"enhance_outlier_points took {elapsed_time:.2f} seconds")
    
    return meaningful_df, new_df


# Cell 10
# =================n_segment need to change!!!!===============
# The sample data is a half of one station, so n_segment should change when using entire station point cloud. 
meaningful_df, new_df = enhance_outlier_points(df_support_filtered_curva, 
                                               depth_threshold_low=depth_threshold_low,
                                               depth_threshold_high=depth_threshold_high,
                                               inter_radius=inter_radius,
                                               num_interpolations=num_interpolations,
                                               duplicate_threshold=duplicate_threshold,
                                               n_segment=[n_segment_start, n_segment_end],
                                               resolution=resolution)

df_enhance_joint = pd.concat([meaningful_df, new_df], ignore_index=False)

# Cell 11
# update pred 0 using meaningful_df, we believe outlier points are belong to background
df_point_cloud.loc[meaningful_df.index, 'pred'] = 0

# Cell 12
df_point_cloud.tail()

# ## 3. projection and record mapping index 

# Cell 15
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from tqdm.notebook import tqdm
from collections import defaultdict

def project_to_depth_map_inter(data1, data2, resolution=0.005, window_size=5, outlier_mode=False):
    """
    Optimized version of the function that projects 2D point cloud data into a depth map.
    This version maintains the original filling logic and interpolation process.

    Parameters:
    data1, data2: pandas DataFrames or dictionaries containing 'x', 'y', 'z', 'pred' keys
    resolution: Float, the resolution of the depth map
    window_size: Integer, the size of the window used for interpolation

    Returns:
    depth_map: numpy array of shape (L, W) representing the depth map
    pixel_to_point: list of dictionaries mapping pixels to point indices
    """
    # Save the original indices of data1
    data1_index = data1['index']

    # Convert input to numpy arrays if they're dictionaries or DataFrames
    def to_numpy_arrays(data):
        if isinstance(data, dict):
            return np.array([data['x'], data['y'], data['z'], data['pred']])
        elif isinstance(data, pd.DataFrame):
            return data[['x', 'y', 'z', 'pred']].values.T
        return data

    data1 = to_numpy_arrays(data1)
    data2 = to_numpy_arrays(data2)

    # Calculate bounding box
    x_min = min(data1[0].min(), data2[0].min())
    x_max = max(data1[0].max(), data2[0].max())
    y_min = min(data1[1].min(), data2[1].min())
    y_max = max(data1[1].max(), data2[1].max())

    # Calculate grid dimensions
    L = int((y_max - y_min) / resolution)
    W = int((x_max - x_min) / resolution)
    print('L', L, 'W', W)

    # Initialize depth map
    depth_map = np.full((L, W), np.nan, dtype=np.float32)

    def process_data(data, index, depth_map, record_mapping=False):
        # Calculate grid indices
        grid_x = np.clip(((data[0] - x_min) / resolution).astype(int), 0, W - 1)
        grid_y = np.clip(((data[1] - y_min) / resolution).astype(int), 0, L - 1)

        # Use defaultdict to collect z values for each pixel
        pixel_z_values = defaultdict(list)
        pixel_to_point = []

        # If index is None, use default range
        if index is None:
            index = range(len(data[0]))

        for idx, (x, y, z, pred) in zip(index, zip(grid_x, grid_y, data[2], data[3])):
            pixel_z_values[(y, x)].append(z)
        
            if record_mapping and pred != 8:
                pixel_to_point.append({'pixel_x': x, 'pixel_y': y, 'index': idx})

        # Calculate median z value for each pixel and update depth map
        for (y, x), z_values in pixel_z_values.items():
            depth_map[y, x] = np.mean(z_values)

        return pixel_to_point if record_mapping else None

    # Process data1 and data2
    with tqdm(total=2 if not outlier_mode else 1, desc="Processing point clouds") as pbar:
        # Process data1 with index
        if outlier_mode == False:
            pixel_to_point = process_data(data1, data1_index, depth_map, record_mapping=True)
            pbar.update(1)
        # Process data2 without index (None is passed)
        process_data(data2, None, depth_map)
        pbar.update(1)

    if outlier_mode == False:
        print(f"Total mapped points: {len(pixel_to_point)}")

    # Use a sliding window to check if there is valid data in the neighborhood
    valid_points = []
    if window_size != 1:
        for i in tqdm(range(window_size // 2, L - window_size // 2), desc="Checking neighborhood"):
            for j in range(window_size // 2, W - window_size // 2):
                if np.isnan(depth_map[i, j]):
                    # Check if there is valid data in the window_size x window_size neighborhood
                    window = depth_map[i - window_size // 2 : i + window_size // 2 + 1,
                                       j - window_size // 2 : j + window_size // 2 + 1]
                    if np.any(~np.isnan(window)):
                        valid_points.append((i, j))

    # Get the valid (x, y) coordinates and corresponding z-values for interpolation
    interp_points = np.array(valid_points)
    if interp_points.size > 0:
        known_points = np.argwhere(~np.isnan(depth_map))
        known_values = depth_map[~np.isnan(depth_map)]
        
        # Perform interpolation using the nearest method
        with tqdm(total=1, desc="Interpolating") as pbar:
            interp_values = griddata(known_points, known_values, interp_points, method='nearest')
            pbar.update(1)
        
        # Fill in the interpolated results into the depth map
        depth_map[interp_points[:, 0], interp_points[:, 1]] = interp_values

    if outlier_mode==True:
        pixel_to_point = []

    return depth_map, pixel_to_point


# Cell 16
data_segment = {
    'index': df_enhance_segment.index,
    'x': df_enhance_segment['h'],
    'y': df_enhance_segment['theta'],
    'z': df_enhance_segment['r'],
    'pred': df_enhance_segment['pred']
}

data_joint = {
    'x': df_enhance_joint['h'],
    'y': df_enhance_joint['theta'],
    'z': df_enhance_joint['r'],
    'pred': df_enhance_joint['pred']
}

# depth map generation, and record pixel to point
depth_map, pixel_to_point = project_to_depth_map_inter(data_segment, data_joint, resolution=resolution, window_size=window_size)
# save pixel to point
os.makedirs(base_dir, exist_ok=True)
pixel_to_point_file = os.path.join(base_dir, "pixel_to_point.pkl")
with open(pixel_to_point_file, 'wb') as f:
    pickle.dump(pixel_to_point, f)


# Cell 18
import matplotlib.pyplot as plt

def save_depth_map_exact(depth_map, resolution, filename="depth_map.png"):
    """
    Save the depth map as an image with the exact dimensions and resolution.

    Parameters:
    depth_map: numpy array, the depth map to be saved.
    resolution: Float, the resolution used to generate the depth map (e.g., 0.005).
    filename: String, the filename to save the image as.
    """
    height, width = depth_map.shape
    dpi = 1.0 / resolution  # Calculate DPI from the resolution

    # Create a figure without any padding or axes
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Add an axes covering the entire figure
    ax.axis('off')  # No axes for this plot
    
    # Display the depth map
    ax.imshow(depth_map, cmap='viridis')

    # Save the depth map with exact dimensions
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


# Cell 19
# save to base_dir
save_depth_map_exact(depth_map, resolution=resolution, filename=f"{base_dir}/depth_map.png")


# for algorithm 4-1
data_joint_2 = {
    'x': df_enhance_joint['h'],
    'y': df_enhance_joint['theta'],
    'z': df_enhance_joint['r'],
    'pred': df_enhance_joint['pred'],
    'intensity': df_enhance_joint['intensity'],
}

df_joint = pd.DataFrame(data_joint_2)

# df_joint = df_joint[df_joint['intensity'] <= -1200]

# Cell 3
# generate map only including outlier point
depth_map_outlier,_ = project_to_depth_map_inter(data_segment, df_joint, window_size=1, outlier_mode=True)

# save depth_map_outlier
depth_map_outlier_file = os.path.join(base_dir, "depth_map_outlier.npy")
np.save(depth_map_outlier_file, depth_map_outlier)

# save df_point_cloud
df_point_cloud.to_csv(f"{base_dir}/enhanced.csv", index=False)