#!/usr/bin/env python3
# AUTO-GENERATED from SAM4Tun.py — do not edit body; re-run generate_modules.py

import sys
import os
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from helpers.pipeline_io import ensure_dir
from helpers.pipeline_state import load_state, save_state

tunnel_id = sys.argv[1]
paths = ensure_dir(tunnel_id)
state = load_state(paths["state"])
df_point_cloud = state["df_point_cloud"]
ring_count = state["ring_count"]
resolution = state.get("resolution", 0.005)

import pandas as pd
from scipy.interpolate import interp1d
from tqdm.notebook import tqdm
from scipy.ndimage import uniform_filter1d
from numba import njit, prange
import time

# Start the timer
start_time = time.time()

# Add a 'pred' column and initialize to 7
df_point_cloud['pred'] = 7

# Initial filter based on 'r' column
mask_r = (df_point_cloud['r'] < 2.7)|(df_point_cloud['r'] > 2.8) # diameter is 5.5
df_point_cloud.loc[mask_r, 'pred'] = 0

# Remaining point cloud data
filtered_df = df_point_cloud[~mask_r].copy()

# Define bins for X, Y, and Z directions
x_points = filtered_df['h'].values
y_points = filtered_df['theta'].values
z_points = filtered_df['r'].values

min_x, max_x = np.min(x_points), np.max(x_points)
min_y, max_y = np.min(y_points), np.max(y_points)
min_z, max_z = np.min(z_points), np.max(z_points)

print('Range of X:', '[', min_x, max_x, ']')

# Set grid sizes
x_step = (max_x - min_x) / ring_count
y_step = 0.5
z_step = 0.001

x_bins = np.arange(min_x, max_x + x_step, x_step)
y_bins = np.arange(min_y, max_y + y_step, y_step)
z_bins = np.arange(min_z, max_z + z_step, z_step)

# Initialize list to store filtered points and count matrices
filtered_points_list = []
count_matrices = []

# Pre-compute useful variables
grad_threshold = 0.2
epsilon = 1e-6

@njit(parallel=True)
def calculate_counts_matrix(y_points_sub, z_points_sub, y_bins, z_bins):
    counts_matrix = np.zeros((len(y_bins) - 1, len(z_bins) - 1))
    for i in prange(len(y_bins) - 1):
        y_min, y_max = y_bins[i], y_bins[i + 1]
        for j in range(len(z_bins) - 1):
            z_min, z_max = z_bins[j], z_bins[j + 1]
            mask = (y_points_sub >= y_min) & (y_points_sub < y_max) & (z_points_sub >= z_min) & (z_points_sub < z_max)
            counts_matrix[i, j] = np.sum(mask)
    return counts_matrix

@njit(parallel=True)
def calculate_cutoff_z_values(counts_matrix, z_bins, grad_threshold, epsilon):
    cutoff_z_values = np.full(counts_matrix.shape[0], 2.7)
    max_z_temp_values = np.zeros(counts_matrix.shape[0])
    
    for i in prange(counts_matrix.shape[0]):
        counts = counts_matrix[i, :]
        
        if np.all(counts == 0):
            continue
        
        max_count_idx = np.argmax(counts)
        grad_counts = np.diff(counts) / (counts[:-1] + epsilon)
        
        max_z_temp_values[i] = z_bins[max_count_idx]
        
        last_non_zero_idx = max_count_idx
        for j in range(max_count_idx, 0, -1):
            if counts[j] != 0:
                last_non_zero_idx = j
                
            if grad_counts[j - 1] < -grad_threshold or (counts[j] == 0 and counts[j - 1] == 0):
                cutoff_z_values[i] = z_bins[last_non_zero_idx]
                break
                
    return cutoff_z_values, max_z_temp_values

# Initialize list to store filtered points and count matrices
filtered_points_list = []
count_matrices = []

# Iterate over X bins
for x_min in x_bins[:-1]:
    x_max = x_min + x_step
    print('x_min-x_max:','[',x_min,x_max,']')
    mask_x = (x_points >= x_min) & (x_points < x_max)
    y_points_sub = y_points[mask_x]
    z_points_sub = z_points[mask_x]

    # Calculate count matrix using numba
    counts_matrix = calculate_counts_matrix(y_points_sub, z_points_sub, y_bins, z_bins)
    count_matrices.append(counts_matrix)

    # Calculate cutoff values using numba
    cutoff_z_values, max_z_temp_values = calculate_cutoff_z_values(counts_matrix, z_bins, grad_threshold, epsilon)

    # Handle NaNs and smoothing
    nan_indices = np.isnan(cutoff_z_values)
    not_nan_indices = ~nan_indices

    if np.any(nan_indices):
        interp_func = interp1d(
            np.where(not_nan_indices)[0],
            cutoff_z_values[not_nan_indices],
            kind='linear',
            fill_value='extrapolate'
        )
        cutoff_z_values[nan_indices] = interp_func(np.where(nan_indices)[0])

    cutoff_z_values_smoothed = uniform_filter1d(cutoff_z_values, size=3, mode='nearest') - 0.003

    index_ranges = [[min_y + i * y_step, min_y + (i + 1) * y_step] for i in range(len(cutoff_z_values))]
    df_visual = pd.DataFrame({
        'Y range': index_ranges,
        'cutoff_z': cutoff_z_values,
        'smoothed_z': cutoff_z_values_smoothed,
        'max_z': max_z_temp_values
    })
    df_visual['Y range'] = df_visual['Y range'].apply(lambda x: [round(val, 4) for val in x])
    print(df_visual)

    # Vectorized filtering based on cutoff values
    y_indices = np.digitize(y_points_sub, y_bins) - 1
    filtered_mask = (z_points_sub >= cutoff_z_values_smoothed[y_indices])
    
    filtered_points_sub = {
        'h': x_points[mask_x][filtered_mask],
        'theta': y_points[mask_x][filtered_mask],
        'r': z_points[mask_x][filtered_mask]
    }
    filtered_points_list.append(filtered_points_sub)

    # Update filtered out points 'pred' to 0
    # Corrected: apply filtered_mask directly to the indices of mask_x
    filtered_out_indices = filtered_df.index[mask_x][~filtered_mask]
    df_point_cloud.loc[filtered_out_indices, 'pred'] = 0

# Output the number of filtered points
filtered_points_count = (df_point_cloud['pred'] == 7).sum()
print(f"Remaining points count: {filtered_points_count}")

# End the timer
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")


df_point_cloud.to_csv(paths["denoised_csv"], index=False)
state["df_point_cloud"] = df_point_cloud
save_state(paths["state"], state)
print(f"Denoising complete -> {paths['denoised_csv']}")

