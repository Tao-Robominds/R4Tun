"""
Local Point Cloud Density-Difference-Based Denoising

This module removes noise from unfolded tunnel point cloud data using
local density analysis in cylindrical coordinates.

Algorithm Overview:
    1. Filter points outside the expected tunnel radius range
    2. Divide the point cloud into slices along the tunnel axis
    3. For each slice, compute point density histograms in the radial direction
    4. Identify the tunnel surface boundary using gradient-based cutoff detection
    5. Remove points below the detected surface boundary
"""

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

# =============================================================================
# Constants
# =============================================================================

# --- Physical Constants (fixed by tunnel design, don't change) ---
TUNNEL_DIAMETER = 5.5  # Nominal tunnel diameter (meters)
TUNNEL_RADIUS = TUNNEL_DIAMETER / 2  # Nominal tunnel radius (meters)

# --- Quality Parameters (tune these for better results) ---
# Radius filtering
RADIUS_MIN = 2.7  # Minimum expected radius (meters)
                   # Points closer than this are likely noise/equipment
RADIUS_MAX = 2.8  # Maximum expected radius (meters)
                   # Points farther than this are likely noise

# Grid resolution for density analysis
THETA_STEP = 0.5  # Angular grid step size (meters along circumference)
                   # Smaller = finer angular resolution, more computation
RADIAL_STEP = 0.001  # Radial grid step size (meters)
                      # Smaller = finer depth resolution, more sensitive

# Gradient-based cutoff detection
GRADIENT_THRESHOLD = 0.2  # Threshold for density gradient to detect surface boundary
                           # Lower = more aggressive denoising
                           # Higher = more conservative, keeps more points
GRADIENT_EPSILON = 1e-6  # Small value to prevent division by zero

# Cutoff smoothing
SMOOTHING_WINDOW = 3  # Window size for smoothing cutoff values
                       # Larger = smoother boundary, may miss details
SMOOTHING_OFFSET = 0.003  # Offset subtracted from smoothed cutoff (meters)
                           # Larger = more aggressive denoising

# --- Performance Parameters ---
# (None for this module - uses Numba parallel processing automatically)


# =============================================================================
# Density Calculation Functions
# =============================================================================

@njit(parallel=True)
def calculate_density_matrix(
    theta_points: np.ndarray,
    radial_points: np.ndarray,
    theta_bins: np.ndarray,
    radial_bins: np.ndarray
) -> np.ndarray:
    """
    Calculate 2D histogram of point density in theta-radial space.
    
    Args:
        theta_points: Array of theta (angular) coordinates.
        radial_points: Array of radial coordinates.
        theta_bins: Bin edges for theta axis.
        radial_bins: Bin edges for radial axis.
        
    Returns:
        2D array of point counts per bin.
    """
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
    epsilon: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial cutoff values for surface boundary detection.
    
    Uses gradient analysis to find where point density drops significantly,
    indicating the transition from tunnel surface to noise.
    
    Args:
        density_matrix: 2D array of point densities.
        radial_bins: Bin edges for radial axis.
        gradient_threshold: Threshold for detecting density drop.
        epsilon: Small value to prevent division by zero.
        
    Returns:
        Tuple of (cutoff_values, peak_radial_values) arrays.
    """
    num_theta_bins = density_matrix.shape[0]
    cutoff_values = np.full(num_theta_bins, RADIUS_MIN)
    peak_radial_values = np.zeros(num_theta_bins)
    
    for i in prange(num_theta_bins):
        counts = density_matrix[i, :]
        
        # Skip empty bins
        if np.all(counts == 0):
            continue
        
        # Find peak density location
        peak_idx = np.argmax(counts)
        peak_radial_values[i] = radial_bins[peak_idx]
        
        # Compute normalized gradient
        gradient = np.diff(counts) / (counts[:-1] + epsilon)
        
        # Search backwards from peak for significant density drop
        last_valid_idx = peak_idx
        for j in range(peak_idx, 0, -1):
            if counts[j] != 0:
                last_valid_idx = j
            
            # Detect boundary: significant negative gradient or gap
            if gradient[j - 1] < -gradient_threshold or (counts[j] == 0 and counts[j - 1] == 0):
                cutoff_values[i] = radial_bins[last_valid_idx]
                break
    
    return cutoff_values, peak_radial_values


# =============================================================================
# Cutoff Smoothing
# =============================================================================

def smooth_cutoff_values(
    cutoff_values: np.ndarray,
    window_size: int = SMOOTHING_WINDOW,
    offset: float = SMOOTHING_OFFSET
) -> np.ndarray:
    """
    Smooth and interpolate cutoff values for robust boundary detection.
    
    Args:
        cutoff_values: Raw cutoff values per angular bin.
        window_size: Smoothing window size.
        offset: Offset to subtract from smoothed values.
        
    Returns:
        Smoothed cutoff values.
    """
    # Interpolate NaN values
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
    
    # Apply uniform smoothing and offset
    smoothed = uniform_filter1d(cutoff_values, size=window_size, mode='nearest')
    return smoothed - offset


# =============================================================================
# Main Denoising Pipeline
# =============================================================================

def denoise_point_cloud(
    df: pd.DataFrame,
    ring_count: int,
    radius_min: float = RADIUS_MIN,
    radius_max: float = RADIUS_MAX,
    theta_step: float = THETA_STEP,
    radial_step: float = RADIAL_STEP,
    gradient_threshold: float = GRADIENT_THRESHOLD
) -> pd.DataFrame:
    """
    Denoise tunnel point cloud using density-based surface detection.
    
    Args:
        df: DataFrame with columns ['h', 'theta', 'r', ...].
        ring_count: Number of rings in the tunnel.
        radius_min: Minimum expected radius for initial filtering.
        radius_max: Maximum expected radius for initial filtering.
        theta_step: Angular bin size.
        radial_step: Radial bin size.
        gradient_threshold: Threshold for surface detection.
        
    Returns:
        DataFrame with 'pred' column updated (0 = noise, 7 = valid).
    """
    # Initialize prediction column
    df = df.copy()
    df['pred'] = 7  # Default: valid point
    
    # Step 1: Initial radius filtering
    radius_mask = (df['r'] < radius_min) | (df['r'] > radius_max)
    df.loc[radius_mask, 'pred'] = 0
    
    # Get points within radius range
    valid_df = df[~radius_mask].copy()
    
    # Extract coordinates
    h_coords = valid_df['h'].values
    theta_coords = valid_df['theta'].values
    radial_coords = valid_df['r'].values
    
    # Compute axis ranges
    h_min, h_max = np.min(h_coords), np.max(h_coords)
    theta_min, theta_max = np.min(theta_coords), np.max(theta_coords)
    radial_min, radial_max = np.min(radial_coords), np.max(radial_coords)
    
    # Create bin edges
    h_step = (h_max - h_min) / ring_count
    h_bins = np.arange(h_min, h_max + h_step, h_step)
    theta_bins = np.arange(theta_min, theta_max + theta_step, theta_step)
    radial_bins = np.arange(radial_min, radial_max + radial_step, radial_step)
    
    # Step 2: Process each axial slice
    for h_idx in range(len(h_bins) - 1):
        h_low, h_high = h_bins[h_idx], h_bins[h_idx] + h_step
        slice_mask = (h_coords >= h_low) & (h_coords < h_high)
        
        theta_slice = theta_coords[slice_mask]
        radial_slice = radial_coords[slice_mask]
        
        if len(theta_slice) == 0:
            continue
        
        # Compute density matrix
        density_matrix = calculate_density_matrix(
            theta_slice, radial_slice, theta_bins, radial_bins
        )
        
        # Compute surface cutoffs
        cutoffs, _ = compute_surface_cutoffs(
            density_matrix, radial_bins, gradient_threshold, GRADIENT_EPSILON
        )
        
        # Smooth cutoff values
        smoothed_cutoffs = smooth_cutoff_values(cutoffs)
        
        # Step 3: Filter points below cutoff
        theta_bin_indices = np.digitize(theta_slice, theta_bins) - 1
        theta_bin_indices = np.clip(theta_bin_indices, 0, len(smoothed_cutoffs) - 1)
        
        below_surface = radial_slice < smoothed_cutoffs[theta_bin_indices]
        
        # Update predictions for filtered points
        slice_indices = valid_df.index[slice_mask]
        filtered_indices = slice_indices[below_surface]
        df.loc[filtered_indices, 'pred'] = 0
    
    return df


# =============================================================================
# Entry Point
# =============================================================================

def main(tunnel_id: str, base_dir: str = "data") -> None:
    """
    Execute the denoising pipeline for a tunnel.
    
    Args:
        tunnel_id: Identifier for the tunnel (e.g., "1-4", "5-1").
        base_dir: Base directory for data files.
    """
    print(f"Processing tunnel: {tunnel_id}")
    
    # Load data
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    unwrapped_file = os.path.join(tunnel_dir, "unwrapped.csv")
    ring_count_file = os.path.join(tunnel_dir, "ring_count.txt")
    
    df = pd.read_csv(unwrapped_file)
    with open(ring_count_file, 'r') as f:
        ring_count = int(f.read().strip())
    
    # Run denoising
    df_denoised = denoise_point_cloud(df, ring_count)
    
    # Save results
    output_file = os.path.join(tunnel_dir, "denoised.csv")
    df_denoised.to_csv(output_file, index=False)
    
    # Report statistics
    total_points = len(df_denoised)
    noise_points = (df_denoised['pred'] == 0).sum()
    valid_points = total_points - noise_points
    print(f"Total points: {total_points}")
    print(f"Removed as noise: {noise_points} ({100 * noise_points / total_points:.1f}%)")
    print(f"Valid points: {valid_points} ({100 * valid_points / total_points:.1f}%)")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 2_denoising_clean.py <tunnel_id>")
        print("Example: python 2_denoising_clean.py 1-4")
        sys.exit(1)
    
    main(sys.argv[1])

