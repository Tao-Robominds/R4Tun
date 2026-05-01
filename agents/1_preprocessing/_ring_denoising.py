"""Ring denoising — density-difference cutoffs (from r4tun/agents/denoising.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

SURFACE_PRED = 7

epsilon = 1e-6


@njit(parallel=True)
def calculate_counts_matrix(y_points_sub, z_points_sub, y_bins, z_bins):
    counts_matrix = np.zeros((len(y_bins) - 1, len(z_bins) - 1))
    for i in prange(len(y_bins) - 1):
        y_min, y_max = y_bins[i], y_bins[i + 1]
        for j in range(len(z_bins) - 1):
            z_min, z_max = z_bins[j], z_bins[j + 1]
            mask = (
                (y_points_sub >= y_min)
                & (y_points_sub < y_max)
                & (z_points_sub >= z_min)
                & (z_points_sub < z_max)
            )
            counts_matrix[i, j] = np.sum(mask)
    return counts_matrix


@njit(parallel=True)
def calculate_cutoff_z_values(counts_matrix, z_bins, grad_threshold, epsilon_, default_cutoff_z):
    cutoff_z_values = np.full(counts_matrix.shape[0], default_cutoff_z)
    max_z_temp_values = np.zeros(counts_matrix.shape[0])

    for i in prange(counts_matrix.shape[0]):
        counts = counts_matrix[i, :]

        if np.all(counts == 0):
            continue

        max_count_idx = np.argmax(counts)
        grad_counts = np.diff(counts) / (counts[:-1] + epsilon_)

        max_z_temp_values[i] = z_bins[max_count_idx]

        last_non_zero_idx = max_count_idx
        for j in range(max_count_idx, 0, -1):
            if counts[j] != 0:
                last_non_zero_idx = j

            if grad_counts[j - 1] < -grad_threshold or (counts[j] == 0 and counts[j - 1] == 0):
                cutoff_z_values[i] = z_bins[last_non_zero_idx]
                break

    return cutoff_z_values, max_z_temp_values


def denoise_ring(
    df_point_cloud: pd.DataFrame,
    ring_count: int,
    mask_r_low: float,
    mask_r_high: float,
    y_step: float,
    z_step: float,
    grad_threshold: float,
    smoothing_window_size: int,
    smoothing_offset: float,
    default_cutoff_z: float,
) -> pd.DataFrame:
    """Same logic as r4tun denoising.py main loop."""
    df_point_cloud = df_point_cloud.copy()
    df_point_cloud["pred"] = SURFACE_PRED

    mask_r = (df_point_cloud["r"] < mask_r_low) | (df_point_cloud["r"] > mask_r_high)
    df_point_cloud.loc[mask_r, "pred"] = 0

    filtered_df = df_point_cloud[~mask_r].copy()

    x_points = filtered_df["h"].values
    y_points = filtered_df["theta"].values
    z_points = filtered_df["r"].values

    min_x, max_x = np.min(x_points), np.max(x_points)
    min_y, max_y = np.min(y_points), np.max(y_points)
    min_z, max_z = np.min(z_points), np.max(z_points)

    x_step = (max_x - min_x) / ring_count if ring_count > 0 else max(max_x - min_x, 1e-9)

    x_bins = np.arange(min_x, max_x + x_step, x_step)
    y_bins = np.arange(min_y, max_y + y_step, y_step)
    z_bins = np.arange(min_z, max_z + z_step, z_step)

    for x_min_bin in x_bins[:-1]:
        x_max_bin = x_min_bin + x_step
        mask_x = (x_points >= x_min_bin) & (x_points < x_max_bin)
        y_points_sub = y_points[mask_x]
        z_points_sub = z_points[mask_x]

        counts_matrix = calculate_counts_matrix(y_points_sub, z_points_sub, y_bins, z_bins)

        cutoff_z_values, _ = calculate_cutoff_z_values(
            counts_matrix, z_bins, grad_threshold, epsilon, default_cutoff_z
        )

        nan_indices = np.isnan(cutoff_z_values)
        not_nan_indices = ~nan_indices

        if np.any(nan_indices):
            if np.any(not_nan_indices):
                interp_func = interp1d(
                    np.where(not_nan_indices)[0],
                    cutoff_z_values[not_nan_indices],
                    kind="linear",
                    fill_value="extrapolate",
                )
                cutoff_z_values[nan_indices] = interp_func(np.where(nan_indices)[0])

        cutoff_z_values_smoothed = (
            uniform_filter1d(cutoff_z_values, size=smoothing_window_size, mode="nearest")
            + smoothing_offset
        )

        y_indices = np.digitize(y_points_sub, y_bins) - 1
        y_indices = np.clip(y_indices, 0, len(cutoff_z_values_smoothed) - 1)
        filtered_mask = z_points_sub >= cutoff_z_values_smoothed[y_indices]

        filtered_out_indices = filtered_df.index[mask_x][~filtered_mask]
        df_point_cloud.loc[filtered_out_indices, "pred"] = 0

    return df_point_cloud
