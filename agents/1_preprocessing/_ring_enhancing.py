"""Ring enhancing + depth maps — ported from r4tun/agents/enhancing.py (no CLI side effects)."""

from __future__ import annotations

import os
import pickle
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.interpolate import griddata
from scipy.spatial import KDTree, cKDTree
from tqdm.auto import tqdm


@njit(parallel=True)
def calculate_curvatures(points, indices, k):
    curvatures = np.zeros(len(points))
    for i in prange(len(points)):
        neighbors = points[indices[i, 1:]]
        cov_matrix = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)
    return curvatures


def compute_curvature(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    points = df[["x", "y", "z"]].values
    tree = KDTree(points)
    _, indices = tree.query(points, k=min(k + 1, len(points)))
    curvatures = calculate_curvatures(points, indices, k)
    out = df.copy()
    out.loc[:, "curvature"] = curvatures
    return out


@njit(parallel=False)
def compute_midpoints_and_filter(points, indices, distances, target_distance, curvature_threshold):
    num_points = len(points)
    max_new_points = num_points * (len(indices[0]) - 1)
    new_points = np.zeros((max_new_points, points.shape[1]), dtype=np.float64)
    new_points_count = 0

    for i in range(len(points)):
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

    for i in range(num_points):
        if removed_indices[i] == 0:
            keep_indices[count] = i
            count += 1
            for j in range(neighbors_array.shape[1]):
                neighbor_idx = neighbors_array[i, j]
                if valid_mask[i, j] and removed_indices[neighbor_idx] == 0:
                    removed_indices[neighbor_idx] = 1

    return keep_indices[:count]


def optimized_radius_filter(df: pd.DataFrame, target_distance: float) -> pd.DataFrame:
    points = df[["h", "theta"]].values
    num_points = len(points)
    if num_points == 0:
        return df.iloc[0:0].reset_index(drop=True)

    r_dist = 0.15 * target_distance
    tree = cKDTree(points)

    neighbors_list = tree.query_ball_point(points, r=r_dist)
    max_neighbors = max((len(neighbors) for neighbors in neighbors_list), default=0)
    if max_neighbors == 0:
        return df.iloc[0:0].reset_index(drop=True)
    neighbors_array = np.full((len(points), max_neighbors), -1, dtype=np.int32)
    valid_mask = np.zeros((len(points), max_neighbors), dtype=np.bool_)

    for i in range(len(points)):
        length = len(neighbors_list[i])
        neighbors_array[i, :length] = neighbors_list[i]
        valid_mask[i, :length] = True

    keep_indices = _filter_points_to_keep(neighbors_array, valid_mask, num_points)
    return df.iloc[keep_indices].reset_index(drop=True)


def enhance_segment_surface(
    df: pd.DataFrame,
    target_distance: float,
    curvature_threshold_param: float,
    num_neighbors_param: int,
) -> pd.DataFrame:
    start_time = time.time()
    points = df[["h", "theta", "r", "curvature", "intensity"]].values
    original_points = points[:, :2]

    original_tree = cKDTree(original_points)
    kq = min(num_neighbors_param + 1, len(points))
    distances, indices = original_tree.query(original_points, k=kq)

    all_new_points = compute_midpoints_and_filter(
        points, indices, distances, target_distance, curvature_threshold_param
    )

    if len(all_new_points) == 0:
        return pd.DataFrame(columns=["h", "theta", "r", "curvature", "intensity", "pred"])

    distances, _ = original_tree.query(all_new_points[:, :2], k=1)
    distances_flat = distances.flatten()
    valid_new_points = all_new_points[distances_flat >= 0.2 * target_distance]

    add_point_df = pd.DataFrame(
        valid_new_points, columns=["h", "theta", "r", "curvature", "intensity"]
    )
    add_point_df = add_point_df[(add_point_df != 0).any(axis=1)]
    add_point_df["pred"] = 8
    add_point_df_rf = optimized_radius_filter(add_point_df, target_distance)

    print(
        f"insert_midpoints target_distance={target_distance:.4f}s took {time.time()-start_time:.2f}s "
        f"(+{len(add_point_df_rf)} pts)"
    )
    return add_point_df_rf.reset_index(drop=True)


@njit(parallel=False)
def interpolate_points_jit(filtered_indices, points, inter_radius, num_interpolations, duplicate_threshold, resolution):
    num_indices = len(filtered_indices)
    max_new_points = num_indices * num_indices * num_interpolations
    new_points = np.zeros((max_new_points, 4))
    count = 0

    for i in range(num_indices):
        index1 = filtered_indices[i]
        point1 = points[index1]
        x1, y1, z1, i1 = point1

        for j in range(i + 1, num_indices):
            index2 = filtered_indices[j]
            point2 = points[index2]
            x2, y2, z2, i2 = point2

            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if not (resolution < dist < inter_radius):
                continue

            for t in np.linspace(0, 1, num=num_interpolations + 2)[1:-1]:
                new_x = (1 - t) * x1 + t * x2
                new_y = (1 - t) * y1 + t * y2
                new_z = (1 - t) * z1 + t * z2
                new_i = (1 - t) * i1 + t * i2

                if count > 0:
                    dists = np.sqrt((new_points[:count, 0] - new_x) ** 2 + (new_points[:count, 1] - new_y) ** 2)
                    if np.any(dists < duplicate_threshold):
                        continue

                new_points[count] = np.array([new_x, new_y, new_z, new_i])
                count += 1

    return new_points[:count]


def enhance_outlier_points_ring(
    df: pd.DataFrame,
    depth_threshold_low: float,
    depth_threshold_high: float,
    inter_radius: float,
    num_interpolations: int,
    duplicate_threshold: float,
    resolution: float,
    num_neighbors: int,
    hd_disabled: bool,
    n_segment: Tuple[float, float],
    x_min_ref: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ring mode: if ``hd_disabled``, use single threshold (no tunnel HD band)."""
    points = df[["h", "theta", "r", "intensity"]].values
    coords = points[:, :2]
    tree = cKDTree(coords)
    kq = min(num_neighbors + 1, len(points))
    distances, indices = tree.query(coords, k=kq)
    z_vals = df["r"].values

    meaningful_mask = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        nei = indices[i, 1:]
        if len(nei) < min(20, num_neighbors):
            continue
        avg_diff = points[i, 2] - np.mean(z_vals[nei])
        if hd_disabled:
            if avg_diff > depth_threshold_low:
                meaningful_mask[i] = True
        else:
            lo, hi = n_segment
            if (x_min_ref + 1.2 * lo) <= points[i, 0] <= (x_min_ref + 1.2 * hi):
                if avg_diff > depth_threshold_high:
                    meaningful_mask[i] = True
            else:
                if avg_diff > depth_threshold_low:
                    meaningful_mask[i] = True

    meaningful_indices = np.where(meaningful_mask)[0]
    print(f"Number of outlier points: {len(meaningful_indices)}")
    meaningful_df = df.iloc[meaningful_indices].copy()

    if hd_disabled:
        filtered_indices = meaningful_indices.astype(np.int64)
    else:
        filtered_high_density_indices = []
        for idx in meaningful_indices:
            x = points[idx, 0]
            lo, hi = n_segment
            if not ((x_min_ref + 1.2 * lo) <= x <= (x_min_ref + 1.2 * hi)):
                filtered_high_density_indices.append(idx)
        filtered_indices = np.array(filtered_high_density_indices, dtype=np.int64)

    _max_pairs = 2800
    _nfi = len(filtered_indices)
    if _nfi > _max_pairs:
        _rng = np.random.default_rng(42)
        _pick = _rng.choice(_nfi, size=_max_pairs, replace=False)
        filtered_indices = filtered_indices[_pick]
        print(f"Note: subsampled outliers for pairwise interpolation {_nfi} → {_max_pairs}")

    new_points_array = interpolate_points_jit(
        filtered_indices, points, inter_radius, num_interpolations, duplicate_threshold, resolution
    )
    new_df = pd.DataFrame(new_points_array, columns=["h", "theta", "r", "intensity"])
    new_df["pred"] = 8
    print(f"Number of new added points: {len(new_df)}")
    return meaningful_df, new_df


def project_to_depth_map_inter(
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    resolution: float,
    window_size: int = 5,
    outlier_mode: bool = False,
    canonical_height_px: Optional[int] = None,
) -> Tuple[np.ndarray, List]:
    """Project (h, theta) → depth map; optional fixed theta height in pixels."""

    def to_numpy_arrays(data):
        if isinstance(data, dict):
            return np.array([data["x"], data["y"], data["z"], data["pred"]])
        return data[["x", "y", "z", "pred"]].values.T

    data1_index = data1["index"]
    data1a = to_numpy_arrays(data1)
    data2a = to_numpy_arrays(data2)

    x_min = min(data1a[0].min(), data2a[0].min())
    x_max = max(data1a[0].max(), data2a[0].max())
    y_min = min(data1a[1].min(), data2a[1].min())
    y_max = max(data1a[1].max(), data2a[1].max())

    W = max(1, int((x_max - x_min) / resolution))
    if canonical_height_px is not None:
        L = max(1, int(canonical_height_px))
    else:
        L = max(1, int((y_max - y_min) / resolution))
    print("Depth map L", L, "W", W, "(canonical_height_px=", canonical_height_px, ")")

    depth_map = np.full((L, W), np.nan, dtype=np.float32)

    def process_data(data, index, record_mapping=False):
        grid_x = np.clip(((data[0] - x_min) / resolution).astype(int), 0, W - 1)
        grid_y = np.clip(((data[1] - y_min) / resolution).astype(int), 0, L - 1)

        pixel_z_values = defaultdict(list)
        pixel_to_point = []

        if index is None:
            index = range(len(data[0]))

        for idx, (x, y, z, pred) in zip(index, zip(grid_x, grid_y, data[2], data[3])):
            pixel_z_values[(y, x)].append(z)
            if record_mapping and pred != 8:
                pixel_to_point.append({"pixel_x": int(x), "pixel_y": int(y), "index": idx})

        for (y, x), z_values in pixel_z_values.items():
            depth_map[y, x] = np.mean(z_values)

        return pixel_to_point if record_mapping else None

    with tqdm(total=2 if not outlier_mode else 1, desc="Processing point clouds") as pbar:
        if not outlier_mode:
            pixel_to_point = process_data(data1a, data1_index, record_mapping=True)
            pbar.update(1)
        else:
            pixel_to_point = []
        process_data(data2a, None, record_mapping=False)
        pbar.update(1)

    if not outlier_mode:
        print(f"Total mapped points: {len(pixel_to_point)}")

    valid_points = []
    if window_size != 1:
        for i in tqdm(range(window_size // 2, L - window_size // 2), desc="Checking neighborhood"):
            for j in range(window_size // 2, W - window_size // 2):
                if np.isnan(depth_map[i, j]):
                    window = depth_map[
                        i - window_size // 2 : i + window_size // 2 + 1,
                        j - window_size // 2 : j + window_size // 2 + 1,
                    ]
                    if np.any(~np.isnan(window)):
                        valid_points.append((i, j))

    interp_points = np.array(valid_points)
    if interp_points.size > 0:
        known_points = np.argwhere(~np.isnan(depth_map))
        known_values = depth_map[~np.isnan(depth_map)]
        with tqdm(total=1, desc="Interpolating") as pbar:
            interp_values = griddata(known_points, known_values, interp_points, method="nearest")
            pbar.update(1)
        depth_map[interp_points[:, 0], interp_points[:, 1]] = interp_values

    return depth_map, pixel_to_point if not outlier_mode else []


def save_depth_map_exact(depth_map: np.ndarray, resolution: float, filename: str) -> None:
    height, width = depth_map.shape
    dpi = 1.0 / resolution
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(depth_map, cmap="viridis")
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def canonical_theta_pixels(tunnel_diameter: float, resolution: float) -> int:
    return int(round(np.pi * float(tunnel_diameter) / float(resolution)))


def run_ring_enhancing(
    df_point_cloud: pd.DataFrame,
    base_dir: str,
    tunnel_diameter: float,
    enhancing_params: Dict[str, Any],
    outlier_hd_disabled: bool,
) -> pd.DataFrame:
    """Run enhancing stages; write depth_map.png, depth_map_outlier.npy, pixel_to_point.pkl, enhanced.csv."""
    e = enhancing_params
    s1 = float(e["upsampling_stage1_target_distance"])
    s2 = float(e["upsampling_stage2_target_distance"])
    s3 = float(e["upsampling_stage3_target_distance"])
    curvature_threshold = float(e["curvature_threshold"])
    depth_threshold_low = float(e["depth_threshold_low"])
    depth_threshold_high = float(e["depth_threshold_high"])
    inter_radius = float(e["inter_radius"])
    duplicate_threshold = float(e["duplicate_threshold"])
    n_segment_start = int(e["n_segment_start"])
    n_segment_end = int(e["n_segment_end"])
    num_neighbors = int(e["num_neighbors"])
    num_interpolations = int(e["num_interpolations"])
    resolution = float(e["resolution"])
    window_size = int(e["window_size"])

    df_support_filtered = df_point_cloud[df_point_cloud["pred"] != 0].copy()
    df_support_filtered_curva = compute_curvature(df_support_filtered)

    df_upsampling_all = df_support_filtered_curva
    for td in (s1, s2, s3):
        df_up = enhance_segment_surface(
            df_upsampling_all,
            target_distance=td,
            curvature_threshold_param=curvature_threshold,
            num_neighbors_param=num_neighbors,
        )
        df_upsampling_all = pd.concat([df_upsampling_all, df_up], ignore_index=False)

    df_enhance_segment = df_upsampling_all

    x_min_ref = float(df_support_filtered_curva["h"].min())
    meaningful_df, new_df = enhance_outlier_points_ring(
        df_support_filtered_curva,
        depth_threshold_low=depth_threshold_low,
        depth_threshold_high=depth_threshold_high,
        inter_radius=inter_radius,
        num_interpolations=num_interpolations,
        duplicate_threshold=duplicate_threshold,
        resolution=resolution,
        num_neighbors=num_neighbors,
        hd_disabled=outlier_hd_disabled or (n_segment_end < 0 or n_segment_start < 0),
        n_segment=(float(n_segment_start), float(n_segment_end)),
        x_min_ref=x_min_ref,
    )

    df_enhance_joint = pd.concat([meaningful_df, new_df], ignore_index=False)
    df_point_cloud.loc[meaningful_df.index, "pred"] = 0

    can_h = canonical_theta_pixels(tunnel_diameter, resolution)

    data_segment = {
        "index": df_enhance_segment.index,
        "x": df_enhance_segment["h"],
        "y": df_enhance_segment["theta"],
        "z": df_enhance_segment["r"],
        "pred": df_enhance_segment["pred"],
    }
    data_joint = {
        "x": df_enhance_joint["h"],
        "y": df_enhance_joint["theta"],
        "z": df_enhance_joint["r"],
        "pred": df_enhance_joint["pred"],
    }

    depth_map, pixel_to_point = project_to_depth_map_inter(
        data_segment,
        data_joint,
        resolution=resolution,
        window_size=window_size,
        outlier_mode=False,
        canonical_height_px=can_h,
    )

    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, "pixel_to_point.pkl"), "wb") as f:
        pickle.dump(pixel_to_point, f)

    save_depth_map_exact(
        depth_map, resolution=resolution, filename=os.path.join(base_dir, "depth_map.png")
    )
    np.save(os.path.join(base_dir, "depth_map.npy"), depth_map)

    data_joint_2 = {
        "x": df_enhance_joint["h"],
        "y": df_enhance_joint["theta"],
        "z": df_enhance_joint["r"],
        "pred": df_enhance_joint["pred"],
        "intensity": df_enhance_joint["intensity"],
    }
    df_joint = pd.DataFrame(data_joint_2)

    depth_map_outlier, _ = project_to_depth_map_inter(
        data_segment,
        df_joint,
        resolution=resolution,
        window_size=1,
        outlier_mode=True,
        canonical_height_px=can_h,
    )
    np.save(os.path.join(base_dir, "depth_map_outlier.npy"), depth_map_outlier)

    new_upsampled = df_enhance_segment[df_enhance_segment["pred"] == 8].copy()
    new_joint = df_enhance_joint[df_enhance_joint["pred"] == 8].copy()

    for col in df_point_cloud.columns:
        if col not in new_upsampled.columns:
            new_upsampled[col] = np.nan if col in ("x", "y", "z") else None
        if col not in new_joint.columns:
            new_joint[col] = np.nan if col in ("x", "y", "z") else None

    all_new = pd.concat([new_upsampled, new_joint], ignore_index=True)
    df_enhanced = pd.concat([df_point_cloud, all_new], ignore_index=True)
    df_enhanced.to_csv(os.path.join(base_dir, "enhanced.csv"), index=False)
    return df_enhanced
