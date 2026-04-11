#!/usr/bin/env python3
"""Raw point cloud: key BO observation fields only (see skills/key_characteristics_observation_space.md)."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import Polygon

from sam4tun.plugins.paths import tunnel_characteristics_dir


def _fit_circle_3pts(p):
    A = np.array(
        [
            [2 * p[0, 0], 2 * p[0, 1], 1],
            [2 * p[1, 0], 2 * p[1, 1], 1],
            [2 * p[2, 0], 2 * p[2, 1], 1],
        ]
    )
    b = np.array(
        [
            p[0, 0] ** 2 + p[0, 1] ** 2,
            p[1, 0] ** 2 + p[1, 1] ** 2,
            p[2, 0] ** 2 + p[2, 1] ** 2,
        ]
    )
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx**2 + cy**2
    if r2 <= 0:
        return None
    return cx, cy, np.sqrt(r2)


def _ransac_circle(points_2d, n_iter=500, inlier_thresh=0.2, r_bounds=(1.0, 10.0)):
    best_inliers = 0
    best_params = None
    n = len(points_2d)
    if n < 10:
        return None
    for _ in range(n_iter):
        idx = np.random.choice(n, 3, replace=False)
        result = _fit_circle_3pts(points_2d[idx])
        if result is None:
            continue
        cx, cy, r = result
        if r < r_bounds[0] or r > r_bounds[1]:
            continue
        dists = np.abs(
            np.sqrt((points_2d[:, 0] - cx) ** 2 + (points_2d[:, 1] - cy) ** 2) - r
        )
        inliers = int(np.sum(dists < inlier_thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_params = (cx, cy, r)
    return best_params


def _estimate_diameter_ransac_cross_sections(
    points_xyz, n_sections=10, n_iter=300, inlier_thresh=0.2
):
    points_2d_xoy = points_xyz[:, :2]
    hull = ConvexHull(points_2d_xoy)
    poly = Polygon(points_2d_xoy[hull.vertices])
    rect = poly.minimum_rotated_rectangle
    rv = np.array(rect.exterior.coords)[:-1]
    edge_lens = [np.linalg.norm(rv[i] - rv[(i + 1) % 4]) for i in range(4)]

    if edge_lens[0] > edge_lens[1]:
        axis = (rv[1] - rv[0]) / edge_lens[0]
    else:
        axis = (rv[2] - rv[1]) / edge_lens[1]
    perp = np.array([-axis[1], axis[0]])

    center_xy = np.mean(points_2d_xoy, axis=0)
    along = (points_2d_xoy - center_xy) @ axis
    across = (points_2d_xoy - center_xy) @ perp
    z = points_xyz[:, 2]

    along_lo, along_hi = np.percentile(along, [10, 90])
    positions = np.linspace(along_lo, along_hi, n_sections)
    half_w = (along_hi - along_lo) / (2 * n_sections)

    diameters = []
    for pos in positions:
        mask = np.abs(along - pos) < half_w
        if np.sum(mask) < 50:
            continue
        sec = np.column_stack([across[mask], z[mask]])
        if len(sec) > 2000:
            sec = sec[np.random.choice(len(sec), 2000, replace=False)]
        result = _ransac_circle(sec, n_iter=n_iter, inlier_thresh=inlier_thresh)
        if result is not None:
            diameters.append(float(2 * result[2]))

    if not diameters:
        return None
    return float(np.median(diameters))


def analyze_point_cloud_key_only(file_path: str, tunnel_id: str | None) -> dict:
    point_cloud_data = np.loadtxt(file_path)
    points_xyz = point_cloud_data[:, :3]

    z_min = float(points_xyz[:, 2].min())
    z_max = float(points_xyz[:, 2].max())
    tunnel_height = z_max - z_min

    points_2d_xoy = points_xyz[:, :2]
    hull = ConvexHull(points_2d_xoy)
    poly = Polygon(points_2d_xoy[hull.vertices])
    rect = poly.minimum_rotated_rectangle
    rv = np.array(rect.exterior.coords)[:-1]
    edge_lens = [np.linalg.norm(rv[i] - rv[(i + 1) % 4]) for i in range(4)]
    tunnel_length = float(max(edge_lens))

    np.random.seed(42)
    ransac_diameter = _estimate_diameter_ransac_cross_sections(points_xyz)
    estimated_diameter = float(ransac_diameter) if ransac_diameter is not None else tunnel_height

    tree = cKDTree(points_xyz)
    distances, _ = tree.query(points_xyz, k=2)
    nearest_distances = distances[:, 1]

    return {
        "tunnel_id": tunnel_id or "unknown",
        "point_cloud_analysis": {
            "basic_statistics": {
                "coordinate_ranges": {
                    "z_range": [z_min, z_max],
                },
            },
            "tunnel_geometry": {
                "dimensions": {
                    "tunnel_length": tunnel_length,
                    "tunnel_height": float(tunnel_height),
                },
                "estimated_diameter": estimated_diameter,
            },
            "point_density": {
                "mean_nearest_neighbor_distance": float(np.mean(nearest_distances)),
                "median_nearest_neighbor_distance": float(np.median(nearest_distances)),
                "min_nearest_neighbor_distance": float(np.min(nearest_distances)),
            },
        },
    }


def main():
    p = argparse.ArgumentParser(description="BO key-only raw characteristics")
    p.add_argument("tunnel_id", type=str, help="Tunnel ID (e.g. 1-1)")
    p.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing {tunnel_id}.txt (default: data)",
    )
    args = p.parse_args()

    data_path = os.path.join(args.data_dir, f"{args.tunnel_id}.txt")
    if not os.path.exists(data_path):
        raise SystemExit(f"Input not found: {data_path}")

    out = analyze_point_cloud_key_only(data_path, args.tunnel_id)
    chars_dir = tunnel_characteristics_dir(args.tunnel_id)
    os.makedirs(chars_dir, exist_ok=True)
    out_path = os.path.join(chars_dir, "raw_characteristics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
