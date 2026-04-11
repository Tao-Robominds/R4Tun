#!/usr/bin/env python3
"""Denoised stage: key BO observation fields only."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sam4tun.plugins.paths import tunnel_characteristics_dir, tunnel_pipeline_dir


def load_denoised_data(tunnel_id: str) -> pd.DataFrame:
    base_dir = tunnel_pipeline_dir(tunnel_id)
    for name in ("denoised.csv", "unwrapped.csv"):
        file_path = os.path.join(base_dir, name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, comment="#" if name == "unwrapped.csv" else None)
            if name == "unwrapped.csv" and "pred" not in df.columns:
                df["pred"] = 7
            return df
    raise FileNotFoundError(f"No denoised data found in {base_dir}")


def point_density_key_only(df: pd.DataFrame) -> dict:
    valid = df[df["pred"] == 7] if "pred" in df.columns else df
    if len(valid) < 2:
        return {
            "mean_nn_distance": None,
            "median_nn_distance": None,
            "std_nn_distance": None,
        }
    np.random.seed(42)
    sample_size = min(10000, len(valid))
    if len(valid) > sample_size:
        idx = np.random.choice(len(valid), sample_size, replace=False)
        sample_coords = valid.iloc[idx][["h", "theta", "r"]].values
    else:
        sample_coords = valid[["h", "theta", "r"]].values
    tree = cKDTree(sample_coords)
    distances, _ = tree.query(sample_coords, k=2)
    nn = distances[:, 1]
    return {
        "mean_nn_distance": float(np.mean(nn)),
        "median_nn_distance": float(np.median(nn)),
        "std_nn_distance": float(np.std(nn)),
    }


def surface_completeness(df: pd.DataFrame) -> float:
    valid_df = df[df["pred"] == 7] if "pred" in df.columns else df
    if len(valid_df) == 0:
        return 0.0
    h_cov = (valid_df["h"].max() - valid_df["h"].min()) / (df["h"].max() - df["h"].min())
    th_cov = (valid_df["theta"].max() - valid_df["theta"].min()) / (
        df["theta"].max() - df["theta"].min()
    )
    return float((h_cov + th_cov) / 2)


def geometry_key_only(df: pd.DataFrame) -> dict:
    valid = df[df["pred"] == 7] if "pred" in df.columns else df
    h_range = [float(valid["h"].min()), float(valid["h"].max())]
    r_range = [float(valid["r"].min()), float(valid["r"].max())]

    h_sections = np.linspace(h_range[0], h_range[1], 10)
    section_curvatures = []
    for i in range(len(h_sections) - 1):
        h_min, h_max = h_sections[i], h_sections[i + 1]
        section_data = valid[(valid["h"] >= h_min) & (valid["h"] < h_max)]
        if len(section_data) > 10:
            section_curvatures.append(float(section_data["r"].std()))

    avg_curvature = float(np.mean(section_curvatures)) if section_curvatures else 0.0
    surface_regularity = float(np.std(section_curvatures)) if section_curvatures else 0.0

    return {
        "average_curvature_estimate": avg_curvature,
        "surface_regularity": surface_regularity,
        "tunnel_length": float(h_range[1] - h_range[0]),
        "section_curvatures": section_curvatures,
        "estimated_diameter": float(2 * np.mean(r_range)),
    }


def characterize(tunnel_id: str) -> dict:
    df = load_denoised_data(tunnel_id)
    return {
        "tunnel_id": tunnel_id,
        "denoising_results": {
            "point_density_analysis": point_density_key_only(df),
            "geometry_characteristics": geometry_key_only(df),
            "denoising_summary": {
                "surface_completeness": surface_completeness(df),
            },
        },
    }


def main():
    p = argparse.ArgumentParser(description="BO key-only denoised characteristics")
    p.add_argument("tunnel_id", type=str)
    args = p.parse_args()

    out = characterize(args.tunnel_id)
    chars_dir = tunnel_characteristics_dir(args.tunnel_id)
    os.makedirs(chars_dir, exist_ok=True)
    out_path = os.path.join(chars_dir, "denoised_characteristics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
