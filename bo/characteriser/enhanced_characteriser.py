#!/usr/bin/env python3
"""Enhanced (Algorithm 3) stage: key BO observation fields only."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sam4tun.plugins.paths import tunnel_characteristics_dir, tunnel_pipeline_dir


def load_enhanced_data(tunnel_id: str) -> pd.DataFrame:
    base_dir = tunnel_pipeline_dir(tunnel_id)
    enhanced_path = os.path.join(base_dir, "enhanced.csv")
    if not os.path.exists(enhanced_path):
        raise FileNotFoundError(enhanced_path)
    # Require upstream artefact so BO runs match a full pipeline tree
    pre_ok = any(
        os.path.exists(os.path.join(base_dir, n)) for n in ("denoised.csv", "unwrapped.csv")
    )
    if not pre_ok:
        raise FileNotFoundError(f"No denoised.csv or unwrapped.csv in {base_dir}")
    return pd.read_csv(enhanced_path, comment="#")


def final_nn_key_only(enhanced_df: pd.DataFrame) -> dict:
    if "pred" in enhanced_df.columns:
        valid = enhanced_df[enhanced_df["pred"] != 0]
    else:
        valid = enhanced_df

    np.random.seed(42)
    if len(valid) > 1000:
        sample_size = min(10000, len(valid))
        idx = np.random.choice(len(valid), sample_size, replace=False)
        sample_coords = valid.iloc[idx][["h", "theta", "r"]].values
        tree = cKDTree(sample_coords)
        distances, _ = tree.query(sample_coords, k=2)
        nn = distances[:, 1]
    else:
        nn = np.array([0.1], dtype=float)

    return {
        "mean": float(np.mean(nn)),
        "median": float(np.median(nn)),
    }


def coverage_uniformity_only(enhanced_df: pd.DataFrame) -> float:
    valid_e = enhanced_df[enhanced_df["pred"] != 0] if "pred" in enhanced_df.columns else enhanced_df
    if len(valid_e) == 0:
        return 0.0

    h_range = [valid_e["h"].min(), valid_e["h"].max()]
    theta_range = [valid_e["theta"].min(), valid_e["theta"].max()]
    h_bins = np.linspace(h_range[0], h_range[1], 20)
    theta_bins = np.linspace(theta_range[0], theta_range[1], 20)

    enhanced_coverage = np.zeros((len(h_bins) - 1, len(theta_bins) - 1))
    for i, (h_min, h_max) in enumerate(zip(h_bins[:-1], h_bins[1:])):
        for j, (t_min, t_max) in enumerate(zip(theta_bins[:-1], theta_bins[1:])):
            mask = (valid_e["h"] >= h_min) & (valid_e["h"] < h_max) & (
                valid_e["theta"] >= t_min
            ) & (valid_e["theta"] < t_max)
            enhanced_coverage[i, j] = mask.sum()

    flat = enhanced_coverage.flatten()
    nonzero = flat[flat > 0]
    if len(nonzero) == 0:
        return 0.0
    return float(1 / (1 + np.std(nonzero)))


def segmentation_readiness_key_only(median_spacing: float) -> dict:
    reference_template_spacing_m = 0.05
    suitability = (
        min(1.0, reference_template_spacing_m / median_spacing)
        if median_spacing > 0
        else 0.0
    )
    return {
        "template_spacing_suitability": float(suitability),
        "reference_template_spacing_m": float(reference_template_spacing_m),
        "current_median_spacing": float(median_spacing),
    }


def characterize(tunnel_id: str) -> dict:
    enhanced_df = load_enhanced_data(tunnel_id)
    nn = final_nn_key_only(enhanced_df)
    cov_uni = coverage_uniformity_only(enhanced_df)
    readiness = segmentation_readiness_key_only(nn["median"])

    return {
        "tunnel_id": tunnel_id,
        "algorithm_3_results": {
            "enhanced_density": {
                "total_points_after": int(len(enhanced_df)),
                "final_nn_distances": {
                    "median": nn["median"],
                    "mean": nn["mean"],
                },
            },
            "upsampling_quality": {
                "coverage_uniformity": cov_uni,
            },
            "segmentation_readiness": readiness,
        },
    }


def main():
    p = argparse.ArgumentParser(description="BO key-only enhanced characteristics")
    p.add_argument("tunnel_id", type=str)
    args = p.parse_args()

    out = characterize(args.tunnel_id)
    chars_dir = tunnel_characteristics_dir(args.tunnel_id)
    os.makedirs(chars_dir, exist_ok=True)
    out_path = os.path.join(chars_dir, "enhanced_characteristics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
