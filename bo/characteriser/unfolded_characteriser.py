#!/usr/bin/env python3
"""Unfolded stage: key BO observation fields only."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sam4tun.plugins.paths import tunnel_characteristics_dir, tunnel_pipeline_dir


def analyze_unwrapped_key_only(csv_path: str, tunnel_id: str) -> dict:
    df = pd.read_csv(csv_path, comment="#")
    expected = ["x", "y", "z", "intensity", "r", "theta", "h"]
    available = [c for c in expected if c in df.columns]
    df = df[available]

    np.random.seed(42)

    cyl = {}
    if all(c in df.columns for c in ["r", "theta", "h"]):
        r_values = df["r"]
        r_pcts = np.percentile(r_values, [10, 99])
        theta_min = float(df["theta"].min())
        theta_max = float(df["theta"].max())
        cyl = {
            "r_percentiles": {
                "p10": round(float(r_pcts[0]), 4),
                "p99": round(float(r_pcts[1]), 4),
            },
            "h_span": float(df["h"].max() - df["h"].min()),
            "theta_span": float(theta_max - theta_min),
            # Key field is theta_range[0] only (see plan / key_characteristics doc)
            "theta_range": [theta_min],
        }

    intensity_analysis = {}
    if "intensity" in df.columns:
        intensity_analysis = {
            "median": float(df["intensity"].median()),
            "min": float(df["intensity"].min()),
        }

    point_density = {}
    if all(c in df.columns for c in ["x", "y", "z"]):
        sample_size = min(10000, len(df))
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        sample_points = df.iloc[sample_indices][["x", "y", "z"]].values
        tree = cKDTree(sample_points)
        distances, _ = tree.query(sample_points, k=2)
        nn = distances[:, 1]
        point_density = {
            "median_nn_distance": float(np.median(nn)),
            "std_nn_distance": float(np.std(nn)),
        }

    return {
        "tunnel_id": tunnel_id,
        "unfolding_results": {
            "cylindrical_coordinates": cyl,
            "intensity_analysis": intensity_analysis,
            "point_density": point_density,
        },
    }


def main():
    p = argparse.ArgumentParser(description="BO key-only unfolded characteristics")
    p.add_argument("tunnel_id", type=str)
    args = p.parse_args()

    csv_path = os.path.join(tunnel_pipeline_dir(args.tunnel_id), "unwrapped.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"Input not found: {csv_path}")

    out = analyze_unwrapped_key_only(csv_path, args.tunnel_id)
    chars_dir = tunnel_characteristics_dir(args.tunnel_id)
    os.makedirs(chars_dir, exist_ok=True)
    out_path = os.path.join(chars_dir, "unfolded_characteristics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
