#!/usr/bin/env python3
"""Detected / SAM prompts stage: key BO observation fields only."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sam4tun.plugins.paths import tunnel_characteristics_dir, tunnel_pipeline_dir


def load_sam_workflow_data(tunnel_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = tunnel_pipeline_dir(tunnel_id)
    det_path = os.path.join(base, "detected.csv")
    enh_path = os.path.join(base, "enhanced.csv")
    if not os.path.exists(det_path):
        raise FileNotFoundError(det_path)
    if not os.path.exists(enh_path):
        raise FileNotFoundError(enh_path)
    return pd.read_csv(det_path, comment="#"), pd.read_csv(enh_path, comment="#")


def prompt_distribution_key_only(detection_points_df: pd.DataFrame) -> dict:
    if not ("X" in detection_points_df.columns and "Y" in detection_points_df.columns):
        raise ValueError(
            f"Expected X, Y in detection points; got {list(detection_points_df.columns)}"
        )
    valid = detection_points_df.dropna(subset=["X", "Y"])
    valid_count = len(valid)
    sam_template_distribution: dict = {
        "prompt_density": 0.0,
        "coverage_area": 0.0,
    }
    spacing_analysis: dict = {"potential_template_overlap": None}

    if valid_count == 0:
        return {
            "sam_template_distribution": sam_template_distribution,
            "prompt_spacing_analysis": spacing_analysis,
        }

    x0, x1 = float(valid["X"].min()), float(valid["X"].max())
    y0, y1 = float(valid["Y"].min()), float(valid["Y"].max())
    coverage_area = (x1 - x0) * (y1 - y0)
    sam_template_distribution = {
        "coverage_area": float(coverage_area),
        "prompt_density": float(valid_count / coverage_area) if coverage_area > 0 else 0.0,
    }

    sam_template_width_px = 1250
    if valid_count > 1:
        coords = valid[["X", "Y"]].values
        tree = cKDTree(coords)
        k_nn = min(3, valid_count)
        distances, _ = tree.query(coords, k=k_nn)
        if distances.ndim == 2 and distances.shape[1] > 1:
            nn_distances = distances[:, 1]
            overlap = float(np.sum(nn_distances < sam_template_width_px) / len(nn_distances))
        else:
            overlap = None
    else:
        overlap = None

    spacing_analysis = {"potential_template_overlap": overlap}

    return {
        "sam_template_distribution": sam_template_distribution,
        "prompt_spacing_analysis": spacing_analysis,
    }


def prompt_effectiveness_key_only(
    detection_points_df: pd.DataFrame, enhanced_df: pd.DataFrame
) -> dict:
    valid_detection = detection_points_df.dropna(subset=["X", "Y"])
    if "pred" in enhanced_df.columns:
        target_points = enhanced_df[enhanced_df["pred"] == 7]
    else:
        target_points = enhanced_df

    if len(valid_detection) == 0 or len(target_points) == 0:
        return {
            "prompt_to_target_ratio": 0.0,
            "sam_coverage_analysis": {"estimated_template_coverage": 0.0},
        }

    prompt_to_target_ratio = len(valid_detection) / len(target_points)

    template_area = 1250 * 3240
    total_detection_coverage = len(valid_detection) * template_area
    x0, x1 = float(valid_detection["X"].min()), float(valid_detection["X"].max())
    y0, y1 = float(valid_detection["Y"].min()), float(valid_detection["Y"].max())
    depth_map_area = (x1 - x0) * (y1 - y0)
    coverage_efficiency = (
        min(1.0, total_detection_coverage / depth_map_area) if depth_map_area > 0 else 0.0
    )

    return {
        "prompt_to_target_ratio": float(prompt_to_target_ratio),
        "sam_coverage_analysis": {
            "estimated_template_coverage": float(coverage_efficiency),
        },
    }


def characterize(tunnel_id: str) -> dict:
    det_df, enh_df = load_sam_workflow_data(tunnel_id)
    return {
        "tunnel_id": tunnel_id,
        "sam_workflow_analysis": {
            "prompt_distribution": prompt_distribution_key_only(det_df),
            "prompt_effectiveness": prompt_effectiveness_key_only(det_df, enh_df),
        },
    }


def main():
    p = argparse.ArgumentParser(description="BO key-only detected characteristics")
    p.add_argument("tunnel_id", type=str)
    args = p.parse_args()

    out = characterize(args.tunnel_id)
    chars_dir = tunnel_characteristics_dir(args.tunnel_id)
    os.makedirs(chars_dir, exist_ok=True)
    out_path = os.path.join(chars_dir, "detected_characteristics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
