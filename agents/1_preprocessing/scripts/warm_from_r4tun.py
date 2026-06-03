#!/usr/bin/env python3
"""Map r4tun reference stage JSONs into ``parameters_preprocessing.json`` for a ring folder.

Example::

    ./venv/bin/python agents/1_preprocessing/scripts/warm_from_r4tun.py 4-1 110 \\
        --reference-dir r4tun/references/data/4-1/parameters
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tunnel_id", help="e.g. 4-1")
    ap.add_argument("ring_id", type=int, help="e.g. 110")
    ap.add_argument(
        "--reference-dir",
        type=Path,
        default=REPO_ROOT / "r4tun" / "references" / "data" / "4-1" / "parameters",
        help="Directory with parameters_unfolding_*.json etc.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output dir (default: agents/1_preprocessing/parameters/<tunnel>/r<ring>/)",
    )
    args = ap.parse_args()

    ref_dir: Path = args.reference_dir
    if not ref_dir.is_dir():
        print(f"Missing reference dir: {ref_dir}", file=sys.stderr)
        return 1

    def _load_glob(pattern: str) -> dict:
        matches = list(ref_dir.glob(pattern))
        if len(matches) != 1:
            print(f"Expected exactly one {pattern}, got {matches}", file=sys.stderr)
            sys.exit(1)
        with open(matches[0], encoding="utf-8") as f:
            return json.load(f)

    unfolding = _load_glob("parameters_unfolding_*.json")
    denoising = _load_glob("parameters_denoising_*.json")
    enhancing = _load_glob("parameters_enhancing_*.json")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = (
            REPO_ROOT
            / "agents"
            / "1_preprocessing"
            / "parameters"
            / args.tunnel_id
            / f"r{int(args.ring_id)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessing = {
        "tunnel_diameter": unfolding["diameter"],
        "ring_spacing": 1.2,
        "depth_map_resolution": enhancing["resolution"],
        "vertical_filter_window": unfolding["vertical_filter_window"],
        "ransac_threshold": unfolding["ransac_threshold"],
        "ransac_probability": unfolding["ransac_probability"],
        "ransac_inlier_ratio": unfolding["ransac_inlier_ratio"],
        "ransac_sample_size": unfolding["ransac_sample_size"],
        "ransac_initial_iterations": unfolding["ransac_initial_iterations"],
        "ransac_inlier_threshold_multiplier": unfolding["ransac_inlier_threshold_multiplier"],
        "radius_min": denoising["mask_r_low"],
        "radius_max": denoising["mask_r_high"],
        "y_step": denoising["y_step"],
        "z_step": denoising["z_step"],
        "gradient_threshold": denoising["grad_threshold"],
        "smoothing_window_size": denoising["smoothing_window_size"],
        "smoothing_offset": denoising["smoothing_offset"],
        "default_cutoff_z": denoising["default_cutoff_z"],
        "double_zero_cutoff": True,
        "target_distances": [
            enhancing["upsampling_stage1_target_distance"],
            enhancing["upsampling_stage2_target_distance"],
            enhancing["upsampling_stage3_target_distance"],
        ],
        "interpolation_window": enhancing["window_size"],
        "curvature_threshold_enh": enhancing["curvature_threshold"],
        "depth_threshold_low": enhancing["depth_threshold_low"],
        "depth_threshold_high": enhancing["depth_threshold_high"],
        "inter_radius": enhancing["inter_radius"],
        "duplicate_threshold": enhancing["duplicate_threshold"],
        "n_segment_start": -1,
        "n_segment_end": -1,
        "num_neighbors": enhancing["num_neighbors"],
        "num_interpolations": enhancing["num_interpolations"],
        "outlier_high_density_ring_start": -1,
        "outlier_high_density_ring_end": -1,
        "max_outlier_points": 5000,
        "outlier_interpolation_radius": enhancing["inter_radius"],
        "outlier_num_interpolations": enhancing["num_interpolations"],
        "outlier_duplicate_threshold": enhancing["duplicate_threshold"],
        "outlier_bidirectional": False,
        "outlier_depth_map_window": 1,
        "_warm_source": str(ref_dir.relative_to(REPO_ROOT)),
    }

    out_path = out_dir / "parameters_preprocessing.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preprocessing, f, indent=2)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")

    # Detection JSON schema differs from r4tun ``parameters_detecting_*`` — copy a ring BO template
    # that includes ``per_ring_offsets`` (required by agents/2_detection).
    det_template = (
        REPO_ROOT
        / "agents"
        / "2_detection"
        / "parameters"
        / "_bo_v1"
        / args.tunnel_id
        / "r116"
        / "parameters_detection.json"
    )
    if det_template.is_file():
        det_root = REPO_ROOT / "agents" / "2_detection" / "parameters" / args.tunnel_id / f"r{int(args.ring_id)}"
        det_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(det_template, det_root / "parameters_detection.json")
        print(f"Wrote {det_root.relative_to(REPO_ROOT)}/parameters_detection.json (from _bo_v1 template)")
    else:
        print(
            f"Warning: no detection template at {det_template.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
