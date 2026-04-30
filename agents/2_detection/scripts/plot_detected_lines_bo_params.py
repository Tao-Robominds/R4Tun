"""
Generate data/<tunnel_id>/detected_lines.png using BO-best K detection params
so the visualization matches what the regulator actually sees.

Usage (from repo root):
  ./venv/bin/python agents/2_detection/scripts/plot_detected_lines_bo_params.py 5-1 --data-dir data
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root (scripts -> 2_detection -> agents -> repo)
SCRIPT_DIR = Path(__file__).resolve().parent
IRREGULAR_2 = SCRIPT_DIR.parent  # 2_detection
REPO_ROOT = IRREGULAR_2.parent.parent  # agents, repo
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(IRREGULAR_2))

# Detection module (lines + visualization)
import importlib.util
_spec_det = importlib.util.spec_from_file_location("detection", IRREGULAR_2 / "2_detection.py")
_detection = importlib.util.module_from_spec(_spec_det)
_spec_det.loader.exec_module(_detection)
detect_lines = _detection.detect_lines
visualize_detection = _detection.visualize_detection

# K detection (p4tun)
P4TUN = REPO_ROOT / "p4tun"
_spec_k = importlib.util.spec_from_file_location("k_detection", P4TUN / "4-1-1_geo_k_detection.py")
_k_mod = importlib.util.module_from_spec(_spec_k)
_spec_k.loader.exec_module(_k_mod)
run_k_detection = _k_mod.run_k_detection


def main():
    parser = argparse.ArgumentParser(description="Plot detected lines using BO-best params")
    parser.add_argument("tunnel_id", help="e.g. 5-1")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--bo-trial", default=None, help="BO trial JSON path or e.g. 085 for k_detect_<tunnel>_085.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    tunnel_dir = data_dir / args.tunnel_id
    depth_path = tunnel_dir / "depth_map_outlier.npy"
    ring_path = tunnel_dir / "ring_count.txt"
    if not depth_path.exists() or not ring_path.exists():
        raise FileNotFoundError(f"Need {depth_path} and {ring_path}")

    # Load BO-best params
    if args.bo_trial is None or args.bo_trial.isdigit():
        trial_id = args.bo_trial or "085"
        logs_dir = REPO_ROOT / "p4tun" / "bo" / "k_logs" / "dbscan"
        pattern = f"k_detect_{args.tunnel_id}_{int(trial_id):03d}.json"
        trial_path = logs_dir / pattern
        if not trial_path.exists():
            # Find best by objective
            best_score = 1e9
            trial_path = None
            for p in logs_dir.glob(f"k_detect_{args.tunnel_id}_*.json"):
                with open(p) as f:
                    d = json.load(f)
                if d.get("objective_value", 1e9) < best_score:
                    best_score = d["objective_value"]
                    trial_path = p
            if trial_path is None:
                raise FileNotFoundError(f"No BO logs for {args.tunnel_id} in {logs_dir}")
        bo_path = trial_path
    else:
        bo_path = Path(args.bo_trial)
    with open(bo_path) as f:
        trial = json.load(f)
    params = trial["params"]
    print(f"Using BO params from {bo_path.name} (objective={trial.get('objective_value', 'N/A')})")

    depth = np.load(str(depth_path))
    ring_count = int(ring_path.read_text().strip())

    # Line detection with BO params
    line_data = detect_lines(depth, params)
    print(f"  Lines: {len(line_data['positive_lines'])} pos, {len(line_data['negative_lines'])} neg")

    # K positions with BO params (DBSCAN + regulator)
    k_positions = run_k_detection(
        depth, ring_count, "dbscan", params,
        tunnel_id=args.tunnel_id, base_dir=str(data_dir), verbose=False,
    )
    k_positions = k_positions.sort_values("X").reset_index(drop=True)
    print(f"  K positions: {len(k_positions)}")

    # Draw and save
    visualize_detection(line_data, k_positions, str(tunnel_dir), all_segments=None)
    out_path = tunnel_dir / "detected_lines.png"
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
