"""
Build all_segments.csv from detected K (e.g. detected_k_dbscan.csv) + group_offsets.
GT-free at inference: no all_segments_gt.csv required.

Usage:
  python build_all_segments_group_offsets.py 4-1 --data-dir data [--k-file detected_k_dbscan.csv]
  Run from repo root, or set data-dir to point to data/.
"""

import os
import sys
import json
import argparse

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IRREGULAR_ROOT = os.path.dirname(SCRIPT_DIR)
DETECTION_PATH = os.path.join(IRREGULAR_ROOT, "2_detection", "2_detection.py")
import importlib.util
_spec = importlib.util.spec_from_file_location("detection", DETECTION_PATH)
_detection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detection)
expand_k_with_grouped_offsets = _detection.expand_k_with_grouped_offsets


def main():
    parser = argparse.ArgumentParser(description="Build all_segments.csv from K + group_offsets (GT-free)")
    parser.add_argument("tunnel_id", help="e.g. 4-1")
    parser.add_argument("--data-dir", default="data", help="Base data directory (default: data)")
    parser.add_argument("--k-file", default="detected_k_dbscan.csv", help="K positions CSV (relative to tunnel dir)")
    args = parser.parse_args()

    # Resolve data_dir relative to repo root if needed
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        repo_root = os.path.dirname(os.path.dirname(IRREGULAR_ROOT))
        data_dir = os.path.join(repo_root, data_dir)

    tunnel_dir = os.path.join(data_dir, args.tunnel_id)
    k_path = os.path.join(tunnel_dir, args.k_file)
    if not os.path.exists(k_path):
        raise FileNotFoundError(f"K file not found: {k_path}")

    params_path = os.path.join(
        IRREGULAR_ROOT, "2_detection", "parameters", args.tunnel_id,
        "parameters_detection.json",
    )
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters not found: {params_path}")

    depth_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    k_positions = pd.read_csv(k_path)
    if "Ring" not in k_positions.columns:
        k_positions.insert(0, "Ring", range(len(k_positions)))
    k_positions = k_positions.sort_values("Ring").reset_index(drop=True)

    with open(params_path) as f:
        params = json.load(f)
    stagger_groups = params.get("stagger_groups", {"A": list(range(len(k_positions)))})
    group_offsets = params.get("group_offsets", {})

    import numpy as np
    depth = np.load(depth_path)
    img_height = int(depth.shape[0])

    all_segments = expand_k_with_grouped_offsets(
        k_positions,
        img_height=img_height,
        stagger_groups=stagger_groups,
        group_offsets=group_offsets,
    )
    out_path = os.path.join(tunnel_dir, "all_segments.csv")
    all_segments.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(all_segments)} segments, from {len(k_positions)} K + group_offsets)")


if __name__ == "__main__":
    main()
