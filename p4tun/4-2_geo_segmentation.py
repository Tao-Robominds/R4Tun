"""
Geometric-only segmentation for complex staggered tunnels (4-1, 5-1).

Assigns every point in the depth map to a block using only geometric regions:
segment centres from all_segments.csv + per-block-type half-heights and width.
Y-axis wraps (cylindrical); overlaps resolved by nearest centre. Optional
boundary snapping to groove edges. No GPU, no SAM — fast.

Pipeline:
    Preprocessing → depth_map.png, enhanced.csv, pixel_to_point.pkl
    Detection or GT → all_segments.csv (Ring, Block, X, Y in pixels)
    4-2_geo_segmentation.py → final.csv (and optionally final_geo.csv)

Previously validated: geometric segmentation scores higher mIoU than SAM on
these tunnels (see reports/journal_*.md). Use this script to produce
final.csv for evaluation.
"""

import os
import sys
import json
import argparse
import shutil
import importlib.util

# Load geometric segmentation from agents (same logic as 3_geometric.py)
_agents_geo_path = os.path.join(
    os.path.dirname(__file__), "..", "agents", "irregular", "3_segmentation", "3_geometric.py"
)
_spec = importlib.util.spec_from_file_location("_geo", _agents_geo_path)
_geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geo)
run_geometric = _geo.run_geometric


def load_p4tun_parameters(tunnel_id: str) -> dict:
    """Load geometric parameters from p4tun/parameters if present."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "parameters", tunnel_id, "parameters_geometric.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Geometric-only segmentation (no SAM) for complex staggered tunnels."
    )
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1, 5-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument(
        "--segments-file",
        default=None,
        help="Segments CSV (default: <data_dir>/<tunnel_id>/all_segments.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Also copy final.csv to this name (e.g. final_geo.csv) for comparison with SAM",
    )
    args = parser.parse_args()

    override = load_p4tun_parameters(args.tunnel_id)
    result = run_geometric(
        args.tunnel_id,
        base_dir=args.data_dir,
        segments_file=args.segments_file,
        override_params=override if override else None,
    )

    print(f"Groove alignment score: {result['groove_score']:.4f}")

    if args.output:
        tunnel_dir = os.path.join(args.data_dir, args.tunnel_id)
        src = os.path.join(tunnel_dir, "final.csv")
        dst = os.path.join(tunnel_dir, args.output)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied final.csv -> {args.output}")


if __name__ == "__main__":
    main()
