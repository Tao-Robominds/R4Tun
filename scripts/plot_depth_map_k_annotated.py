#!/usr/bin/env python3
"""
Plot depth map with GT K positions only (R0K..R6K).

Reads all_segments_gt.csv and depth_map.png (or depth_map_outlier.npy), draws
red circles + labels for each K, saves data/<tunnel_id>/depth_map_annotated.png.

Run from repo root:
  python scripts/plot_depth_map_k_annotated.py [--tunnel 4-1] [--data-dir data]

Example:
  python scripts/plot_depth_map_k_annotated.py --tunnel 4-1
"""
import os
import sys

# Run from repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.chdir(_REPO_ROOT)

from p4tun.plot_depth_map_k_annotated import main

if __name__ == "__main__":
    main()
