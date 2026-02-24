#!/usr/bin/env python3
"""
Build data/<tunnel_id>/all_segments_gt.csv from pipeline ground truth.

Re-generates segment centres (Ring, Block, X, Y) from unwrapped.csv so every
block falls inside the depth map. Uses depth_map_grid.json when present, else
fits unwrapped extent into the depth map shape.

Run from repo root:
  python scripts/build_all_segments_gt.py <tunnel_id> [--data-dir data] [--output all_segments_gt.csv]

Example:
  python scripts/build_all_segments_gt.py 4-1
"""
import os
import sys

# Run from repo root; p4tun is sibling of scripts/
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Change to p4tun so relative paths in build_all_segments_gt (e.g. parameters/) resolve
_script_dir = os.path.join(_REPO_ROOT, "p4tun")
os.chdir(_REPO_ROOT)

from p4tun.build_all_segments_gt import main

if __name__ == "__main__":
    main()
