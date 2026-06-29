#!/usr/bin/env python3
"""Run all six SAM4Tun pipeline stages for a tunnel id."""

import os
import subprocess
import sys

STAGES = [
    "1_upfolding.py",
    "2_denoising.py",
    "3_enhancing.py",
    "4_detection.py",
    "5_sam.py",
    "6_evaluation.py",
]

SAM4TUN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <tunnel_id>")
        sys.exit(1)
    tunnel_id = sys.argv[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(SAM4TUN_ROOT, "segment-anything")
    py = sys.executable
    for stage in STAGES:
        path = os.path.join(SAM4TUN_ROOT, stage)
        print(f"\n=== {stage} ===")
        subprocess.run([py, path, tunnel_id], cwd=SAM4TUN_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
