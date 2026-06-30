#!/usr/bin/env python3
"""Run all five parameterized SAM4Tun agent stages for a tunnel id."""

import os
import subprocess
import sys

STAGES = [
    "unfolding.py",
    "denoising.py",
    "enhancing.py",
    "detecting.py",
    "sam.py",
]

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
SAM4TUN_ROOT = os.path.dirname(AGENTS_DIR)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <tunnel_id>")
        sys.exit(1)
    tunnel_id = sys.argv[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(SAM4TUN_ROOT, "segment-anything")
    py = sys.executable
    for stage in STAGES:
        path = os.path.join(AGENTS_DIR, stage)
        print(f"\n=== {stage} ===")
        subprocess.run([py, path, tunnel_id], cwd=SAM4TUN_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
