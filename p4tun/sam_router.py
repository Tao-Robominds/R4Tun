"""
Route SAM stage by pattern type (from pattern_type.json).

- simple_staggered (T1, T2): 4-2_sam.py (standard)
- continuous (T3):           4-2_sam.py (standard; sam_continuous performs worse, A/B verified)
- complex_staggered (T4, T5): 4-2_sam_wrap_around.py (always wraparound)

Usage:
  python -m p4tun.sam_router <tunnel_dir>
  Prints script name (e.g. 4-2_sam.py) to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

SCRIPTS = {
    "simple_staggered": "4-2_sam.py",
    "continuous": "4-2_sam.py",  # standard; sam_continuous worse on 3-1 (mIoU 0.457 vs 0.594)
    "complex_staggered": "4-2_sam_wrap_around.py",
}
DEFAULT_SCRIPT = "4-2_sam.py"


def choose(tunnel_dir: str) -> str:
    path = os.path.join(tunnel_dir, "pattern_type.json")
    if not os.path.exists(path):
        return DEFAULT_SCRIPT
    try:
        with open(path) as f:
            data = json.load(f)
        pt = data.get("pattern_type") or ""
        return SCRIPTS.get(pt, DEFAULT_SCRIPT)
    except (json.JSONDecodeError, OSError):
        return DEFAULT_SCRIPT


def main() -> None:
    ap = argparse.ArgumentParser(description="Choose SAM script by pattern type")
    ap.add_argument("tunnel_dir", help="Tunnel directory (e.g. data/3-1)")
    args = ap.parse_args()
    print(choose(args.tunnel_dir))


if __name__ == "__main__":
    main()
