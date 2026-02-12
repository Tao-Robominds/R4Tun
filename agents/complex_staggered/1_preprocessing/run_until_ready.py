#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run analyst then coder in a loop until pre_ready_for_detection is true.
Use the same Python (e.g. venv) to run this script so analyst and coder use it too:
  /path/to/P4Tun_Off/venv/bin/python run_until_ready.py 1-4
"""

import json
import subprocess
import sys
from pathlib import Path


def main():
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "1-4"
    max_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "parameters" / tunnel_id
    intrinsics_path = params_dir / "intrinsics.json"
    analyst_script = script_dir / "analyst.py"
    coder_script = script_dir / "coder.py"
    python = sys.executable

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{max_rounds} — tunnel {tunnel_id}")
        print("="*60)

        print("\n--- Running analyst ---")
        r1 = subprocess.run([python, str(analyst_script), tunnel_id], cwd=script_dir.parent.parent.parent)
        if r1.returncode != 0:
            print(f"Analyst exited with {r1.returncode}; stopping.")
            sys.exit(1)

        print("\n--- Running coder ---")
        r2 = subprocess.run([python, str(coder_script), tunnel_id], cwd=script_dir.parent.parent.parent)
        if r2.returncode != 0:
            print(f"Coder exited with {r2.returncode}; stopping.")
            sys.exit(1)

        if not intrinsics_path.exists():
            print("intrinsics.json not found; continuing next round.")
            continue

        with open(intrinsics_path) as f:
            intrinsics = json.load(f)
        ready = intrinsics.get("pre_ready_for_detection", False)
        print(f"\npre_ready_for_detection: {ready}")
        if ready:
            print(f"\n✅ Done after {round_num} round(s).")
            return

    print(f"\n❌ Stopped after {max_rounds} rounds without pre_ready_for_detection true.")
    sys.exit(1)


if __name__ == "__main__":
    main()
