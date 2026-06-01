#!/usr/bin/env python3
"""Build stratified 25/25 Stage A / Stage B split from ring descriptors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BO_DIR = Path(__file__).resolve().parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.ceiling_gate import REPO_ROOT  # noqa: E402
from lib.stage_split import build_stage_split, write_split_outputs  # noqa: E402

import pandas as pd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptors", default=str(REPO_ROOT / "logs" / "stage_a_candidates_v1" / "ring_descriptors.csv"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "logs" / "stage_a_candidates_v1"))
    ap.add_argument("--seed", type=int, default=20260529)
    args = ap.parse_args()

    df = pd.read_csv(args.descriptors)
    result = build_stage_split(df, seed=args.seed)
    paths = write_split_outputs(result, Path(args.out_dir).resolve())
    summary = {**paths, "balance_passed": result["balance"]["passed"]}
    print(json.dumps(summary, indent=2))
    return 0 if result["balance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
