#!/usr/bin/env python3
"""CLI wrapper for official fixed B+C+D ring-level preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESSING_DIR = SCRIPT_DIR.parent
REPO_ROOT = PREPROCESSING_DIR.parent.parent
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))
from context_preprocessing import run_context_trial  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tunnel-id", required=True)
    p.add_argument("--ring-id", required=True, type=int)
    p.add_argument("--context-radius", type=int, default=1)
    p.add_argument("--output-root", default="logs/context_preprocessing_v1")
    p.add_argument(
        "--reference-base-dir",
        default="data/ablation/baseline",
        help="Only used for reading parameters_preprocessing.json fallback path.",
    )
    args = p.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    out_dir = run_context_trial(
        tunnel_id=str(args.tunnel_id),
        ring_id=int(args.ring_id),
        context_radius=int(args.context_radius),
        output_root=output_root,
        reference_base_dir=str(args.reference_base_dir),
    )
    meta_path = out_dir / "trial_meta.json"
    if meta_path.is_file():
        print(json.dumps(json.loads(meta_path.read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
