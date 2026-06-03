#!/usr/bin/env python3
"""Build ring descriptors for held-out 50-ring panel."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BO_DIR = Path(__file__).resolve().parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.ceiling_gate import REPO_ROOT  # noqa: E402
from lib.held_out_descriptors import build_panel_descriptors  # noqa: E402

DEFAULT_PANEL = REPO_ROOT / "data" / "held-out" / "_manifests" / "data_v6_50ring_calibration_panel.csv"
DEFAULT_HELD_OUT = REPO_ROOT / "data" / "held-out"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--held-out-root", default=str(DEFAULT_HELD_OUT))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "logs" / "stage_a_candidates_v1"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_panel_descriptors(Path(args.panel), Path(args.held_out_root))
    csv_path = out_dir / "ring_descriptors.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "n_rings": int(len(df)),
        "panel": str(Path(args.panel).resolve()),
        "held_out_root": str(Path(args.held_out_root).resolve()),
        "density_tiers": df["density_tier"].value_counts().to_dict(),
        "k_span_tiers": df["k_span_tier"].value_counts().to_dict(),
        "direction_tiers": df["direction_tier"].value_counts().to_dict(),
        "coverage_tiers": df["coverage_tier"].value_counts().to_dict(),
    }
    meta_path = out_dir / "ring_descriptors.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "meta": str(meta_path), "n": len(df)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
