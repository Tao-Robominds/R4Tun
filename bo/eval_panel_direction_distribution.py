#!/usr/bin/env python3
"""Re-evaluate BO calibration (6) and held-out (50) direction tiers (plus/minus)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "bo") not in sys.path:
    sys.path.insert(0, str(REPO / "bo"))

from lib.ceiling_gate import REPO_ROOT
from lib.held_out_descriptors import build_ring_descriptor


def _ring_keys_from_bo_manifest(manifest_path: Path) -> list[str]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [str(r["ring_key"]) for r in data["rings"]]


def _evaluate_panel(ring_keys: list[str], data_root: Path, panel_name: str) -> pd.DataFrame:
    rows = []
    for ring_key in ring_keys:
        rows.append(build_ring_descriptor(ring_key, data_root))
    df = pd.DataFrame(rows)
    df["panel"] = panel_name
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "direction_panel_eval_v1",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bo_root = REPO_ROOT / "data" / "bo_calibration"
    held_root = REPO_ROOT / "data" / "held-out"
    bo_keys = _ring_keys_from_bo_manifest(bo_root / "MANIFEST.json")
    held_panel = held_root / "_manifests" / "data_v6_50ring_calibration_panel.csv"
    held_keys = pd.read_csv(held_panel)["ring_key"].astype(str).tolist()

    bo_df = _evaluate_panel(bo_keys, bo_root, "bo_calibration")
    held_df = _evaluate_panel(held_keys, held_root, "held_out")
    all_df = pd.concat([bo_df, held_df], ignore_index=True)
    all_df.to_csv(out_dir / "ring_direction.csv", index=False)

    def _counts(df: pd.DataFrame) -> dict[str, int]:
        vc = df["direction_tier"].value_counts()
        return {str(k): int(v) for k, v in vc.items()}

    summary = {
        "bo_calibration": {
            "n_rings": len(bo_df),
            "direction_tiers": _counts(bo_df),
            "plus": int((bo_df["direction_tier"] == "plus").sum()),
            "minus": int((bo_df["direction_tier"] == "minus").sum()),
            "unknown": int((bo_df["direction_tier"] == "unknown").sum()),
        },
        "held_out": {
            "n_rings": len(held_df),
            "direction_tiers": _counts(held_df),
            "plus": int((held_df["direction_tier"] == "plus").sum()),
            "minus": int((held_df["direction_tier"] == "minus").sum()),
            "unknown": int((held_df["direction_tier"] == "unknown").sum()),
        },
        "data_roots": {
            "bo_calibration": str(bo_root),
            "held_out": str(held_root),
        },
        "method": "gt_layout spatial_order_by_label: plus=rotation of [1..n], minus=rotation of [n..1]",
    }
    (out_dir / "panel_direction_counts.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
