#!/usr/bin/env python3
"""Build unified experience bank from v3/v4/v5 BO trial pools."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_BO_DIR = Path(__file__).resolve().parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.ceiling_gate import REPO_ROOT  # noqa: E402
from lib.experience_bank import POOLS, build_experience_bank  # noqa: E402

SCHEMA = {
    "description": "Unified BO experience bank (v3 random + v4 SAM4Tun + v5 GT-derived)",
    "normalisation": {
        "layout_k_center_norm": "k_y / image_height (circumferential axis)",
        "layout_k_center_norm_w": "k_y / image_width (secondary)",
        "layout_k_width_norm": "K arc span / image_height",
        "layout_ab_offset_norm_json": "per-block offset / image_height",
        "form_boundary_gap_norm": "det_min_y_gap_px / image_height",
    },
    "source_types": {k: v["source_type"] for k, v in POOLS.items()},
    "expected_rows": 1440,
    "pools": list(POOLS.keys()),
}


def _gate(bank: pd.DataFrame, expected_n: int) -> dict:
    criteria = {
        "row_count": len(bank) == expected_n,
        "has_all_pools": set(bank["experience_pool"].unique()) == set(POOLS.keys()),
        "has_source_types": bool(bank["source_type"].notna().all()),
        "has_miou": bool(bank["label_gt_miou"].notna().all()),
        "has_layout_norm": bool(bank["layout_k_center_norm"].notna().all()),
        "has_ring_features": bool(bank["ring_segment_count"].notna().all()),
    }
    per_pool = bank.groupby("experience_pool").size().to_dict()
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "n_rows": int(len(bank)),
        "expected_n": expected_n,
        "per_pool_counts": per_pool,
        "n_rings": int(bank["ring_id"].nunique()),
        "mean_gt_miou_by_pool": {
            str(k): round(float(v), 4)
            for k, v in bank.groupby("source_type")["label_gt_miou"].mean().items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "logs" / "experience_bank_v1"))
    ap.add_argument("--manifest", default=str(REPO_ROOT / "data" / "bo_calibration" / "MANIFEST.json"))
    ap.add_argument("--corpus-dir", default=str(REPO_ROOT / "data" / "bo_calibration"))
    ap.add_argument("--expected-n", type=int, default=1440)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = build_experience_bank(
        manifest_path=Path(args.manifest),
        corpus_dir=Path(args.corpus_dir),
    )

    csv_path = out_dir / "experience_bank.csv"
    bank.to_csv(csv_path, index=False)

    schema_path = out_dir / "experience_bank_schema.json"
    schema = dict(SCHEMA)
    schema["columns"] = list(bank.columns)
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    gate = _gate(bank, args.expected_n)
    gate["evidence_path"] = str(out_dir / "experience_bank_gate.json")
    gate_path = out_dir / "experience_bank_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    summary = {
        "n_rows": len(bank),
        "experience_bank": str(csv_path),
        "schema": str(schema_path),
        "gate": str(gate_path),
        "gate_passed": gate["passed"],
    }
    (out_dir / "experience_bank_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
