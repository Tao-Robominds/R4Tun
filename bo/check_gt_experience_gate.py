#!/usr/bin/env python3
"""Validate GT-anchor BO experience trial pools (gt_layout warm-start present)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_BO_DIR = Path(__file__).resolve().parent
REPO_ROOT = _BO_DIR.parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.layout_bo import ORACLE_TRIAL_KINDS  # noqa: E402
from lib.manifest import load_manifest_rings, n_evals_for_ring_entry  # noqa: E402


def check_trials_df(
    df: pd.DataFrame,
    *,
    case_id: str = "panel",
    expected_n: int | None = None,
) -> dict:
    criteria: dict[str, bool] = {}
    details: dict[str, object] = {"case_id": case_id, "n_trials": int(len(df))}

    if df.empty:
        criteria["non_empty"] = False
        return {"passed": False, "criteria": criteria, "details": details}

    criteria["non_empty"] = True
    if expected_n is not None:
        criteria["n_trials_expected"] = len(df) == expected_n
        details["expected_n"] = expected_n

    gt_ceiling = df[df["kind"] == "gt_layout_ceiling_r"]
    criteria["gt_ceiling_warm_present"] = len(gt_ceiling) >= 1
    details["n_gt_layout_ceiling_r"] = int(len(gt_ceiling))
    if not gt_ceiling.empty:
        gt_miou_by_ring = gt_ceiling.groupby("case_id")["gt_miou"].first() if "case_id" in gt_ceiling.columns else gt_ceiling["gt_miou"]
        gt_miou_min = float(gt_miou_by_ring.min())
        details["gt_layout_ceiling_r_miou_min"] = round(gt_miou_min, 4)
        if "case_id" in gt_ceiling.columns and len(gt_ceiling["case_id"].unique()) > 1:
            details["gt_layout_ceiling_r_miou"] = round(float(gt_ceiling["gt_miou"].mean()), 4)
        else:
            details["gt_layout_ceiling_r_miou"] = round(float(gt_ceiling["gt_miou"].iloc[0]), 4)
        criteria["gt_ceiling_warm_miou_ok"] = gt_miou_min >= 0.80
    else:
        criteria["gt_ceiling_warm_miou_ok"] = False

    n_oracle = int(df["kind"].isin(ORACLE_TRIAL_KINDS).sum())
    details["n_oracle_trials"] = n_oracle
    criteria["has_oracle_warm"] = n_oracle >= 1

    if "direction_select_enabled" in df.columns:
        dir_ok = df["direction_select_enabled"].fillna(False).astype(bool).all()
        criteria["direction_select_enabled"] = bool(dir_ok)
    else:
        criteria["direction_select_enabled"] = False

    if "gt_miou" in df.columns and len(df) > 1:
        criteria["miou_spread_ok"] = float(df["gt_miou"].std()) > 0.02
        details["miou_std"] = round(float(df["gt_miou"].std()), 4)
    else:
        criteria["miou_spread_ok"] = False

    passed = all(criteria.values())
    return {"passed": passed, "criteria": criteria, "details": details}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="Experience run root under logs/")
    ap.add_argument("--manifest", default=str(REPO_ROOT / "data" / "bo_calibration" / "MANIFEST.json"))
    ap.add_argument("--expected-n", type=int, default=480, help="Panel trial count (0 = skip)")
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    panel_path = run_root / "bo_trials.csv"
    per_ring_paths = sorted(run_root.glob("*/*/bo_trials.csv"))

    ring_checks: list[dict] = []
    if per_ring_paths:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        expected_by_ring = {
            r["ring_key"]: n_evals_for_ring_entry(r)
            for r in manifest.get("rings", [])
        }
        for p in per_ring_paths:
            case_id = f"{p.parent.parent.name}/{p.parent.name}"
            df = pd.read_csv(p)
            ring_checks.append(
                check_trials_df(
                    df,
                    case_id=case_id,
                    expected_n=expected_by_ring.get(case_id),
                )
            )

    panel_check: dict | None = None
    if panel_path.is_file():
        df_panel = pd.read_csv(panel_path)
        panel_check = check_trials_df(
            df_panel,
            case_id="panel",
            expected_n=args.expected_n if args.expected_n > 0 else None,
        )

    all_ring_pass = all(r["passed"] for r in ring_checks) if ring_checks else True
    panel_pass = panel_check["passed"] if panel_check else True
    # Panel passes when all per-ring checks pass (panel row check is informational).
    passed = all_ring_pass if ring_checks else panel_pass

    out = {
        "run_root": str(run_root),
        "passed": passed,
        "panel": panel_check,
        "rings": ring_checks,
        "n_rings": len(ring_checks),
        "n_rings_passed": sum(1 for r in ring_checks if r["passed"]),
    }
    out_path = run_root / "gt_experience_gate.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
