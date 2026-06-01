#!/usr/bin/env python3
"""Validate honest BO experience trial pools (no oracle layout trials)."""
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

ORACLE_KIND_PREFIX = "gt_layout"


def check_trials_df(df: pd.DataFrame, *, case_id: str = "panel") -> dict:
    criteria: dict[str, bool] = {}
    details: dict[str, object] = {"case_id": case_id, "n_trials": int(len(df))}

    if df.empty:
        criteria["non_empty"] = False
        return {"passed": False, "criteria": criteria, "details": details}

    criteria["non_empty"] = True
    oracle_mask = df["kind"].astype(str).isin(ORACLE_TRIAL_KINDS) | df["kind"].astype(str).str.startswith(ORACLE_KIND_PREFIX)
    n_oracle = int(oracle_mask.sum())
    criteria["no_oracle_trials"] = n_oracle == 0
    details["n_oracle_trials"] = n_oracle
    if n_oracle:
        details["oracle_kinds"] = sorted(df.loc[oracle_mask, "kind"].astype(str).unique().tolist())

    if "direction_select_enabled" in df.columns:
        dir_ok = df["direction_select_enabled"].fillna(False).astype(bool).all()
        criteria["direction_select_enabled"] = bool(dir_ok)
    else:
        criteria["direction_select_enabled"] = False

    if "order_branch" in df.columns and "direction_score_plus" in df.columns:
        criteria["branch_selection_logged"] = True
    else:
        criteria["branch_selection_logged"] = "order_branch" in df.columns

    if "gt_miou" in df.columns and len(df) > 1:
        criteria["miou_spread_ok"] = float(df["gt_miou"].std()) > 0.02
        details["gt_miou_std"] = round(float(df["gt_miou"].std()), 4)
    else:
        criteria["miou_spread_ok"] = False

    passed = all(criteria.values())
    return {"passed": passed, "criteria": criteria, "details": details}


def check_run_root(run_root: Path) -> dict:
    trials_path = run_root / "bo_trials.csv"
    per_ring_paths = sorted(run_root.glob("*/*/bo_trials.csv"))
    per_ring: list[dict] = []
    for ring_dir in per_ring_paths:
        ring_df = pd.read_csv(ring_dir)
        tid = ring_dir.parent.parent.name
        rid = ring_dir.parent.name
        per_ring.append(check_trials_df(ring_df, case_id=f"{tid}/{rid}"))

    if trials_path.is_file():
        df = pd.read_csv(trials_path)
        result = check_trials_df(df, case_id="panel")
    elif per_ring_paths:
        df = pd.concat([pd.read_csv(p) for p in per_ring_paths], ignore_index=True)
        result = check_trials_df(df, case_id="panel")
    else:
        return {"passed": False, "error": f"no trials under {run_root}"}

    result["per_ring"] = per_ring
    result["all_rings_passed"] = all(r["passed"] for r in per_ring) if per_ring else True
    result["passed"] = bool(result["passed"] and result["all_rings_passed"])
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Honest BO experience gate")
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="Write honesty_gate.json (default: run-root/honesty_gate.json)")
    args = ap.parse_args()

    run_root = args.run_root.resolve()
    result = check_run_root(run_root)
    out = args.out or (run_root / "honesty_gate.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
