#!/usr/bin/env python3
"""Check v2 BO gate: r_surface_min varies and regret threshold met."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_BO_DIR = Path(__file__).resolve().parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.ceiling_gate import REPO_ROOT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring-key", default="1-4/r206")
    ap.add_argument("--run-root", type=Path, default=REPO_ROOT / "logs" / "bo_experience_v2")
    ap.add_argument("--regret-max", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tid, rid = args.ring_key.split("/")
    ring_dir = args.run_root / tid / rid
    trials_path = ring_dir / "bo_trials.csv"
    exp_gate_path = ring_dir / "experience_gate.json"
    if not trials_path.exists():
        raise FileNotFoundError(trials_path)

    df = pd.read_csv(trials_path)
    r_col = "r_surface_min" if "r_surface_min" in df.columns else "r_surface_min_fixed"
    r_std = float(df[r_col].std()) if len(df) > 1 else 0.0
    r_unique = int(df[r_col].nunique())
    reclass_std = float(df["n_reclassified_by_r_filter"].std()) if "n_reclassified_by_r_filter" in df.columns else 0.0

    exp_gate = json.loads(exp_gate_path.read_text(encoding="utf-8")) if exp_gate_path.exists() else {}
    best_bo = float(exp_gate.get("best_bo_miou") or df["gt_miou"].max())
    regret = float(exp_gate.get("regret_vs_ceiling") or 0.0)
    ceiling = float(exp_gate.get("ceiling_miou_reference") or 0.0)

    gate = {
        "ring_key": args.ring_key,
        "run_root": str(args.run_root),
        "ceiling_miou_reference": ceiling,
        "best_bo_miou": best_bo,
        "regret_vs_ceiling": regret,
        "r_surface_min_std": r_std,
        "r_surface_min_unique": r_unique,
        "n_reclassified_std": reclass_std,
        "criteria": {
            "r_surface_varies": r_std > 0.0 and r_unique > 1,
            "regret_ok": regret <= args.regret_max,
            "reclass_varies": reclass_std > 0.0,
            "experience_gate_passed": bool(exp_gate.get("passed", False)),
        },
        "pass": bool(
            r_std > 0.0
            and r_unique > 1
            and regret <= args.regret_max
            and reclass_std > 0.0
            and exp_gate.get("passed", False)
        ),
    }
    out = args.out or (ring_dir / "gate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
