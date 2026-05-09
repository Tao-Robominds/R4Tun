from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bo.v3._paths import assert_writable

SCOREBOARD = REPO / "logs" / "v3_direction_stabilisation_v1" / "scoreboard.csv"
PROXY_PRED = REPO / "logs" / "v3_binary_order_bo_r4tun_v1" / "proxy" / "proxy_loso_predictions.csv"
RUN_ROOT = REPO / "logs" / "v3_arm_b_proxy_stabilisation_v1"


def _summary(df: pd.DataFrame, selected_col: str) -> dict[str, Any]:
    selected = df[selected_col].to_numpy(dtype=float)
    plus = df["plus_miou"].to_numpy(dtype=float)
    oracle = df["oracle_miou"].to_numpy(dtype=float)
    lifts = selected - plus
    return {
        "n_rings": int(len(df)),
        "mean_miou": float(np.mean(selected)),
        "lift_vs_k_only_plus": float(np.mean(selected) - np.mean(plus)),
        "oracle_recovered_fraction": float((np.mean(selected) - np.mean(plus)) / max(1e-9, np.mean(oracle) - np.mean(plus))),
        "n_degrade_lt_minus_0p01": int(np.sum(lifts < -0.01)),
    }


def finalize(run_root: Path, strategy: str) -> dict[str, Any]:
    sb = pd.read_csv(SCOREBOARD)
    pp = pd.read_csv(PROXY_PRED)
    pp = pp[pp["strategy"] == strategy].copy()
    if pp.empty:
        raise SystemExit(f"proxy strategy {strategy!r} not found in {PROXY_PRED}")
    if pp["ring_key"].nunique() != 40:
        raise SystemExit(f"expected 40 rings for {strategy}, got {pp['ring_key'].nunique()}")

    merged = sb.merge(
        pp[
            [
                "ring_key",
                "section",
                "proxy_plus_miou",
                "proxy_minus_miou",
                "proxy_margin_minus_plus",
                "pred_minus",
                "selected_miou",
                "lift_vs_s0",
            ]
        ],
        on="ring_key",
        how="inner",
        suffixes=("", "_proxy"),
    )
    if len(merged) != 40:
        raise SystemExit(f"expected merged 40 rings, got {len(merged)}")

    merged["section"] = merged["ring_key"].str.split("/").str[0]
    merged["bottom_baseline_miou"] = merged["bottom_baseline_miou"].astype(float)
    merged["k_only_plus_miou"] = merged["plus_miou"].astype(float)
    merged["oracle_miou"] = merged[["plus_miou", "minus_miou"]].max(axis=1)
    merged["oracle_order"] = np.where(merged["minus_miou"] > merged["plus_miou"], "minus", "plus")
    merged["selected_order"] = np.where(merged["pred_minus"], "minus", "plus")
    merged["selected_proxy_miou"] = merged["selected_miou"].astype(float)
    merged["lift_proxy_minus_k_only"] = merged["selected_proxy_miou"] - merged["k_only_plus_miou"]
    merged["lift_proxy_minus_bottom"] = merged["selected_proxy_miou"] - merged["bottom_baseline_miou"]
    merged["lift_k_only_minus_bottom"] = merged["k_only_plus_miou"] - merged["bottom_baseline_miou"]
    merged["degrade_lt_minus_0p01"] = merged["lift_proxy_minus_k_only"] < -0.01
    merged["missed_oracle_gain_gt_0p05"] = (
        (merged["oracle_miou"] - merged["selected_proxy_miou"]) > 0.05
    )

    scoreboard_cols = [
        "ring_key",
        "section",
        "split",
        "pattern_type",
        "bottom_baseline_miou",
        "k_only_plus_miou",
        "proxy_plus_miou",
        "proxy_minus_miou",
        "proxy_margin_minus_plus",
        "selected_order",
        "selected_proxy_miou",
        "oracle_order",
        "oracle_miou",
        "lift_k_only_minus_bottom",
        "lift_proxy_minus_k_only",
        "lift_proxy_minus_bottom",
        "degrade_lt_minus_0p01",
        "missed_oracle_gain_gt_0p05",
    ]
    merged[scoreboard_cols].sort_values("ring_key").to_csv(
        run_root / "arm_b_final_scoreboard.csv", index=False
    )

    failures = merged[
        merged["degrade_lt_minus_0p01"] | merged["missed_oracle_gain_gt_0p05"]
    ][
        [
            "ring_key",
            "section",
            "split",
            "selected_order",
            "k_only_plus_miou",
            "selected_proxy_miou",
            "oracle_order",
            "oracle_miou",
            "lift_proxy_minus_k_only",
            "degrade_lt_minus_0p01",
            "missed_oracle_gain_gt_0p05",
        ]
    ].sort_values(["degrade_lt_minus_0p01", "lift_proxy_minus_k_only", "ring_key"])
    failures.to_csv(run_root / "arm_b_final_failures.csv", index=False)

    summary = {
        "method_name": f"arm_b_k_anchor_plus_{strategy}_binary_switch",
        "strategy": strategy,
        "n_rings": int(len(merged)),
        "panels": {
            "cross_section_n": int((merged["split"] == "cross_section").sum()),
            "within_section_n": int((merged["split"] == "within_section").sum()),
        },
        "metrics": {
            "bottom_baseline": {
                "mean_miou": float(merged["bottom_baseline_miou"].mean()),
            },
            "k_only_plus": _summary(merged, "k_only_plus_miou"),
            "proxy_switch": _summary(merged, "selected_proxy_miou"),
            "oracle_binary_order": _summary(merged, "oracle_miou"),
        },
        "counts": {
            "selected_minus": int((merged["selected_order"] == "minus").sum()),
            "selected_plus": int((merged["selected_order"] == "plus").sum()),
            "degrade_lt_minus_0p01": int(merged["degrade_lt_minus_0p01"].sum()),
            "missed_oracle_gain_gt_0p05": int(merged["missed_oracle_gain_gt_0p05"].sum()),
        },
    }
    (run_root / "arm_b_final_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "source_files": {
            "scoreboard": str(SCOREBOARD.resolve()),
            "proxy_predictions": str(PROXY_PRED.resolve()),
        },
        "output_files": [
            "arm_b_final_scoreboard.csv",
            "arm_b_final_summary.json",
            "arm_b_final_failures.csv",
            "arm_b_final_manifest.json",
        ],
        "method_name": summary["method_name"],
    }
    (run_root / "arm_b_final_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Finalize Arm B proxy stabilisation outputs")
    p.add_argument("--run-root", default=str(RUN_ROOT))
    p.add_argument("--strategy", default="proxy_rf")
    ns = p.parse_args(argv)
    out_root = assert_writable(Path(ns.run_root).resolve())
    out_root.mkdir(parents=True, exist_ok=True)
    summary = finalize(out_root, strategy=str(ns.strategy))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
