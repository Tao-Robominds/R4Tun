from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bo.v5.run_t3_gt_range_recovery_v1 as base
SRC_RUN = REPO_ROOT / "logs" / "v5_t3_gt_range_recovery_v1"
NEW_RUN = REPO_ROOT / "logs" / "v5_t3_gt_range_recovery_v1_wide_mismatch_rerun"


def main() -> int:
    base._assert_writable(NEW_RUN)
    base._require_depth_contract()
    NEW_RUN.mkdir(parents=True, exist_ok=True)

    prev = pd.read_csv(SRC_RUN / "t3_gt_range_scoreboard.csv")
    mismatch = prev[prev["failure_mode"].astype(str).eq("range_mismatch")].copy()
    if mismatch.empty:
        raise RuntimeError("No range_mismatch rings found in previous T3 run.")
    ring_keys = mismatch["ring_key"].astype(str).tolist()

    prev_summary = json.loads((SRC_RUN / "t3_gt_range_summary.json").read_text(encoding="utf-8"))
    low = float(prev_summary["gt_range_low_frac"])
    high = float(prev_summary["gt_range_high_frac"])
    widened_low = max(0.0, low - 0.08)
    widened_high = min(0.999, high + 0.08)
    if widened_high <= widened_low:
        widened_high = min(0.999, widened_low + 0.1)

    base.RUN_ROOT = NEW_RUN
    cfgs = base._build_range_cfgs(widened_low, widened_high)

    v5 = pd.read_csv(base.V5_SCORE)
    v5_t3 = v5[v5["family"].astype(int).eq(3)][["ring_key", "seeded_initial_miou", "stabilised_miou"]].copy()

    rows: list[dict] = []
    all_cands = []
    for rk in ring_keys:
        base._stage_ring(rk)
        cands = base._run_candidates_for_ring(rk, cfgs)
        all_cands.append(cands)
        sel = base._select_candidate(cands)
        selected_miou = float(sel["miou"])
        stabilised = float(v5_t3[v5_t3["ring_key"] == rk]["stabilised_miou"].iloc[0])
        floor_abstain = bool(selected_miou + 1e-9 < stabilised)
        final_miou = stabilised if floor_abstain else selected_miou
        best = cands.loc[cands["miou"].idxmax()]
        rows.append(
            {
                "ring_key": rk,
                "selected_runtime_tag": f"{sel['det_tag']}_{sel['branch']}_rot{int(sel['rotation_shift'])}",
                "selected_runtime_miou": selected_miou,
                "stabilised_floor_miou": stabilised,
                "floor_abstain": floor_abstain,
                "intrinsic_final_miou": final_miou,
                "oracle_best_miou": float(best["miou"]),
                "oracle_best_tag": f"{best['det_tag']}_{best['branch']}_rot{int(best['rotation_shift'])}",
            }
        )

    cand_df = pd.concat(all_cands, ignore_index=True)
    cand_df.to_csv(NEW_RUN / "t3_wide_mismatch_candidate_scores.csv", index=False)
    out = pd.DataFrame(rows).sort_values("ring_key").reset_index(drop=True)
    out.to_csv(NEW_RUN / "t3_wide_mismatch_scoreboard.csv", index=False)

    summary = {
        "source_run": str(SRC_RUN),
        "run_root": str(NEW_RUN),
        "source_gt_range": [low, high],
        "widened_gt_range": [widened_low, widened_high],
        "n_rings": int(len(out)),
        "mean_intrinsic_final_miou": float(out["intrinsic_final_miou"].mean()),
        "n_ge_0_5": int((out["intrinsic_final_miou"] >= 0.5).sum()),
        "n_floor_abstain": int(out["floor_abstain"].sum()),
    }
    (NEW_RUN / "t3_wide_mismatch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
