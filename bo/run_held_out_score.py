#!/usr/bin/env python3
"""Score Stage A candidate pools with frozen proxies + structural guardrails."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_BO_DIR = Path(__file__).resolve().parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.candidate_eval import evaluate_candidate, load_ring_context  # noqa: E402
from lib.ceiling_gate import REPO_ROOT  # noqa: E402
from lib.stage_a_score import default_models, load_failure_tables, select_from_pool  # noqa: E402

DEFAULT_CANDIDATES = REPO_ROOT / "logs" / "stage_a_candidates_v1"
DEFAULT_SCORE = REPO_ROOT / "logs" / "stage_a_score_v1"
DEFAULT_HELD_OUT = REPO_ROOT / "data" / "held-out"
DEFAULT_EXPERIENCE = REPO_ROOT / "methods" / "paper" / "experience"


def _load_split(manifest_path: Path, split: str) -> list[str]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(data.get(split, []))


def score_ring(
    ring_key: str,
    *,
    candidates_root: Path,
    score_root: Path,
    held_out_root: Path,
    experience_root: Path,
    models: dict,
    failures: pd.DataFrame,
    rules: pd.DataFrame,
) -> dict:
    tunnel, rpart = ring_key.split("/")
    pool_path = candidates_root / tunnel / rpart / "candidate_pool.json"
    if not pool_path.is_file():
        raise FileNotFoundError(f"No candidate pool at {pool_path}")

    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    nearest = str(pool.get("nearest_calib_ring", ""))
    rho_k = float(pool.get("rho_K", 0.0))
    rho_ab = float(pool.get("rho_AB", 0.0))
    valid_line_anchor = rho_k >= 0.2

    ring_score_dir = score_root / tunnel / rpart
    ring_score_dir.mkdir(parents=True, exist_ok=True)

    ctx, pre7 = load_ring_context(ring_key, held_out_root=held_out_root, score_root=score_root)
    rows: list[dict] = []
    for cand in pool["candidates"]:
        cid = int(cand["candidate_id"])
        rec = evaluate_candidate(ctx, cand["search_x"], candidate_id=cid, pre7=pre7)
        rec["candidate_type"] = cand.get("candidate_type")
        rec["penalised_at_proposal"] = bool(cand.get("penalised", False))
        rec["rho_K"] = rho_k
        rec["rho_AB"] = rho_ab
        rows.append(rec)

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(ring_score_dir / "candidate_eval.csv", index=False)

    selections: dict[str, dict] = {}
    for variant, model in models.items():
        sel = select_from_pool(
            eval_df,
            model=model,
            variant=variant,
            nearest_calib_ring=nearest,
            failures=failures,
            rules=rules,
            rho_k=rho_k,
            rho_ab=rho_ab,
            valid_line_anchor=valid_line_anchor,
        )
        scored = sel.pop("scored_df")
        scored.to_csv(ring_score_dir / f"candidate_scores_{variant}.csv", index=False)
        selections[variant] = {k: v for k, v in sel.items() if k != "scored_df"}

    payload = {
        "ring_key": ring_key,
        "nearest_calib_ring": nearest,
        "n_candidates": len(rows),
        "n_agent_errors": int(eval_df["agent_error"].astype(bool).sum()),
        "rho_K": rho_k,
        "rho_AB": rho_ab,
        "valid_line_anchor": valid_line_anchor,
        "selections": selections,
    }
    (ring_score_dir / "selection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _gate_payload(ring_result: dict) -> dict:
    eval_path = None
    ring_key = ring_result["ring_key"]
    tunnel, rpart = ring_key.split("/")
    base = DEFAULT_SCORE / tunnel / rpart
    eval_csv = base / "candidate_eval.csv"
    eval_df = pd.read_csv(eval_csv) if eval_csv.is_file() else pd.DataFrame()

    p11_cols = default_models()["p11"]["feature_columns"]
    finite_p11 = True
    if not eval_df.empty:
        for col in p11_cols:
            if col not in eval_df.columns:
                finite_p11 = False
                break
            if not pd.to_numeric(eval_df[col], errors="coerce").notna().all():
                finite_p11 = False
                break

    criteria = {
        "pool_evaluated": ring_result["n_candidates"] >= 18,
        "agent_error_rate_ok": ring_result["n_agent_errors"] <= ring_result["n_candidates"],
        "p11_features_finite": finite_p11,
        "composite_selector_ran": "p11" in ring_result.get("selections", {}),
        "abstention_exercised": any(
            s.get("abstained_to_c0") is not None for s in ring_result.get("selections", {}).values()
        ),
    }
    return {
        "ring_key": ring_key,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "selections": ring_result.get("selections", {}),
        "lineage": "bo/run_held_out_score.py",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-root", type=Path, default=DEFAULT_CANDIDATES)
    ap.add_argument("--score-root", type=Path, default=DEFAULT_SCORE)
    ap.add_argument("--held-out-root", type=Path, default=DEFAULT_HELD_OUT)
    ap.add_argument("--experience-root", type=Path, default=DEFAULT_EXPERIENCE)
    ap.add_argument("--split", default="stage_a_proxy_select")
    ap.add_argument("--split-manifest", type=Path, default=DEFAULT_CANDIDATES / "stage_split_manifest.json")
    ap.add_argument("--only-ring", default=None)
    ap.add_argument("--gate", action="store_true", help="Write single-instance gate after one ring")
    args = ap.parse_args()

    args.score_root.mkdir(parents=True, exist_ok=True)
    failures, rules = load_failure_tables(args.experience_root)
    models = default_models()

    if args.only_ring:
        rings = [args.only_ring]
    else:
        rings = _load_split(args.split_manifest, args.split)

    results: list[dict] = []
    for ring_key in rings:
        print(f"Score {ring_key} ...", flush=True)
        rec = score_ring(
            ring_key,
            candidates_root=args.candidates_root,
            score_root=args.score_root,
            held_out_root=args.held_out_root,
            experience_root=args.experience_root,
            models=models,
            failures=failures,
            rules=rules,
        )
        flat = {"ring_key": ring_key}
        for variant, sel in rec["selections"].items():
            for k, v in sel.items():
                flat[f"{variant}_{k}"] = v
        results.append(flat)

    summary_df = pd.DataFrame(results)
    summary_path = args.score_root / "stage_a_score_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    panel = {
        "n_rings": int(len(results)),
        "mean_p11_selected_gt_miou": float(summary_df["p11_selected_gt_miou"].mean()) if len(results) else None,
        "mean_a3_slim_selected_gt_miou": float(summary_df["a3_slim_selected_gt_miou"].mean()) if len(results) else None,
        "mean_oracle_gt_miou": float(summary_df["p11_oracle_gt_miou"].mean()) if len(results) else None,
        "mean_c0_gt_miou": float(summary_df["p11_c0_gt_miou"].mean()) if len(results) else None,
        "p11_abstain_rate": float(summary_df["p11_abstained_to_c0"].mean()) if len(results) else None,
        "p11_regression_rate": float((summary_df["p11_lift_vs_c0"] < -0.01).mean()) if len(results) else None,
    }
    (args.score_root / "stage_a_score_panel.json").write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(panel, indent=2))

    if args.gate and args.only_ring:
        gtunnel, grpart = args.only_ring.split("/")
        ring_result = json.loads((args.score_root / gtunnel / grpart / "selection.json").read_text())
        gate = _gate_payload(ring_result)
        gate_path = args.score_root / "single_instance_gate.json"
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(gate, indent=2))
        return 0 if gate["passed"] else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
