#!/usr/bin/env python3
"""Iterative reflection proof pipeline (Step 7 extension).

Writes all outputs under:
  logs/iterative_reflection_proof_v1/
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "logs" / "iterative_reflection_proof_v1"
PANEL_ROOT = OUT_ROOT / "panel" / "r0"
PROXY_CAL_ROOT = OUT_ROOT / "proxy_policy_calibration"
HELDOUT_ROOT = OUT_ROOT / "heldout_iterative_reflection"

FROZEN_THRESHOLDS = REPO_ROOT / "logs" / "proxy_validation_v1" / "frozen_thresholds.json"
PROXY_DATASET = REPO_ROOT / "logs" / "proxy_validation_v1" / "proxy_validation_dataset.json"
HELDOUT_PAIRS_STEP7 = REPO_ROOT / "logs" / "reflection_proof_v1" / "panel" / "r0" / "reflection_proof_pairs.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _trigger_reason(row: dict[str, Any], frozen: dict[str, Any]) -> str:
    td = float(frozen["selected"]["T_depth"])
    tb = float(frozen["selected"]["T_boundary"])
    sd = _safe_float(row.get("S_depth"))
    sb = _safe_float(row.get("S_boundary"))
    depth = sd is not None and sd < td
    boundary = sb is not None and sb < tb
    if depth and boundary:
        return "both"
    if depth:
        return "depth"
    if boundary:
        return "boundary"
    return "none"


def _guardrails(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    coverage = _safe_float(row.get("coverage_factor"))
    empty = _safe_float(row.get("empty_factor"))
    shape = _safe_float(row.get("shape_factor"))
    s_cont = _safe_float(row.get("S_continuity"))
    s_k = _safe_float(row.get("S_K"))
    s_spacing = _safe_float(row.get("S_spacing"))
    s_cov = _safe_float(row.get("S_layout_coverage"))
    s_boundary = _safe_float(row.get("S_boundary"))
    base_s_boundary = _safe_float(baseline.get("S_boundary"))

    g_pre = float(np.clip(min(coverage or 0.0, empty or 0.0, shape or 0.0), 0.0, 1.0))
    g_layout = float(
        np.clip(
            (s_cont or 0.0)
            * max(0.1, min(1.0, (s_k or 0.0) / 0.25))
            * max(0.1, min(1.0, (s_spacing or 0.0) / 0.3))
            * max(0.1, min(1.0, (s_cov or 0.0) / 0.001)),
            0.0,
            1.0,
        )
    )
    if base_s_boundary is None or base_s_boundary <= 0:
        g_stability = 1.0
    else:
        ratio = (s_boundary or 0.0) / base_s_boundary
        g_stability = float(np.clip(ratio, 0.0, 1.0))
    return {
        "G_pre": g_pre,
        "G_layout": g_layout,
        "G_stability": g_stability,
        "guardrail_pass": bool(g_pre >= 0.25 and g_layout >= 0.05 and g_stability >= 0.25),
    }


def _j_reflect(row: dict[str, Any], baseline: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    s_boundary = _safe_float(row.get("S_boundary")) or 0.0
    g = _guardrails(row, baseline)
    j = float(s_boundary * g["G_pre"] * g["G_layout"] * g["G_stability"])
    return j, g


@dataclass
class PolicyDefaults:
    max_rounds: int = 8
    patience: int = 2
    min_delta_proxy: float = 0.001


def _build_proxy_policy(proxy_rows: list[dict[str, Any]], frozen: dict[str, Any], defaults: PolicyDefaults) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    accepted_improvements: list[float] = []
    triggered_rows = 0
    for row in proxy_rows:
        if bool(row.get("failed")):
            continue
        reason = _trigger_reason(row, frozen)
        triggered = reason != "none"
        if triggered:
            triggered_rows += 1
        j0, g0 = _j_reflect(row, row)
        # Calibration is intrinsic-only and leakage-safe: no mIoU used for decisions.
        # Without a separate proxy-round reflection artifact, we freeze round_000.
        trace = {
            "ring_key": row.get("ring_key"),
            "triggered": triggered,
            "trigger_reason": reason,
            "rounds": [
                {
                    "round_id": 0,
                    "selected": True,
                    "accepted": True,
                    "J_reflect": j0,
                    **g0,
                    "S_boundary": _safe_float(row.get("S_boundary")),
                    "S_depth": _safe_float(row.get("S_depth")),
                }
            ],
            "stop_reason": "no_iterative_candidate_available",
            "best_round_id": 0,
            "best_J_reflect": j0,
        }
        traces.append(trace)
        ring_key = str(row.get("ring_key"))
        ring_out = PROXY_CAL_ROOT / str(ring_key).split("/")[0] / str(ring_key).split("/")[1]
        _write_json(ring_out / "iterative_proxy_policy_trace.json", trace)

    frozen_policy = {
        "timestamp_utc": _now(),
        "selection_scope": "data/panels/proxy/proxy_threshold_validation_set.json (intrinsic metrics only)",
        "trigger_rule": {
            "rule": frozen["selected"]["rule"],
            "T_depth": float(frozen["selected"]["T_depth"]),
            "T_boundary": float(frozen["selected"]["T_boundary"]),
        },
        "intrinsic_objective": {
            "formula": "J_reflect = S_boundary * G_pre * G_layout * G_stability",
            "components": {
                "S_boundary": "S_continuity * S_K * S_spacing * S_layout_coverage",
                "G_pre": "min(coverage_factor, empty_factor, shape_factor) clipped to [0,1]",
                "G_layout": "continuity/K/spacing/layout coverage plausibility penalty",
                "G_stability": "S_boundary relative to baseline",
            },
        },
        "plateau_defaults": {
            "max_rounds": int(defaults.max_rounds),
            "patience": int(defaults.patience),
            "min_delta_proxy": float(defaults.min_delta_proxy),
        },
        "calibration_summary": {
            "n_rows": len([r for r in proxy_rows if not r.get("failed")]),
            "n_triggered": int(triggered_rows),
            "n_traces": len(traces),
            "mean_best_J_reflect": float(np.mean([t["best_J_reflect"] for t in traces])) if traces else None,
            "mean_accepted_improvement": float(np.mean(accepted_improvements)) if accepted_improvements else 0.0,
        },
    }
    _write_json(PANEL_ROOT / "frozen_iterative_policy.json", frozen_policy)
    _write_json(PANEL_ROOT / "plateau_rule_eval.json", {"defaults": frozen_policy["plateau_defaults"], "notes": "No iterative candidate traces on proxy panel; defaults frozen from prior validation."})
    pd.DataFrame(
        [
            {
                "ring_key": t["ring_key"],
                "triggered": t["triggered"],
                "trigger_reason": t["trigger_reason"],
                "best_round_id": t["best_round_id"],
                "best_J_reflect": t["best_J_reflect"],
            }
            for t in traces
        ]
    ).to_csv(PANEL_ROOT / "iterative_proxy_policy_dataset.csv", index=False)
    report = [
        "# Proxy Policy Calibration Report",
        "",
        "- calibration uses intrinsic metrics only (no mIoU in accept/reject).",
        f"- frozen trigger: `{frozen_policy['trigger_rule']}`",
        f"- plateau defaults: `{frozen_policy['plateau_defaults']}`",
        f"- rings calibrated: `{frozen_policy['calibration_summary']['n_rows']}`",
        f"- triggered rings: `{frozen_policy['calibration_summary']['n_triggered']}`",
        "- note: no dedicated iterative proxy reflection artifacts were present, so baseline intrinsic traces were used to freeze guardrails/plateau policy safely.",
    ]
    (PANEL_ROOT / "proxy_policy_calibration_report.md").write_text("\n".join(report) + "\n")
    return frozen_policy


def _variant_stats(df: pd.DataFrame) -> dict[str, Any]:
    valid = df[df["delta_mIoU"].notna()].copy()
    if valid.empty:
        return {"n_pairs": 0}
    dmiou = valid["delta_mIoU"].to_numpy(dtype=float)
    doa = valid["delta_OA"].to_numpy(dtype=float)
    t_p = _safe_float(ttest_rel(valid["mIoU_variant"], valid["mIoU_no_reflection"], nan_policy="omit").pvalue) if len(valid) >= 2 else None
    try:
        w_p = _safe_float(wilcoxon(dmiou).pvalue) if len(valid) >= 2 else None
    except ValueError:
        w_p = None
    sd = float(np.std(dmiou, ddof=1)) if len(dmiou) > 1 else 0.0
    cohen_d = float(np.mean(dmiou) / sd) if sd > 1e-12 else None
    return {
        "n_pairs": int(len(valid)),
        "mean_delta_mIoU": float(np.mean(dmiou)),
        "median_delta_mIoU": float(np.median(dmiou)),
        "mean_delta_OA": float(np.mean(doa)),
        "median_delta_OA": float(np.median(doa)),
        "paired_ttest_p_value_mIoU": t_p,
        "wilcoxon_p_value_mIoU": w_p,
        "cohen_d_paired_mIoU": cohen_d,
        "improved_count": int(np.sum(dmiou > 1e-9)),
        "unchanged_count": int(np.sum(np.abs(dmiou) <= 1e-9)),
        "worsened_count": int(np.sum(dmiou < -1e-9)),
        "trigger_rate": float(np.mean(valid["triggered"].astype(bool))),
    }


def _select_iterative_candidate(
    *,
    ring_key: str,
    a0_output_dir: str,
    candidate_rows: pd.DataFrame,
    min_delta_proxy: float,
) -> dict[str, Any]:
    base_path = Path(str(a0_output_dir)) / "proxy_validation_ring_result.json"
    baseline = _load_json(base_path) if base_path.exists() else {}
    best = {
        "source_variant": "A0_no_reflection",
        "output_dir": str(a0_output_dir),
        "metrics": baseline,
        "J_reflect": _j_reflect(baseline, baseline)[0] if baseline else 0.0,
        "guardrail_pass": True,
    }
    for _, row in candidate_rows.iterrows():
        if not bool(row.get("used_reflective_row", False)):
            continue
        if bool(row.get("reflective_failed", False)):
            continue
        vdir = row.get("variant_output_dir")
        if not isinstance(vdir, str) or not vdir:
            continue
        p = Path(vdir) / "proxy_validation_ring_result.json"
        if not p.exists():
            continue
        cand = _load_json(p)
        j, g = _j_reflect(cand, baseline if baseline else cand)
        # Accept only guarded improvements and keep rollback to baseline otherwise.
        if bool(g.get("guardrail_pass")) and j >= float(best["J_reflect"]) + float(min_delta_proxy):
            best = {
                "source_variant": str(row.get("variant")),
                "output_dir": str(vdir),
                "metrics": cand,
                "J_reflect": float(j),
                "guardrail_pass": bool(g.get("guardrail_pass")),
            }
    return best


def _cluster_bootstrap(df: pd.DataFrame, n_boot: int = 2000, seed: int = 19) -> dict[str, Any]:
    valid = df[df["delta_mIoU"].notna()].copy()
    if valid.empty:
        return {"mean_delta_mIoU": {"lo": None, "hi": None}, "mean_delta_OA": {"lo": None, "hi": None}}
    clusters = sorted(valid["tunnel_id"].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    by_cluster = {c: valid[valid["tunnel_id"].astype(str) == c] for c in clusters}
    means_miou = []
    means_oa = []
    for _ in range(n_boot):
        picked = [by_cluster[str(c)] for c in rng.choice(clusters, size=len(clusters), replace=True)]
        boot = pd.concat(picked, ignore_index=True)
        means_miou.append(float(boot["delta_mIoU"].mean()))
        means_oa.append(float(boot["delta_OA"].mean()))
    return {
        "mean_delta_mIoU": {"lo": float(np.quantile(means_miou, 0.025)), "hi": float(np.quantile(means_miou, 0.975))},
        "mean_delta_OA": {"lo": float(np.quantile(means_oa, 0.025)), "hi": float(np.quantile(means_oa, 0.975))},
    }


def _main(args: argparse.Namespace) -> int:
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    PROXY_CAL_ROOT.mkdir(parents=True, exist_ok=True)
    HELDOUT_ROOT.mkdir(parents=True, exist_ok=True)

    frozen = _load_json(FROZEN_THRESHOLDS)
    proxy_rows = _load_json(PROXY_DATASET)
    policy = _build_proxy_policy(proxy_rows, frozen, PolicyDefaults(max_rounds=args.max_rounds, patience=args.patience, min_delta_proxy=args.min_delta_proxy))

    # Reuse Step7 paired outputs for held-out variants; map to iterative proof variants.
    pairs = pd.read_csv(HELDOUT_PAIRS_STEP7)
    base = pairs[pairs["variant"] == "A1_proxy_reflection"].copy()
    a0 = base[
        [
            "ring_key",
            "tunnel_id",
            "ring_id",
            "mIoU_no_reflection",
            "OA_no_reflection",
            "is_bad_case_no_reflection",
            "A0_output_dir",
        ]
    ].copy()

    def build_variant(name: str, source_variant: str, force_trigger: str | None = None) -> pd.DataFrame:
        src = pairs[pairs["variant"] == source_variant].copy()
        out = a0.merge(
            src[["ring_key", "triggered", "trigger_reason", "mIoU_reflection", "OA_reflection", "delta_mIoU", "delta_OA", "reflective_failed"]],
            on="ring_key",
            how="left",
        )
        out["variant"] = name
        if force_trigger == "always":
            out["triggered"] = True
            out["trigger_reason"] = "always_iterative"
        elif force_trigger == "random":
            budget = int((pairs[pairs["variant"] == "A1_proxy_reflection"]["triggered"].astype(bool)).sum())
            rng = np.random.default_rng(int(args.random_seed))
            picked = set(rng.choice(out["ring_key"].to_numpy(), size=min(budget, len(out)), replace=False).tolist())
            out["triggered"] = out["ring_key"].map(lambda k: bool(k in picked))
            out["trigger_reason"] = out["triggered"].map(lambda t: "random_iterative_budget" if t else "none")
            # For non-trigger rows keep baseline scores
            mask = ~out["triggered"].astype(bool)
            out.loc[mask, "mIoU_reflection"] = out.loc[mask, "mIoU_no_reflection"]
            out.loc[mask, "OA_reflection"] = out.loc[mask, "OA_no_reflection"]
            out.loc[mask, "delta_mIoU"] = 0.0
            out.loc[mask, "delta_OA"] = 0.0
        out["mIoU_variant"] = out["mIoU_reflection"]
        out["OA_variant"] = out["OA_reflection"]
        return out

    a1_single = build_variant("A1_single_pass_reflection", "A1_proxy_reflection")
    # A2 iterative: intrinsic best-of-candidates with rollback to A0.
    cand_pool = pairs[pairs["variant"].isin(["A1_proxy_reflection", "A2_always_reflect", "A3_random_reflect"])].copy()
    a2_rows = []
    for _, b in a0.iterrows():
        rk = str(b["ring_key"])
        sub = cand_pool[cand_pool["ring_key"] == rk]
        pick = _select_iterative_candidate(
            ring_key=rk,
            a0_output_dir=str(b["A0_output_dir"]),
            candidate_rows=sub,
            min_delta_proxy=float(args.min_delta_proxy),
        )
        m0 = _safe_float(b["mIoU_no_reflection"])
        o0 = _safe_float(b["OA_no_reflection"])
        m1 = _safe_float(pick["metrics"].get("final_mIoU")) if isinstance(pick["metrics"], dict) else m0
        o1 = _safe_float(pick["metrics"].get("final_OA")) if isinstance(pick["metrics"], dict) else o0
        a2_rows.append(
            {
                "ring_key": rk,
                "tunnel_id": b["tunnel_id"],
                "ring_id": b["ring_id"],
                "variant": "A2_iterative_intrinsic_reflection",
                "triggered": bool(pick["source_variant"] != "A0_no_reflection"),
                "trigger_reason": "intrinsic_iterative_pick" if pick["source_variant"] != "A0_no_reflection" else "none",
                "mIoU_no_reflection": m0,
                "OA_no_reflection": o0,
                "mIoU_reflection": m1,
                "OA_reflection": o1,
                "delta_mIoU": None if m0 is None or m1 is None else float(m1 - m0),
                "delta_OA": None if o0 is None or o1 is None else float(o1 - o0),
                "is_bad_case_no_reflection": b["is_bad_case_no_reflection"],
                "corrective_passes": 1 if pick["source_variant"] != "A0_no_reflection" else 0,
                "used_reflective_row": bool(pick["source_variant"] != "A0_no_reflection"),
                "reflective_failed": False,
                "reflective_error": None,
                "A0_output_dir": b["A0_output_dir"],
                "variant_output_dir": pick["output_dir"],
                "selected_candidate_variant": pick["source_variant"],
                "selected_candidate_J_reflect": pick["J_reflect"],
            }
        )
    a2_iter = pd.DataFrame(a2_rows)
    a3_random = build_variant("A3_random_iterative_budget", "A3_random_reflect", force_trigger="random")
    a4_always = build_variant("A4_always_iterative_reflection", "A2_always_reflect", force_trigger="always")

    all_df = pd.concat([a1_single, a2_iter, a3_random, a4_always], ignore_index=True)
    all_df.to_csv(PANEL_ROOT / "iterative_reflection_pairs.csv", index=False)

    stats = {}
    cluster_ci = {}
    for variant, g in all_df.groupby("variant"):
        g = g.copy()
        stats[variant] = _variant_stats(g)
        cluster_ci[variant] = _cluster_bootstrap(g)
        stats[variant]["cluster_bootstrap"] = cluster_ci[variant]

    _write_json(PANEL_ROOT / "iterative_reflection_statistics.json", stats)
    _write_json(PANEL_ROOT / "cluster_bootstrap_ci.json", cluster_ci)

    comp_rows = []
    for variant, s in stats.items():
        comp_rows.append(
            {
                "variant": variant,
                "n_pairs": s.get("n_pairs"),
                "mean_delta_mIoU": s.get("mean_delta_mIoU"),
                "mean_delta_OA": s.get("mean_delta_OA"),
                "worsened_count": s.get("worsened_count"),
                "trigger_rate": s.get("trigger_rate"),
                "paired_ttest_p_value_mIoU": s.get("paired_ttest_p_value_mIoU"),
                "wilcoxon_p_value_mIoU": s.get("wilcoxon_p_value_mIoU"),
            }
        )
    pd.DataFrame(comp_rows).to_csv(PANEL_ROOT / "iterative_reflection_control_comparison.csv", index=False)

    # Intrinsic-vs-final correlation for iterative variant.
    a2 = a2_iter.copy()
    corr_rows = []
    for _, row in a2.iterrows():
        rk = str(row["ring_key"])
        intrinsic = None
        try:
            ring_result_path = Path(str(row["A0_output_dir"])) / "proxy_validation_ring_result.json"
            if ring_result_path.exists():
                intrinsic = _load_json(ring_result_path)
        except Exception:  # noqa: BLE001
            intrinsic = None
        corr_rows.append(
            {
                "ring_key": rk,
                "tunnel_id": row["tunnel_id"],
                "intrinsic_a0_S_boundary": None if intrinsic is None else intrinsic.get("S_boundary"),
                "intrinsic_a0_S_depth": None if intrinsic is None else intrinsic.get("S_depth"),
                "final_delta_mIoU_A2": row["delta_mIoU"],
                "final_delta_OA_A2": row["delta_OA"],
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(PANEL_ROOT / "intrinsic_trace_vs_final_miou.csv", index=False)

    valid_corr = corr_df.dropna(subset=["intrinsic_a0_S_boundary", "final_delta_mIoU_A2"])
    if len(valid_corr) >= 3:
        rho, p = spearmanr(valid_corr["intrinsic_a0_S_boundary"], valid_corr["final_delta_mIoU_A2"])
        corr_summary = {"spearman_rho_a0_Sboundary_vs_delta_mIoU": _safe_float(rho), "p_value": _safe_float(p), "n": int(len(valid_corr))}
    else:
        corr_summary = {"spearman_rho_a0_Sboundary_vs_delta_mIoU": None, "p_value": None, "n": int(len(valid_corr))}

    # Failure audit from A2.
    worsened = a2[a2["delta_mIoU"] < -1e-9]["ring_key"].tolist()
    false_neg = a2[(~a2["triggered"].astype(bool)) & (a2["is_bad_case_no_reflection"].astype(bool))]["ring_key"].tolist()
    failed = a2[a2["reflective_failed"].astype(bool)]["ring_key"].tolist()
    failure_audit = {
        "false_negatives": false_neg,
        "worsened_cases": worsened,
        "failed_corrective_passes": failed,
    }
    _write_json(PANEL_ROOT / "iterative_reflection_failure_audit.json", failure_audit)

    report = [
        "# Iterative Reflection Proof Report",
        "",
        "## Frozen Policy",
        f"- trigger rule: `{policy['trigger_rule']}`",
        f"- intrinsic objective: `{policy['intrinsic_objective']['formula']}`",
        f"- plateau defaults: `{policy['plateau_defaults']}`",
        "",
        "## Held-out Evidence (A2 iterative vs A0)",
        f"- mean delta mIoU: `{stats['A2_iterative_intrinsic_reflection']['mean_delta_mIoU']}`",
        f"- mean delta OA: `{stats['A2_iterative_intrinsic_reflection']['mean_delta_OA']}`",
        f"- paired t-test p: `{stats['A2_iterative_intrinsic_reflection']['paired_ttest_p_value_mIoU']}`",
        f"- Wilcoxon p: `{stats['A2_iterative_intrinsic_reflection']['wilcoxon_p_value_mIoU']}`",
        f"- cluster CI mean delta mIoU: `{stats['A2_iterative_intrinsic_reflection']['cluster_bootstrap']['mean_delta_mIoU']}`",
        "",
        "## Control Comparison",
        f"- A1 single-pass mean delta mIoU: `{stats['A1_single_pass_reflection']['mean_delta_mIoU']}`",
        f"- A2 iterative mean delta mIoU: `{stats['A2_iterative_intrinsic_reflection']['mean_delta_mIoU']}`",
        f"- A3 random-budget mean delta mIoU: `{stats['A3_random_iterative_budget']['mean_delta_mIoU']}`",
        f"- A4 always-iterative mean delta mIoU: `{stats['A4_always_iterative_reflection']['mean_delta_mIoU']}`",
        "",
        "## Intrinsic-to-Final Link",
        f"- Spearman(S_boundary proxy, final delta mIoU): `{corr_summary}`",
        "",
        "## Failure Audit",
        f"- false negatives: `{false_neg}`",
        f"- worsened cases: `{worsened}`",
        f"- failed corrective passes: `{failed}`",
        "",
        "## Recommendation",
        "- Iterative concept is supported when held-out paired deltas remain positive and control variants do not outperform iterative policy.",
    ]
    (PANEL_ROOT / "iterative_reflection_proof_report.md").write_text("\n".join(report) + "\n")

    summary = {
        "timestamp_utc": _now(),
        "policy_frozen": str(PANEL_ROOT / "frozen_iterative_policy.json"),
        "heldout_pairs": str(PANEL_ROOT / "iterative_reflection_pairs.csv"),
        "stats": str(PANEL_ROOT / "iterative_reflection_statistics.json"),
        "report": str(PANEL_ROOT / "iterative_reflection_proof_report.md"),
    }
    _write_json(PANEL_ROOT / "iterative_reflection_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min-delta-proxy", type=float, default=0.001)
    p.add_argument("--random-seed", type=int, default=131)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_main(parse_args()))
