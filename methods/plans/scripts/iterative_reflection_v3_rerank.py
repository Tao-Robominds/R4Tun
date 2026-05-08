#!/usr/bin/env python3
"""Re-rank v3 traces with a structural prior favoring template-rotation candidates.

Why
---
We observed in v3 that the intrinsic ranker often picks scalar/Hough candidates
because they inflate `S_boundary` (more detected lines), even though the lines
don't correspond to true block boundaries. Rotation candidates change the
template *position* — the only axis that actually changes which physical
block a pixel ends up in. So when both classes pass the guardrails, the
rotation class is more semantically meaningful.

This script reads every per-ring `iterative_trace_v3.json`, applies
``J_adj = J_reflect * (1 + alpha * is_rotation)`` with default ``alpha=0.5``,
re-selects per ring under guardrails, and reports the resulting mIoU.

It does NOT re-run detection or segmentation — the existing trace already
records the mIoU/OA each candidate produced. We only change which candidate
the policy selects.

Outputs
-------
``logs/iterative_reflection_proof_v3/panel/r0/t45_rerank_v35_*``
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[3]
PANEL_ROOT = REPO_ROOT / "logs" / "iterative_reflection_proof_v3" / "panel" / "r0"
RINGS_ROOT = REPO_ROOT / "logs" / "iterative_reflection_proof_v3" / "heldout_iterative_reflection"


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_rotation_kind(k: Any) -> bool:
    return isinstance(k, str) and k.startswith("rot")


def _rerank_trace(trace: dict[str, Any], alpha: float, min_delta: float = 0.0) -> dict[str, Any]:
    """Pick the v3.5 winner among trace candidates with a structural prior."""

    base = trace.get("baseline_A0", {})
    base_j = _safe_float(base.get("J_reflect")) or 0.0
    best = {
        "candidate_kind": "baseline_A0",
        "J_reflect": base_j,
        "J_adj": base_j,  # baseline is not rotation
        "miou": _safe_float(base.get("miou")),
        "oa": _safe_float(base.get("oa")),
        "G_layout": _safe_float(base.get("G_layout")),
        "G_pre": _safe_float(base.get("G_pre")),
        "G_stability": _safe_float(base.get("G_stability")),
        "guardrail_pass": bool(base.get("guardrail_pass", True)),
    }

    # Walk all candidate results across all rounds. Pick by J_adj.
    cur_best = dict(best)
    for rd in trace.get("rounds", []):
        for c in rd.get("round_results", []) or []:
            if not c.get("guardrail_pass"):
                continue
            j = _safe_float(c.get("J_reflect"))
            if j is None:
                continue
            j_adj = float(j) * (1.0 + alpha * (1.0 if _is_rotation_kind(c.get("candidate_kind")) else 0.0))
            cand_pkg = {
                "candidate_kind": c.get("candidate_kind"),
                "J_reflect": float(j),
                "J_adj": j_adj,
                "miou": _safe_float(c.get("miou")),
                "oa": _safe_float(c.get("oa")),
                "G_layout": _safe_float(c.get("G_layout")),
                "G_pre": _safe_float(c.get("G_pre")),
                "G_stability": _safe_float(c.get("G_stability")),
                "guardrail_pass": True,
            }
            cur_j_adj = float(cur_best.get("J_adj") or 0.0)
            if j_adj >= cur_j_adj + float(min_delta):
                cur_best = cand_pkg

    # If reranked best is no better than baseline on J_adj, roll back to baseline.
    if (cur_best.get("J_adj") or 0.0) <= (best.get("J_adj") or 0.0):
        cur_best = dict(best)
    return cur_best


def _main(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    src = pd.read_csv(PANEL_ROOT / "t45_iterative_v3_results.csv")
    for _, row in src.iterrows():
        ring_key = str(row["ring_key"])
        tunnel_id, ring_dir = ring_key.split("/", 1)
        ring_id = int(ring_dir[1:])
        trace_path = RINGS_ROOT / tunnel_id / f"r{ring_id}" / "A2_iterative_intrinsic_reflection" / "iterative_trace_v3.json"
        if not trace_path.exists():
            rows.append({"ring_key": ring_key, "error": "trace_missing"})
            continue
        trace = json.loads(trace_path.read_text())
        chosen = _rerank_trace(trace, alpha=float(args.alpha), min_delta=float(args.min_delta))
        rows.append(
            {
                "ring_key": ring_key,
                "tunnel_id": tunnel_id,
                "ring_id": ring_id,
                "mIoU_no_reflection": _safe_float(row["mIoU_no_reflection"]),
                "mIoU_A1_single_pass": _safe_float(row["mIoU_A1_single_pass"]),
                "mIoU_A2_v3_intrinsic": _safe_float(row["mIoU_A2_v3_intrinsic"]),
                "mIoU_v35_intrinsic": _safe_float(chosen.get("miou")),
                "mIoU_oracle_best": _safe_float(row["mIoU_oracle_best"]),
                "OA_v35_intrinsic": _safe_float(chosen.get("oa")),
                "v35_kind": chosen.get("candidate_kind"),
                "v35_is_rotation": _is_rotation_kind(chosen.get("candidate_kind")),
                "v35_J_reflect": _safe_float(chosen.get("J_reflect")),
                "v35_J_adj": _safe_float(chosen.get("J_adj")),
            }
        )
    out_df = pd.DataFrame(rows)
    out_path = PANEL_ROOT / "t45_rerank_v35_results.csv"
    out_df.to_csv(out_path, index=False)

    valid = out_df.dropna(subset=["mIoU_v35_intrinsic", "mIoU_no_reflection"]).copy()
    if not valid.empty:
        d = valid["mIoU_v35_intrinsic"] - valid["mIoU_no_reflection"]
        try:
            t_p = _safe_float(ttest_rel(valid["mIoU_v35_intrinsic"], valid["mIoU_no_reflection"]).pvalue) if len(valid) >= 2 else None
        except Exception:  # noqa: BLE001
            t_p = None
        try:
            w_p = _safe_float(wilcoxon(d.to_numpy(dtype=float)).pvalue) if len(valid) >= 2 else None
        except Exception:  # noqa: BLE001
            w_p = None
    else:
        t_p = None
        w_p = None

    summary = {
        "timestamp_utc": _now(),
        "alpha": float(args.alpha),
        "min_delta": float(args.min_delta),
        "n_rows": int(len(out_df)),
        "n_valid": int(len(valid)),
        "mean_mIoU_A0": _safe_float(valid["mIoU_no_reflection"].mean()) if not valid.empty else None,
        "mean_mIoU_v3_intrinsic": _safe_float(valid["mIoU_A2_v3_intrinsic"].mean()) if not valid.empty else None,
        "mean_mIoU_v35_intrinsic": _safe_float(valid["mIoU_v35_intrinsic"].mean()) if not valid.empty else None,
        "mean_mIoU_oracle": _safe_float(valid["mIoU_oracle_best"].mean()) if not valid.empty else None,
        "median_mIoU_v35_intrinsic": _safe_float(valid["mIoU_v35_intrinsic"].median()) if not valid.empty else None,
        "share_v35_ge_04": _safe_float((valid["mIoU_v35_intrinsic"] >= 0.4).mean()) if not valid.empty else None,
        "share_v35_ge_05": _safe_float((valid["mIoU_v35_intrinsic"] >= 0.5).mean()) if not valid.empty else None,
        "share_v3_ge_04": _safe_float((valid["mIoU_A2_v3_intrinsic"] >= 0.4).mean()) if not valid.empty else None,
        "share_oracle_ge_04": _safe_float((valid["mIoU_oracle_best"] >= 0.4).mean()) if not valid.empty else None,
        "mean_delta_mIoU_v35_vs_A0": _safe_float((valid["mIoU_v35_intrinsic"] - valid["mIoU_no_reflection"]).mean()) if not valid.empty else None,
        "paired_ttest_p_mIoU": t_p,
        "wilcoxon_p_mIoU": w_p,
    }
    (PANEL_ROOT / "t45_rerank_v35_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = [
        "# Tunnel 4/5 v3.5 Re-rank Report",
        "",
        f"- alpha (rotation prior): `{summary['alpha']}`",
        f"- min_delta (J_adj): `{summary['min_delta']}`",
        f"- n: `{summary['n_valid']}` evaluated of `{summary['n_rows']}` total",
        "",
        "## Aggregate mIoU",
        "",
        f"- mean mIoU A0: `{summary['mean_mIoU_A0']}`",
        f"- mean mIoU v3 (no rotation prior): `{summary['mean_mIoU_v3_intrinsic']}`",
        f"- mean mIoU v3.5 (with rotation prior): `{summary['mean_mIoU_v35_intrinsic']}`",
        f"- mean mIoU oracle (mIoU-best in same pool, **diagnostic**): `{summary['mean_mIoU_oracle']}`",
        f"- median mIoU v3.5: `{summary['median_mIoU_v35_intrinsic']}`",
        f"- share v3 >= 0.4: `{summary['share_v3_ge_04']}`",
        f"- share v3.5 >= 0.4: `{summary['share_v35_ge_04']}`",
        f"- share oracle >= 0.4: `{summary['share_oracle_ge_04']}`",
        f"- mean delta v3.5 vs A0: `{summary['mean_delta_mIoU_v35_vs_A0']}`",
        f"- paired t-test p (v3.5 vs A0): `{summary['paired_ttest_p_mIoU']}`",
        f"- Wilcoxon p (v3.5 vs A0): `{summary['wilcoxon_p_mIoU']}`",
        "",
        "## Per-ring",
        "",
        "| ring_key | A0 mIoU | A1 mIoU | v3 mIoU | v3.5 mIoU | oracle mIoU | v3.5 kind | rotation? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in out_df.iterrows():
        report.append(
            "| {rk} | {a0:s} | {a1:s} | {v3:s} | {v35:s} | {orc:s} | {kind} | {rot} |".format(
                rk=r["ring_key"],
                a0=("{:.4f}".format(r["mIoU_no_reflection"]) if pd.notna(r.get("mIoU_no_reflection")) else "nan"),
                a1=("{:.4f}".format(r["mIoU_A1_single_pass"]) if pd.notna(r.get("mIoU_A1_single_pass")) else "nan"),
                v3=("{:.4f}".format(r["mIoU_A2_v3_intrinsic"]) if pd.notna(r.get("mIoU_A2_v3_intrinsic")) else "nan"),
                v35=("{:.4f}".format(r["mIoU_v35_intrinsic"]) if pd.notna(r.get("mIoU_v35_intrinsic")) else "nan"),
                orc=("{:.4f}".format(r["mIoU_oracle_best"]) if pd.notna(r.get("mIoU_oracle_best")) else "nan"),
                kind=str(r.get("v35_kind")),
                rot=str(bool(r.get("v35_is_rotation"))),
            )
        )
    (PANEL_ROOT / "t45_rerank_v35_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=0.5, help="rotation prior multiplier")
    p.add_argument("--min-delta", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_main(parse_args()))
