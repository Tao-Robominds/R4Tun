#!/usr/bin/env python3
"""V3.5 alternate re-rank: rotational-stability-aware selector.

Inside the rotation candidate sweep, the *good* rotation tends to have
a coherent neighborhood: the chosen offset and its two adjacent offsets
(`+/- 1/N`) all keep guardrails satisfied. Bad rotations are isolated
spikes in `J_reflect`. We can pick a rotation whose neighborhood is
stable. This metric is intrinsic (uses J_reflect / S_boundary only).

Selection rule
--------------
1. Filter to candidates with ``guardrail_pass = True``.
2. If any rotation candidate exists, restrict the choice to rotations
   (a soft inductive bias: rotations are the only structural axis).
3. For each rotation offset ``k/N`` recorded in the trace, compute its
   neighborhood score:
   ``score = J_reflect(k) * (1 + 0.5 * sum_{neighbors} J_reflect(neighbor) / J_reflect(k))``
   We use the *log-mean* of the immediate neighbors' J_reflect as a
   stability bonus.
4. Pick the rotation with highest stability-adjusted score.
5. Fall back to baseline if no candidate beats baseline on raw
   J_reflect by ``min_delta`` (rollback).

Outputs
-------
``logs/iterative_reflection_proof_v3/panel/r0/t45_rerank_v36_*``
"""

from __future__ import annotations

import argparse
import json
import re
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


_ROT_RE = re.compile(r"^rot(\d+)/(\d+)$")


def _parse_rot(kind: Any) -> tuple[int, int] | None:
    if not isinstance(kind, str):
        return None
    m = _ROT_RE.match(kind)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _rerank_trace(trace: dict[str, Any], min_delta: float) -> dict[str, Any]:
    base = trace.get("baseline_A0", {})
    base_j = _safe_float(base.get("J_reflect")) or 0.0

    # Aggregate the latest passing candidate per kind across all rounds.
    per_kind_latest: dict[str, dict[str, Any]] = {}
    rotations_kind: dict[tuple[int, int], dict[str, Any]] = {}
    other_kinds: list[dict[str, Any]] = []
    for rd in trace.get("rounds", []):
        for c in rd.get("round_results", []) or []:
            if not c.get("guardrail_pass"):
                continue
            kind = str(c.get("candidate_kind") or "")
            per_kind_latest[kind] = c
            rot = _parse_rot(kind)
            if rot is not None:
                rotations_kind[rot] = c
            else:
                other_kinds.append(c)

    # Build the rotation slate: for each (k/N), gather J_reflect values.
    rotations_score: list[dict[str, Any]] = []
    if rotations_kind:
        # Group by N (in our generator we use a single N per ring, e.g. 12).
        denominators: dict[int, dict[int, dict[str, Any]]] = {}
        for (k, n), entry in rotations_kind.items():
            denominators.setdefault(n, {})[k] = entry
        for n, slate in denominators.items():
            keys_sorted = sorted(slate.keys())
            for k in keys_sorted:
                entry = slate[k]
                k_left = (k - 1) % n
                k_right = (k + 1) % n
                neigh = [
                    _safe_float((slate.get(k_left) or {}).get("J_reflect")) or 0.0,
                    _safe_float((slate.get(k_right) or {}).get("J_reflect")) or 0.0,
                ]
                # Stability score: geometric mean of neighborhood + center.
                ctr = _safe_float(entry.get("J_reflect")) or 0.0
                # Add small epsilon to avoid log(0).
                vals = [v + 1e-12 for v in neigh + [ctr]]
                stab = float(np.exp(np.mean(np.log(vals))))
                rotations_score.append(
                    {
                        "kind": f"rot{k}/{n}",
                        "stability": stab,
                        "J_reflect": ctr,
                        "neighbors_J": neigh,
                        "miou": _safe_float(entry.get("miou")),
                        "oa": _safe_float(entry.get("oa")),
                        "G_layout": _safe_float(entry.get("G_layout")),
                        "G_pre": _safe_float(entry.get("G_pre")),
                        "G_stability": _safe_float(entry.get("G_stability")),
                    }
                )

    # Choose: prefer rotation with best stability over baseline; otherwise
    # pick best non-rotation by raw J_reflect; otherwise stay with baseline.
    cur_best = {
        "candidate_kind": "baseline_A0",
        "J_reflect": base_j,
        "stability": base_j,
        "miou": _safe_float(base.get("miou")),
        "oa": _safe_float(base.get("oa")),
        "G_layout": _safe_float(base.get("G_layout")),
        "G_pre": _safe_float(base.get("G_pre")),
        "G_stability": _safe_float(base.get("G_stability")),
    }

    if rotations_score:
        rotations_score.sort(key=lambda r: float(r["stability"]), reverse=True)
        top_rot = rotations_score[0]
        # Only accept the rotation if its raw J_reflect is at least
        # min_delta above baseline (rollback rule).
        if (top_rot["J_reflect"] or 0.0) >= base_j + float(min_delta):
            cur_best = {
                "candidate_kind": top_rot["kind"],
                "J_reflect": top_rot["J_reflect"],
                "stability": top_rot["stability"],
                "miou": top_rot["miou"],
                "oa": top_rot["oa"],
                "G_layout": top_rot["G_layout"],
                "G_pre": top_rot["G_pre"],
                "G_stability": top_rot["G_stability"],
            }
    if cur_best["candidate_kind"] == "baseline_A0" and other_kinds:
        # No usable rotation; fall back to the highest-J non-rotation.
        other_kinds.sort(key=lambda c: _safe_float(c.get("J_reflect")) or 0.0, reverse=True)
        cand = other_kinds[0]
        if (_safe_float(cand.get("J_reflect")) or 0.0) >= base_j + float(min_delta):
            cur_best = {
                "candidate_kind": cand.get("candidate_kind"),
                "J_reflect": _safe_float(cand.get("J_reflect")),
                "stability": _safe_float(cand.get("J_reflect")),
                "miou": _safe_float(cand.get("miou")),
                "oa": _safe_float(cand.get("oa")),
                "G_layout": _safe_float(cand.get("G_layout")),
                "G_pre": _safe_float(cand.get("G_pre")),
                "G_stability": _safe_float(cand.get("G_stability")),
            }
    cur_best["rotation_slate_size"] = len(rotations_score)
    return cur_best


def _main(args: argparse.Namespace) -> int:
    src = pd.read_csv(PANEL_ROOT / "t45_iterative_v3_results.csv")
    rows = []
    for _, row in src.iterrows():
        ring_key = str(row["ring_key"])
        tunnel_id, ring_dir = ring_key.split("/", 1)
        ring_id = int(ring_dir[1:])
        trace_path = RINGS_ROOT / tunnel_id / f"r{ring_id}" / "A2_iterative_intrinsic_reflection" / "iterative_trace_v3.json"
        if not trace_path.exists():
            rows.append({"ring_key": ring_key, "error": "trace_missing"})
            continue
        trace = json.loads(trace_path.read_text())
        chosen = _rerank_trace(trace, min_delta=float(args.min_delta))
        rows.append(
            {
                "ring_key": ring_key,
                "tunnel_id": tunnel_id,
                "ring_id": ring_id,
                "mIoU_no_reflection": _safe_float(row["mIoU_no_reflection"]),
                "mIoU_A1_single_pass": _safe_float(row["mIoU_A1_single_pass"]),
                "mIoU_A2_v3_intrinsic": _safe_float(row["mIoU_A2_v3_intrinsic"]),
                "mIoU_oracle_best": _safe_float(row["mIoU_oracle_best"]),
                "mIoU_v36_intrinsic": _safe_float(chosen.get("miou")),
                "OA_v36_intrinsic": _safe_float(chosen.get("oa")),
                "v36_kind": chosen.get("candidate_kind"),
                "v36_stability": _safe_float(chosen.get("stability")),
                "v36_J_reflect": _safe_float(chosen.get("J_reflect")),
                "rotation_slate_size": int(chosen.get("rotation_slate_size") or 0),
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(PANEL_ROOT / "t45_rerank_v36_results.csv", index=False)

    valid = out_df.dropna(subset=["mIoU_v36_intrinsic", "mIoU_no_reflection"]).copy()
    if not valid.empty:
        d = valid["mIoU_v36_intrinsic"] - valid["mIoU_no_reflection"]
        try:
            t_p = _safe_float(ttest_rel(valid["mIoU_v36_intrinsic"], valid["mIoU_no_reflection"]).pvalue) if len(valid) >= 2 else None
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
        "min_delta": float(args.min_delta),
        "n_rows": int(len(out_df)),
        "n_valid": int(len(valid)),
        "mean_mIoU_A0": _safe_float(valid["mIoU_no_reflection"].mean()) if not valid.empty else None,
        "mean_mIoU_v3_intrinsic": _safe_float(valid["mIoU_A2_v3_intrinsic"].mean()) if not valid.empty else None,
        "mean_mIoU_v36_intrinsic": _safe_float(valid["mIoU_v36_intrinsic"].mean()) if not valid.empty else None,
        "mean_mIoU_oracle": _safe_float(valid["mIoU_oracle_best"].mean()) if not valid.empty else None,
        "median_mIoU_v36_intrinsic": _safe_float(valid["mIoU_v36_intrinsic"].median()) if not valid.empty else None,
        "share_v36_ge_04": _safe_float((valid["mIoU_v36_intrinsic"] >= 0.4).mean()) if not valid.empty else None,
        "share_v36_ge_05": _safe_float((valid["mIoU_v36_intrinsic"] >= 0.5).mean()) if not valid.empty else None,
        "share_v3_ge_04": _safe_float((valid["mIoU_A2_v3_intrinsic"] >= 0.4).mean()) if not valid.empty else None,
        "share_oracle_ge_04": _safe_float((valid["mIoU_oracle_best"] >= 0.4).mean()) if not valid.empty else None,
        "mean_delta_mIoU_v36_vs_A0": _safe_float((valid["mIoU_v36_intrinsic"] - valid["mIoU_no_reflection"]).mean()) if not valid.empty else None,
        "paired_ttest_p_mIoU": t_p,
        "wilcoxon_p_mIoU": w_p,
    }
    (PANEL_ROOT / "t45_rerank_v36_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # Markdown report.
    report = [
        "# Tunnel 4/5 v3.6 Re-rank Report (rotational stability)",
        "",
        f"- min_delta vs baseline J: `{summary['min_delta']}`",
        f"- n: `{summary['n_valid']}` evaluated of `{summary['n_rows']}` total",
        "",
        "## Aggregate mIoU",
        "",
        f"- mean mIoU A0: `{summary['mean_mIoU_A0']}`",
        f"- mean mIoU v3 (raw J_reflect): `{summary['mean_mIoU_v3_intrinsic']}`",
        f"- mean mIoU v3.6 (rotational stability): `{summary['mean_mIoU_v36_intrinsic']}`",
        f"- mean mIoU oracle (mIoU-best in candidate pool, **diagnostic**): `{summary['mean_mIoU_oracle']}`",
        f"- median v3.6: `{summary['median_mIoU_v36_intrinsic']}`",
        f"- share v3 >= 0.4: `{summary['share_v3_ge_04']}`",
        f"- share v3.6 >= 0.4: `{summary['share_v36_ge_04']}`",
        f"- share v3.6 >= 0.5: `{summary['share_v36_ge_05']}`",
        f"- share oracle >= 0.4: `{summary['share_oracle_ge_04']}`",
        f"- mean delta v3.6 vs A0: `{summary['mean_delta_mIoU_v36_vs_A0']}`",
        f"- paired t-test p: `{summary['paired_ttest_p_mIoU']}`",
        f"- Wilcoxon p: `{summary['wilcoxon_p_mIoU']}`",
        "",
        "## Per-ring",
        "",
        "| ring_key | A0 mIoU | A1 mIoU | v3 mIoU | v3.6 mIoU | oracle mIoU | v3.6 kind |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in out_df.iterrows():
        report.append(
            "| {rk} | {a0:s} | {a1:s} | {v3:s} | {v36:s} | {orc:s} | {kind} |".format(
                rk=r["ring_key"],
                a0=("{:.4f}".format(r["mIoU_no_reflection"]) if pd.notna(r.get("mIoU_no_reflection")) else "nan"),
                a1=("{:.4f}".format(r["mIoU_A1_single_pass"]) if pd.notna(r.get("mIoU_A1_single_pass")) else "nan"),
                v3=("{:.4f}".format(r["mIoU_A2_v3_intrinsic"]) if pd.notna(r.get("mIoU_A2_v3_intrinsic")) else "nan"),
                v36=("{:.4f}".format(r["mIoU_v36_intrinsic"]) if pd.notna(r.get("mIoU_v36_intrinsic")) else "nan"),
                orc=("{:.4f}".format(r["mIoU_oracle_best"]) if pd.notna(r.get("mIoU_oracle_best")) else "nan"),
                kind=str(r.get("v36_kind")),
            )
        )
    (PANEL_ROOT / "t45_rerank_v36_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-delta", type=float, default=-1e-9, help="rollback threshold on raw J_reflect")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_main(parse_args()))
