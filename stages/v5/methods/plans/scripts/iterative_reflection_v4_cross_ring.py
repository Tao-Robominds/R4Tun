#!/usr/bin/env python3
"""Cross-ring intrinsic re-ranker on top of v4.

Idea
----
Held-out tunnel sections contain multiple rings each (e.g. 4-3 has r170
and r171; 4-8 has r330, r334, r337). Physically these rings of one
tunnel section share roughly the same rotational offset, because the
unwrapping is done with a consistent reference for the entire tunnel.
That gives us a strong intrinsic constraint:

    For each tunnel, pick the rotation candidate kind k* that maximizes
    the *aggregate* per-ring J_reflect (and G_structural) across all
    held-out rings of that tunnel.

The same kind constraint is principled because:
  - the template y_frac is in [0, 1] so cyclic rotations are comparable
    across rings of one tunnel,
  - rotation candidate kinds (e.g. ``rot7/12``) are deterministic across
    rings,
  - the structural alignment metric, J_reflect, S_boundary etc are all
    intrinsic.

For each ring we then *commit* to that kind k*'s cached labelmap as the
v4.5 winner. Solo-ring tunnels (no neighbor) keep their v4 result.

This re-ranking does NOT re-run detection; it reads the cached
``logs/iterative_reflection_proof_v4/candidate_labelmaps/<tunnel>/
r<ring>/cand_<idx>/metrics.json`` files written by v4.

Outputs
-------
``logs/iterative_reflection_proof_v4/panel/r0/t45_v45_*``
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[3]
PANEL_ROOT = REPO_ROOT / "logs" / "iterative_reflection_proof_v4" / "panel" / "r0"
CAND_ROOT = REPO_ROOT / "logs" / "iterative_reflection_proof_v4" / "candidate_labelmaps"


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


def _per_ring_candidates(ring_key: str) -> list[dict[str, Any]]:
    """Read all cached candidate metrics for a ring."""

    tunnel_id, ring_dir = ring_key.split("/", 1)
    rd = CAND_ROOT / tunnel_id / ring_dir
    if not rd.exists():
        return []
    out = []
    for d in sorted(rd.iterdir()):
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text())
        except Exception:  # noqa: BLE001
            continue
        out.append(m)
    return out


def _aggregate_kind_score(
    rings_data: dict[str, list[dict[str, Any]]],
    kind: str,
) -> dict[str, Any]:
    """Aggregate J_reflect across rings of a tunnel for a given kind.

    We require the kind to appear in *every* ring (else the constraint
    is broken). We use the geometric mean of J_reflect (so we can't be
    saved by one outlier) and sum of mIoU (diagnostic only, returned
    for analysis but never used to choose).
    """

    j_values: list[float] = []
    miou_values: list[float] = []
    g_struct_values: list[float] = []
    n_pass = 0
    for rk, cands in rings_data.items():
        match = [c for c in cands if c.get("candidate_kind") == kind and c.get("guardrail_pass")]
        if not match:
            return {
                "n_rings_present": 0,
                "j_geomean": 0.0,
                "g_struct_geomean": 0.0,
                "mean_miou": None,
                "rings": rings_data.keys(),
                "kind": kind,
                "n_rings_total": len(rings_data),
            }
        # if a kind appears multiple times across rounds, pick the latest
        c = match[-1]
        j = _safe_float(c.get("J_reflect")) or 0.0
        gs = _safe_float(c.get("G_structural")) or 0.0
        m = _safe_float(c.get("miou"))
        j_values.append(j)
        g_struct_values.append(gs)
        if m is not None:
            miou_values.append(m)
        n_pass += 1
    if not j_values:
        return {
            "n_rings_present": 0,
            "j_geomean": 0.0,
            "g_struct_geomean": 0.0,
            "mean_miou": None,
            "rings": list(rings_data.keys()),
            "kind": kind,
            "n_rings_total": len(rings_data),
        }
    j_geomean = float(np.exp(np.mean(np.log([max(1e-12, v) for v in j_values]))))
    g_struct_geomean = float(np.exp(np.mean(np.log([max(1e-12, v) for v in g_struct_values]))))
    return {
        "kind": kind,
        "n_rings_present": int(n_pass),
        "n_rings_total": int(len(rings_data)),
        "j_geomean": j_geomean,
        "g_struct_geomean": g_struct_geomean,
        "mean_miou": float(np.mean(miou_values)) if miou_values else None,
        "rings": list(rings_data.keys()),
    }


def _select_per_tunnel_kind(rings_data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """For one tunnel, pick the rotation kind that maximizes aggregate J.

    Falls back to per-ring v4 baseline when no rotation passes for all
    rings simultaneously.
    """

    # Collect all rotation kinds seen in any ring.
    kinds: set[str] = set()
    for cands in rings_data.values():
        for c in cands:
            k = c.get("candidate_kind")
            if _is_rotation_kind(k):
                kinds.add(str(k))
    aggregated = []
    for kind in sorted(kinds):
        agg = _aggregate_kind_score(rings_data, kind)
        if agg["n_rings_present"] == agg["n_rings_total"]:
            aggregated.append(agg)
    if not aggregated:
        return {"selected_kind": None, "aggregated": []}
    # Pick by j_geomean * g_struct_geomean (both intrinsic).
    aggregated.sort(key=lambda a: a["j_geomean"] * (0.05 + 0.95 * a["g_struct_geomean"]), reverse=True)
    return {"selected_kind": aggregated[0]["kind"], "aggregated": aggregated[:5]}


def _main(args: argparse.Namespace) -> int:
    src_path = PANEL_ROOT / "t45_iterative_v4_results.csv"
    if not src_path.exists():
        raise FileNotFoundError(f"Expected v4 results at {src_path}")
    src = pd.read_csv(src_path)

    # Group by tunnel.
    by_tunnel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, r in src.iterrows():
        by_tunnel[str(r["tunnel_id"])].append(dict(r))

    rows: list[dict[str, Any]] = []
    tunnel_sel: dict[str, dict[str, Any]] = {}
    for t, ring_rows in by_tunnel.items():
        if len(ring_rows) >= 2:
            rings_data = {}
            for r in ring_rows:
                rk = r["ring_key"]
                cands = _per_ring_candidates(rk)
                rings_data[rk] = cands
            sel = _select_per_tunnel_kind(rings_data)
            tunnel_sel[t] = sel
        else:
            sel = {"selected_kind": None, "aggregated": []}
            tunnel_sel[t] = sel

        for r in ring_rows:
            rk = r["ring_key"]
            cands = _per_ring_candidates(rk)
            chosen_kind = sel.get("selected_kind")
            if chosen_kind is not None:
                # Use the cached candidate's mIoU as v4.5 mIoU (oracle of
                # the v4 candidate pool restricted to chosen_kind).
                match = [c for c in cands if c.get("candidate_kind") == chosen_kind and c.get("guardrail_pass")]
                if match:
                    c = match[-1]
                    rows.append(
                        {
                            "ring_key": rk,
                            "tunnel_id": t,
                            "ring_id": int(r["ring_id"]),
                            "mIoU_no_reflection": _safe_float(r["mIoU_no_reflection"]),
                            "mIoU_A1_single_pass": _safe_float(r["mIoU_A1_single_pass"]),
                            "mIoU_v4_intrinsic": _safe_float(r["mIoU_v4_intrinsic"]),
                            "mIoU_v45_intrinsic": _safe_float(c.get("miou")),
                            "OA_v45_intrinsic": _safe_float(c.get("oa")),
                            "mIoU_oracle_best": _safe_float(r["mIoU_oracle_best"]),
                            "v45_kind": chosen_kind,
                            "v45_J_reflect": _safe_float(c.get("J_reflect")),
                            "v45_G_structural": _safe_float(c.get("G_structural")),
                            "v45_chosen_by": "tunnel_consensus",
                        }
                    )
                    continue
            # Fallback to v4 result.
            rows.append(
                {
                    "ring_key": rk,
                    "tunnel_id": t,
                    "ring_id": int(r["ring_id"]),
                    "mIoU_no_reflection": _safe_float(r["mIoU_no_reflection"]),
                    "mIoU_A1_single_pass": _safe_float(r["mIoU_A1_single_pass"]),
                    "mIoU_v4_intrinsic": _safe_float(r["mIoU_v4_intrinsic"]),
                    "mIoU_v45_intrinsic": _safe_float(r["mIoU_v4_intrinsic"]),
                    "OA_v45_intrinsic": None,
                    "mIoU_oracle_best": _safe_float(r["mIoU_oracle_best"]),
                    "v45_kind": r.get("intrinsic_best_kind"),
                    "v45_J_reflect": None,
                    "v45_G_structural": _safe_float(r.get("G_structural_at_winner")),
                    "v45_chosen_by": "fallback_v4",
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(PANEL_ROOT / "t45_v45_results.csv", index=False)
    sel_payload = {t: {"selected_kind": s.get("selected_kind"), "aggregated_top5": s.get("aggregated")} for t, s in tunnel_sel.items()}
    (PANEL_ROOT / "t45_v45_tunnel_selection.json").write_text(json.dumps(sel_payload, indent=2, sort_keys=True) + "\n")

    valid = out_df.dropna(subset=["mIoU_v45_intrinsic", "mIoU_no_reflection"]).copy()
    if not valid.empty:
        d = valid["mIoU_v45_intrinsic"] - valid["mIoU_no_reflection"]
        try:
            t_p = _safe_float(ttest_rel(valid["mIoU_v45_intrinsic"], valid["mIoU_no_reflection"]).pvalue) if len(valid) >= 2 else None
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
        "n_rows": int(len(out_df)),
        "n_valid": int(len(valid)),
        "mean_mIoU_A0": _safe_float(valid["mIoU_no_reflection"].mean()) if not valid.empty else None,
        "mean_mIoU_v4": _safe_float(valid["mIoU_v4_intrinsic"].mean()) if not valid.empty else None,
        "mean_mIoU_v45": _safe_float(valid["mIoU_v45_intrinsic"].mean()) if not valid.empty else None,
        "mean_mIoU_oracle": _safe_float(out_df["mIoU_oracle_best"].dropna().mean()) if not out_df.empty else None,
        "median_mIoU_v45": _safe_float(valid["mIoU_v45_intrinsic"].median()) if not valid.empty else None,
        "share_v45_ge_04": _safe_float((valid["mIoU_v45_intrinsic"] >= 0.4).mean()) if not valid.empty else None,
        "share_v45_ge_05": _safe_float((valid["mIoU_v45_intrinsic"] >= 0.5).mean()) if not valid.empty else None,
        "share_v4_ge_04": _safe_float((valid["mIoU_v4_intrinsic"] >= 0.4).mean()) if not valid.empty else None,
        "share_oracle_ge_04": _safe_float((out_df["mIoU_oracle_best"] >= 0.4).mean()) if not out_df["mIoU_oracle_best"].dropna().empty else None,
        "mean_delta_v45_vs_A0": _safe_float((valid["mIoU_v45_intrinsic"] - valid["mIoU_no_reflection"]).mean()) if not valid.empty else None,
        "mean_delta_v45_vs_v4": _safe_float((valid["mIoU_v45_intrinsic"] - valid["mIoU_v4_intrinsic"]).mean()) if not valid.empty else None,
        "paired_ttest_p_mIoU_vs_A0": t_p,
        "wilcoxon_p_mIoU_vs_A0": w_p,
        "n_tunnels_with_consensus": int(sum(1 for s in tunnel_sel.values() if s.get("selected_kind") is not None)),
    }
    (PANEL_ROOT / "t45_v45_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = [
        "# Tunnel 4/5 v4.5 Cross-Ring Re-rank Report",
        "",
        f"- n: `{summary['n_valid']}` evaluated of `{summary['n_rows']}` total",
        f"- tunnels with cross-ring consensus: `{summary['n_tunnels_with_consensus']}`",
        "",
        "## Aggregate mIoU",
        "",
        f"- mean A0: `{summary['mean_mIoU_A0']}`",
        f"- mean v4 intrinsic: `{summary['mean_mIoU_v4']}`",
        f"- mean v4.5 (cross-ring intrinsic): `{summary['mean_mIoU_v45']}`",
        f"- mean oracle (mIoU-best in candidate pool, **diagnostic**): `{summary['mean_mIoU_oracle']}`",
        f"- median v4.5: `{summary['median_mIoU_v45']}`",
        f"- share v4.5 >= 0.4: `{summary['share_v45_ge_04']}`",
        f"- share v4.5 >= 0.5: `{summary['share_v45_ge_05']}`",
        f"- share v4 >= 0.4: `{summary['share_v4_ge_04']}`",
        f"- share oracle >= 0.4: `{summary['share_oracle_ge_04']}`",
        f"- mean delta v4.5 vs A0: `{summary['mean_delta_v45_vs_A0']}`",
        f"- mean delta v4.5 vs v4: `{summary['mean_delta_v45_vs_v4']}`",
        f"- paired t-test p (v4.5 vs A0): `{summary['paired_ttest_p_mIoU_vs_A0']}`",
        f"- Wilcoxon p (v4.5 vs A0): `{summary['wilcoxon_p_mIoU_vs_A0']}`",
        "",
        "## Per-tunnel consensus rotation",
        "",
        "| tunnel | selected kind |",
        "|---|---|",
    ]
    for t, s in sorted(tunnel_sel.items()):
        report.append(f"| {t} | {s.get('selected_kind') or '-'} |")
    report += [
        "",
        "## Per-ring",
        "",
        "| ring_key | A0 | v4 | v4.5 | oracle | v4.5 kind | chosen_by |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in out_df.iterrows():
        report.append(
            "| {rk} | {a0:s} | {v4:s} | {v45:s} | {orc:s} | {kind} | {by} |".format(
                rk=r["ring_key"],
                a0=("{:.4f}".format(r["mIoU_no_reflection"]) if pd.notna(r.get("mIoU_no_reflection")) else "nan"),
                v4=("{:.4f}".format(r["mIoU_v4_intrinsic"]) if pd.notna(r.get("mIoU_v4_intrinsic")) else "nan"),
                v45=("{:.4f}".format(r["mIoU_v45_intrinsic"]) if pd.notna(r.get("mIoU_v45_intrinsic")) else "nan"),
                orc=("{:.4f}".format(r["mIoU_oracle_best"]) if pd.notna(r.get("mIoU_oracle_best")) else "nan"),
                kind=str(r.get("v45_kind")),
                by=str(r.get("v45_chosen_by")),
            )
        )
    (PANEL_ROOT / "t45_v45_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_main(parse_args()))
