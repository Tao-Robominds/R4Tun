#!/usr/bin/env python3
"""Step 7 reflection proof runner.

Builds held-out proof artifacts under:
  logs/reflection_proof_v1/panel/r0/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "logs" / "reflection_proof_v1" / "panel" / "r0"
HELDOUT_ROOT = REPO_ROOT / "logs" / "reflection_proof_v1" / "heldout_reflection_test"
HELDOUT_PANEL = REPO_ROOT / "data" / "panels" / "heldout" / "heldout_reflection_test_set.json"
FROZEN_THRESHOLDS = REPO_ROOT / "logs" / "proxy_validation_v1" / "frozen_thresholds.json"
EXISTING_TRACES = REPO_ROOT / "logs" / "proxy_validation_v1" / "reflection_traces.json"
PV_SCRIPT = REPO_ROOT / "methods" / "plans" / "scripts" / "proxy_validation_v1.py"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _ring_key(tunnel_id: str, ring_id: int) -> str:
    return f"{tunnel_id}/r{int(ring_id)}"


def _load_proxy_validation_module():
    spec = importlib.util.spec_from_file_location("pv_step7", PV_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {PV_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    # Allow writing under reflection_proof_v1 instead of proxy_validation_v1.
    mod.OUT_ROOT = REPO_ROOT / "logs" / "reflection_proof_v1"
    mod.HELDOUT_ROOT = HELDOUT_ROOT
    return mod


def _trigger_reason(a0_row: dict[str, Any], frozen: dict[str, Any]) -> str:
    td = float(frozen["selected"]["T_depth"])
    tb = float(frozen["selected"]["T_boundary"])
    sd = _safe_float(a0_row.get("S_depth"))
    sb = _safe_float(a0_row.get("S_boundary"))
    depth = sd is not None and sd < td
    boundary = sb is not None and sb < tb
    if depth and boundary:
        return "both"
    if depth:
        return "depth"
    if boundary:
        return "boundary"
    return "none"


def _ensure_reflection_row(
    *,
    pv: Any,
    panel_row: dict[str, Any],
    variant: str,
    reflection: bool,
    existing_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rk = _ring_key(str(panel_row["tunnel_id"]), int(panel_row["ring_id"]))
    current = existing_cache.get(rk)
    if current is not None and not current.get("failed"):
        return current
    try:
        result = pv._run_ring(panel_row, output_root=HELDOUT_ROOT, reflection=reflection, variant=variant)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        tunnel_id = str(panel_row["tunnel_id"])
        ring_id = int(panel_row["ring_id"])
        ring_dir = HELDOUT_ROOT / tunnel_id / f"r{ring_id}" / variant
        ring_dir.mkdir(parents=True, exist_ok=True)
        result = {
            **panel_row,
            "ring_key": rk,
            "variant": variant,
            "failed": True,
            "timeout": False,
            "error": str(exc),
            "output_dir": str(ring_dir),
            "timestamp_utc": _now(),
        }
        _write_json(ring_dir / "proxy_validation_ring_result.json", result)
    existing_cache[rk] = result
    return result


def _paired_rows(
    *,
    variant_name: str,
    a0_rows: list[dict[str, Any]],
    reflective_rows_by_key: dict[str, dict[str, Any]],
    triggered_by_key: dict[str, bool],
    trigger_reason_by_key: dict[str, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a0 in a0_rows:
        rk = str(a0["ring_key"])
        trig = bool(triggered_by_key.get(rk, False))
        reason = str(trigger_reason_by_key.get(rk, "none"))
        reflective = reflective_rows_by_key.get(rk)
        used_reflective = trig and reflective is not None and not reflective.get("failed")
        final_row = reflective if used_reflective else a0
        miou0 = _safe_float(a0.get("final_mIoU"))
        oa0 = _safe_float(a0.get("final_OA"))
        miou1 = _safe_float(final_row.get("final_mIoU"))
        oa1 = _safe_float(final_row.get("final_OA"))
        out.append(
            {
                "variant": variant_name,
                "ring_key": rk,
                "tunnel_id": a0.get("tunnel_id"),
                "ring_id": a0.get("ring_id"),
                "triggered": trig,
                "trigger_reason": reason if trig else "none",
                "mIoU_no_reflection": miou0,
                "mIoU_reflection": miou1,
                "delta_mIoU": None if miou0 is None or miou1 is None else float(miou1 - miou0),
                "OA_no_reflection": oa0,
                "OA_reflection": oa1,
                "delta_OA": None if oa0 is None or oa1 is None else float(oa1 - oa0),
                "is_bad_case_no_reflection": bool(a0.get("is_bad_case")),
                "corrective_passes": 1 if trig else 0,
                "used_reflective_row": bool(used_reflective),
                "reflective_failed": bool(trig and reflective is not None and reflective.get("failed")),
                "reflective_error": None if reflective is None else reflective.get("error"),
                "A0_output_dir": a0.get("output_dir"),
                "variant_output_dir": None if reflective is None else reflective.get("output_dir"),
            }
        )
    return out


def _cluster_bootstrap(rows: list[dict[str, Any]], *, n_boot: int = 2000, seed: int = 41) -> dict[str, Any]:
    valid = [r for r in rows if _safe_float(r.get("delta_mIoU")) is not None]
    clusters = sorted({str(r["tunnel_id"]) for r in valid})
    if not clusters:
        return {
            "mean_delta_mIoU": {"lo": None, "hi": None},
            "median_delta_mIoU": {"lo": None, "hi": None},
            "mean_delta_OA": {"lo": None, "hi": None},
            "trigger_rate": {"lo": None, "hi": None},
        }
    rng = np.random.default_rng(seed)
    grouped = {c: [r for r in valid if str(r["tunnel_id"]) == c] for c in clusters}
    samples = {
        "mean_delta_mIoU": [],
        "median_delta_mIoU": [],
        "mean_delta_OA": [],
        "trigger_rate": [],
    }
    for _ in range(n_boot):
        picked: list[dict[str, Any]] = []
        for c in rng.choice(clusters, size=len(clusters), replace=True):
            picked.extend(grouped[str(c)])
        dmiou = np.array([float(r["delta_mIoU"]) for r in picked], dtype=float)
        doa = np.array([float(r["delta_OA"]) for r in picked if _safe_float(r.get("delta_OA")) is not None], dtype=float)
        samples["mean_delta_mIoU"].append(float(np.mean(dmiou)))
        samples["median_delta_mIoU"].append(float(np.median(dmiou)))
        samples["mean_delta_OA"].append(float(np.mean(doa)) if doa.size else 0.0)
        samples["trigger_rate"].append(float(np.mean([bool(r["triggered"]) for r in picked])))
    return {k: {"lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975))} for k, v in samples.items()}


def _variant_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if _safe_float(r.get("delta_mIoU")) is not None]
    dmiou = np.array([float(r["delta_mIoU"]) for r in valid], dtype=float)
    doa = np.array([float(r["delta_OA"]) for r in valid if _safe_float(r.get("delta_OA")) is not None], dtype=float)
    p_t = None
    p_w = None
    if len(valid) >= 2:
        p_t = _safe_float(
            ttest_rel(
                [float(r["mIoU_reflection"]) for r in valid],
                [float(r["mIoU_no_reflection"]) for r in valid],
                nan_policy="omit",
            ).pvalue
        )
        try:
            p_w = _safe_float(wilcoxon(dmiou).pvalue)
        except ValueError:
            p_w = None
    sd = float(np.std(dmiou, ddof=1)) if len(dmiou) > 1 else 0.0
    cohen_d = float(np.mean(dmiou) / sd) if sd > 1e-12 else None
    return {
        "n_pairs": int(len(valid)),
        "mean_delta_mIoU": float(np.mean(dmiou)) if len(dmiou) else None,
        "median_delta_mIoU": float(np.median(dmiou)) if len(dmiou) else None,
        "mean_delta_OA": float(np.mean(doa)) if len(doa) else None,
        "median_delta_OA": float(np.median(doa)) if len(doa) else None,
        "paired_ttest_p_value_mIoU": p_t,
        "wilcoxon_p_value_mIoU": p_w,
        "cohen_d_paired_mIoU": cohen_d,
        "improved_count": int(np.sum(dmiou > 1e-9)) if len(dmiou) else 0,
        "unchanged_count": int(np.sum(np.abs(dmiou) <= 1e-9)) if len(dmiou) else 0,
        "worsened_count": int(np.sum(dmiou < -1e-9)) if len(dmiou) else 0,
        "trigger_rate": float(np.mean([bool(r["triggered"]) for r in valid])) if valid else None,
        "trigger_precision_on_bad_labels": (
            float(
                sum(bool(r["triggered"]) and bool(r["is_bad_case_no_reflection"]) for r in valid)
                / sum(bool(r["triggered"]) for r in valid)
            )
            if sum(bool(r["triggered"]) for r in valid)
            else None
        ),
        "cluster_bootstrap": _cluster_bootstrap(valid),
    }


def _build_failure_audit(
    *,
    a1_pairs: list[dict[str, Any]],
    a0_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    a0_by_key = {str(r["ring_key"]): r for r in a0_rows}
    false_negatives = [
        rk
        for rk, row in a0_by_key.items()
        if bool(row.get("is_bad_case"))
        and not any(p["ring_key"] == rk and bool(p["triggered"]) for p in a1_pairs)
    ]
    worsened = [
        {
            "ring_key": r["ring_key"],
            "delta_mIoU": r["delta_mIoU"],
            "delta_OA": r["delta_OA"],
            "trigger_reason": r["trigger_reason"],
        }
        for r in a1_pairs
        if _safe_float(r.get("delta_mIoU")) is not None and float(r["delta_mIoU"]) < -1e-9
    ]
    failed_reflections = [
        {
            "ring_key": r["ring_key"],
            "trigger_reason": r["trigger_reason"],
            "reflective_error": r["reflective_error"],
            "variant_output_dir": r["variant_output_dir"],
        }
        for r in a1_pairs
        if bool(r.get("reflective_failed"))
    ]
    return {
        "false_negatives_proxy_trigger": false_negatives,
        "worsened_cases_proxy_trigger": worsened,
        "failed_corrective_passes_proxy_trigger": failed_reflections,
    }


def _report(
    *,
    frozen: dict[str, Any],
    stats_by_variant: dict[str, dict[str, Any]],
    failure_audit: dict[str, Any],
    random_seed: int,
    random_budget: int,
) -> str:
    a1 = stats_by_variant["A1_proxy_reflection"]
    a2 = stats_by_variant["A2_always_reflect"]
    a3 = stats_by_variant["A3_random_reflect"]
    mean_a1 = _safe_float(a1.get("mean_delta_mIoU"))
    mean_a3 = _safe_float(a3.get("mean_delta_mIoU"))
    proof_pass = bool(
        mean_a1 is not None
        and mean_a3 is not None
        and mean_a1 > 0.0
        and mean_a1 > mean_a3
        and (a1.get("worsened_count", 0) <= a3.get("worsened_count", 0))
    )
    lines = [
        "# Step 7 Reflection Proof Report",
        "",
        "## Setup",
        f"- frozen trigger input: `logs/proxy_validation_v1/frozen_thresholds.json`",
        f"- selected rule: `{frozen['selected']['rule']}`",
        f"- T_depth: `{float(frozen['selected']['T_depth']):.6f}`",
        f"- T_boundary: `{float(frozen['selected']['T_boundary']):.6f}`",
        f"- held-out panel: `data/panels/heldout/heldout_reflection_test_set.json`",
        "",
        "## Variant Comparison (vs A0 baseline)",
        f"- A1 proxy reflection mean delta mIoU: `{a1.get('mean_delta_mIoU')}`",
        f"- A1 proxy reflection mean delta OA: `{a1.get('mean_delta_OA')}`",
        f"- A1 paired t-test p (mIoU): `{a1.get('paired_ttest_p_value_mIoU')}`",
        f"- A1 Wilcoxon p (mIoU): `{a1.get('wilcoxon_p_value_mIoU')}`",
        f"- A2 always reflect mean delta mIoU: `{a2.get('mean_delta_mIoU')}`",
        f"- A3 random reflect mean delta mIoU: `{a3.get('mean_delta_mIoU')}`",
        f"- A3 random config: seed=`{random_seed}`, budget=`{random_budget}`",
        "",
        "## Failure Audit",
        f"- false negatives (A1): `{failure_audit['false_negatives_proxy_trigger']}`",
        f"- worsened cases (A1): `{[r['ring_key'] for r in failure_audit['worsened_cases_proxy_trigger']]}`",
        f"- failed corrective passes (A1): `{[r['ring_key'] for r in failure_audit['failed_corrective_passes_proxy_trigger']]}`",
        "",
        "## Decision",
        f"- reflection_proof_supported: `{proof_pass}`",
        "- criterion: A1 must improve over A0, outperform random-trigger A3, and not have worse worsening profile than A3.",
    ]
    return "\n".join(lines) + "\n"


def _main(args: argparse.Namespace) -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    HELDOUT_ROOT.mkdir(parents=True, exist_ok=True)

    frozen = _load_json(FROZEN_THRESHOLDS)
    heldout_panel = _load_json(HELDOUT_PANEL)
    existing = _load_json(EXISTING_TRACES)
    pv = _load_proxy_validation_module()

    a0_existing = {str(r["ring_key"]): r for r in existing.get("A0", [])}
    a1_existing = {str(r["ring_key"]): r for r in existing.get("A1", [])}

    a0_rows: list[dict[str, Any]] = []
    for row in heldout_panel:
        rk = _ring_key(str(row["tunnel_id"]), int(row["ring_id"]))
        a0 = a0_existing.get(rk)
        if a0 is None or a0.get("failed"):
            a0 = _ensure_reflection_row(
                pv=pv,
                panel_row=row,
                variant="A0_no_reflection",
                reflection=False,
                existing_cache=a0_existing,
            )
        a0_rows.append(a0)

    trigger_reason_a1 = {str(r["ring_key"]): _trigger_reason(r, frozen) for r in a0_rows}
    triggered_a1 = {k: (v != "none") for k, v in trigger_reason_a1.items()}
    a1_budget = int(sum(triggered_a1.values()))

    # Reuse existing A1 reflection rows where available.
    a1_reflections: dict[str, dict[str, Any]] = {}
    for row in heldout_panel:
        rk = _ring_key(str(row["tunnel_id"]), int(row["ring_id"]))
        if not triggered_a1.get(rk, False):
            continue
        existing_row = a1_existing.get(rk)
        if existing_row is not None and not existing_row.get("failed"):
            a1_reflections[rk] = existing_row
            continue
        ran = _ensure_reflection_row(
            pv=pv,
            panel_row=row,
            variant="A1_proxy_reflection",
            reflection=True,
            existing_cache=a1_reflections,
        )
        a1_reflections[rk] = ran

    # A2 always reflect: run reflection for all rings (reuse A1 rows first).
    a2_reflections: dict[str, dict[str, Any]] = {}
    for row in heldout_panel:
        rk = _ring_key(str(row["tunnel_id"]), int(row["ring_id"]))
        if rk in a1_reflections and not a1_reflections[rk].get("failed"):
            a2_reflections[rk] = a1_reflections[rk]
            continue
        a2_reflections[rk] = _ensure_reflection_row(
            pv=pv,
            panel_row=row,
            variant="A2_always_reflect",
            reflection=True,
            existing_cache=a2_reflections,
        )

    # A3 random reflect with same budget as A1.
    rng = np.random.default_rng(int(args.random_seed))
    all_keys = [_ring_key(str(r["tunnel_id"]), int(r["ring_id"])) for r in heldout_panel]
    picked_keys = set(rng.choice(all_keys, size=min(a1_budget, len(all_keys)), replace=False).tolist())
    trigger_reason_a3 = {k: ("random_budget" if k in picked_keys else "none") for k in all_keys}
    triggered_a3 = {k: (k in picked_keys) for k in all_keys}
    a3_reflections: dict[str, dict[str, Any]] = {}
    for row in heldout_panel:
        rk = _ring_key(str(row["tunnel_id"]), int(row["ring_id"]))
        if not triggered_a3[rk]:
            continue
        if rk in a1_reflections and not a1_reflections[rk].get("failed"):
            a3_reflections[rk] = a1_reflections[rk]
            continue
        a3_reflections[rk] = _ensure_reflection_row(
            pv=pv,
            panel_row=row,
            variant="A3_random_reflect",
            reflection=True,
            existing_cache=a3_reflections,
        )

    a1_pairs = _paired_rows(
        variant_name="A1_proxy_reflection",
        a0_rows=a0_rows,
        reflective_rows_by_key=a1_reflections,
        triggered_by_key=triggered_a1,
        trigger_reason_by_key=trigger_reason_a1,
    )
    a2_pairs = _paired_rows(
        variant_name="A2_always_reflect",
        a0_rows=a0_rows,
        reflective_rows_by_key=a2_reflections,
        triggered_by_key={k: True for k in all_keys},
        trigger_reason_by_key={k: "always_reflect" for k in all_keys},
    )
    a3_pairs = _paired_rows(
        variant_name="A3_random_reflect",
        a0_rows=a0_rows,
        reflective_rows_by_key=a3_reflections,
        triggered_by_key=triggered_a3,
        trigger_reason_by_key=trigger_reason_a3,
    )

    all_pairs = a1_pairs + a2_pairs + a3_pairs
    _write_csv(OUT_ROOT / "reflection_proof_pairs.csv", all_pairs)

    stats_by_variant = {
        "A1_proxy_reflection": _variant_stats(a1_pairs),
        "A2_always_reflect": _variant_stats(a2_pairs),
        "A3_random_reflect": _variant_stats(a3_pairs),
    }
    _write_json(OUT_ROOT / "reflection_proof_statistics.json", stats_by_variant)
    _write_json(
        OUT_ROOT / "cluster_bootstrap_ci.json",
        {k: v.get("cluster_bootstrap", {}) for k, v in stats_by_variant.items()},
    )

    control_rows = []
    for name, stats in stats_by_variant.items():
        control_rows.append(
            {
                "variant": name,
                "n_pairs": stats.get("n_pairs"),
                "mean_delta_mIoU": stats.get("mean_delta_mIoU"),
                "mean_delta_OA": stats.get("mean_delta_OA"),
                "worsened_count": stats.get("worsened_count"),
                "trigger_rate": stats.get("trigger_rate"),
                "trigger_precision_on_bad_labels": stats.get("trigger_precision_on_bad_labels"),
                "paired_ttest_p_value_mIoU": stats.get("paired_ttest_p_value_mIoU"),
                "wilcoxon_p_value_mIoU": stats.get("wilcoxon_p_value_mIoU"),
            }
        )
    _write_csv(OUT_ROOT / "reflection_control_comparison.csv", control_rows)

    failure_audit = _build_failure_audit(a1_pairs=a1_pairs, a0_rows=a0_rows)
    _write_json(OUT_ROOT / "reflection_failure_audit.json", failure_audit)

    report = _report(
        frozen=frozen,
        stats_by_variant=stats_by_variant,
        failure_audit=failure_audit,
        random_seed=int(args.random_seed),
        random_budget=a1_budget,
    )
    (OUT_ROOT / "reflection_proof_report.md").write_text(report)

    summary = {
        "timestamp_utc": _now(),
        "heldout_rows": len(heldout_panel),
        "a1_trigger_budget": a1_budget,
        "random_seed": int(args.random_seed),
        "paths": {
            "pairs_csv": str(OUT_ROOT / "reflection_proof_pairs.csv"),
            "comparison_csv": str(OUT_ROOT / "reflection_control_comparison.csv"),
            "stats_json": str(OUT_ROOT / "reflection_proof_statistics.json"),
            "failure_audit_json": str(OUT_ROOT / "reflection_failure_audit.json"),
            "report_md": str(OUT_ROOT / "reflection_proof_report.md"),
        },
    }
    _write_json(OUT_ROOT / "reflection_proof_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--random-seed", type=int, default=73)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_main(parse_args()))
