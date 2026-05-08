#!/usr/bin/env python3
"""Build proxy calibration artifacts for plan steps 3-6.

Outputs are written under:
  logs/proxy_calibration_v1/panel/r0/05_proxy_and_calibration/
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "logs" / "proxy_calibration_v1" / "panel" / "r0" / "05_proxy_and_calibration"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _norm(path_text: str | None) -> str | None:
    if not path_text:
        return None
    return str(Path(path_text).resolve())


def _ring_key(tunnel_id: str, ring_id: int) -> str:
    return f"{tunnel_id}/r{ring_id}"


def _miou_threshold(tunnel_id: str) -> float:
    if tunnel_id == "1-1":
        return 0.60
    if tunnel_id.startswith("3-"):
        return 0.50
    return 0.40


@dataclass
class PreRecord:
    tunnel_id: str
    ring_id: int
    ring_key: str
    source_run: str
    source_type: str
    trial_id: str
    output_dir: str
    target_foreground_recall: float | None
    coverage_factor: float | None
    empty_factor: float | None
    shape_factor: float | None
    guarded_score: float | None
    summary_json: str | None

    @property
    def s_depth(self) -> float | None:
        if self.coverage_factor is None or self.empty_factor is None:
            return None
        return float(self.coverage_factor * self.empty_factor)


def _collect_preprocessing_records() -> tuple[dict[str, PreRecord], dict[str, PreRecord]]:
    by_output_dir: dict[str, PreRecord] = {}
    selected_by_ring: dict[str, PreRecord] = {}

    summary_roots = [
        REPO_ROOT / "logs" / "preprocessing_bo_v2" / "summary_preprocessing_bo_v2.json",
    ]
    for summary_path in summary_roots:
        if not summary_path.exists():
            continue
        payload = _load_json(summary_path)
        for row in payload.get("rings", []):
            tunnel_id = str(row["tunnel_id"])
            ring_id = int(row["ring_id"])
            rk = _ring_key(tunnel_id, ring_id)
            selected_output_dir = _norm(row.get("selected_output_dir"))
            if selected_output_dir:
                r = PreRecord(
                    tunnel_id=tunnel_id,
                    ring_id=ring_id,
                    ring_key=rk,
                    source_run=str(payload.get("run_id", "unknown")),
                    source_type="selected",
                    trial_id="selected",
                    output_dir=selected_output_dir,
                    target_foreground_recall=None,
                    coverage_factor=None,
                    empty_factor=None,
                    shape_factor=None,
                    guarded_score=_safe_float(row.get("best_guarded_score")),
                    summary_json=str(row.get("summary_json")) if row.get("summary_json") else None,
                )
                selected_by_ring[rk] = r
                by_output_dir[selected_output_dir] = r

    # Detailed preprocessing metrics: add baseline/best/trials with explicit factors.
    summary_globs = [
        REPO_ROOT / "logs" / "preprocessing_bo_v2" / "metrics",
        REPO_ROOT / "logs" / "preprocessing_bo_v2_fix" / "metrics",
    ]
    for root in summary_globs:
        if not root.exists():
            continue
        for summary_path in root.glob("**/summary.json"):
            payload = _load_json(summary_path)
            tunnel_id = str(payload.get("tunnel_id", ""))
            ring_id = int(payload.get("ring_id", -1))
            if not tunnel_id or ring_id < 0:
                continue
            rk = _ring_key(tunnel_id, ring_id)

            # Baseline record.
            baseline = payload.get("baseline", {})
            baseline_out = _norm(baseline.get("output_dir"))
            if baseline_out:
                rec = PreRecord(
                    tunnel_id=tunnel_id,
                    ring_id=ring_id,
                    ring_key=rk,
                    source_run=str(payload.get("run_id", "unknown")),
                    source_type="baseline",
                    trial_id="baseline",
                    output_dir=baseline_out,
                    target_foreground_recall=_safe_float(baseline.get("target_foreground_recall")),
                    coverage_factor=_safe_float(baseline.get("coverage_factor")),
                    empty_factor=_safe_float(baseline.get("empty_factor")),
                    shape_factor=_safe_float(baseline.get("shape_factor", 1.0)),
                    guarded_score=_safe_float(baseline.get("guarded_score")),
                    summary_json=str(summary_path.relative_to(REPO_ROOT)),
                )
                by_output_dir[rec.output_dir] = rec

            # Best record.
            best_metrics = payload.get("best", {}).get("metrics", {})
            best_out = _norm(best_metrics.get("output_dir"))
            if best_out:
                rec = PreRecord(
                    tunnel_id=tunnel_id,
                    ring_id=ring_id,
                    ring_key=rk,
                    source_run=str(payload.get("run_id", "unknown")),
                    source_type="best",
                    trial_id="best",
                    output_dir=best_out,
                    target_foreground_recall=_safe_float(best_metrics.get("target_foreground_recall")),
                    coverage_factor=_safe_float(best_metrics.get("coverage_factor")),
                    empty_factor=_safe_float(best_metrics.get("empty_factor")),
                    shape_factor=_safe_float(best_metrics.get("shape_factor", 1.0)),
                    guarded_score=_safe_float(best_metrics.get("guarded_score")),
                    summary_json=str(summary_path.relative_to(REPO_ROOT)),
                )
                by_output_dir[rec.output_dir] = rec
                # Keep selected-by-ring richer if it matches selected output directory.
                selected = selected_by_ring.get(rk)
                if selected and selected.output_dir == rec.output_dir:
                    selected_by_ring[rk] = rec

            # Trial records.
            for trial_path in sorted(summary_path.parent.glob("trial_*.json")):
                t = _load_json(trial_path)
                metrics = t.get("metrics", {})
                out = _norm(metrics.get("output_dir"))
                if not out:
                    continue
                trial_id = int(t.get("trial_id", 0))
                rec = PreRecord(
                    tunnel_id=tunnel_id,
                    ring_id=ring_id,
                    ring_key=rk,
                    source_run=str(payload.get("run_id", "unknown")),
                    source_type="trial",
                    trial_id=f"trial_{trial_id:03d}",
                    output_dir=out,
                    target_foreground_recall=_safe_float(metrics.get("target_foreground_recall")),
                    coverage_factor=_safe_float(metrics.get("coverage_factor")),
                    empty_factor=_safe_float(metrics.get("empty_factor")),
                    shape_factor=_safe_float(metrics.get("shape_factor", 1.0)),
                    guarded_score=_safe_float(metrics.get("guarded_score")),
                    summary_json=str(trial_path.relative_to(REPO_ROOT)),
                )
                by_output_dir[rec.output_dir] = rec

    return by_output_dir, selected_by_ring


def _collect_detection_rows(pre_by_output_dir: dict[str, PreRecord], pre_selected_by_ring: dict[str, PreRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    ring_meta_map: dict[str, dict[str, Any]] = {}
    final_summary = REPO_ROOT / "logs" / "detection_boundary_bo_v1" / "summary_detection_boundary_bo_v1.json"
    if final_summary.exists():
        payload = _load_json(final_summary)
        for row in payload.get("rings", []):
            rk = _ring_key(str(row["tunnel_id"]), int(row["ring_id"]))
            ring_meta_map[rk] = {
                "intrinsic_only": row.get("intrinsic_only"),
                "calibration_source": row.get("calibration_source"),
                "selected_source": row.get("selected_source"),
                "deployable_valid": row.get("deployable_valid"),
            }

    detection_summary_paths: list[Path] = []
    detection_summary_paths.extend((REPO_ROOT / "logs" / "detection_boundary_bo_v1" / "metrics").glob("**/summary.json"))
    detection_summary_paths.extend((REPO_ROOT / "logs" / "detection_boundary_structural_panel_v3" / "metrics").glob("**/summary.json"))
    detection_summary_paths.extend((REPO_ROOT / "logs" / "detection_boundary_structural_panel_iter").glob("metrics_iter*/**/summary.json"))

    for summary_path in sorted(set(detection_summary_paths)):
        payload = _load_json(summary_path)
        tunnel_id = str(payload.get("tunnel_id", ""))
        ring_id = int(payload.get("ring_id", -1))
        if not tunnel_id or ring_id < 0:
            continue
        rk = _ring_key(tunnel_id, ring_id)
        run_id = str(payload.get("run_id", "unknown"))
        preprocessing_source_dir = _norm(payload.get("preprocessing_source_dir"))
        ring_meta = ring_meta_map.get(rk, {})

        # Selected/best row.
        selection = payload.get("selection", {})
        best = payload.get("best", {})
        best_metrics = best.get("metrics", {})
        selected_output_dir = _norm(selection.get("selected_output_dir")) or _norm(best_metrics.get("output_dir"))
        selected_source = str(selection.get("selected_source", ring_meta.get("selected_source", "unknown")))
        final_miou = _safe_float(best.get("segmentation_mIoU"))
        if final_miou is None:
            final_miou = _safe_float(best_metrics.get("segmentation_mIoU"))

        selected_row = {
            "row_id": f"{run_id}:{rk}:selected",
            "tunnel_id": tunnel_id,
            "ring_id": ring_id,
            "ring_key": rk,
            "source_stage": "detection_boundary",
            "source_run": run_id,
            "trial_id": "selected",
            "selected_source": selected_source,
            "preprocessing_output_dir": preprocessing_source_dir,
            "detection_output_dir": selected_output_dir,
            "summary_json": str(summary_path.relative_to(REPO_ROOT)),
            "BoundaryF1": _safe_float(best.get("BoundaryF1")),
            "S_continuity": _safe_float(best.get("S_continuity")),
            "S_K": _safe_float(best.get("S_K")),
            "S_spacing": _safe_float(best.get("S_spacing")),
            "S_layout_coverage": _safe_float(best.get("S_layout_coverage")),
            "S_boundary": _safe_float(best.get("S_boundary")),
            "J_det": _safe_float(best.get("J_det")),
            "final_mIoU": final_miou,
            "deployable_valid": bool(ring_meta.get("deployable_valid", best.get("meaningful_layout_passed", False))),
            "meaningful_layout_score": _safe_float(best.get("meaningful_layout_score")),
            "intrinsic_only": ring_meta.get("intrinsic_only", payload.get("intrinsic_only")),
            "calibration_source": ring_meta.get("calibration_source", payload.get("calibration_source")),
            "row_kind": "selected",
        }
        rows.append(selected_row)

        # Trial rows.
        for trial_path in sorted(summary_path.parent.glob("trial_*.json")):
            t = _load_json(trial_path)
            metrics = t.get("metrics", {})
            trial_id = int(t.get("trial_id", 0))
            tr = {
                "row_id": f"{run_id}:{rk}:trial_{trial_id:03d}",
                "tunnel_id": tunnel_id,
                "ring_id": ring_id,
                "ring_key": rk,
                "source_stage": "detection_boundary",
                "source_run": run_id,
                "trial_id": f"trial_{trial_id:03d}",
                "selected_source": "trial",
                "preprocessing_output_dir": preprocessing_source_dir,
                "detection_output_dir": _norm(metrics.get("output_dir")),
                "summary_json": str(trial_path.relative_to(REPO_ROOT)),
                "BoundaryF1": _safe_float(t.get("BoundaryF1")),
                "S_continuity": _safe_float(t.get("S_continuity")),
                "S_K": _safe_float(t.get("S_K")),
                "S_spacing": _safe_float(t.get("S_spacing")),
                "S_layout_coverage": _safe_float(t.get("S_layout_coverage")),
                "S_boundary": _safe_float(t.get("S_boundary")),
                "J_det": _safe_float(t.get("J_det")),
                "final_mIoU": _safe_float(t.get("segmentation_mIoU")),
                "deployable_valid": bool(ring_meta.get("deployable_valid", t.get("meaningful_layout_passed", False))),
                "meaningful_layout_score": _safe_float(t.get("meaningful_layout_score")),
                "intrinsic_only": ring_meta.get("intrinsic_only", payload.get("intrinsic_only")),
                "calibration_source": ring_meta.get("calibration_source", payload.get("calibration_source")),
                "row_kind": "trial",
            }
            rows.append(tr)

    # Attach S_depth/preprocessing metrics.
    for row in rows:
        pre: PreRecord | None = None
        pre_out = _norm(row.get("preprocessing_output_dir"))
        if pre_out:
            pre = pre_by_output_dir.get(pre_out)
        if pre is None:
            pre = pre_selected_by_ring.get(row["ring_key"])
        if pre is not None:
            row["target_foreground_recall"] = pre.target_foreground_recall
            row["coverage_factor"] = pre.coverage_factor
            row["empty_factor"] = pre.empty_factor
            row["shape_factor"] = pre.shape_factor
            row["S_depth"] = pre.s_depth
            row["guarded_score"] = pre.guarded_score
            row["pre_summary_json"] = pre.summary_json
            row["pre_source_type"] = pre.source_type
            row["pre_trial_id"] = pre.trial_id
            if not row.get("preprocessing_output_dir"):
                row["preprocessing_output_dir"] = pre.output_dir
        else:
            row["target_foreground_recall"] = None
            row["coverage_factor"] = None
            row["empty_factor"] = None
            row["shape_factor"] = None
            row["S_depth"] = None
            row["guarded_score"] = None
            row["pre_summary_json"] = None
            row["pre_source_type"] = None
            row["pre_trial_id"] = None

        # Labels.
        threshold = _miou_threshold(str(row["tunnel_id"]))
        row["mIoU_threshold"] = threshold
        miou = _safe_float(row.get("final_mIoU"))
        row["is_low_mIoU"] = None if miou is None else bool(miou < threshold)
        if miou is None:
            row["label_status"] = "missing_mIoU"
        elif row.get("S_depth") is None:
            row["label_status"] = "missing_S_depth"
        elif row.get("S_boundary") is None:
            row["label_status"] = "missing_S_boundary"
        else:
            row["label_status"] = "ok"

    # Deduplicate by row_id while preserving last (latest source path sort may include overrides).
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup[row["row_id"]] = row
    return list(dedup.values())


def _spearman_with_bootstrap(
    x: np.ndarray, y: np.ndarray, n_boot: int = 1000
) -> dict[str, Any]:
    if len(x) < 3:
        return {"n": int(len(x)), "rho": None, "p_value": None, "ci95": None}
    rho, p = spearmanr(x, y)
    if np.isnan(rho):
        return {"n": int(len(x)), "rho": None, "p_value": None, "ci95": None}
    out = {"n": int(len(x)), "rho": float(rho), "p_value": float(p), "ci95": None}
    if len(x) >= 20:
        rng = np.random.default_rng(42)
        boot = []
        idx = np.arange(len(x))
        for _ in range(n_boot):
            sample = rng.choice(idx, size=len(idx), replace=True)
            brho, _ = spearmanr(x[sample], y[sample])
            if not np.isnan(brho):
                boot.append(float(brho))
        if boot:
            out["ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return out


def _compute_spearman(df: pd.DataFrame) -> dict[str, Any]:
    subsets = {
        "all_labelled": df,
        "selected_only": df[df["row_kind"] == "selected"],
        "trial_only": df[df["row_kind"] == "trial"],
        "intrinsic_only": df[df["intrinsic_only"] == True],  # noqa: E712
        "gt_calibrated": df[df["calibration_source"].notna()],
    }
    result: dict[str, Any] = {}
    for name, sub in subsets.items():
        sub = sub.dropna(subset=["S_depth", "S_boundary", "final_mIoU"])
        x_depth = sub["S_depth"].to_numpy(dtype=float)
        x_boundary = sub["S_boundary"].to_numpy(dtype=float)
        y = sub["final_mIoU"].to_numpy(dtype=float)
        result[name] = {
            "S_depth_vs_mIoU": _spearman_with_bootstrap(x_depth, y),
            "S_boundary_vs_mIoU": _spearman_with_bootstrap(x_boundary, y),
        }
    return result


def _rule_metrics(df: pd.DataFrame, trigger: pd.Series) -> dict[str, Any]:
    is_bad = df["is_low_mIoU"].astype(bool)
    total_bad = int(is_bad.sum())
    triggered = trigger.astype(bool)
    tp_bad = int((triggered & is_bad).sum())
    fn_bad = int((~triggered & is_bad).sum())
    total_trigger = int(triggered.sum())
    fp = int((triggered & ~is_bad).sum())
    accepted = df.loc[~triggered, "final_mIoU"]
    return {
        "n_rows": int(len(df)),
        "n_bad": total_bad,
        "n_triggered": total_trigger,
        "tp_bad": tp_bad,
        "fp": fp,
        "fn_bad": fn_bad,
        "bad_case_recall": None if total_bad == 0 else float(tp_bad / total_bad),
        "trigger_precision": None if total_trigger == 0 else float(tp_bad / total_trigger),
        "false_negative_rate": None if total_bad == 0 else float(fn_bad / total_bad),
        "accepted_mIoU_mean": None if accepted.empty else float(accepted.mean()),
        "accepted_mIoU_median": None if accepted.empty else float(accepted.median()),
    }


def _search_thresholds(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    # Candidate grids over empirical quantiles.
    depth_vals = np.sort(df["S_depth"].dropna().unique())
    boundary_vals = np.sort(df["S_boundary"].dropna().unique())
    depth_grid = np.unique(np.quantile(depth_vals, np.linspace(0.05, 0.95, 30)))
    boundary_grid = np.unique(np.quantile(boundary_vals, np.linspace(0.05, 0.95, 30)))

    rows = []
    for td in depth_grid:
        for tb in boundary_grid:
            depth_trigger = df["S_depth"] < float(td)
            boundary_trigger = df["S_boundary"] < float(tb)
            union_trigger = depth_trigger | boundary_trigger
            and_trigger = depth_trigger & boundary_trigger
            for rule_name, trigger in [
                ("depth_only", depth_trigger),
                ("boundary_only", boundary_trigger),
                ("union", union_trigger),
                ("and", and_trigger),
            ]:
                m = _rule_metrics(df, trigger)
                rows.append(
                    {
                        "T_depth": float(td),
                        "T_boundary": float(tb),
                        "rule": rule_name,
                        **m,
                    }
                )

    grid = pd.DataFrame(rows)
    if grid.empty:
        raise RuntimeError("Threshold grid is empty; cannot select thresholds.")

    def _pick(rule: str) -> dict[str, Any]:
        sub = grid[grid["rule"] == rule].copy()
        sub["bad_case_recall"] = sub["bad_case_recall"].fillna(-1.0)
        sub["trigger_precision"] = sub["trigger_precision"].fillna(-1.0)
        sub["false_negative_rate"] = sub["false_negative_rate"].fillna(1.0)
        sub["accepted_mIoU_mean"] = sub["accepted_mIoU_mean"].fillna(-1.0)
        # Maximize recall first, then precision, then lower false negatives, then higher accepted mIoU.
        sub = sub.sort_values(
            by=["bad_case_recall", "trigger_precision", "false_negative_rate", "accepted_mIoU_mean"],
            ascending=[False, False, True, False],
        )
        return sub.iloc[0].to_dict()

    chosen = {
        "primary_rule": "union",
        "selected": _pick("union"),
        "alternatives": {
            "depth_only": _pick("depth_only"),
            "boundary_only": _pick("boundary_only"),
            "and": _pick("and"),
        },
    }
    return chosen, grid


def _write_markdown(
    out_path: Path,
    spearman_eval: dict[str, Any],
    thresholds: dict[str, Any],
    proxy_eval: dict[str, Any],
    df_all: pd.DataFrame,
    df_labelled: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines += ["# Proxy Calibration Report", ""]
    lines += [
        "- dataset scope: selected outputs + trial outputs",
        f"- rows (raw): **{len(df_all)}**",
        f"- rows (labelled): **{len(df_labelled)}**",
        "- bad-case thresholds: `1-1 < 0.60`, `3-* < 0.50`, `4/5-* < 0.40`",
        "",
        "## Spearman",
        "",
        "| Subset | n | rho(S_depth,mIoU) | rho(S_boundary,mIoU) |",
        "|---|---:|---:|---:|",
    ]
    for subset, result in spearman_eval.items():
        d = result["S_depth_vs_mIoU"]
        b = result["S_boundary_vs_mIoU"]
        lines.append(
            f"| `{subset}` | {d['n']} | "
            f"{'' if d['rho'] is None else f'{d['rho']:.4f}'} | "
            f"{'' if b['rho'] is None else f'{b['rho']:.4f}'} |"
        )

    sel = thresholds["selected"]
    lines += [
        "",
        "## Selected Thresholds",
        "",
        f"- primary rule: `{thresholds['primary_rule']}`",
        f"- `T_depth`: **{sel['T_depth']:.6f}**",
        f"- `T_boundary`: **{sel['T_boundary']:.6f}**",
        f"- bad-case recall: **{sel['bad_case_recall']:.4f}**",
        f"- trigger precision: **{sel['trigger_precision']:.4f}**",
        f"- false-negative rate: **{sel['false_negative_rate']:.4f}**",
        "",
        "## Failure Capture",
        "",
        f"- n_bad: **{proxy_eval['n_bad']}**",
        f"- n_triggered: **{proxy_eval['n_triggered']}**",
        f"- bad-case recall: **{proxy_eval['bad_case_recall']:.4f}**",
        f"- trigger precision: **{proxy_eval['trigger_precision']:.4f}**",
        f"- false-negative rate: **{proxy_eval['false_negative_rate']:.4f}**",
        "",
        "## Caveat",
        "",
        "Results include both intrinsic-only rows and GT-calibrated design-time template rows. These groups are reported separately in the JSON outputs and should not be conflated for deployment confidence.",
        "",
    ]
    out_path.write_text("\n".join(lines))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pre_by_output_dir, pre_selected_by_ring = _collect_preprocessing_records()
    all_rows = _collect_detection_rows(pre_by_output_dir, pre_selected_by_ring)
    df_all = pd.DataFrame(all_rows).sort_values(["tunnel_id", "ring_id", "source_run", "trial_id"]).reset_index(drop=True)

    # Labelled rows for stats and threshold fitting.
    df_labelled = df_all[df_all["label_status"] == "ok"].copy()

    # Save datasets.
    raw_csv = OUT_DIR / "proxy_dataset_raw.csv"
    labelled_csv = OUT_DIR / "proxy_dataset_labelled.csv"
    df_all.to_csv(raw_csv, index=False)
    df_labelled.to_csv(labelled_csv, index=False)

    dataset_summary = {
        "raw_rows": int(len(df_all)),
        "labelled_rows": int(len(df_labelled)),
        "missing_mIoU_rows": int((df_all["label_status"] == "missing_mIoU").sum()),
        "missing_S_depth_rows": int((df_all["label_status"] == "missing_S_depth").sum()),
        "missing_S_boundary_rows": int((df_all["label_status"] == "missing_S_boundary").sum()),
        "row_kind_counts": df_all["row_kind"].value_counts(dropna=False).to_dict(),
        "source_run_counts": df_all["source_run"].value_counts(dropna=False).to_dict(),
        "ring_counts": df_all["ring_key"].value_counts(dropna=False).to_dict(),
        "deployable_valid_counts": df_labelled["deployable_valid"].value_counts(dropna=False).to_dict(),
    }
    (OUT_DIR / "proxy_dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2) + "\n")

    # Step 4: Spearman.
    spearman_eval = _compute_spearman(df_labelled)
    (OUT_DIR / "spearman_proxy_eval.json").write_text(json.dumps(spearman_eval, indent=2) + "\n")

    # Step 5: threshold search.
    thresholds, threshold_grid = _search_thresholds(df_labelled)
    threshold_grid.to_csv(OUT_DIR / "threshold_selection_grid.csv", index=False)
    (OUT_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")

    # Step 6: evaluate selected trigger and diagnostic sets.
    sel = thresholds["selected"]
    trigger = (df_labelled["S_depth"] < float(sel["T_depth"])) | (df_labelled["S_boundary"] < float(sel["T_boundary"]))
    proxy_eval = _rule_metrics(df_labelled, trigger)
    proxy_eval["rule"] = thresholds["primary_rule"]
    proxy_eval["T_depth"] = float(sel["T_depth"])
    proxy_eval["T_boundary"] = float(sel["T_boundary"])
    (OUT_DIR / "proxy_eval.json").write_text(json.dumps(proxy_eval, indent=2) + "\n")

    false_neg = df_labelled[(~trigger) & (df_labelled["is_low_mIoU"] == True)].copy()  # noqa: E712
    false_pos = df_labelled[(trigger) & (df_labelled["is_low_mIoU"] == False)].copy()  # noqa: E712
    false_neg.to_csv(OUT_DIR / "false_negative_cases.csv", index=False)
    false_pos.to_csv(OUT_DIR / "false_positive_cases.csv", index=False)

    # Human-readable markdown outputs.
    _write_markdown(
        out_path=OUT_DIR / "proxy_calibration_report.md",
        spearman_eval=spearman_eval,
        thresholds=thresholds,
        proxy_eval=proxy_eval,
        df_all=df_all,
        df_labelled=df_labelled,
    )
    # Alias files requested in step descriptions.
    (OUT_DIR / "proxy_eval.md").write_text((OUT_DIR / "proxy_calibration_report.md").read_text())
    (OUT_DIR / "proxy_thresholds.md").write_text(json.dumps(thresholds, indent=2) + "\n")
    (OUT_DIR / "spearman_proxy_eval.md").write_text(json.dumps(spearman_eval, indent=2) + "\n")

    print(f"Wrote proxy calibration artifacts to: {OUT_DIR}")
    print(f"Rows raw={len(df_all)} labelled={len(df_labelled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
