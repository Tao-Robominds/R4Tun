#!/usr/bin/env python3
"""Guarded BO for official fixed B+C+D preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from skopt import gp_minimize
from skopt.space import Integer, Real

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = REPO_ROOT / "agents" / "1_preprocessing"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from bo.preprocessing_iou_metrics import (  # noqa: E402
    compute_foreground_mask_iou_metrics,
    compute_target_guarded_metrics,
)
from context_preprocessing import run_context_trial  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _build_space(base: Dict[str, Any]) -> Tuple[List[str], List[Any], List[float]]:
    keys = [
        "radius_min",
        "radius_max",
        "gradient_threshold",
        "smoothing_offset",
        "curvature_neighbors",
        "interpolation_window",
        "target_distance_1",
        "target_distance_2",
        "target_distance_3",
        "outlier_interpolation_radius",
        "outlier_num_interpolations",
        "outlier_depth_map_window",
        "outlier_neighbors",
    ]
    td = list(base.get("target_distances", [0.06, 0.03, 0.015]))
    while len(td) < 3:
        td.append(0.015)
    x0 = [
        float(base.get("radius_min", 2.3)),
        float(base.get("radius_max", 3.0)),
        float(base.get("gradient_threshold", 0.15)),
        float(base.get("smoothing_offset", -0.002)),
        int(base.get("curvature_neighbors", base.get("num_neighbors", 20))),
        int(base.get("interpolation_window", 9)),
        float(td[0]),
        float(td[1]),
        float(td[2]),
        float(base.get("outlier_interpolation_radius", base.get("inter_radius", 0.03))),
        int(base.get("outlier_num_interpolations", base.get("num_interpolations", 2))),
        int(base.get("outlier_depth_map_window", 1)),
        int(base.get("outlier_neighbors", 20)),
    ]
    dims = [
        Real(1.8, 3.8, name="radius_min"),
        Real(2.0, 4.2, name="radius_max"),
        Real(0.03, 0.40, name="gradient_threshold"),
        Real(-0.02, 0.02, name="smoothing_offset"),
        Integer(8, 40, name="curvature_neighbors"),
        Integer(1, 15, name="interpolation_window"),
        Real(0.03, 0.12, name="target_distance_1"),
        Real(0.015, 0.06, name="target_distance_2"),
        Real(0.008, 0.04, name="target_distance_3"),
        Real(0.01, 0.08, name="outlier_interpolation_radius"),
        Integer(1, 5, name="outlier_num_interpolations"),
        Integer(1, 9, name="outlier_depth_map_window"),
        Integer(8, 40, name="outlier_neighbors"),
    ]
    x0 = [
        _clip(float(x0[i]), float(dims[i].low), float(dims[i].high)) if isinstance(dims[i], Real)
        else int(_clip(float(x0[i]), float(dims[i].low), float(dims[i].high)))
        for i in range(len(dims))
    ]
    return keys, dims, x0


def _candidate_from_x(base: Dict[str, Any], keys: List[str], x: List[float]) -> Dict[str, Any]:
    c = dict(base)
    m = {k: v for k, v in zip(keys, x)}
    c["radius_min"] = float(m["radius_min"])
    c["radius_max"] = float(max(m["radius_max"], c["radius_min"] + 0.05))
    c["gradient_threshold"] = float(m["gradient_threshold"])
    c["smoothing_offset"] = float(m["smoothing_offset"])
    c["curvature_neighbors"] = int(round(m["curvature_neighbors"]))
    c["num_neighbors"] = int(round(m["curvature_neighbors"]))
    c["interpolation_window"] = int(round(m["interpolation_window"]))
    c["target_distances"] = sorted(
        [float(m["target_distance_1"]), float(m["target_distance_2"]), float(m["target_distance_3"])],
        reverse=True,
    )
    c["outlier_interpolation_radius"] = float(m["outlier_interpolation_radius"])
    c["inter_radius"] = float(m["outlier_interpolation_radius"])
    c["outlier_num_interpolations"] = int(round(m["outlier_num_interpolations"]))
    c["num_interpolations"] = int(round(m["outlier_num_interpolations"]))
    c["outlier_depth_map_window"] = int(round(m["outlier_depth_map_window"]))
    c["outlier_neighbors"] = int(round(m["outlier_neighbors"]))
    return c


def _run_trial_once(
    *,
    tunnel_id: str,
    ring_id: int,
    context_radius: int,
    output_root: Path,
    reference_base_dir: str,
    params: Dict[str, Any],
    baseline_valid_ratio_ref: float,
    min_coverage_ratio: float,
    max_empty_row_band_ratio: float,
) -> Dict[str, Any]:
    out_dir = run_context_trial(
        tunnel_id=tunnel_id,
        ring_id=ring_id,
        context_radius=context_radius,
        output_root=output_root,
        reference_base_dir=reference_base_dir,
        params_override=params,
    )
    metrics = compute_target_guarded_metrics(
        out_dir,
        baseline_valid_ratio=baseline_valid_ratio_ref,
        min_coverage_ratio=min_coverage_ratio,
        max_empty_row_band_ratio=max_empty_row_band_ratio,
    )
    metrics["iou_diagnostic"] = compute_foreground_mask_iou_metrics(out_dir)
    metrics["output_dir"] = str(out_dir)
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tunnel-id", required=True)
    p.add_argument("--ring-id", required=True, type=int)
    p.add_argument("--context-radius", type=int, default=1)
    p.add_argument("--base-dir", default="data/bo/preprocessing")
    p.add_argument("--baseline-dir", default="logs/context_preprocessing_v1")
    p.add_argument("--reference-base-dir", default="data/ablation/baseline")
    p.add_argument("--n-calls", type=int, default=8)
    p.add_argument("--n-initial-points", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default="context_bcd_guarded_v1")
    p.add_argument("--logs-root", default="logs/preprocessing_context_bo")
    p.add_argument("--min-coverage-ratio", type=float, default=0.70)
    p.add_argument("--max-empty-row-band-ratio", type=float, default=0.45)
    args = p.parse_args()

    tunnel_id = str(args.tunnel_id)
    ring_id = int(args.ring_id)
    ring_key = f"r{ring_id}"
    base_dir = (REPO_ROOT / args.base_dir).resolve()
    baseline_dir = (REPO_ROOT / args.baseline_dir).resolve()
    baseline_ring_dir = baseline_dir / tunnel_id / ring_key
    if not baseline_ring_dir.exists():
        raise FileNotFoundError(f"Baseline ring directory not found: {baseline_ring_dir}")

    # Trial dirs are isolated: data/bo/preprocessing/<tunnel>/r<ring>/trial_###
    ring_trials_root = base_dir / tunnel_id / ring_key
    ring_trials_root.mkdir(parents=True, exist_ok=True)

    params_path = PREPROCESSING_DIR / "parameters" / tunnel_id / ring_key / "parameters_preprocessing.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing preprocessing parameters: {params_path}")
    base_params = _load_json(params_path)

    logs_dir = (REPO_ROOT / args.logs_root / args.run_id / tunnel_id / ring_key).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    keys, dims, x0 = _build_space(base_params)
    baseline_ref = compute_target_guarded_metrics(
        baseline_ring_dir,
        baseline_valid_ratio=None,
        min_coverage_ratio=float(args.min_coverage_ratio),
        max_empty_row_band_ratio=float(args.max_empty_row_band_ratio),
    )
    baseline_valid_ratio_ref = float(baseline_ref["valid_ratio"])

    # BO baseline run (candidate params = base params), isolated directory.
    t0 = time.time()
    baseline_metrics = _run_trial_once(
        tunnel_id=tunnel_id,
        ring_id=ring_id,
        context_radius=int(args.context_radius),
        output_root=ring_trials_root / "baseline",
        reference_base_dir=str(args.reference_base_dir),
        params=base_params,
        baseline_valid_ratio_ref=baseline_valid_ratio_ref,
        min_coverage_ratio=float(args.min_coverage_ratio),
        max_empty_row_band_ratio=float(args.max_empty_row_band_ratio),
    )
    baseline_metrics["elapsed_sec"] = round(time.time() - t0, 3)
    baseline_metrics["objective"] = -float(baseline_metrics["guarded_score"])

    trial_rows: List[Dict[str, Any]] = []

    def evaluate(x: List[float], source: str) -> float:
        candidate = _candidate_from_x(base_params, keys, x)
        trial_id = len(trial_rows) + 1
        t0 = time.time()
        error: str | None = None
        try:
            metrics = _run_trial_once(
                tunnel_id=tunnel_id,
                ring_id=ring_id,
                context_radius=int(args.context_radius),
                output_root=ring_trials_root / f"trial_{trial_id:03d}",
                reference_base_dir=str(args.reference_base_dir),
                params=candidate,
                baseline_valid_ratio_ref=baseline_valid_ratio_ref,
                min_coverage_ratio=float(args.min_coverage_ratio),
                max_empty_row_band_ratio=float(args.max_empty_row_band_ratio),
            )
            obj = -float(metrics["guarded_score"])
        except Exception as e:  # noqa: BLE001
            error = repr(e)
            metrics = {
                "guarded_score": 0.0,
                "target_foreground_recall": 0.0,
                "foreground_mask_iou": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "empty_row_band_ratio": 1.0,
                "largest_empty_row_band": 0,
                "valid_ratio": 0.0,
                "gt_foreground_ratio": 0.0,
                "coverage_ok": False,
                "empty_band_ok": False,
                "coverage_factor": 0.0,
                "empty_factor": 0.0,
                "depth_shape_h": 0,
                "depth_shape_w": 0,
                "iou_diagnostic": {
                    "foreground_mask_iou": 0.0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "valid_ratio": 0.0,
                    "gt_foreground_ratio": 0.0,
                    "depth_shape_h": 0,
                    "depth_shape_w": 0,
                },
                "output_dir": str(ring_trials_root / f"trial_{trial_id:03d}" / tunnel_id / ring_key),
            }
            obj = 1.0
        elapsed = time.time() - t0
        row = {
            "trial_id": trial_id,
            "source": source,
            "objective": obj,
            "guarded_score": float(metrics["guarded_score"]),
            "target_foreground_recall": float(metrics["target_foreground_recall"]),
            "foreground_mask_iou": float(metrics["foreground_mask_iou"]),
            "metrics": metrics,
            "params": candidate,
            "error": error,
            "elapsed_sec": round(elapsed, 3),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        trial_rows.append(row)
        _write_json(logs_dir / f"trial_{trial_id:03d}.json", row)
        print(
            f"[trial {trial_id:03d}] score={row['guarded_score']:.5f} "
            f"recall={row['target_foreground_recall']:.5f} iou_diag={row['foreground_mask_iou']:.5f} "
            f"obj={row['objective']:.5f} elapsed={row['elapsed_sec']:.1f}s"
        )
        return obj

    result = gp_minimize(
        func=lambda x: evaluate(x, source="bo"),
        dimensions=dims,
        n_calls=int(args.n_calls),
        n_initial_points=int(args.n_initial_points),
        x0=x0,
        random_state=int(args.seed),
        acq_func="EI",
    )

    best_params = _candidate_from_x(base_params, keys, list(result.x))
    best_metrics = _run_trial_once(
        tunnel_id=tunnel_id,
        ring_id=ring_id,
        context_radius=int(args.context_radius),
        output_root=ring_trials_root / "best",
        reference_base_dir=str(args.reference_base_dir),
        params=best_params,
        baseline_valid_ratio_ref=baseline_valid_ratio_ref,
        min_coverage_ratio=float(args.min_coverage_ratio),
        max_empty_row_band_ratio=float(args.max_empty_row_band_ratio),
    )

    improved = float(best_metrics["guarded_score"]) > float(baseline_metrics["guarded_score"])
    selected_source = "bo_best" if improved else "fixed_baseline"
    selected_output_dir = (
        str(ring_trials_root / "best" / tunnel_id / ring_key)
        if improved
        else str(baseline_ring_dir)
    )

    summary = {
        "run_id": args.run_id,
        "tunnel_id": tunnel_id,
        "ring_id": ring_id,
        "ring_key": ring_key,
        "base_dir": str(base_dir),
        "baseline_dir": str(baseline_dir),
        "objective": "maximize guarded_score = target_foreground_recall * coverage_guard * empty_band_guard",
        "guardrails": {
            "min_coverage_ratio": float(args.min_coverage_ratio),
            "max_empty_row_band_ratio": float(args.max_empty_row_band_ratio),
            "baseline_valid_ratio_reference": baseline_valid_ratio_ref,
        },
        "baseline": baseline_metrics,
        "best": {
            "guarded_score": float(best_metrics["guarded_score"]),
            "target_foreground_recall": float(best_metrics["target_foreground_recall"]),
            "foreground_mask_iou_diagnostic": float(best_metrics["foreground_mask_iou"]),
            "metrics": best_metrics,
            "params": best_params,
        },
        "selection": {
            "improved": improved,
            "selected_source": selected_source,
            "selected_output_dir": selected_output_dir,
        },
        "delta_guarded_score": float(best_metrics["guarded_score"]) - float(baseline_metrics["guarded_score"]),
        "delta_iou_diagnostic": float(best_metrics["foreground_mask_iou"]) - float(
            baseline_metrics["foreground_mask_iou"]
        ),
        "n_trials": len(trial_rows),
        "trials_json": str((logs_dir / "trial_001.json").relative_to(REPO_ROOT)) if trial_rows else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(logs_dir / "summary.json", summary)

    md_lines = [
        f"# Preprocessing Guarded BO — {tunnel_id}/{ring_key}",
        "",
        f"- objective: `guarded_score`",
        f"- baseline_guarded_score: **{baseline_metrics['guarded_score']:.6f}**",
        f"- best_guarded_score: **{best_metrics['guarded_score']:.6f}**",
        f"- delta_guarded_score: **{summary['delta_guarded_score']:+.6f}**",
        f"- selection: **{selected_source}**",
        f"- selected_output_dir: `{selected_output_dir}`",
        f"- n_trials: **{len(trial_rows)}**",
        "",
        "## Diagnostics (best)",
        f"- target_foreground_recall: {best_metrics['target_foreground_recall']:.6f}",
        f"- precision: {best_metrics['precision']:.6f}",
        f"- valid_ratio: {best_metrics['valid_ratio']:.6f}",
        f"- gt_foreground_ratio: {best_metrics['gt_foreground_ratio']:.6f}",
        f"- empty_row_band_ratio: {best_metrics['empty_row_band_ratio']:.6f}",
        f"- coverage_ok: {best_metrics['coverage_ok']}",
        f"- empty_band_ok: {best_metrics['empty_band_ok']}",
    ]
    (logs_dir / "summary.md").write_text("\n".join(md_lines) + "\n")

    print(f"[done] baseline_guarded_score={baseline_metrics['guarded_score']:.6f}")
    print(f"[done] best_guarded_score={best_metrics['guarded_score']:.6f}")
    print(f"[done] selected={selected_source}")
    print(f"[done] logs={logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
