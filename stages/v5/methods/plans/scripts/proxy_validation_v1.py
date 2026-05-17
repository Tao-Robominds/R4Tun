#!/usr/bin/env python3
"""Proxy threshold validation and held-out reflection evaluation.

All outputs are written under:
  logs/proxy_validation_v1/
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "logs" / "proxy_validation_v1"
PROXY_ROOT = OUT_ROOT / "proxy_threshold_validation"
HELDOUT_ROOT = OUT_ROOT / "heldout_reflection_test"
CAL_THRESHOLDS = REPO_ROOT / "logs" / "proxy_calibration_v1" / "panel" / "r0" / "05_proxy_and_calibration" / "thresholds.json"
PROXY_PANEL = REPO_ROOT / "data" / "panels" / "proxy" / "proxy_threshold_validation_set.json"
HELDOUT_PANEL = REPO_ROOT / "data" / "panels" / "heldout" / "heldout_reflection_test_set.json"
PRE_SUMMARY = REPO_ROOT / "logs" / "preprocessing_bo_v2" / "summary_preprocessing_bo_v2.json"
DET_SUMMARY = REPO_ROOT / "logs" / "detection_boundary_bo_v1" / "summary_detection_boundary_bo_v1.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _assert_safe_output(path: Path) -> None:
    protected = [
        REPO_ROOT / "data" / "ablation",
        REPO_ROOT / "data" / "bo",
        REPO_ROOT / "data" / "baseline",
        REPO_ROOT / "data" / "preprocessing_qa",
        REPO_ROOT / "data" / "represents",
        REPO_ROOT / "logs" / "context_preprocessing_v1",
        REPO_ROOT / "r4tun" / "data",
        REPO_ROOT / "r4tun" / "references",
        REPO_ROOT / "methods" / "plans" / "output",
    ]
    resolved = path.resolve()
    if not _is_within(resolved, OUT_ROOT):
        raise ValueError(f"Refusing output outside {OUT_ROOT}: {resolved}")
    for root in protected:
        if _is_within(resolved, root):
            raise ValueError(f"Refusing protected output path: {resolved}")


def _module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _miou_threshold(tunnel_id: str) -> float:
    if tunnel_id.startswith("1-"):
        return 0.60
    if tunnel_id.startswith("3-"):
        return 0.50
    return 0.40


def _ring_key(tunnel_id: str, ring_id: int) -> str:
    return f"{tunnel_id}/r{int(ring_id)}"


def _segment_count(row: dict[str, Any], tunnel_id: str) -> int:
    if row.get("segment_count") is not None:
        return int(row["segment_count"])
    return 6 if tunnel_id.startswith(("1-", "2-", "3-")) else 7


@dataclass(frozen=True)
class FrozenConfig:
    pre_params: dict[str, Any] | None
    det_params: dict[str, Any] | None
    pre_source: str
    det_source: str


def _load_frozen_pre_params() -> list[dict[str, Any]]:
    if not PRE_SUMMARY.exists():
        return []
    rows: list[dict[str, Any]] = []
    for item in _load_json(PRE_SUMMARY).get("rings", []):
        params = None
        source = str(item.get("summary_json") or "")
        if source:
            summary_path = REPO_ROOT / source
            if summary_path.exists():
                payload = _load_json(summary_path)
                params = payload.get("best", {}).get("params")
        rows.append(
            {
                "tunnel_id": str(item["tunnel_id"]),
                "ring_id": int(item["ring_id"]),
                "regime": str(item.get("regime", "")),
                "segment_count": 6 if str(item["tunnel_id"]).startswith(("1-", "2-", "3-")) else 7,
                "params": params,
                "source": source or str(item.get("selected_output_dir", "")),
            }
        )
    return rows


def _load_frozen_det_params() -> list[dict[str, Any]]:
    if not DET_SUMMARY.exists():
        return []
    rows: list[dict[str, Any]] = []
    for item in _load_json(DET_SUMMARY).get("rings", []):
        out = Path(str(item.get("selected_output_dir", "")))
        param_path = out / "parameters_detection.json"
        params = _load_json(param_path) if param_path.exists() else None
        rows.append(
            {
                "tunnel_id": str(item["tunnel_id"]),
                "ring_id": int(item["ring_id"]),
                "segment_count": 6 if str(item["tunnel_id"]).startswith(("1-", "2-", "3-")) else 7,
                "deployable_valid": bool(item.get("deployable_valid", False)),
                "params": params,
                "source": str(param_path) if param_path.exists() else str(out),
            }
        )
    return rows


def _nearest_config(panel_row: dict[str, Any], *, reflection: bool) -> FrozenConfig:
    tunnel_id = str(panel_row["tunnel_id"])
    ring_id = int(panel_row["ring_id"])
    seg_count = _segment_count(panel_row, tunnel_id)
    pre_rows = _load_frozen_pre_params()
    det_rows = _load_frozen_det_params()

    def pick(rows: list[dict[str, Any]], *, want_deployable: bool = False) -> dict[str, Any] | None:
        usable = [r for r in rows if r.get("params")]
        if want_deployable:
            deployable = [r for r in usable if bool(r.get("deployable_valid"))]
            if deployable:
                usable = deployable
        if not usable:
            return None
        def score(r: dict[str, Any]) -> tuple[int, int, int]:
            same_tunnel = 0 if r["tunnel_id"] == tunnel_id else 1
            same_seg = 0 if int(r.get("segment_count", seg_count)) == seg_count else 1
            distance = abs(int(r["ring_id"]) - ring_id) if r["tunnel_id"] == tunnel_id else 10_000 + abs(int(r["ring_id"]) - ring_id)
            return (same_tunnel, same_seg, distance)
        return min(usable, key=score)

    pre = pick(pre_rows)
    det = pick(det_rows, want_deployable=reflection)
    return FrozenConfig(
        pre_params=None if pre is None else pre.get("params"),
        det_params=None if det is None else det.get("params"),
        pre_source="default_parameters" if pre is None else str(pre.get("source")),
        det_source="default_parameters" if det is None else str(det.get("source")),
    )


def _run_ring(row: dict[str, Any], *, output_root: Path, reflection: bool, variant: str) -> dict[str, Any]:
    tunnel_id = str(row["tunnel_id"])
    ring_id = int(row["ring_id"])
    ring_root = output_root / tunnel_id / f"r{ring_id}"
    if variant != "proxy":
        ring_root = ring_root / variant
    _assert_safe_output(ring_root)
    ring_root.mkdir(parents=True, exist_ok=True)

    context_pre = _module_from_path(
        "context_preprocessing_validation",
        REPO_ROOT / "agents" / "1_preprocessing" / "context_preprocessing.py",
    )
    detection = _module_from_path(
        "detection_validation",
        REPO_ROOT / "agents" / "2_detection" / "2_detection.py",
    )
    segmentation = _module_from_path(
        "segmentation_validation",
        REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py",
    )
    evaluation = _module_from_path("evaluation_validation", REPO_ROOT / "agents" / "evaluation.py")
    pre_metrics_mod = _module_from_path("pre_metrics_validation", REPO_ROOT / "bo" / "preprocessing_iou_metrics.py")
    det_metrics_mod = _module_from_path("det_metrics_validation", REPO_ROOT / "bo" / "detection_boundary_metrics.py")

    cfg = _nearest_config(row, reflection=reflection)
    work_root = output_root if variant == "proxy" else output_root / f"_{variant}_work"
    _assert_safe_output(work_root)
    work_ring = context_pre.run_context_trial(
        tunnel_id=tunnel_id,
        ring_id=ring_id,
        context_radius=0,
        output_root=work_root,
        reference_base_dir=str(REPO_ROOT / "data" / "ablation" / "baseline"),
        params_override=cfg.pre_params,
    )

    if cfg.det_params is not None:
        (work_ring / "parameters_detection.json").write_text(json.dumps(cfg.det_params, indent=2, sort_keys=True) + "\n")

    stage_base = work_root if variant != "proxy" else output_root
    detection.run_detection(tunnel_id, ring_id, base_dir=str(stage_base))
    segmentation.run_segmentation(tunnel_id, ring_id, base_dir=str(stage_base))

    eval_base = str(stage_base)
    mirror = work_ring
    eval_results = evaluation.evaluate(tunnel_id, ring_id, base_dir=eval_base, segment_count=_segment_count(row, tunnel_id))
    if variant != "proxy":
        if ring_root.exists():
            shutil.rmtree(ring_root)
        shutil.move(str(work_ring), str(ring_root))
        parent = work_ring.parent
        while parent != work_root and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
        work_ring = ring_root

    pre_metrics = pre_metrics_mod.compute_target_guarded_metrics(work_ring)
    det_metrics = det_metrics_mod.compute_detection_boundary_metrics(work_ring)
    s_depth = float(pre_metrics.get("coverage_factor", 0.0) * pre_metrics.get("empty_factor", 0.0))
    s_boundary = float(det_metrics.get("S_boundary", 0.0))
    miou = float(eval_results.get("mIoU", 0.0))
    oa = float(eval_results.get("OA", 0.0))
    threshold = _miou_threshold(tunnel_id)
    result = {
        **{k: row.get(k) for k in row.keys()},
        "ring_key": _ring_key(tunnel_id, ring_id),
        "variant": variant,
        "output_dir": str(work_ring),
        "pre_params_source": cfg.pre_source,
        "det_params_source": cfg.det_source,
        "S_depth": s_depth,
        "target_foreground_recall": _safe_float(pre_metrics.get("target_foreground_recall")),
        "coverage_factor": _safe_float(pre_metrics.get("coverage_factor")),
        "empty_factor": _safe_float(pre_metrics.get("empty_factor")),
        "shape_factor": _safe_float(pre_metrics.get("shape_factor")),
        "guarded_score": _safe_float(pre_metrics.get("guarded_score")),
        "S_boundary": s_boundary,
        "S_continuity": _safe_float(det_metrics.get("S_continuity")),
        "S_K": _safe_float(det_metrics.get("S_K")),
        "S_spacing": _safe_float(det_metrics.get("S_spacing")),
        "S_layout_coverage": _safe_float(det_metrics.get("S_layout_coverage")),
        "BoundaryF1": _safe_float(det_metrics.get("BoundaryF1")),
        "final_mIoU": miou,
        "final_OA": oa,
        "mIoU_threshold": threshold,
        "is_bad_case": bool(miou < threshold),
        "timestamp_utc": _now(),
    }
    _write_json(work_ring / "proxy_validation_ring_result.json", result)
    return result


def _run_ring_entry(args: argparse.Namespace) -> int:
    panel = _load_json(PROXY_PANEL if args.dataset == "proxy" else HELDOUT_PANEL)
    match = None
    for row in panel:
        if str(row["tunnel_id"]) == args.tunnel_id and int(row["ring_id"]) == int(args.ring_id):
            match = row
            break
    if match is None:
        raise KeyError(f"Missing ring in {args.dataset} panel: {args.tunnel_id}/r{args.ring_id}")
    root = PROXY_ROOT if args.dataset == "proxy" else HELDOUT_ROOT
    result = _run_ring(match, output_root=root, reflection=bool(args.reflection), variant=args.variant)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _read_ring_result(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return _load_json(path)
    return None


def _run_panel(panel_path: Path, *, dataset: str, variant: str, reflection: bool, timeout_sec: int, limit: int | None) -> list[dict[str, Any]]:
    panel = _load_json(panel_path)
    if limit is not None:
        panel = panel[: int(limit)]
    root = PROXY_ROOT if dataset == "proxy" else HELDOUT_ROOT
    results: list[dict[str, Any]] = []
    venv_python = REPO_ROOT / "venv" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(f"Missing venv python: {venv_python}")
    for idx, row in enumerate(panel, start=1):
        tunnel_id = str(row["tunnel_id"])
        ring_id = int(row["ring_id"])
        ring_root = root / tunnel_id / f"r{ring_id}"
        if variant != "proxy":
            ring_root = ring_root / variant
        result_path = ring_root / "proxy_validation_ring_result.json"
        existing = _read_ring_result(result_path)
        if existing is not None and not existing.get("failed"):
            print(f"[skip {idx}/{len(panel)}] {dataset}:{variant} {tunnel_id}/r{ring_id} already complete")
            results.append(existing)
            continue
        print(f"[run {idx}/{len(panel)}] {dataset}:{variant} {tunnel_id}/r{ring_id}")
        cmd = [
            str(venv_python),
            str(Path(__file__).resolve()),
            "--run-ring",
            "--dataset",
            dataset,
            "--variant",
            variant,
            "--tunnel-id",
            tunnel_id,
            "--ring-id",
            str(ring_id),
        ]
        if reflection:
            cmd.append("--reflection")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            failed = {
                **row,
                "ring_key": _ring_key(tunnel_id, ring_id),
                "variant": variant,
                "failed": True,
                "timeout": True,
                "error": f"timeout after {timeout_sec}s",
                "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            }
            _write_json(ring_root / "proxy_validation_ring_result.json", failed)
            results.append(failed)
            continue
        if proc.returncode != 0:
            failed = {
                **row,
                "ring_key": _ring_key(tunnel_id, ring_id),
                "variant": variant,
                "failed": True,
                "timeout": False,
                "error": f"exit code {proc.returncode}",
                "stdout": proc.stdout[-4000:],
            }
            _write_json(ring_root / "proxy_validation_ring_result.json", failed)
            results.append(failed)
            print(proc.stdout[-2000:])
            continue
        result = _read_ring_result(result_path)
        if result is None:
            raise FileNotFoundError(f"Ring completed but result missing: {result_path}")
        results.append(result)
    return results


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and key != "stdout":
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rule_trigger(row: dict[str, Any], t_depth: float, t_boundary: float, rule: str) -> bool:
    sd = _safe_float(row.get("S_depth"))
    sb = _safe_float(row.get("S_boundary"))
    depth = sd is not None and sd < t_depth
    boundary = sb is not None and sb < t_boundary
    if rule == "depth_only":
        return depth
    if rule == "boundary_only":
        return boundary
    if rule == "and":
        return depth and boundary
    return depth or boundary


def _rule_metrics(rows: list[dict[str, Any]], t_depth: float, t_boundary: float, rule: str) -> dict[str, Any]:
    valid = [r for r in rows if not r.get("failed") and _safe_float(r.get("final_mIoU")) is not None]
    n_bad = sum(bool(r.get("is_bad_case")) for r in valid)
    triggered = [_rule_trigger(r, t_depth, t_boundary, rule) for r in valid]
    tp = sum(bool(r.get("is_bad_case")) and trig for r, trig in zip(valid, triggered))
    fp = sum((not bool(r.get("is_bad_case"))) and trig for r, trig in zip(valid, triggered))
    fn = sum(bool(r.get("is_bad_case")) and (not trig) for r, trig in zip(valid, triggered))
    accepted = [r for r, trig in zip(valid, triggered) if not trig]
    return {
        "T_depth": float(t_depth),
        "T_boundary": float(t_boundary),
        "rule": rule,
        "n_rows": len(valid),
        "n_bad": int(n_bad),
        "n_triggered": int(sum(triggered)),
        "tp_bad": int(tp),
        "fp": int(fp),
        "fn_bad": int(fn),
        "bad_case_recall": float(tp / n_bad) if n_bad else None,
        "trigger_precision": float(tp / (tp + fp)) if (tp + fp) else None,
        "false_negative_rate": float(fn / n_bad) if n_bad else None,
        "accepted_mIoU_mean": float(np.mean([r["final_mIoU"] for r in accepted])) if accepted else None,
        "accepted_mIoU_median": float(np.median([r["final_mIoU"] for r in accepted])) if accepted else None,
        "accepted_OA_mean": float(np.mean([r["final_OA"] for r in accepted])) if accepted else None,
    }


def _threshold_candidates(values: list[float], seed: float | None) -> list[float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return [float(seed or 0.0)]
    qs = np.linspace(0.0, 1.0, 31)
    out = {float(np.quantile(vals, q)) for q in qs}
    if seed is not None:
        out.add(float(seed))
    return sorted(out)


def _freeze_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_seed = _load_json(CAL_THRESHOLDS).get("selected", {}) if CAL_THRESHOLDS.exists() else {}
    seed_depth = _safe_float(selected_seed.get("T_depth"))
    seed_boundary = _safe_float(selected_seed.get("T_boundary"))
    valid = [r for r in rows if not r.get("failed") and _safe_float(r.get("S_depth")) is not None and _safe_float(r.get("S_boundary")) is not None]
    d_candidates = _threshold_candidates([float(r["S_depth"]) for r in valid], seed_depth)
    b_candidates = _threshold_candidates([float(r["S_boundary"]) for r in valid], seed_boundary)
    grid: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for td in d_candidates:
        for tb in b_candidates:
            for rule in ("depth_only", "boundary_only", "union", "and"):
                m = _rule_metrics(valid, td, tb, rule)
                grid.append(m)
                recall = m["bad_case_recall"] if m["bad_case_recall"] is not None else -1.0
                precision = m["trigger_precision"] if m["trigger_precision"] is not None else -1.0
                accepted = m["accepted_mIoU_mean"] if m["accepted_mIoU_mean"] is not None else -1.0
                score = (recall, precision, accepted, -m["n_triggered"])
                if best is None or score > best["_score"]:
                    best = {**m, "_score": score}
    assert best is not None
    best.pop("_score", None)
    alternatives = {
        rule: max(
            (g for g in grid if g["rule"] == rule),
            key=lambda m: (
                m["bad_case_recall"] if m["bad_case_recall"] is not None else -1.0,
                m["trigger_precision"] if m["trigger_precision"] is not None else -1.0,
                m["accepted_mIoU_mean"] if m["accepted_mIoU_mean"] is not None else -1.0,
                -m["n_triggered"],
            ),
        )
        for rule in ("depth_only", "boundary_only", "union", "and")
    }
    payload = {
        "timestamp_utc": _now(),
        "selection_scope": "data/panels/proxy/proxy_threshold_validation_set.json only",
        "seed_thresholds": {"T_depth": seed_depth, "T_boundary": seed_boundary, "rule": selected_seed.get("rule", "union")},
        "selected": best,
        "alternatives": alternatives,
    }
    _write_table(OUT_ROOT / "threshold_validation_grid.csv", grid)
    _write_json(OUT_ROOT / "threshold_validation_eval.json", payload)
    _write_json(OUT_ROOT / "frozen_thresholds.json", payload)
    return payload


def _trigger_reason(row: dict[str, Any], frozen: dict[str, Any]) -> str:
    sel = frozen["selected"]
    td = float(sel["T_depth"])
    tb = float(sel["T_boundary"])
    depth = _safe_float(row.get("S_depth")) is not None and float(row["S_depth"]) < td
    boundary = _safe_float(row.get("S_boundary")) is not None and float(row["S_boundary"]) < tb
    if depth and boundary:
        return "both"
    if depth:
        return "depth"
    if boundary:
        return "boundary"
    return "none"


def _heldout_pairs(a0_rows: list[dict[str, Any]], a1_rows: list[dict[str, Any]], frozen: dict[str, Any]) -> list[dict[str, Any]]:
    by_key_a1 = {r["ring_key"]: r for r in a1_rows}
    pairs: list[dict[str, Any]] = []
    for a0 in a0_rows:
        key = a0["ring_key"]
        reason = _trigger_reason(a0, frozen)
        triggered = reason != "none"
        a1 = by_key_a1.get(key)
        if not triggered or a1 is None or a1.get("failed"):
            a1 = a0
        miou0 = _safe_float(a0.get("final_mIoU"))
        miou1 = _safe_float(a1.get("final_mIoU"))
        oa0 = _safe_float(a0.get("final_OA"))
        oa1 = _safe_float(a1.get("final_OA"))
        pairs.append(
            {
                "ring_key": key,
                "tunnel_id": a0.get("tunnel_id"),
                "ring_id": a0.get("ring_id"),
                "triggered": bool(triggered),
                "trigger_reason": reason,
                "mIoU_no_reflection": miou0,
                "mIoU_reflection": miou1,
                "delta_mIoU": None if miou0 is None or miou1 is None else float(miou1 - miou0),
                "OA_no_reflection": oa0,
                "OA_reflection": oa1,
                "delta_OA": None if oa0 is None or oa1 is None else float(oa1 - oa0),
                "is_bad_case_no_reflection": bool(a0.get("is_bad_case")),
                "corrective_passes": 1 if triggered else 0,
                "timeout": bool(a0.get("timeout")) or bool(a1.get("timeout")),
                "failed": bool(a0.get("failed")) or (triggered and bool(a1.get("failed"))),
                "A0_output_dir": a0.get("output_dir"),
                "A1_output_dir": a1.get("output_dir"),
            }
        )
    return pairs


def _cluster_bootstrap(rows: list[dict[str, Any]], *, n_boot: int = 2000, seed: int = 13) -> dict[str, Any]:
    valid = [r for r in rows if _safe_float(r.get("delta_mIoU")) is not None and _safe_float(r.get("delta_OA")) is not None]
    clusters = sorted({str(r["tunnel_id"]) for r in valid})
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"mean_delta_mIoU": [], "median_delta_mIoU": [], "mean_delta_OA": [], "trigger_rate": []}
    by_cluster = {c: [r for r in valid if str(r["tunnel_id"]) == c] for c in clusters}
    if not clusters:
        return {k: {"lo": None, "hi": None} for k in samples}
    for _ in range(n_boot):
        picked: list[dict[str, Any]] = []
        for c in rng.choice(clusters, size=len(clusters), replace=True):
            picked.extend(by_cluster[str(c)])
        dmiou = np.array([float(r["delta_mIoU"]) for r in picked], dtype=float)
        doa = np.array([float(r["delta_OA"]) for r in picked], dtype=float)
        samples["mean_delta_mIoU"].append(float(np.mean(dmiou)))
        samples["median_delta_mIoU"].append(float(np.median(dmiou)))
        samples["mean_delta_OA"].append(float(np.mean(doa)))
        samples["trigger_rate"].append(float(np.mean([bool(r["triggered"]) for r in picked])))
    return {
        k: {"lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975))}
        for k, v in samples.items()
    }


def _paired_statistics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in pairs if _safe_float(r.get("delta_mIoU")) is not None and not r.get("failed")]
    dmiou = np.array([float(r["delta_mIoU"]) for r in valid], dtype=float)
    doa = np.array([float(r["delta_OA"]) for r in valid], dtype=float)
    t_p = None
    w_p = None
    if len(valid) >= 2:
        t_p = _safe_float(ttest_rel(
            [float(r["mIoU_reflection"]) for r in valid],
            [float(r["mIoU_no_reflection"]) for r in valid],
            nan_policy="omit",
        ).pvalue)
        try:
            w_p = _safe_float(wilcoxon(dmiou).pvalue)
        except ValueError:
            w_p = None
    return {
        "n_pairs": len(valid),
        "mean_delta_mIoU": float(np.mean(dmiou)) if len(dmiou) else None,
        "median_delta_mIoU": float(np.median(dmiou)) if len(dmiou) else None,
        "mean_delta_OA": float(np.mean(doa)) if len(doa) else None,
        "median_delta_OA": float(np.median(doa)) if len(doa) else None,
        "paired_ttest_p_value_mIoU": t_p,
        "wilcoxon_p_value_mIoU": w_p,
        "improved_count": int(np.sum(dmiou > 1e-9)) if len(dmiou) else 0,
        "unchanged_count": int(np.sum(np.abs(dmiou) <= 1e-9)) if len(dmiou) else 0,
        "worsened_count": int(np.sum(dmiou < -1e-9)) if len(dmiou) else 0,
        "trigger_rate": float(np.mean([bool(r["triggered"]) for r in valid])) if valid else None,
        "trigger_precision_heldout_label": (
            float(sum(bool(r["triggered"]) and bool(r["is_bad_case_no_reflection"]) for r in valid) / sum(bool(r["triggered"]) for r in valid))
            if sum(bool(r["triggered"]) for r in valid)
            else None
        ),
        "accepted_mIoU_mean": (
            float(np.mean([float(r["mIoU_no_reflection"]) for r in valid if not bool(r["triggered"])]))
            if any(not bool(r["triggered"]) for r in valid)
            else None
        ),
        "accepted_OA_mean": (
            float(np.mean([float(r["OA_no_reflection"]) for r in valid if not bool(r["triggered"])]))
            if any(not bool(r["triggered"]) for r in valid)
            else None
        ),
        "cluster_bootstrap": _cluster_bootstrap(valid),
    }


def _write_proxy_summary(rows: list[dict[str, Any]], frozen: dict[str, Any]) -> None:
    valid = [r for r in rows if not r.get("failed")]
    lines = [
        "# Proxy Validation Summary",
        "",
        f"- rows: {len(rows)}",
        f"- completed: {len(valid)}",
        f"- bad cases: {sum(bool(r.get('is_bad_case')) for r in valid)}",
        f"- frozen rule: `{frozen['selected']['rule']}`",
        f"- T_depth: `{frozen['selected']['T_depth']:.6f}`",
        f"- T_boundary: `{frozen['selected']['T_boundary']:.6f}`",
        f"- validation bad-case recall: `{frozen['selected']['bad_case_recall']}`",
        f"- validation trigger precision: `{frozen['selected']['trigger_precision']}`",
        "",
        "| Ring | S_depth | S_boundary | mIoU | OA | Bad |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in valid:
        lines.append(
            f"| {r['ring_key']} | {float(r['S_depth']):.4f} | {float(r['S_boundary']):.6f} | "
            f"{float(r['final_mIoU']):.4f} | {float(r['final_OA']):.4f} | {bool(r['is_bad_case'])} |"
        )
    (OUT_ROOT / "proxy_validation_summary.md").write_text("\n".join(lines) + "\n")
    (OUT_ROOT / "threshold_validation_report.md").write_text("\n".join(lines[:16]) + "\n")


def _write_final_report(proxy_rows: list[dict[str, Any]], frozen: dict[str, Any], pairs: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    false_negatives = [r for r in proxy_rows if bool(r.get("is_bad_case")) and not _rule_trigger(r, frozen["selected"]["T_depth"], frozen["selected"]["T_boundary"], frozen["selected"]["rule"])]
    worsened = [r for r in pairs if _safe_float(r.get("delta_mIoU")) is not None and float(r["delta_mIoU"]) < -1e-9]
    lines = [
        "# Proxy / Reflection Validation Report",
        "",
        "BO rings were used for tuning only. Proxy validation rings were used for threshold selection only. Held-out rings were used for paired reflection evidence only.",
        "",
        "## Frozen Trigger",
        "",
        f"- rule: `{frozen['selected']['rule']}`",
        f"- T_depth: `{frozen['selected']['T_depth']:.6f}`",
        f"- T_boundary: `{frozen['selected']['T_boundary']:.6f}`",
        f"- proxy validation recall: `{frozen['selected']['bad_case_recall']}`",
        f"- proxy validation precision: `{frozen['selected']['trigger_precision']}`",
        "",
        "## Held-Out Paired Evidence",
        "",
        f"- paired rows: `{stats['n_pairs']}`",
        f"- mean delta mIoU: `{stats['mean_delta_mIoU']}`",
        f"- median delta mIoU: `{stats['median_delta_mIoU']}`",
        f"- mean delta OA: `{stats['mean_delta_OA']}`",
        f"- paired t-test p-value: `{stats['paired_ttest_p_value_mIoU']}`",
        f"- Wilcoxon p-value: `{stats['wilcoxon_p_value_mIoU']}`",
        f"- improved / unchanged / worsened: `{stats['improved_count']} / {stats['unchanged_count']} / {stats['worsened_count']}`",
        f"- cluster bootstrap CI: `{stats['cluster_bootstrap']}`",
        "",
        "## Failure Audit",
        "",
        f"- proxy false negatives: `{[r['ring_key'] for r in false_negatives]}`",
        f"- held-out worsened cases: `{[r['ring_key'] for r in worsened]}`",
        "",
        "## Deployment Recommendation",
        "",
        "Use the frozen trigger only as a reflection gate when GT is unavailable; do not use BO-panel statistics for final claims.",
    ]
    (OUT_ROOT / "proxy_reflection_validation_report.md").write_text("\n".join(lines) + "\n")


def _main(args: argparse.Namespace) -> int:
    _assert_safe_output(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    proxy_rows = _run_panel(PROXY_PANEL, dataset="proxy", variant="proxy", reflection=False, timeout_sec=args.ring_timeout_sec, limit=args.limit)
    _write_table(OUT_ROOT / "proxy_validation_dataset.csv", proxy_rows)
    _write_json(OUT_ROOT / "proxy_validation_dataset.json", proxy_rows)

    frozen = _freeze_thresholds(proxy_rows)
    _write_proxy_summary(proxy_rows, frozen)

    a0_rows = _run_panel(HELDOUT_PANEL, dataset="heldout", variant="A0_no_reflection", reflection=False, timeout_sec=args.ring_timeout_sec, limit=args.limit)
    # Only run reflection pass for held-out rings that would actually trigger. Non-triggered rows reuse A0.
    panel = _load_json(HELDOUT_PANEL)
    if args.limit is not None:
        panel = panel[: int(args.limit)]
    by_key_a0 = {r["ring_key"]: r for r in a0_rows}
    triggered_panel = [r for r in panel if _trigger_reason(by_key_a0[_ring_key(str(r["tunnel_id"]), int(r["ring_id"]))], frozen) != "none" and not by_key_a0[_ring_key(str(r["tunnel_id"]), int(r["ring_id"]))].get("failed")]
    temp_panel = OUT_ROOT / "_triggered_heldout_panel.json"
    _write_json(temp_panel, triggered_panel)
    # Reuse the same runner logic directly so the official held-out panel remains untouched.
    a1_rows: list[dict[str, Any]] = []
    for row in triggered_panel:
        existing = _read_ring_result(HELDOUT_ROOT / str(row["tunnel_id"]) / f"r{int(row['ring_id'])}" / "A1_reflection" / "proxy_validation_ring_result.json")
        if existing and not existing.get("failed"):
            a1_rows.append(existing)
            continue
        cmd = [
            str(REPO_ROOT / "venv" / "bin" / "python"),
            str(Path(__file__).resolve()),
            "--run-ring",
            "--dataset",
            "heldout",
            "--variant",
            "A1_reflection",
            "--reflection",
            "--tunnel-id",
            str(row["tunnel_id"]),
            "--ring-id",
            str(row["ring_id"]),
        ]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.ring_timeout_sec, check=False)
        result_path = HELDOUT_ROOT / str(row["tunnel_id"]) / f"r{int(row['ring_id'])}" / "A1_reflection" / "proxy_validation_ring_result.json"
        if proc.returncode == 0 and result_path.exists():
            a1_rows.append(_load_json(result_path))
        else:
            failed = {**row, "ring_key": _ring_key(str(row["tunnel_id"]), int(row["ring_id"])), "variant": "A1_reflection", "failed": True, "error": proc.stdout[-4000:]}
            _write_json(result_path, failed)
            a1_rows.append(failed)

    pairs = _heldout_pairs(a0_rows, a1_rows, frozen)
    _write_table(OUT_ROOT / "paired_reflection_results.csv", pairs)
    _write_json(OUT_ROOT / "reflection_traces.json", {"A0": a0_rows, "A1": a1_rows, "pairs": pairs})
    heldout_summary = {"timestamp_utc": _now(), "n_A0": len(a0_rows), "n_A1_triggered": len(a1_rows), "n_pairs": len(pairs)}
    _write_json(OUT_ROOT / "heldout_reflection_summary.json", heldout_summary)

    stats = _paired_statistics(pairs)
    _write_json(OUT_ROOT / "paired_statistics.json", stats)
    _write_json(OUT_ROOT / "cluster_bootstrap_ci.json", stats["cluster_bootstrap"])
    (OUT_ROOT / "paired_statistics_report.md").write_text(
        "# Paired Statistics\n\n"
        f"- n_pairs: `{stats['n_pairs']}`\n"
        f"- mean_delta_mIoU: `{stats['mean_delta_mIoU']}`\n"
        f"- median_delta_mIoU: `{stats['median_delta_mIoU']}`\n"
        f"- mean_delta_OA: `{stats['mean_delta_OA']}`\n"
        f"- trigger_rate: `{stats['trigger_rate']}`\n"
        f"- cluster_bootstrap: `{stats['cluster_bootstrap']}`\n"
    )
    _write_final_report(proxy_rows, frozen, pairs, stats)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-ring", action="store_true")
    p.add_argument("--dataset", choices=["proxy", "heldout"], default="proxy")
    p.add_argument("--variant", default="proxy")
    p.add_argument("--reflection", action="store_true")
    p.add_argument("--tunnel-id")
    p.add_argument("--ring-id", type=int)
    p.add_argument("--ring-timeout-sec", type=int, default=900)
    p.add_argument("--limit", type=int, default=None, help="Debug limit; omit for full plan.")
    return p.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    try:
        if ns.run_ring:
            raise SystemExit(_run_ring_entry(ns))
        raise SystemExit(_main(ns))
    except Exception:
        traceback.print_exc()
        raise
