"""GT-free plus/minus direction selection after detection.

Detection writes direction_plus and direction_minus artifacts (same geometry,
relabeled blocks). This module scores both branches with label-free metrics,
runs segmentation on each, and commits the higher-scoring branch.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.ceiling_gate import REPO_ROOT, SEG_CLI, VENV_PY
from lib.guardrail_utils import apply_guardrail_fields, guardrail_passed

BRANCHES = ("plus", "minus")
BRANCH_ARTIFACTS: dict[str, tuple[str, str]] = {
    "plus": ("all_segments_direction_plus.csv", "boundaries_per_ring_direction_plus.json"),
    "minus": ("all_segments_direction_minus.csv", "boundaries_per_ring_direction_minus.json"),
}
DIRECTION_META = "direction_hypotheses_meta.json"
SELECTION_FILENAME = "direction_selection.json"
MARGIN_TIE_EPS = 1e-6
LOW_CONFIDENCE_MARGIN = 0.05
TEMPLATE_SWITCH_MARGIN = 0.08

BLOCKS_6 = ["K", "B1", "A1", "A2", "A3", "B2"]
BLOCKS_7 = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]
PRIOR_K_SMALL_7 = np.array([0.07, 0.15, 0.15, 0.15, 0.15, 0.15, 0.18])
PRIOR_K_SMALL_6 = np.array([0.07, 0.18, 0.18, 0.18, 0.18, 0.21])


def _import_extract_metrics():
    agents_scripts = REPO_ROOT / "agents" / "2_detection" / "scripts"
    if str(agents_scripts) not in sys.path:
        sys.path.insert(0, str(agents_scripts))
    from extract_intrinsics import extract_detection_metrics  # noqa: WPS433

    return extract_detection_metrics


def direction_hypotheses_available(ring_dir: Path) -> bool:
    meta_path = ring_dir / DIRECTION_META
    if not meta_path.is_file():
        return False
    for branch in BRANCHES:
        seg_name, bnd_name = BRANCH_ARTIFACTS[branch]
        if not (ring_dir / seg_name).is_file() or not (ring_dir / bnd_name).is_file():
            return False
    return True


def activate_branch_artifacts(ring_dir: Path, branch: str) -> None:
    seg_name, bnd_name = BRANCH_ARTIFACTS[branch]
    shutil.copy2(ring_dir / seg_name, ring_dir / "all_segments.csv")
    shutil.copy2(ring_dir / bnd_name, ring_dir / "boundaries_per_ring.json")


def _seg_missing_class_count(ring_dir: Path) -> int:
    meta_path = ring_dir / "segment_completion_meta_segmentation.json"
    if not meta_path.is_file():
        return 0
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    after = meta.get("completion_after_projection") or {}
    missing = after.get("missing_ids_before") or []
    return len(missing)


def _load_image_height(ring_dir: Path) -> int | None:
    npy = ring_dir / "depth_map.npy"
    if not npy.is_file():
        return None
    return int(np.load(npy).shape[0])


def _arc_width_profile(ring_dir: Path) -> tuple[list[str], np.ndarray] | None:
    seg_path = ring_dir / "all_segments.csv"
    if not seg_path.is_file():
        return None
    H = _load_image_height(ring_dir)
    if H is None:
        return None
    df = pd.read_csv(seg_path)
    if df.empty:
        return None
    ring_df = df.sort_values("Y")
    ys = ring_df["Y"].to_numpy(dtype=float)
    if len(ys) < 2:
        return None
    widths = [float(ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]
    widths.append(float((H - ys[-1]) + ys[0]))
    w = np.array(widths, dtype=float)
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    blocks = ring_df["Block"].astype(str).tolist()
    return blocks, w


def template_match_score(blocks: list[str], widths: np.ndarray, segment_count: int) -> float:
    """Correlate Y-ordered arc widths with canonical block width prior."""
    canonical = BLOCKS_7 if segment_count == 7 else BLOCKS_6
    prior = PRIOR_K_SMALL_7 if segment_count == 7 else PRIOR_K_SMALL_6
    if len(blocks) != len(canonical):
        return 0.0
    try:
        expected = np.array([prior[canonical.index(b)] for b in blocks], dtype=float)
    except ValueError:
        return 0.0
    expected = expected / expected.sum()
    if float(np.std(widths)) < 1e-9 or float(np.std(expected)) < 1e-9:
        return 0.0
    return float(np.corrcoef(widths, expected)[0, 1])


def composite_branch_score(metrics: dict[str, Any]) -> float:
    """Label-free branch quality score (higher = better)."""
    score = 0.0

    tmpl = metrics.get("template_match_score")
    if tmpl is not None:
        score += 3.0 * float(tmpl)

    cov = metrics.get("det_y_coverage_pct")
    if cov is not None:
        score += 0.5 * max(0.0, 1.0 - abs(float(cov) - 100.0) / 100.0)

    gap = metrics.get("det_min_y_gap_px")
    if gap is not None:
        score += 0.001 * min(float(gap), 500.0)

    if guardrail_passed(metrics):
        score += 0.25

    score -= 0.25 * int(metrics.get("seg_missing_class_count", 0))
    return float(score)


def score_branch(ring_dir: Path, segment_count: int = 7) -> dict[str, Any]:
    extract_detection_metrics = _import_extract_metrics()
    raw = apply_guardrail_fields(extract_detection_metrics(str(ring_dir)))
    missing = _seg_missing_class_count(ring_dir)
    profile = _arc_width_profile(ring_dir)
    tmpl = 0.0
    if profile is not None:
        blocks, widths = profile
        tmpl = template_match_score(blocks, widths, segment_count)
    metrics = {
        "template_match_score": tmpl,
        "det_y_order_consistency": raw.get("det_y_order_consistency"),
        "det_y_coverage_pct": raw.get("det_y_coverage_pct"),
        "det_min_y_gap_px": raw.get("det_min_y_gap_px"),
        "det_guardrail_passed": raw.get("det_guardrail_passed"),
        "det_guardrail_violations": raw.get("det_guardrail_violations"),
        "seg_missing_class_count": missing,
    }
    metrics["composite_score"] = composite_branch_score(metrics)
    return metrics


def _run_segmentation(
    *,
    tunnel_id: str,
    ring_id: int,
    sandbox_data: Path,
    ring_dir: Path,
    tag: str,
    branch: str,
) -> bool:
    log = ring_dir / "logs" / f"{tag}_3_segmentation_{branch}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(__import__("os").environ)
    env["INTRINSIC_PARAMS_BASE_DIR_ONLY"] = "1"
    proc = subprocess.run(
        [str(VENV_PY), str(SEG_CLI), tunnel_id, str(ring_id), "--data-dir", str(sandbox_data)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        timeout=900,
        check=False,
    )
    return proc.returncode == 0


def _archive_branch_outputs(ring_dir: Path, branch: str) -> None:
    final = ring_dir / "final.csv"
    if final.is_file():
        shutil.copy2(final, ring_dir / f"final_direction_{branch}.csv")
    meta = ring_dir / "segment_completion_meta_segmentation.json"
    if meta.is_file():
        shutil.copy2(meta, ring_dir / f"segment_completion_meta_direction_{branch}.json")


def _gt_miou_from_branch_final(ring_dir: Path, branch: str, max_class: int) -> float | None:
    path = ring_dir / f"final_direction_{branch}.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "segment" not in df.columns or "pred" not in df.columns:
        return None
    tmp = df[["segment", "pred"]].dropna(subset=["segment", "pred"]).copy()
    if tmp.empty:
        return None
    gt = pd.to_numeric(tmp["segment"], errors="coerce").fillna(0).astype(int).to_numpy()
    pred = pd.to_numeric(tmp["pred"], errors="coerce").fillna(0).astype(int).to_numpy()
    valid = (gt >= 0) & (gt <= max_class) & (pred >= 0) & (pred <= max_class)
    gt, pred = gt[valid], pred[valid]
    if gt.size == 0:
        return None
    labels = sorted(set(gt.tolist()) | set(pred.tolist()))
    ious = []
    for cls in labels:
        g, p = gt == cls, pred == cls
        union = np.logical_or(g, p).sum()
        if union:
            ious.append(float(np.logical_and(g, p).sum() / union))
    return float(np.mean(ious)) if ious else None


def select_direction_and_segment(
    *,
    tunnel_id: str,
    ring_id: int,
    sandbox_data: Path,
    ring_dir: Path,
    tag: str,
    prefer_branch: str = "plus",
    segment_count: int = 7,
    force_branch: str | None = None,
    log_twin_gt_miou: bool = False,
) -> dict[str, Any]:
    """Score plus/minus, run seg on both, commit winner to canonical outputs."""
    ring_dir = Path(ring_dir)
    if not direction_hypotheses_available(ring_dir):
        ok = _run_segmentation(
            tunnel_id=tunnel_id,
            ring_id=ring_id,
            sandbox_data=sandbox_data,
            ring_dir=ring_dir,
            tag=tag,
            branch="default",
        )
        return {
            "status": "fallback_single",
            "selected_branch": prefer_branch,
            "agent_error": not ok,
            "direction_select_enabled": False,
        }

    branch_results: dict[str, Any] = {}
    for branch in BRANCHES:
        activate_branch_artifacts(ring_dir, branch)
        ok = _run_segmentation(
            tunnel_id=tunnel_id,
            ring_id=ring_id,
            sandbox_data=sandbox_data,
            ring_dir=ring_dir,
            tag=tag,
            branch=branch,
        )
        if not ok:
            branch_results[branch] = {"agent_error": True, "composite_score": float("-inf")}
            continue
        _archive_branch_outputs(ring_dir, branch)
        metrics = score_branch(ring_dir, segment_count=segment_count)
        metrics["agent_error"] = False
        branch_results[branch] = metrics

    if all(branch_results.get(b, {}).get("agent_error") for b in BRANCHES):
        return {
            "status": "error",
            "selected_branch": prefer_branch,
            "agent_error": True,
            "direction_select_enabled": True,
            "branches": branch_results,
        }

    score_plus = float(branch_results.get("plus", {}).get("composite_score", float("-inf")))
    score_minus = float(branch_results.get("minus", {}).get("composite_score", float("-inf")))
    tmpl_plus = float(branch_results.get("plus", {}).get("template_match_score", 0.0))
    tmpl_minus = float(branch_results.get("minus", {}).get("template_match_score", 0.0))
    margin = abs(score_plus - score_minus)
    tmpl_margin = tmpl_minus - tmpl_plus

    prefer = prefer_branch if prefer_branch in BRANCHES else "plus"
    if tmpl_minus > tmpl_plus + TEMPLATE_SWITCH_MARGIN:
        intrinsic_selected = "minus"
    else:
        intrinsic_selected = prefer

    if force_branch in BRANCHES:
        selected = force_branch
    else:
        selected = intrinsic_selected

    gt_miou_plus = gt_miou_minus = None
    if log_twin_gt_miou:
        gt_miou_plus = _gt_miou_from_branch_final(ring_dir, "plus", segment_count)
        gt_miou_minus = _gt_miou_from_branch_final(ring_dir, "minus", segment_count)

    activate_branch_artifacts(ring_dir, selected)
    winner_final = ring_dir / f"final_direction_{selected}.csv"
    if winner_final.is_file():
        shutil.copy2(winner_final, ring_dir / "final.csv")
    else:
        _run_segmentation(
            tunnel_id=tunnel_id,
            ring_id=ring_id,
            sandbox_data=sandbox_data,
            ring_dir=ring_dir,
            tag=tag,
            branch=selected,
        )

    selection = {
        "status": "ok",
        "selected_branch": selected,
        "intrinsic_selected_branch": intrinsic_selected,
        "score_plus": score_plus,
        "score_minus": score_minus,
        "margin": margin,
        "template_margin_minus_plus": tmpl_margin,
        "template_switch_margin": TEMPLATE_SWITCH_MARGIN,
        "low_confidence": margin < LOW_CONFIDENCE_MARGIN,
        "direction_select_enabled": True,
        "branches": branch_results,
        "template_match_score_plus": tmpl_plus,
        "template_match_score_minus": tmpl_minus,
        "gt_miou_plus": gt_miou_plus,
        "gt_miou_minus": gt_miou_minus,
        "force_branch": force_branch,
        "prefer_branch": prefer_branch,
    }
    (ring_dir / SELECTION_FILENAME).write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return selection


def write_selection_to_out_dir(ring_dir: Path, out_dir: Path) -> None:
    src = ring_dir / SELECTION_FILENAME
    if src.is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_dir / SELECTION_FILENAME)
