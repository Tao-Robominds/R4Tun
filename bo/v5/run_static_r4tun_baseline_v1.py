from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / "venv" / "bin" / "python"
RUN_ROOT = REPO_ROOT / "logs" / "v5_static_r4tun_baseline_v1"
PANEL_CSV = REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv"

PRE_CLI = REPO_ROOT / "agents" / "1_preprocessing" / "1_preprocessing.py"
DET_CLI = REPO_ROOT / "agents" / "2_detection" / "2_detection.py"
SEG_CLI = REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py"
EVAL_CLI = REPO_ROOT / "agents" / "evaluation.py"

R4TUN_PRE = REPO_ROOT / "r4tun" / "sample" / "parameters_preprocessing.json"
R4TUN_UNF = REPO_ROOT / "r4tun" / "sample" / "parameters_unfolding.json"
R4TUN_DET = REPO_ROOT / "r4tun" / "sample" / "parameters_detection.json"
R4TUN_SAM = REPO_ROOT / "r4tun" / "sample" / "parameters_sam.json"

DEFAULT_PRE = REPO_ROOT / "agents" / "1_preprocessing" / "parameters" / "_default_irregular" / "parameters_preprocessing.json"
DEFAULT_DET = REPO_ROOT / "agents" / "2_detection" / "parameters" / "_default_irregular" / "parameters_detection.json"
DEFAULT_SEG = REPO_ROOT / "agents" / "3_segmentation" / "parameters" / "_default_irregular" / "parameters_segmentation.json"
DEFAULT_DET_INIT = REPO_ROOT / "logs" / "v5_t123_depth_contract_v1" / "1-1" / "r18" / "parameters_detection.json"

SCOREBOARD_CSV = RUN_ROOT / "static_r4tun_50ring_scoreboard.csv"
SUMMARY_JSON = RUN_ROOT / "static_r4tun_summary.json"
TRANSLATION_JSON = RUN_ROOT / "schema_translation_manifest.json"
FAILED_JSONL = RUN_ROOT / "failed_rings.jsonl"

PROTECTED_PREFIXES = (
    REPO_ROOT / "data" / "ablation",
    REPO_ROOT / "data" / "bo",
    REPO_ROOT / "data" / "baseline",
    REPO_ROOT / "data" / "preprocessing_qa",
    REPO_ROOT / "data" / "represents",
    REPO_ROOT / "logs" / "context_preprocessing_v1",
    REPO_ROOT / "data",
    REPO_ROOT / "r4tun" / "data",
    REPO_ROOT / "r4tun" / "references",
    REPO_ROOT / "methods" / "plans" / "output",
    REPO_ROOT / "stages" / "v4",
)

# Fixed static contract. Do not optimize per ring/family.
STATIC_SEGMENT_COUNT = 6
STATIC_SEGMENT_ORDER = ["K", "B1", "A1", "A2", "A3", "B2"]
STATIC_OFFSETS = {"K": 0.0, "B1": 181.9, "A1": 727.5, "A2": 1273.2, "A3": -1636.9, "B2": -545.6}


@dataclass
class StageResult:
    ok: bool
    error: str | None = None


def _assert_writable(path: Path) -> None:
    resolved = path.resolve()
    logs_root = (REPO_ROOT / "logs").resolve()
    try:
        resolved.relative_to(logs_root)
    except ValueError as exc:
        raise ValueError(f"Output path must be under logs/: {resolved}") from exc

    for prefix in PROTECTED_PREFIXES:
        if not prefix.exists():
            continue
        pref = prefix.resolve()
        if resolved == pref:
            raise ValueError(f"Refusing protected output path: {resolved}")
        try:
            resolved.relative_to(pref)
            raise ValueError(f"Refusing protected output path: {resolved}")
        except ValueError:
            pass


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _parse_ring(ring_key: str) -> tuple[str, int]:
    tid, rr = ring_key.split("/")
    return tid, int(rr.lstrip("r"))


def _ring_dir(base: Path, ring_key: str) -> Path:
    tid, rid = _parse_ring(ring_key)
    return base / tid / f"r{rid}"


def _source_pointcloud(tunnel_id: str, ring_id: int) -> Path:
    src = REPO_ROOT / "data" / "rings" / f"{tunnel_id.replace('-', '_')}_ring{ring_id}.txt"
    if not src.exists():
        raise FileNotFoundError(f"Missing source point cloud: {src}")
    return src


def _run_cmd(cmd: list[str], log_path: Path, timeout_sec: float = 1800.0) -> StageResult:
    env = dict(os.environ)
    env["INTRINSIC_PARAMS_BASE_DIR_ONLY"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
    if proc.returncode != 0:
        return StageResult(False, f"exit_{proc.returncode}")
    return StageResult(True, None)


def _flatten_r4tun_preprocessing(
    sample_pre: dict[str, Any],
    sample_unf: dict[str, Any],
    fallback_pre: dict[str, Any],
    fallback_log: list[dict[str, Any]],
) -> dict[str, Any]:
    pre_out = dict(fallback_pre)
    src_unf = sample_pre.get("unfolding", {}) if isinstance(sample_pre.get("unfolding"), dict) else {}
    src_den = sample_pre.get("denoising", {}) if isinstance(sample_pre.get("denoising"), dict) else {}
    src_enh = sample_pre.get("enhancing", {}) if isinstance(sample_pre.get("enhancing"), dict) else {}
    if not src_unf:
        src_unf = sample_unf

    mapped = {
        "tunnel_diameter": src_unf.get("physical_constants", {}).get("tunnel_diameter"),
        "ring_spacing": src_unf.get("physical_constants", {}).get("ring_spacing"),
        "samples_per_ring": src_unf.get("arc_length", {}).get("samples_per_ring"),
        "ransac_inlier_ratio": src_unf.get("ransac_ellipse", {}).get("inlier_ratio"),
        "ransac_probability": src_unf.get("ransac_ellipse", {}).get("confidence"),
        "ransac_sample_size": src_unf.get("ransac_ellipse", {}).get("min_samples"),
        "ransac_inlier_threshold_multiplier": src_unf.get("ransac_ellipse", {}).get("inlier_threshold"),
        "radius_min": src_den.get("radius_filtering", {}).get("radius_min"),
        "radius_max": src_den.get("radius_filtering", {}).get("radius_max"),
        "gradient_threshold": src_den.get("gradient_detection", {}).get("gradient_threshold"),
        "smoothing_window_size": src_den.get("cutoff_smoothing", {}).get("smoothing_window"),
        "smoothing_offset": src_den.get("cutoff_smoothing", {}).get("smoothing_offset"),
        "target_distances": src_enh.get("upsampling", {}).get("target_distances"),
        "curvature_neighbors": src_enh.get("curvature", {}).get("curvature_neighbors"),
        "depth_map_resolution": src_enh.get("depth_map", {}).get("resolution"),
        "interpolation_window": src_enh.get("depth_map", {}).get("interpolation_window"),
        "outlier_depth_threshold_low": src_enh.get("outlier_detection", {}).get("depth_threshold_low"),
        "outlier_depth_threshold_high": src_enh.get("outlier_detection", {}).get("depth_threshold_high"),
        "outlier_high_density_ring_start": src_enh.get("outlier_detection", {}).get("high_density_ring_start"),
        "outlier_high_density_ring_end": src_enh.get("outlier_detection", {}).get("high_density_ring_end"),
        "outlier_neighbors": src_enh.get("outlier_detection", {}).get("outlier_neighbors"),
        "outlier_interpolation_radius": src_enh.get("outlier_interpolation", {}).get("interpolation_radius"),
        "outlier_num_interpolations": src_enh.get("outlier_interpolation", {}).get("num_interpolations"),
        "outlier_duplicate_threshold": src_enh.get("outlier_interpolation", {}).get("duplicate_threshold"),
        "max_outlier_points": src_enh.get("outlier_interpolation", {}).get("max_outlier_points"),
    }
    for k, v in mapped.items():
        if v is not None:
            pre_out[k] = v
        else:
            fallback_log.append({"field": k, "reason": "missing_in_r4tun_pre", "fallback": fallback_pre.get(k)})

    # Required static compatibility defaults only.
    if "gravity_anchor" not in pre_out or not isinstance(pre_out.get("gravity_anchor"), dict):
        pre_out["gravity_anchor"] = {"enabled": True, "n_bins": 360}
        fallback_log.append({"field": "gravity_anchor", "reason": "required_for_cli", "fallback": pre_out["gravity_anchor"]})
    else:
        pre_out["gravity_anchor"]["enabled"] = True
        pre_out["gravity_anchor"].setdefault("n_bins", 360)

    pre_out.setdefault("double_zero_cutoff", False)
    pre_out.setdefault("r_surface_min", None)
    pre_out["_static_baseline_contract"] = "r4tun_sample_schema_compatibility_only"
    return pre_out


def _translate_detection(
    sample_det: dict[str, Any],
    fallback_det: dict[str, Any],
    fallback_det_init: dict[str, Any],
    fallback_log: list[dict[str, Any]],
) -> dict[str, Any]:
    out = dict(fallback_det)
    # Map nested sample keys.
    out["binary_threshold"] = sample_det.get("preprocessing", {}).get("binary_threshold", out.get("binary_threshold"))
    out["dilation_kernel_size"] = sample_det.get("preprocessing", {}).get("dilation_kernel_size", out.get("dilation_kernel_size", 3))
    out["dilation_iterations"] = sample_det.get("preprocessing", {}).get("dilation_iterations", out.get("dilation_iterations", 1))
    out["hough_threshold"] = sample_det.get("hough_oblique", {}).get("threshold", out.get("hough_threshold"))
    out["hough_min_length"] = sample_det.get("hough_oblique", {}).get("min_length", out.get("hough_min_length"))
    out["hough_max_gap"] = sample_det.get("hough_oblique", {}).get("max_gap", out.get("hough_max_gap"))
    out["angle_pos_min"] = sample_det.get("hough_oblique", {}).get("angle_positive_min", out.get("angle_pos_min"))
    out["angle_pos_max"] = sample_det.get("hough_oblique", {}).get("angle_positive_max", out.get("angle_pos_max"))
    out["angle_neg_min"] = sample_det.get("hough_oblique", {}).get("angle_negative_min", out.get("angle_neg_min"))
    out["angle_neg_max"] = sample_det.get("hough_oblique", {}).get("angle_negative_max", out.get("angle_neg_max"))
    out["hough_horizontal_threshold"] = sample_det.get("hough_horizontal", {}).get("threshold", out.get("hough_horizontal_threshold", 50))
    out["hough_horizontal_min_length"] = sample_det.get("hough_horizontal", {}).get("min_length", out.get("hough_horizontal_min_length", 100))
    out["hough_horizontal_max_gap"] = sample_det.get("hough_horizontal", {}).get("max_gap", out.get("hough_horizontal_max_gap", 10))
    out["horizontal_angle_tolerance"] = sample_det.get("hough_horizontal", {}).get("angle_tolerance", out.get("horizontal_angle_tolerance", 1.0))
    out["hough_vertical_threshold"] = sample_det.get("hough_vertical", {}).get("threshold", out.get("hough_vertical_threshold", 500))
    out["merge_distance_threshold"] = sample_det.get("line_processing", {}).get("merge_distance_threshold", out.get("merge_distance_threshold", 3.0))
    out["k_expected_height_px"] = float(sample_det.get("physical_constants", {}).get("k_height_mm", 1079.92)) / float(sample_det.get("physical_constants", {}).get("resolution", 0.005))

    # Fixed static contract; no optimization across families.
    out["segment_count"] = STATIC_SEGMENT_COUNT
    out["enabled_blocks"] = list(STATIC_SEGMENT_ORDER)
    out["detector_mode"] = out.get("detector_mode", "single_ring_local")
    if "per_ring_offsets" not in out or not isinstance(out["per_ring_offsets"], dict):
        init_offsets = fallback_det_init.get("per_ring_offsets") if isinstance(fallback_det_init.get("per_ring_offsets"), dict) else None
        if init_offsets and isinstance(init_offsets.get("0"), dict):
            src0 = init_offsets["0"]
            out["per_ring_offsets"] = {"0": {k: float(src0.get(k, STATIC_OFFSETS[k])) for k in STATIC_SEGMENT_ORDER}}
            fallback_log.append({"field": "per_ring_offsets", "reason": "required_for_cli", "fallback": "from_initialized_detection"})
        else:
            out["per_ring_offsets"] = {"0": dict(STATIC_OFFSETS)}
            fallback_log.append({"field": "per_ring_offsets", "reason": "required_for_cli", "fallback": "static_offsets"})
    out["_static_baseline_contract"] = "fixed_segment_6_from_r4tun_sample"
    return out


def _translate_segmentation(
    sample_sam: dict[str, Any],
    fallback_seg: dict[str, Any],
    fallback_log: list[dict[str, Any]],
) -> dict[str, Any]:
    out = dict(fallback_seg)
    if "k_cap" not in out:
        out["k_cap"] = 130
        fallback_log.append({"field": "k_cap", "reason": "required_for_cli", "fallback": 130})
    if "ab_cap" not in out:
        out["ab_cap"] = 390
        fallback_log.append({"field": "ab_cap", "reason": "required_for_cli", "fallback": 390})
    out["segment_count"] = int(sample_sam.get("segment_per_ring", STATIC_SEGMENT_COUNT))
    out["segment_order"] = list(sample_sam.get("segment_order", STATIC_SEGMENT_ORDER))
    out["_static_baseline_contract"] = "fixed_segment_6_from_r4tun_sample"
    return out


def _stage_ring(ring_key: str, pre: dict[str, Any], det: dict[str, Any], seg: dict[str, Any], clean: bool) -> Path:
    dst = _ring_dir(RUN_ROOT, ring_key)
    tid, rid = _parse_ring(ring_key)
    if clean and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    src_txt = _source_pointcloud(tid, rid)
    dst_txt = dst / f"{tid}_r{rid}.txt"
    shutil.copy2(src_txt, dst_txt)
    _write_json(dst / "parameters_preprocessing.json", pre)
    _write_json(dst / "parameters_detection.json", det)
    _write_json(dst / "parameters_segmentation.json", seg)
    (dst / "logs").mkdir(exist_ok=True)
    return dst


def _largest_false_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur = 0
        else:
            cur += 1
            if cur > best:
                best = cur
    return int(best)


def _depth_audit(ring_dir: Path) -> dict[str, Any]:
    depth = ring_dir / "depth_map.npy"
    row = {
        "depth_map_path": str(depth.relative_to(REPO_ROOT)) if depth.exists() else None,
        "finite_ratio": None,
        "row_nonempty_ratio": None,
        "largest_empty_vertical_gap_px": None,
        "largest_empty_vertical_gap_frac": None,
    }
    if not depth.exists():
        return row
    arr = np.load(depth)
    finite = np.isfinite(arr)
    if finite.size == 0:
        return row
    row_ok = finite.any(axis=1)
    row["finite_ratio"] = float(finite.mean())
    row["row_nonempty_ratio"] = float(row_ok.mean())
    gap_px = _largest_false_run(row_ok)
    row["largest_empty_vertical_gap_px"] = int(gap_px)
    row["largest_empty_vertical_gap_frac"] = float(gap_px / max(1, int(arr.shape[0])))
    return row


def _compute_metrics_from_final(final_csv: Path, max_class: int = STATIC_SEGMENT_COUNT) -> tuple[float | None, float | None]:
    if not final_csv.exists():
        return None, None
    df = pd.read_csv(final_csv)
    if "segment" not in df.columns or "pred" not in df.columns:
        return None, None
    gt = pd.to_numeric(df["segment"], errors="coerce").fillna(0).astype(int).to_numpy()
    pred = pd.to_numeric(df["pred"], errors="coerce").fillna(0).astype(int).to_numpy()
    valid = (gt >= 1) & (gt <= max_class)
    if not np.any(valid):
        return None, None
    gt = gt[valid]
    pred = pred[valid]
    oa = float((gt == pred).mean())
    ious: list[float] = []
    for c in range(1, max_class + 1):
        g = gt == c
        p = pred == c
        union = np.logical_or(g, p).sum()
        if union == 0:
            continue
        inter = np.logical_and(g, p).sum()
        ious.append(float(inter / union))
    miou = float(np.mean(ious)) if ious else None
    return oa, miou


def _run_ring(ring_key: str, clean: bool, pre: dict[str, Any], det: dict[str, Any], seg: dict[str, Any]) -> dict[str, Any]:
    tid, rid = _parse_ring(ring_key)
    ring_path = _stage_ring(ring_key, pre, det, seg, clean=clean)
    out: dict[str, Any] = {"ring_key": ring_key, "tunnel_id": tid, "ring_id": rid}

    s1 = _run_cmd([str(VENV_PY), str(PRE_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_path / "logs" / "stage1_preprocessing.log")
    out["stage1_ok"] = s1.ok
    if not s1.ok:
        out["error_stage"] = "preprocessing"
        out["error"] = s1.error
        out.update(_depth_audit(ring_path))
        return out

    s2 = _run_cmd([str(VENV_PY), str(DET_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_path / "logs" / "stage2_detection.log")
    out["stage2_ok"] = s2.ok
    if not s2.ok:
        out["error_stage"] = "detection"
        out["error"] = s2.error
        out.update(_depth_audit(ring_path))
        return out

    s3 = _run_cmd([str(VENV_PY), str(SEG_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_path / "logs" / "stage3_segmentation.log")
    out["stage3_ok"] = s3.ok
    if not s3.ok:
        out["error_stage"] = "segmentation"
        out["error"] = s3.error
        out.update(_depth_audit(ring_path))
        return out

    s4 = _run_cmd(
        [str(VENV_PY), str(EVAL_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT), "--segments", str(STATIC_SEGMENT_COUNT)],
        ring_path / "logs" / "stage4_evaluation.log",
    )
    out["stage4_ok"] = s4.ok
    out["error_stage"] = None if s4.ok else "evaluation"
    out["error"] = None if s4.ok else s4.error

    final_csv = ring_path / "final.csv"
    oa, miou = _compute_metrics_from_final(final_csv, max_class=STATIC_SEGMENT_COUNT)
    out["oa"] = oa
    out["miou"] = miou
    out["segment_count_static"] = STATIC_SEGMENT_COUNT
    out["final_csv"] = str(final_csv.relative_to(REPO_ROOT)) if final_csv.exists() else None
    out.update(_depth_audit(ring_path))
    return out


def _write_failures(df: pd.DataFrame) -> None:
    fails = df[df["error"].notna()].copy()
    with FAILED_JSONL.open("w", encoding="utf-8") as f:
        for _, r in fails.iterrows():
            obj = {
                "ring_key": r["ring_key"],
                "error_stage": r.get("error_stage"),
                "error": r.get("error"),
                "logs_dir": str((_ring_dir(RUN_ROOT, str(r["ring_key"])) / "logs").relative_to(REPO_ROOT)),
            }
            f.write(json.dumps(obj) + "\n")


def _summarize(df: pd.DataFrame) -> dict[str, Any]:
    miou = pd.to_numeric(df["miou"], errors="coerce")
    ok = df["error"].isna()
    out = {
        "run_root": str(RUN_ROOT.relative_to(REPO_ROOT)),
        "rows": int(len(df)),
        "successful_rows": int(ok.sum()),
        "failed_rows": int((~ok).sum()),
        "static_segment_count": STATIC_SEGMENT_COUNT,
        "mean_miou_successful": float(miou[ok].mean()) if ok.any() else None,
        "median_miou_successful": float(miou[ok].median()) if ok.any() else None,
        "mean_miou_all_rows": float(miou.mean()) if miou.notna().any() else None,
    }
    if "family" in df.columns:
        fam_summary = (
            df.assign(miou_num=pd.to_numeric(df["miou"], errors="coerce"))
            .groupby("family", dropna=False)["miou_num"]
            .mean()
            .reset_index()
        )
        out["mean_miou_by_family"] = [
            {"family": int(r["family"]), "mean_miou": None if pd.isna(r["miou_num"]) else float(r["miou_num"])}
            for _, r in fam_summary.iterrows()
        ]
    return out


def run(scope: str, clean: bool) -> int:
    _assert_writable(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_CSV)
    if scope == "smoke":
        panel = panel[panel["ring_key"].isin(["1-1/r18", "4-1/r110"])].copy()
    panel = panel.sort_values("panel_idx").reset_index(drop=True)

    sample_pre = _load_json(R4TUN_PRE)
    sample_unf = _load_json(R4TUN_UNF)
    sample_det = _load_json(R4TUN_DET)
    sample_sam = _load_json(R4TUN_SAM)
    fallback_pre = _load_json(DEFAULT_PRE)
    fallback_det = _load_json(DEFAULT_DET)
    fallback_seg = _load_json(DEFAULT_SEG)
    fallback_det_init = _load_json(DEFAULT_DET_INIT) if DEFAULT_DET_INIT.exists() else {}

    pre_fallbacks: list[dict[str, Any]] = []
    det_fallbacks: list[dict[str, Any]] = []
    seg_fallbacks: list[dict[str, Any]] = []
    pre_params = _flatten_r4tun_preprocessing(sample_pre, sample_unf, fallback_pre, pre_fallbacks)
    det_params = _translate_detection(sample_det, fallback_det, fallback_det_init, det_fallbacks)
    seg_params = _translate_segmentation(sample_sam, fallback_seg, seg_fallbacks)

    translation_manifest = {
        "scope": scope,
        "hard_constraint": "non_adaptive_static_baseline_no_optimization",
        "static_segment_count": STATIC_SEGMENT_COUNT,
        "static_segment_order": STATIC_SEGMENT_ORDER,
        "sources": {
            "r4tun_preprocessing": str(R4TUN_PRE.relative_to(REPO_ROOT)),
            "r4tun_unfolding": str(R4TUN_UNF.relative_to(REPO_ROOT)),
            "r4tun_detection": str(R4TUN_DET.relative_to(REPO_ROOT)),
            "r4tun_sam": str(R4TUN_SAM.relative_to(REPO_ROOT)),
            "fallback_pre": str(DEFAULT_PRE.relative_to(REPO_ROOT)),
            "fallback_det": str(DEFAULT_DET.relative_to(REPO_ROOT)),
            "fallback_seg": str(DEFAULT_SEG.relative_to(REPO_ROOT)),
        },
        "fallbacks": {
            "preprocessing": pre_fallbacks,
            "detection": det_fallbacks,
            "segmentation": seg_fallbacks,
        },
    }
    _write_json(TRANSLATION_JSON, translation_manifest)

    rows: list[dict[str, Any]] = []
    for _, rec in panel.iterrows():
        ring_key = str(rec["ring_key"])
        row = _run_ring(ring_key, clean=clean, pre=pre_params, det=det_params, seg=seg_params)
        row["family"] = int(rec["family"]) if "family" in rec and pd.notna(rec["family"]) else None
        row["panel_idx"] = int(rec["panel_idx"]) if "panel_idx" in rec and pd.notna(rec["panel_idx"]) else None
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("panel_idx").reset_index(drop=True)
    out_df.to_csv(SCOREBOARD_CSV, index=False)
    _write_failures(out_df)
    _write_json(SUMMARY_JSON, _summarize(out_df))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run static r4tun baseline on held-out rings (schema compatibility only).")
    parser.add_argument("--scope", choices=["smoke", "all50"], default="all50")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete per-ring sandbox folder before running.")
    args = parser.parse_args()
    return run(scope=args.scope, clean=not args.no_clean)


if __name__ == "__main__":
    raise SystemExit(main())

