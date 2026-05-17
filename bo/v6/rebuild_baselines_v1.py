from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / "venv" / "bin" / "python"
PANEL_CSV = REPO_ROOT / "data" / "v6" / "_manifests" / "data_v6_50ring_calibration_panel.csv"
DATA_V6 = REPO_ROOT / "data" / "v6"

SAM_STATIC_ROOT = REPO_ROOT / "logs" / "v6_sam4tun_static_baseline_v1"
DETERMINISTIC_ROOT = REPO_ROOT / "logs" / "v6_deterministic_baseline_v1"

PRE_CLI = REPO_ROOT / "agents" / "1_preprocessing" / "1_preprocessing.py"
DET_CLI = REPO_ROOT / "agents" / "2_detection" / "2_detection.py"
SEG_CLI = REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py"
EVAL_CLI = REPO_ROOT / "agents" / "evaluation.py"

STATIC_HELPER_DIR = REPO_ROOT / "bo" / "v5"
if str(STATIC_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(STATIC_HELPER_DIR))

import run_static_r4tun_baseline_v1 as static_base  # noqa: E402


STATIC_SEGMENT_COUNT = 6
SMOKE_RINGS = {"4-6/r275"}

PROTECTED_PREFIXES = (
    REPO_ROOT / "data" / "ablation",
    REPO_ROOT / "data" / "bo",
    REPO_ROOT / "data" / "baseline",
    REPO_ROOT / "data" / "preprocessing_qa",
    REPO_ROOT / "data" / "represents",
    REPO_ROOT / "data" / "rings",
    REPO_ROOT / "data" / "subsets",
    REPO_ROOT / "logs" / "context_preprocessing_v1",
    REPO_ROOT / "r4tun" / "data",
    REPO_ROOT / "r4tun" / "references",
    REPO_ROOT / "methods" / "plans" / "output",
    REPO_ROOT / "stages" / "v4",
)


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _parse_ring_key(ring_key: str) -> tuple[str, int]:
    tunnel_id, ring = ring_key.split("/")
    return tunnel_id, int(ring.removeprefix("r"))


def _ring_dir(root: Path, ring_key: str) -> Path:
    tunnel_id, ring_id = _parse_ring_key(ring_key)
    return root / tunnel_id / f"r{ring_id}"


def _load_panel(scope: str) -> pd.DataFrame:
    panel = pd.read_csv(PANEL_CSV)
    if scope == "smoke":
        panel = panel[panel["ring_key"].astype(str).isin(SMOKE_RINGS)].copy()
    if panel.empty:
        raise RuntimeError(f"No rings selected for scope={scope}")
    return panel.sort_values("panel_idx").reset_index(drop=True)


def _run_cmd(cmd: list[str], log_path: Path, timeout_sec: float = 1800.0) -> tuple[bool, str | None]:
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
        return False, f"exit_{proc.returncode}"
    return True, None


def _static_params() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    pre_fallbacks: list[dict[str, Any]] = []
    det_fallbacks: list[dict[str, Any]] = []
    seg_fallbacks: list[dict[str, Any]] = []
    pre_params = static_base._flatten_r4tun_preprocessing(
        static_base._load_json(static_base.R4TUN_PRE),
        static_base._load_json(static_base.R4TUN_UNF),
        static_base._load_json(static_base.DEFAULT_PRE),
        pre_fallbacks,
    )
    det_params = static_base._translate_detection(
        static_base._load_json(static_base.R4TUN_DET),
        static_base._load_json(static_base.DEFAULT_DET),
        static_base._load_json(static_base.DEFAULT_DET_INIT) if static_base.DEFAULT_DET_INIT.exists() else {},
        det_fallbacks,
    )
    seg_params = static_base._translate_segmentation(
        static_base._load_json(static_base.R4TUN_SAM),
        static_base._load_json(static_base.DEFAULT_SEG),
        seg_fallbacks,
    )
    manifest = {
        "static_segment_count": STATIC_SEGMENT_COUNT,
        "paper_names": {
            "sam4tun_static": "original SAM4Tun static parameter baseline",
            "deterministic_baseline": "SAM4Tun static + verified v6 preprocessing + fixed deterministic downstream rules",
        },
        "fallbacks": {
            "preprocessing": pre_fallbacks,
            "detection": det_fallbacks,
            "segmentation": seg_fallbacks,
        },
    }
    return pre_params, det_params, seg_params, manifest


def _depth_audit(ring_dir: Path) -> dict[str, Any]:
    depth = ring_dir / "depth_map.npy"
    row: dict[str, Any] = {
        "depth_map_path": str(depth.relative_to(REPO_ROOT)) if depth.exists() else None,
        "finite_ratio": None,
        "row_nonempty_ratio": None,
        "largest_empty_vertical_gap_px": None,
        "largest_empty_vertical_gap_frac": None,
    }
    if not depth.exists():
        return row
    arr = np.load(depth)
    finite = np.isfinite(arr) & (arr > 0)
    row_ok = finite.any(axis=1)
    best = cur = 0
    for value in row_ok:
        if value:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    row["finite_ratio"] = float(finite.mean())
    row["row_nonempty_ratio"] = float(row_ok.mean())
    row["largest_empty_vertical_gap_px"] = int(best)
    row["largest_empty_vertical_gap_frac"] = float(best / max(1, int(arr.shape[0])))
    return row


def _compute_metrics(final_csv: Path, max_class: int = STATIC_SEGMENT_COUNT) -> tuple[float | None, float | None]:
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
    for cls_id in range(1, max_class + 1):
        g = gt == cls_id
        p = pred == cls_id
        union = np.logical_or(g, p).sum()
        if union == 0:
            continue
        ious.append(float(np.logical_and(g, p).sum() / union))
    return oa, float(np.mean(ious)) if ious else None


def _copy_data_v6_ring(dst_root: Path, ring_key: str, det_params: dict[str, Any], seg_params: dict[str, Any], clean: bool) -> Path:
    src = _ring_dir(DATA_V6, ring_key)
    if not src.exists():
        raise FileNotFoundError(f"Missing verified data/v6 ring: {src}")
    dst = _ring_dir(dst_root, ring_key)
    if clean and dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)
        # Link verified preprocessing artifacts instead of duplicating multi-GB CSVs.
        # Downstream detection/segmentation writes new outputs into dst.
        for item in src.iterdir():
            target = dst / item.name
            if item.is_file():
                target.symlink_to(os.path.relpath(item, dst))
            elif item.is_dir():
                shutil.copytree(item, target, symlinks=True)
    _write_json(dst / "parameters_detection.json", det_params)
    _write_json(dst / "parameters_segmentation.json", seg_params)
    (dst / "logs").mkdir(exist_ok=True)
    return dst


def _run_downstream_ring(dst_root: Path, ring_key: str, det_params: dict[str, Any], seg_params: dict[str, Any], clean: bool) -> dict[str, Any]:
    tunnel_id, ring_id = _parse_ring_key(ring_key)
    ring_path = _copy_data_v6_ring(dst_root, ring_key, det_params, seg_params, clean)
    row: dict[str, Any] = {"ring_key": ring_key, "tunnel_id": tunnel_id, "ring_id": ring_id}
    ok, err = _run_cmd([str(VENV_PY), str(DET_CLI), tunnel_id, str(ring_id), "--data-dir", str(dst_root)], ring_path / "logs" / "stage2_detection.log")
    row["stage2_ok"] = ok
    if not ok:
        row.update({"error_stage": "detection", "error": err})
        row.update(_depth_audit(ring_path))
        return row
    ok, err = _run_cmd([str(VENV_PY), str(SEG_CLI), tunnel_id, str(ring_id), "--data-dir", str(dst_root)], ring_path / "logs" / "stage3_segmentation.log")
    row["stage3_ok"] = ok
    if not ok:
        row.update({"error_stage": "segmentation", "error": err})
        row.update(_depth_audit(ring_path))
        return row
    ok, err = _run_cmd([str(VENV_PY), str(EVAL_CLI), tunnel_id, str(ring_id), "--data-dir", str(dst_root), "--segments", str(STATIC_SEGMENT_COUNT)], ring_path / "logs" / "stage4_evaluation.log")
    row["stage4_ok"] = ok
    row["error_stage"] = None if ok else "evaluation"
    row["error"] = None if ok else err
    final_csv = ring_path / "final.csv"
    oa, miou = _compute_metrics(final_csv)
    row["oa"] = oa
    row["miou"] = miou
    row["segment_count"] = STATIC_SEGMENT_COUNT
    row["final_csv"] = str(final_csv.relative_to(REPO_ROOT)) if final_csv.exists() else None
    row.update(_depth_audit(ring_path))
    return row


def _run_sam_static_ring(dst_root: Path, ring_key: str, pre_params: dict[str, Any], det_params: dict[str, Any], seg_params: dict[str, Any], clean: bool) -> dict[str, Any]:
    # Reuse the v5 static helper's full raw-point-cloud pipeline, but direct it to v6 roots.
    original_root = static_base.RUN_ROOT
    try:
        static_base.RUN_ROOT = dst_root
        return static_base._run_ring(ring_key, clean=clean, pre=pre_params, det=det_params, seg=seg_params)
    finally:
        static_base.RUN_ROOT = original_root


def _write_outputs(root: Path, panel: pd.DataFrame, rows: list[dict[str, Any]], baseline_name: str, manifest: dict[str, Any]) -> None:
    df = pd.DataFrame(rows).merge(panel[["ring_key", "panel_idx", "family", "is_replacement", "replaces"]], on="ring_key", how="left")
    df = df.sort_values("panel_idx").reset_index(drop=True)
    df.to_csv(root / f"{baseline_name}_scoreboard.csv", index=False)
    miou = pd.to_numeric(df["miou"], errors="coerce")
    ok = df["error"].isna() if "error" in df.columns else miou.notna()
    summary = {
        "baseline_name": baseline_name,
        "run_root": str(root.relative_to(REPO_ROOT)),
        "rows": int(len(df)),
        "successful_rows": int(ok.sum()),
        "failed_rows": int((~ok).sum()),
        "mean_miou_successful": float(miou[ok].mean()) if ok.any() else None,
        "median_miou_successful": float(miou[ok].median()) if ok.any() else None,
        "mean_miou_all_rows": float(miou.mean()) if miou.notna().any() else None,
        "mean_miou_by_family": [
            {"family": int(fam), "mean_miou": float(g.mean()) if g.notna().any() else None}
            for fam, g in df.assign(miou_num=miou).groupby("family")["miou_num"]
        ],
        "manifest": manifest,
    }
    _write_json(root / f"{baseline_name}_summary.json", summary)


def run(which: str, scope: str, clean: bool) -> None:
    panel = _load_panel(scope)
    pre_params, det_params, seg_params, manifest = _static_params()
    if which in {"sam_static", "both"}:
        _assert_writable(SAM_STATIC_ROOT)
        SAM_STATIC_ROOT.mkdir(parents=True, exist_ok=True)
        _write_json(SAM_STATIC_ROOT / "baseline_manifest.json", {**manifest, "baseline": "sam4tun_static", "panel": str(PANEL_CSV.relative_to(REPO_ROOT))})
        rows = [_run_sam_static_ring(SAM_STATIC_ROOT, str(rec["ring_key"]), pre_params, det_params, seg_params, clean) for _, rec in panel.iterrows()]
        _write_outputs(SAM_STATIC_ROOT, panel, rows, "sam4tun_static", manifest)
    if which in {"deterministic", "both"}:
        _assert_writable(DETERMINISTIC_ROOT)
        DETERMINISTIC_ROOT.mkdir(parents=True, exist_ok=True)
        det_manifest = {
            **manifest,
            "baseline": "deterministic_baseline",
            "preprocessing_source": "data/v6 verified preprocessing artifacts",
            "panel": str(PANEL_CSV.relative_to(REPO_ROOT)),
        }
        _write_json(DETERMINISTIC_ROOT / "baseline_manifest.json", det_manifest)
        rows = [_run_downstream_ring(DETERMINISTIC_ROOT, str(rec["ring_key"]), det_params, seg_params, clean) for _, rec in panel.iterrows()]
        _write_outputs(DETERMINISTIC_ROOT, panel, rows, "deterministic_baseline", det_manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild v6 SAM4Tun static and deterministic baselines.")
    parser.add_argument("--which", choices=["sam_static", "deterministic", "both"], default="both")
    parser.add_argument("--scope", choices=["smoke", "all50"], default="smoke")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()
    run(which=args.which, scope=args.scope, clean=not args.no_clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
