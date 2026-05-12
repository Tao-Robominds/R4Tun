from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / "venv" / "bin" / "python"
PRE_CLI = REPO_ROOT / "agents" / "1_preprocessing" / "1_preprocessing.py"

RUN_ROOT = REPO_ROOT / "logs" / "v5_t45_depth_contract_v1"
SRC_STAGE = REPO_ROOT / "stages" / "v4" / "logs" / "v4_remaining_40_v1"
PANEL = REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv"
DEFAULT_PRE = (
    REPO_ROOT
    / "agents"
    / "1_preprocessing"
    / "parameters"
    / "_default_irregular"
    / "parameters_preprocessing.json"
)

DEPTH_GATE_THRESHOLDS = {
    "min_finite_ratio": 0.60,
    "min_row_nonempty_ratio": 0.90,
    "max_largest_empty_vertical_gap_frac": 0.08,
}

PROTECTED_PREFIXES = (
    REPO_ROOT / "data" / "ablation",
    REPO_ROOT / "data" / "bo",
    REPO_ROOT / "data" / "baseline",
    REPO_ROOT / "data" / "preprocessing_qa",
    REPO_ROOT / "data" / "represents",
    REPO_ROOT / "logs" / "context_preprocessing_v1",
    REPO_ROOT / "r4tun" / "data",
    REPO_ROOT / "r4tun" / "references",
    REPO_ROOT / "methods" / "plans" / "output",
    REPO_ROOT / "stages" / "v4",
)

GATE_RINGS = ["4-1/r110", "5-1/r118"]


def assert_writable(path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to((REPO_ROOT / "logs").resolve())
    except ValueError as exc:
        raise ValueError(f"Output must be under logs/: {resolved}") from exc
    for prefix in PROTECTED_PREFIXES:
        if not prefix.exists():
            continue
        pref = prefix.resolve()
        if resolved == pref:
            raise ValueError(f"Protected output path: {resolved}")
        try:
            resolved.relative_to(pref)
            raise ValueError(f"Protected output path: {resolved}")
        except ValueError:
            pass


def parse_ring_key(ring_key: str) -> tuple[str, int]:
    tunnel_id, ring_id = ring_key.split("/")
    return tunnel_id, int(ring_id.lstrip("r"))


def ring_dir(root: Path, ring_key: str) -> Path:
    tunnel_id, ring_id = parse_ring_key(ring_key)
    return root / tunnel_id / f"r{ring_id}"


def load_t45_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL)
    panel = panel[panel["family"].astype(int).isin([4, 5])].copy()
    panel["segment_count"] = 6
    panel["segmentation_ontology"] = "k_bearing"
    return panel.sort_values(["family", "tunnel_id", "ring_id"]).reset_index(drop=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run(cmd: list[str], log_path: Path, timeout_sec: float = 1800.0) -> None:
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
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _ensure_pre_params(dst_ring: Path) -> None:
    pre_path = dst_ring / "parameters_preprocessing.json"
    if pre_path.exists():
        return
    if not DEFAULT_PRE.exists():
        raise FileNotFoundError(f"Missing default preprocessing parameters: {DEFAULT_PRE}")
    pre_path.write_text(DEFAULT_PRE.read_text(encoding="utf-8"), encoding="utf-8")


def stage_ring(ring_key: str, *, clean: bool = True) -> Path:
    src = ring_dir(SRC_STAGE, ring_key)
    dst = ring_dir(RUN_ROOT, ring_key)
    if clean and dst.exists():
        shutil.rmtree(dst)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)
    _ensure_pre_params(dst)
    (dst / "logs").mkdir(parents=True, exist_ok=True)
    return dst


def apply_r4tun_depth_contract(ring_path: Path, *, interpolation_window: int = 9) -> dict[str, Any]:
    pre_path = ring_path / "parameters_preprocessing.json"
    params = json.loads(pre_path.read_text(encoding="utf-8"))
    gravity_anchor = params.get("gravity_anchor")
    if not isinstance(gravity_anchor, dict):
        gravity_anchor = {"n_bins": 360}
    gravity_anchor["enabled"] = True
    gravity_anchor.setdefault("n_bins", 360)
    params["gravity_anchor"] = gravity_anchor
    params.pop("gravity_anchor_enabled", None)

    params["depth_height_mode"] = "observed_gap_aligned"
    params["depth_map_resolution"] = 0.005
    params["interpolation_window"] = int(interpolation_window)
    params["outlier_high_density_ring_start"] = 0
    params["outlier_high_density_ring_end"] = 5
    params["n_segment_start"] = 0
    params["n_segment_end"] = 5
    params["outlier_interpolation_radius"] = 0.06
    params["inter_radius"] = 0.06
    params["outlier_num_interpolations"] = 2
    params["num_interpolations"] = 2
    params["outlier_duplicate_threshold"] = 0.02
    params["duplicate_threshold"] = 0.02
    write_json(pre_path, params)
    return params


def preprocess_ring(ring_key: str) -> Path:
    ring_path = stage_ring(ring_key)
    tunnel_id, ring_id = parse_ring_key(ring_key)
    for interpolation_window in (9, 17, 31, 51):
        apply_r4tun_depth_contract(ring_path, interpolation_window=interpolation_window)
        run(
            [str(VENV_PY), str(PRE_CLI), tunnel_id, str(ring_id), "--data-dir", str(RUN_ROOT)],
            ring_path / "logs" / f"pre_depth_contract_w{interpolation_window}.log",
        )
        audit = audit_depth_map(ring_key)
        if bool(audit["depth_gate_pass"]):
            (ring_path / "depth_contract_selected.json").write_text(
                json.dumps(
                    {
                        "ring_key": ring_key,
                        "interpolation_window": interpolation_window,
                        "audit": audit,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return ring_path
    (ring_path / "depth_contract_selected.json").write_text(
        json.dumps(
            {
                "ring_key": ring_key,
                "interpolation_window": 51,
                "audit": audit_depth_map(ring_key),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ring_path


def _largest_false_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in mask:
        if value:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return int(best)


def _audit_depth_array(arr: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(arr)
    rows_nonempty = finite.any(axis=1)
    largest_gap = _largest_false_run(rows_nonempty)
    height = max(1, int(arr.shape[0]))
    finite_ratio = float(finite.mean())
    row_nonempty_ratio = float(rows_nonempty.mean())
    largest_gap_frac = float(largest_gap / height)
    reasons: list[str] = []
    if finite_ratio < DEPTH_GATE_THRESHOLDS["min_finite_ratio"]:
        reasons.append("finite_ratio_low")
    if row_nonempty_ratio < DEPTH_GATE_THRESHOLDS["min_row_nonempty_ratio"]:
        reasons.append("row_nonempty_ratio_low")
    if largest_gap_frac > DEPTH_GATE_THRESHOLDS["max_largest_empty_vertical_gap_frac"]:
        reasons.append("large_empty_vertical_gap")
    return {
        "height_px": int(arr.shape[0]),
        "width_px": int(arr.shape[1]) if arr.ndim == 2 else None,
        "finite_ratio": finite_ratio,
        "row_nonempty_ratio": row_nonempty_ratio,
        "largest_empty_vertical_gap_px": int(largest_gap),
        "largest_empty_vertical_gap_frac": largest_gap_frac,
        "depth_gate_pass": not reasons,
        "depth_gate_reason": "pass" if not reasons else ";".join(reasons),
    }


def audit_source_depth_map(ring_key: str) -> dict[str, Any]:
    path = ring_dir(SRC_STAGE, ring_key) / "depth_map.npy"
    rec: dict[str, Any] = {
        "ring_key": ring_key,
        "depth_map_path": str(path.relative_to(REPO_ROOT)),
        "depth_map_png": str((path.with_suffix(".png")).relative_to(REPO_ROOT)),
        "selected_interpolation_window": None,
    }
    if not path.exists():
        rec.update(
            {
                "height_px": None,
                "width_px": None,
                "finite_ratio": None,
                "row_nonempty_ratio": None,
                "largest_empty_vertical_gap_px": None,
                "largest_empty_vertical_gap_frac": None,
                "depth_gate_pass": False,
                "depth_gate_reason": "missing_depth_map",
            }
        )
        return rec
    rec.update(_audit_depth_array(np.load(path)))
    return rec


def audit_depth_map(ring_key: str) -> dict[str, Any]:
    path = ring_dir(RUN_ROOT, ring_key) / "depth_map.npy"
    selected_path = ring_dir(RUN_ROOT, ring_key) / "depth_contract_selected.json"
    selected_window = None
    if selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        selected_window = selected.get("interpolation_window")
    rec: dict[str, Any] = {
        "ring_key": ring_key,
        "depth_map_path": str(path.relative_to(REPO_ROOT)),
        "depth_map_png": str((path.with_suffix(".png")).relative_to(REPO_ROOT)),
        "selected_interpolation_window": selected_window,
    }
    if not path.exists():
        rec.update(
            {
                "height_px": None,
                "width_px": None,
                "finite_ratio": None,
                "row_nonempty_ratio": None,
                "largest_empty_vertical_gap_px": None,
                "largest_empty_vertical_gap_frac": None,
                "depth_gate_pass": False,
                "depth_gate_reason": "missing_depth_map",
            }
        )
        return rec
    rec.update(_audit_depth_array(np.load(path)))
    return rec


def audit_many(ring_keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame([audit_depth_map(ring_key) for ring_key in ring_keys])


def audit_many_source(ring_keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame([audit_source_depth_map(ring_key) for ring_key in ring_keys])


def write_summary(name: str, audit: pd.DataFrame) -> dict[str, Any]:
    failed = audit[~audit["depth_gate_pass"].astype(bool)].copy()
    summary = {
        "run_root": str(RUN_ROOT),
        "gate": name,
        "thresholds": DEPTH_GATE_THRESHOLDS,
        "n_rings": int(len(audit)),
        "all_depth_maps_pass": bool(failed.empty),
        "failed_rings": failed["ring_key"].astype(str).tolist(),
    }
    write_json(RUN_ROOT / f"{name}_summary.json", summary)
    return summary
