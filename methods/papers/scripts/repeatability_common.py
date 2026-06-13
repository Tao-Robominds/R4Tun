"""Shared helpers for LLM repeatability experiment (run 1 vs run 2)."""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

ABLATION_FOLDER = "memory+state+knowledge"
COND_TAG = "m_s_k"
PARAM_SUFFIX = "_m_s_k_"

MODELS = ["opus4.6", "gpt5.4", "gemini3flash"]
MODEL_TO_VENDOR = {
    "opus4.6": "anthropic",
    "gpt5.4": "gpt",
    "gemini3flash": "gemini",
}
HARVEST_TUNNELS = ["4-4", "5-3", "5-4"]

ORCHESTRATORS = {
    "opus4.6": "run_memory_state_knowledge.py",
    "gpt5.4": "run_memory_state_knowledge_gpt.py",
    "gemini3flash": "run_memory_state_knowledge_gemini.py",
}

STAGE_FILES = [
    "unfolding",
    "denoising",
    "enhancing",
    "detecting",
    "sam",
]

# 18 critical parameters (Table 7) mapped to flat keys stage.json_key
CRITICAL_FLAT_KEYS = [
    "unfolding.diameter",
    "denoising.mask_r_low",
    "denoising.mask_r_high",
    "denoising.default_cutoff_z",
    "denoising.z_step",
    "denoising.smoothing_window_size",
    "denoising.smoothing_offset",
    "denoising.grad_threshold",
    "denoising.y_step",
    "enhancing.inter_radius",
    "enhancing.upsampling_stage1_target_distance",
    "enhancing.curvature_threshold",
    "enhancing.depth_threshold_low",
    "enhancing.depth_threshold_high",
    "detecting.hough_threshold_oblique",
    "detecting.hough_threshold_horizontal",
    "detecting.hough_threshold_vertical",
    "sam.processing.padding",
]

PARAM_BASE = REPO_ROOT / "agents" / "ablation" / ABLATION_FOLDER / "parameters"


def get_tunnel_ids() -> list[str]:
    if not PARAM_BASE.exists():
        return []
    return sorted(
        d.name for d in PARAM_BASE.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def param_json_name(stage: str, model: str) -> str:
    return f"parameters_{stage}{PARAM_SUFFIX}{model}.json"


def vendor_data_dir(tunnel: str, model: str) -> Path:
    vendor = MODEL_TO_VENDOR[model]
    return REPO_ROOT / "data" / f"ablation_{vendor}" / ABLATION_FOLDER / tunnel


def std_data_dir(tunnel: str) -> Path:
    return REPO_ROOT / "data" / "ablation" / ABLATION_FOLDER / tunnel


def run1_dir(tunnel: str, model: str) -> Path:
    return REPO_ROOT / "logs" / tunnel / "repeatability" / "run1" / model


def run2_harvested_dir(tunnel: str, model: str) -> Path:
    return REPO_ROOT / "logs" / tunnel / "repeatability" / "run2_harvested" / model


def find_performance_md(base: Path) -> Path | None:
    for name in ("evaluation", "evaluation_7", "evaluation_6"):
        perf = base / name / "performance.md"
        if perf.exists():
            return perf
    return None


def extract_miou(base: Path) -> float | None:
    perf = find_performance_md(base)
    if perf is None:
        perf = base / "data" / "evaluation" / "performance.md"
    if not perf.exists():
        perf = base / "performance.md"
    if not perf.exists():
        return None
    text = perf.read_text()
    for pat in (
        r"Mean IoU \(mIoU\):\s*([\d.]+)",
        r"mIoU\D*([0-9.]+)",
    ):
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None


def load_flat_params(param_dir: Path, model: str) -> dict[str, float]:
    flat: dict[str, float] = {}
    if not param_dir.exists():
        return flat
    for stage in STAGE_FILES:
        pf = param_dir / param_json_name(stage, model)
        if not pf.exists():
            continue
        try:
            data = json.loads(pf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for key, val in data.items():
            if isinstance(val, (int, float)):
                flat[f"{stage}.{key}"] = float(val)
            elif key == "processing" and isinstance(val, dict):
                if "padding" in val and isinstance(val["padding"], (int, float)):
                    flat[f"{stage}.processing.padding"] = float(val["padding"])
    return flat


def critical_param_stats(run_a: dict[str, float], run_b: dict[str, float]) -> tuple[int, int, float]:
    """Return (n_identical, n_compared, pct_identical) for critical params."""
    compared = 0
    identical = 0
    for key in CRITICAL_FLAT_KEYS:
        if key not in run_a or key not in run_b:
            continue
        compared += 1
        if run_a[key] == run_b[key]:
            identical += 1
    pct = 100.0 * identical / compared if compared else 0.0
    return identical, compared, pct


def params_identical(run_a: dict[str, float], run_b: dict[str, float]) -> bool:
    keys = set(run_a) | set(run_b)
    if not keys:
        return False
    for k in keys:
        if run_a.get(k) != run_b.get(k):
            return False
    return True


def copy_params_tree(src: Path, dst: Path, model: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for stage in STAGE_FILES:
        name = param_json_name(stage, model)
        if (src / name).exists():
            shutil_copy(src / name, dst / name)


def shutil_copy(src: Path, dst: Path) -> None:
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_performance(src_base: Path, dst_base: Path) -> None:
    import shutil
    perf = find_performance_md(src_base)
    if perf is None:
        return
    rel = perf.relative_to(src_base)
    out = dst_base / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(perf, out)


def has_run2(tunnel: str, model: str) -> bool:
    rep = REPO_ROOT / "logs" / tunnel / "repeatability"
    if run2_harvested_dir(tunnel, model).exists():
        return True
    for p in rep.glob("run2_*/" + model):
        if (p / "parameters").exists() or any(
            (p / f"parameters_{s}{PARAM_SUFFIX}{model}.json").exists() for s in STAGE_FILES
        ):
            params = p / "parameters"
            if params.exists() or load_flat_params(p, model):
                return True
    return False


def latest_run2_dir(tunnel: str, model: str) -> Path | None:
    rep = REPO_ROOT / "logs" / tunnel / "repeatability"
    harvested = run2_harvested_dir(tunnel, model)
    if harvested.exists():
        return harvested
    candidates = sorted(rep.glob(f"run2_*/{model}"))
    for p in reversed(candidates):
        if load_flat_params(p / "parameters", model) or load_flat_params(p, model):
            return p
    return None


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return st.mean(values), st.pstdev(values)
