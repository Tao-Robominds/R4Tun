from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / "venv" / "bin" / "python"

SOURCE_DATA_ROOT = REPO_ROOT / "data" / "v6"
SOURCE_PARAM_ROOT = REPO_ROOT / "logs" / "v6_deterministic_baseline_v1"
SMOKE_ROOT = REPO_ROOT / "logs" / "v6_agents_parameterized_smoke_v1"

DET_CLI = REPO_ROOT / "agents" / "2_detection" / "2_detection.py"
SEG_CLI = REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py"
EVAL_CLI = REPO_ROOT / "agents" / "evaluation.py"

REQUIRED_LINKS = (
    "depth_map.png",
    "depth_map.npy",
    "depth_map_outlier.npy",
    "denoised.csv",
    "enhanced.csv",
    "pixel_to_point.pkl",
    "ring_count.txt",
    "parameters_preprocessing.json",
)
REQUIRED_OUTPUTS = (
    "depth_map.npy",
    "pixel_to_point.pkl",
    "all_segments.csv",
    "boundaries_per_ring.json",
    "final.csv",
    "evaluation/performance.md",
)


def _ring_dir(root: Path, tunnel_id: str, ring_id: int) -> Path:
    return root / tunnel_id / f"r{ring_id}"


def _safe_relpath(target: Path, start: Path) -> str:
    return os.path.relpath(str(target), str(start))


def _prepare_sandbox_ring(tunnel_id: str, ring_id: int, clean: bool) -> Path:
    src_ring = _ring_dir(SOURCE_DATA_ROOT, tunnel_id, ring_id)
    if not src_ring.exists():
        raise FileNotFoundError(f"Missing source ring under data/v6: {src_ring}")

    param_ring = _ring_dir(SOURCE_PARAM_ROOT, tunnel_id, ring_id)
    det_param = param_ring / "parameters_detection.json"
    seg_param = param_ring / "parameters_segmentation.json"
    if not det_param.exists() or not seg_param.exists():
        raise FileNotFoundError(
            f"Missing detection/segmentation params in {param_ring}. "
            "Expected parameters_detection.json and parameters_segmentation.json."
        )

    dst_ring = _ring_dir(SMOKE_ROOT, tunnel_id, ring_id)
    if clean and dst_ring.exists():
        shutil.rmtree(dst_ring)
    dst_ring.mkdir(parents=True, exist_ok=True)
    (dst_ring / "logs").mkdir(exist_ok=True)

    for name in REQUIRED_LINKS:
        src = src_ring / name
        if not src.exists():
            raise FileNotFoundError(f"Required source artifact missing: {src}")
        dst = dst_ring / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(_safe_relpath(src, dst_ring))

    shutil.copy2(det_param, dst_ring / "parameters_detection.json")
    shutil.copy2(seg_param, dst_ring / "parameters_segmentation.json")
    return dst_ring


def _run_cmd(cmd: list[str], log_path: Path) -> None:
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
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _parse_metrics(performance_md: Path) -> dict[str, float | None]:
    txt = performance_md.read_text(encoding="utf-8")

    def _extract(metric: str) -> float | None:
        pat = rf"\|\s*{re.escape(metric)}\s*\|\s*([0-9]*\.?[0-9]+)\s*\|"
        match = re.search(pat, txt)
        return float(match.group(1)) if match else None

    return {
        "oa": _extract("Overall Accuracy (OA)"),
        "f1_macro": _extract("F1 Score (macro)"),
        "miou": _extract("Mean IoU (mIoU)"),
    }


def _verify_outputs(ring_dir: Path) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for rel in REQUIRED_OUTPUTS:
        checks[rel] = (ring_dir / rel).exists()
    return checks


def run(tunnel_id: str, ring_id: int, clean: bool) -> dict[str, object]:
    ring_dir = _prepare_sandbox_ring(tunnel_id=tunnel_id, ring_id=ring_id, clean=clean)
    data_dir = str(SMOKE_ROOT)

    _run_cmd(
        [str(VENV_PY), str(DET_CLI), tunnel_id, str(ring_id), "--data-dir", data_dir],
        ring_dir / "logs" / "stage2_detection.log",
    )
    _run_cmd(
        [str(VENV_PY), str(SEG_CLI), tunnel_id, str(ring_id), "--data-dir", data_dir],
        ring_dir / "logs" / "stage3_segmentation.log",
    )
    _run_cmd(
        [str(VENV_PY), str(EVAL_CLI), tunnel_id, str(ring_id), "--data-dir", data_dir],
        ring_dir / "logs" / "stage4_evaluation.log",
    )

    checks = _verify_outputs(ring_dir)
    perf_path = ring_dir / "evaluation" / "performance.md"
    metrics = _parse_metrics(perf_path) if perf_path.exists() else {"oa": None, "f1_macro": None, "miou": None}
    all_present = all(checks.values())

    summary = {
        "ring_key": f"{tunnel_id}/r{ring_id}",
        "sandbox_path": str(ring_dir.relative_to(REPO_ROOT)),
        "data_source": str(_ring_dir(SOURCE_DATA_ROOT, tunnel_id, ring_id).relative_to(REPO_ROOT)),
        "params_source": str(_ring_dir(SOURCE_PARAM_ROOT, tunnel_id, ring_id).relative_to(REPO_ROOT)),
        "artifact_checks": checks,
        "all_required_artifacts_present": all_present,
        "metrics": metrics,
    }
    out_json = SMOKE_ROOT / "smoke_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-ring parameterized agents smoke test on data/v6.")
    parser.add_argument("--tunnel-id", default="4-6")
    parser.add_argument("--ring-id", type=int, default=275)
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    summary = run(tunnel_id=args.tunnel_id, ring_id=args.ring_id, clean=not args.no_clean)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
