#!/usr/bin/env python3
"""Graded hint-mode ladder for regular tunnels (1-*, 2-*).

Reuses frozen preprocessing from data/ablation_anthropic; reruns detecting→SAM→eval only.

Usage:
    python3 methods/papers/scripts/run_regular_hint_loop.py --level L5 --tunnel 2-2
    python3 methods/papers/scripts/run_regular_hint_loop.py --all-levels --gate
    python3 methods/papers/scripts/run_regular_hint_loop.py --all-levels --all-tunnels
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from repeatability_common import (  # noqa: E402
    ABLATION_FOLDER,
    extract_miou,
    param_json_name,
    PARAM_BASE,
)
from regular_hint_lib import hint_level_to_mode  # noqa: E402

PYTHON = str(REPO_ROOT / "venv" / "bin" / "python")
if not Path(PYTHON).is_file():
    PYTHON = sys.executable

MODEL = "opus4.6"
ABLATION = "m_s_k"
TS = os.environ.get("REGULAR_HINT_LOOP_TS") or datetime.now().strftime("%Y%m%d_%H%M%S")

REGULAR_TUNNELS = [
    "1-1", "1-2", "1-3", "1-4", "1-5",
    "2-1", "2-2", "2-3", "2-4", "2-5",
]

HINT_LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
GATE_TUNNELS = ["2-2", "1-3"]  # T1 gate: 1-3 (not 1-4 — pathological weak detection)

ANTHROPIC_SRC = REPO_ROOT / "data" / "ablation_anthropic" / ABLATION_FOLDER
LOOP_ROOT = REPO_ROOT / "data" / "regular_hint_loop"
LOG_ROOT = REPO_ROOT / "logs" / "regular_hint_loop"

UPSTREAM_FILES = [
    "enhanced.csv",
    "depth_map.png",
    "depth_map_outlier.npy",
    "pixel_to_point.pkl",
    "ring_count.txt",
    "unwrapped.csv",
    "denoised.csv",
    "final.csv",
]

CALIB_FILE = "detected_calib.csv"

AGENTS_REGULAR = REPO_ROOT / "agents_regular"
PARAM_FALLBACK = REPO_ROOT / "agents_regular" / "ablation" / ABLATION_FOLDER / "parameters"


def load_base_detecting_params(tunnel: str) -> dict:
    candidates = [
        PARAM_BASE / tunnel / param_json_name("detecting", MODEL),
        PARAM_FALLBACK / tunnel / param_json_name("detecting", MODEL),
    ]
    for path in candidates:
        if path.is_file():
            data = json.loads(path.read_text())
            if data.get("binary_threshold") is not None:
                return data
    return {}


def _ensure_venv_on_path() -> None:
    venv_site = (
        REPO_ROOT / "venv" / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    sam_root = REPO_ROOT / "sam4tun" / "segment-anything"
    parts: list[str] = []
    if venv_site.is_dir():
        parts.append(str(venv_site))
    if sam_root.is_dir():
        parts.append(str(sam_root))
    if parts:
        cur = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.pathsep.join(parts + ([cur] if cur else []))


def level_out_prefix(level: str) -> Path:
    return LOOP_ROOT / level


def seed_tunnel(level: str, tunnel: str) -> Path:
    """Symlink upstream artefacts from ablation_anthropic (no copy)."""
    dst = level_out_prefix(level) / tunnel
    src = ANTHROPIC_SRC / tunnel
    dst.mkdir(parents=True, exist_ok=True)
    for name in UPSTREAM_FILES:
        sp = src / name
        dp = dst / name
        if not sp.is_file():
            continue
        if dp.exists() or dp.is_symlink():
            dp.unlink()
        dp.symlink_to(sp.resolve())
    det = dst / "detected.csv"
    calib = dst / CALIB_FILE
    if det.exists() or det.is_symlink():
        if calib.exists() or calib.is_symlink():
            calib.unlink()
        calib.symlink_to(det.resolve())
    return dst


def inject_hint_params(tunnel: str, hint_mode: str) -> tuple[Path, dict | None]:
    """Patch detecting params with hint_mode; return path and backup dict."""
    param_path = PARAM_BASE / tunnel / param_json_name("detecting", MODEL)
    if param_path.is_file():
        backup = json.loads(param_path.read_text())
        if backup.get("binary_threshold") is None:
            backup = load_base_detecting_params(tunnel)
    else:
        backup = load_base_detecting_params(tunnel)
    params = dict(backup)
    params["hint_mode"] = hint_mode
    params.setdefault("k_consensus_version", "v3")
    params.setdefault("hint_y_levels", [1210, 1640])
    params.setdefault("hint_gt_k_rings", 2)
    params.setdefault("k_pattern_step_px", 431.87)
    param_path.parent.mkdir(parents=True, exist_ok=True)
    param_path.write_text(json.dumps(params, indent=2) + "\n")
    return param_path, backup


def restore_params(param_path: Path, backup: dict | None) -> None:
    if backup is None or backup.get("binary_threshold") is None:
        return
    param_path.write_text(json.dumps(backup, indent=2) + "\n")


def run_stage(script: str, tunnel: str, env: dict) -> subprocess.CompletedProcess:
    cmd = [
        PYTHON,
        str(AGENTS_REGULAR / script),
        tunnel,
        "--ablation",
        ABLATION,
    ]
    if script != "evaluation.py":
        cmd.extend(["--model", MODEL])
    else:
        cmd.extend(["--schema", "auto"])
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=1800,
    )


def run_tunnel_level(level: str, tunnel: str, skip_existing: bool = False) -> dict:
    hint_mode = hint_level_to_mode(level)
    out_prefix = level_out_prefix(level)
    tunnel_out = out_prefix / tunnel
    log_dir = LOG_ROOT / level / tunnel
    log_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "level": level,
        "hint_mode": hint_mode,
        "tunnel": tunnel,
        "miou": None,
        "status": "pending",
    }

    if skip_existing and extract_miou(tunnel_out) is not None:
        row["miou"] = extract_miou(tunnel_out)
        row["status"] = "skipped"
        return row

    seed_tunnel(level, tunnel)
    param_path, backup = inject_hint_params(tunnel, hint_mode)

    env = os.environ.copy()
    env["R4TUN_PIPELINE_OUT_PREFIX"] = str(out_prefix.relative_to(REPO_ROOT))
    env.setdefault("MPLBACKEND", "Agg")

    try:
        t0 = time.time()
        det = run_stage("detecting.py", tunnel, env)
        (log_dir / "detecting.log").write_text(
            f"exit={det.returncode}\nSTDOUT:\n{det.stdout}\nSTDERR:\n{det.stderr}"
        )
        if det.returncode != 0:
            row["status"] = "detecting_fail"
            return row

        sam = run_stage("sam.py", tunnel, env)
        (log_dir / "sam.log").write_text(
            f"exit={sam.returncode}\nSTDOUT:\n{sam.stdout}\nSTDERR:\n{sam.stderr}"
        )
        if sam.returncode != 0:
            row["status"] = "sam_fail"
            return row

        ev = run_stage("evaluation.py", tunnel, env)
        (log_dir / "evaluation.log").write_text(
            f"exit={ev.returncode}\nSTDOUT:\n{ev.stdout}\nSTDERR:\n{ev.stderr}"
        )
        if ev.returncode != 0:
            row["status"] = "eval_fail"
            return row

        row["miou"] = extract_miou(tunnel_out)
        row["status"] = "ok"
        row["elapsed_s"] = round(time.time() - t0, 1)
        print(f"  {level} {tunnel}: mIoU={row['miou']} ({row['elapsed_s']}s)")
    finally:
        restore_params(param_path, backup)

    return row


def append_csv(path: Path, row: dict) -> None:
    fields = ["level", "hint_mode", "tunnel", "miou", "status", "elapsed_s"]
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def count_passing(results: list[dict], threshold: float = 0.8) -> int:
    return sum(1 for r in results if r.get("miou") is not None and r["miou"] >= threshold)


def main() -> None:
    os.chdir(REPO_ROOT)
    _ensure_venv_on_path()

    parser = argparse.ArgumentParser(description="Regular tunnel hint ladder")
    parser.add_argument("--level", choices=HINT_LEVELS)
    parser.add_argument("--tunnel")
    parser.add_argument("--all-levels", action="store_true")
    parser.add_argument("--all-tunnels", action="store_true")
    parser.add_argument("--gate", action="store_true", help="Run gate tunnels only (2-2, 1-3)")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.gate:
        tunnels = GATE_TUNNELS
    elif args.all_tunnels:
        tunnels = REGULAR_TUNNELS
    elif args.tunnel:
        tunnels = [args.tunnel]
    else:
        tunnels = GATE_TUNNELS

    if args.all_levels:
        levels = HINT_LEVELS
    elif args.level:
        levels = [args.level]
    else:
        levels = ["L5"]

    summary_csv = LOG_ROOT / f"summary_{TS}.csv"
    all_rows: list[dict] = []

    for level in levels:
        print(f"\n=== Level {level} ({hint_level_to_mode(level)}) ===")
        for tunnel in tunnels:
            if tunnel not in REGULAR_TUNNELS:
                continue
            row = run_tunnel_level(level, tunnel, skip_existing=args.skip_existing)
            append_csv(summary_csv, row)
            all_rows.append(row)

    print(f"\nSummary CSV: {summary_csv}")
    for level in levels:
        lvl_rows = [r for r in all_rows if r["level"] == level]
        passing = count_passing(lvl_rows)
        mean_miou = None
        mious = [r["miou"] for r in lvl_rows if r.get("miou") is not None]
        if mious:
            mean_miou = sum(mious) / len(mious)
        print(f"  {level}: pass>={0.8}: {passing}/{len(lvl_rows)} mean_mIoU={mean_miou}")


if __name__ == "__main__":
    main()
