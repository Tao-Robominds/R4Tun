#!/usr/bin/env python3
"""
Regular-tunnel K-pattern consensus v2 experiment.

Swaps agents_regular/detecting.py + detecting docs, seeds run-1 upstream params,
removes detecting+sam params to force those stages to rerun with consensus code.

Usage:
    python3 methods/papers/scripts/run_regular_hint_v2.py --dry-run
    python3 methods/papers/scripts/run_regular_hint_v2.py --skip-existing
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from repeatability_common import (  # noqa: E402
    ABLATION_FOLDER,
    MODELS,
    ORCHESTRATORS,
    PARAM_BASE,
    copy_performance,
    extract_miou,
    param_json_name,
    run1_dir,
    std_data_dir,
    vendor_data_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
TS = os.environ.get("REGULAR_HINT_V2_TS") or datetime.now().strftime("%Y%m%d_%H%M%S")

REGULAR_TUNNELS = [
    "1-1", "1-2", "1-3", "1-4", "1-5",
    "2-1", "2-2", "2-3", "2-4", "2-5",
    "3-1-1", "3-1-2", "3-1-3",
]

LIVE_DETECTING_PY = REPO_ROOT / "agents" / "detecting.py"
HINT_DETECTING_PY = REPO_ROOT / "agents_regular" / "detecting.py"
LIVE_AGENTS = REPO_ROOT / "agents" / "ablation" / ABLATION_FOLDER / "agents" / "detecting"
HINT_AGENTS = REPO_ROOT / "agents_regular" / "ablation" / ABLATION_FOLDER / "agents" / "detecting"
DOC_NAMES = ("knowledge.md", "cot.md")
UPSTREAM_STAGES = ("unfolding", "denoising", "enhancing")
FORCE_RERUN_STAGES = ("detecting", "sam")


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


def hint_dir(tunnel: str, model: str) -> Path:
    return REPO_ROOT / "logs" / tunnel / "regular_hint_v2" / model


def has_hint_run(tunnel: str, model: str) -> bool:
    return extract_miou(hint_dir(tunnel, model)) is not None


@contextmanager
def swapped_v2_artifacts():
    backup_dir = REPO_ROOT / "logs" / f"regular_hint_v2_backup_{TS}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed_files: list[tuple[Path, Path]] = []
    try:
        if HINT_DETECTING_PY.exists():
            bak = backup_dir / "detecting.py"
            if LIVE_DETECTING_PY.exists():
                shutil.copy2(LIVE_DETECTING_PY, bak)
                backed_files.append((bak, LIVE_DETECTING_PY))
            shutil.copy2(HINT_DETECTING_PY, LIVE_DETECTING_PY)
            print("  swapped agents/detecting.py <- agents_regular")
        for name in DOC_NAMES:
            live = LIVE_AGENTS / name
            hint = HINT_AGENTS / name
            if hint.exists():
                bak = backup_dir / name
                if live.exists():
                    shutil.copy2(live, bak)
                    backed_files.append((bak, live))
                shutil.copy2(hint, live)
                print(f"  swapped {name} <- agents_regular")
        yield
    finally:
        for bak, live in backed_files:
            shutil.copy2(bak, live)
            print(f"  restored {live.name}")


def seed_upstream_and_clear_detecting_sam(tunnel: str, model: str, param_dir: Path) -> None:
    src = run1_dir(tunnel, model) / "parameters"
    param_dir.mkdir(parents=True, exist_ok=True)
    for stage in UPSTREAM_STAGES:
        name = param_json_name(stage, model)
        if (src / name).exists():
            shutil.copy2(src / name, param_dir / name)
    for stage in FORCE_RERUN_STAGES:
        pf = param_dir / param_json_name(stage, model)
        if pf.exists():
            pf.unlink()


def run_combo(tunnel: str, model: str, dry_run: bool) -> dict:
    vendor = vendor_data_dir(tunnel, model)
    std_data = std_data_dir(tunnel)
    param_dir = PARAM_BASE / tunnel
    log_dir = hint_dir(tunnel, model)
    log_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "tunnel": tunnel,
        "model": model,
        "baseline_miou": extract_miou(run1_dir(tunnel, model)),
        "hint_v1_miou": extract_miou(REPO_ROOT / "logs" / tunnel / "regular_hint" / model),
        "hint_v2_miou": None,
        "status": "pending",
    }

    print(f"\n{'='*70}\n  REGULAR HINT v2: {tunnel} {model}\n{'='*70}")

    if not vendor.exists():
        print(f"  ERROR: vendor data missing: {vendor}")
        row["status"] = "no_vendor_data"
        return row

    param_backup = log_dir / "_param_backup"
    std_backup = log_dir / "_std_data_backup"
    memory_char_backup = None
    memory_char_dir = REPO_ROOT / "data" / "ablation" / "memory" / tunnel / "characteristics"

    if param_dir.exists():
        if param_backup.exists():
            shutil.rmtree(param_backup)
        shutil.copytree(param_dir, param_backup)
    if std_data.exists():
        if std_backup.exists():
            shutil.rmtree(std_backup)
        shutil.copytree(std_data, std_backup)
        shutil.rmtree(std_data)

    shutil.copytree(vendor, std_data)
    seed_upstream_and_clear_detecting_sam(tunnel, model, param_dir)

    if not memory_char_dir.exists():
        for v in ("anthropic", "gpt", "gemini"):
            src = (
                REPO_ROOT / "data" / f"ablation_{v}" / "memory" / tunnel
                / "characteristics" / "raw_characteristics.json"
            )
            if src.exists():
                memory_char_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, memory_char_dir / "raw_characteristics.json")
                memory_char_backup = "created"
                break

    try:
        if dry_run:
            row["status"] = "dry_run"
            return row

        script = ORCHESTRATORS[model]
        cmd = [PYTHON, str(REPO_ROOT / script), tunnel, "--model", model]
        print(f"  Running: {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=1800,
        )
        elapsed = time.time() - t0
        (log_dir / "orchestrator.log").write_text(
            f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}\n"
        )
        print(f"  Orchestrator done in {elapsed:.0f}s (exit={result.returncode})")

        if result.returncode != 0:
            row["status"] = "orchestrator_fail"
            if param_dir.exists():
                snap = log_dir / "parameters"
                if snap.exists():
                    shutil.rmtree(snap)
                shutil.copytree(param_dir, snap)
            return row

        row["hint_v2_miou"] = extract_miou(std_data)
        row["status"] = "ok"
        print(f"  hint v2 mIoU: {row['hint_v2_miou']}")

        if param_dir.exists():
            snap = log_dir / "parameters"
            if snap.exists():
                shutil.rmtree(snap)
            shutil.copytree(param_dir, snap)
        if std_data.exists():
            data_snap = log_dir / "data"
            if data_snap.exists():
                shutil.rmtree(data_snap)
            shutil.copytree(std_data, data_snap)
        char_src = std_data / "characteristics" / "detected_characteristics.json"
        if char_src.exists():
            shutil.copy2(char_src, log_dir / "detected_characteristics.json")
        det_csv = std_data / "detected.csv"
        if det_csv.exists():
            shutil.copy2(det_csv, log_dir / "detected.csv")
        copy_performance(std_data, log_dir)

    finally:
        if std_data.exists():
            shutil.rmtree(std_data)
        if std_backup.exists():
            shutil.copytree(std_backup, std_data)
        if memory_char_backup == "created" and memory_char_dir.exists():
            shutil.rmtree(memory_char_dir.parent)
        if param_backup.exists():
            if param_dir.exists():
                shutil.rmtree(param_dir)
            shutil.copytree(param_backup, param_dir)

    return row


def append_summary(csv_path: Path, row: dict) -> None:
    fields = ["tunnel", "model", "baseline_miou", "hint_v1_miou", "hint_v2_miou", "status"]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fields})


def main() -> None:
    os.chdir(REPO_ROOT)
    _ensure_venv_on_path()

    parser = argparse.ArgumentParser(description="Regular-tunnel K-pattern consensus v2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--tunnel")
    parser.add_argument("--model", choices=MODELS, default="opus4.6")
    args = parser.parse_args()

    tunnels = [args.tunnel] if args.tunnel else REGULAR_TUNNELS
    model = args.model
    summary_csv = REPO_ROOT / "logs" / f"regular_hint_v2_{TS}_summary.csv"

    queue = []
    for tunnel in tunnels:
        if tunnel not in REGULAR_TUNNELS:
            continue
        if args.skip_existing and has_hint_run(tunnel, model):
            print(f"  skip existing: {tunnel} {model}")
            continue
        queue.append(tunnel)

    print(f"Queue: {len(queue)} tunnels, model={model} (TS={TS})")
    if args.dry_run:
        for t in queue:
            print(f"  would run: {t} {model}")
        return

    with swapped_v2_artifacts():
        for tunnel in queue:
            row = run_combo(tunnel, model, dry_run=False)
            append_summary(summary_csv, row)

    print(f"\nSummary: {summary_csv}")


if __name__ == "__main__":
    main()
