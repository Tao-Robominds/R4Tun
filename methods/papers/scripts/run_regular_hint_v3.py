#!/usr/bin/env python3
"""
Continuous-tunnel v3 experiment (prompt-only detecting docs).

Swaps detecting knowledge.md + cot.md from agents_regular into the live
ablation folder, seeds run-1 upstream params, removes detecting+sam params to
force those stages to rerun. Outputs to logs/{tunnel}/regular_hint_v3/{model}/.

Usage:
    ./venv/bin/python methods/papers/scripts/run_regular_hint_v3.py --dry-run
    ./venv/bin/python methods/papers/scripts/run_regular_hint_v3.py --tunnel 3-1-1
    ./venv/bin/python methods/papers/scripts/run_regular_hint_v3.py --skip-existing
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
PYTHON = str(REPO_ROOT / "venv" / "bin" / "python")
if not Path(PYTHON).exists():
    PYTHON = sys.executable
TS = os.environ.get("REGULAR_HINT_V3_TS") or datetime.now().strftime("%Y%m%d_%H%M%S")

CONTINUOUS_TUNNELS = ["3-1-1", "3-1-2", "3-1-3"]

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
    return REPO_ROOT / "logs" / tunnel / "regular_hint_v3" / model


def has_hint_run(tunnel: str, model: str) -> bool:
    return extract_miou(hint_dir(tunnel, model)) is not None


@contextmanager
def swapped_detecting_docs():
    backup_dir = REPO_ROOT / "logs" / f"regular_hint_v3_doc_backup_{TS}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed: list[tuple[Path, Path]] = []
    try:
        for name in DOC_NAMES:
            live = LIVE_AGENTS / name
            hint = HINT_AGENTS / name
            if not hint.exists():
                raise FileNotFoundError(f"Hint doc missing: {hint}")
            bak = backup_dir / name
            if live.exists():
                shutil.copy2(live, bak)
                backed.append((bak, live))
            shutil.copy2(hint, live)
            print(f"  swapped {name} <- agents_regular")
        yield
    finally:
        for bak, live in backed:
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
        "hint_v3_miou": None,
        "status": "pending",
    }

    print(f"\n{'='*70}\n  REGULAR HINT v3: {tunnel} {model}\n{'='*70}")

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
            print("  [dry-run] would run orchestrator")
            row["status"] = "dry_run"
            return row

        script = ORCHESTRATORS[model]
        cmd = [PYTHON, str(REPO_ROOT / script), tunnel, "--model", model]
        print(f"  Running: {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=3600,
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

        row["hint_v3_miou"] = extract_miou(std_data)
        row["status"] = "ok"
        print(f"  hint v3 mIoU: {row['hint_v3_miou']}")

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
    fields = ["tunnel", "model", "baseline_miou", "hint_v1_miou", "hint_v3_miou", "status"]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fields})


def main() -> None:
    os.chdir(REPO_ROOT)
    _ensure_venv_on_path()

    parser = argparse.ArgumentParser(description="Continuous-tunnel v3 (prompt-only detecting docs)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--tunnel", choices=CONTINUOUS_TUNNELS)
    parser.add_argument("--model", choices=MODELS, default="opus4.6")
    args = parser.parse_args()

    tunnels = [args.tunnel] if args.tunnel else CONTINUOUS_TUNNELS
    model = args.model
    summary_csv = REPO_ROOT / "logs" / f"regular_hint_v3_{TS}_summary.csv"

    queue = []
    for tunnel in tunnels:
        if args.skip_existing and has_hint_run(tunnel, model):
            print(f"  skip existing: {tunnel} {model}")
            continue
        queue.append(tunnel)

    print(f"Queue: {queue}, model={model} (TS={TS})")
    if args.dry_run:
        with swapped_detecting_docs():
            for t in queue:
                run_combo(t, model, dry_run=True)
        return

    with swapped_detecting_docs():
        for tunnel in queue:
            row = run_combo(tunnel, model, dry_run=False)
            append_summary(summary_csv, row)

    print(f"\nSummary: {summary_csv}")


if __name__ == "__main__":
    main()
