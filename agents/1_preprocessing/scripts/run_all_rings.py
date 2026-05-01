"""Batch-run the ring-native preprocessing on every canonical ring.

Discovers tunnels via ``r4tun/references/data/<tid>/`` (the active 30-tunnel pool).
For each tunnel that lacks ``agents/1_preprocessing/parameters/<tid>/r<rid>/parameters_preprocessing.json``,
calls :mod:`warm_from_r4tun` once for the lowest ring id and copies the resulting JSON
into every other ring directory of that tunnel (warm content is identical across rings).

Then iterates ``data/rings/<tid_underscores>_ring<rid>.txt`` and runs
:func:`agents.1_preprocessing.1_preprocessing.run_preprocessing` in-process.
Skips rings whose ``data/<tid>/r<rid>/depth_map.npy`` already exists unless ``--force``.

Per-ring outcomes are appended to ``data/rings/preprocessing_log.csv`` with columns:
``timestamp_utc, tunnel_id, ring_id, status, elapsed_s, n_points, nan_ratio, error``.

Run::

    ./venv/bin/python agents/1_preprocessing/scripts/run_all_rings.py
    ./venv/bin/python agents/1_preprocessing/scripts/run_all_rings.py --tunnels 4-1 5-3
    ./venv/bin/python agents/1_preprocessing/scripts/run_all_rings.py --force
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREPROCESSING_DIR = PROJECT_ROOT / "agents" / "1_preprocessing"
RINGS_DIR = PROJECT_ROOT / "data" / "rings"
SUBSETS_DIR = PROJECT_ROOT / "data" / "subsets"
REFS_DIR = PROJECT_ROOT / "r4tun" / "references" / "data"
PARAMS_ROOT = PREPROCESSING_DIR / "parameters"
DATA_ROOT = PROJECT_ROOT / "data"
LOG_PATH = RINGS_DIR / "preprocessing_log.csv"

WARM_SCRIPT = PREPROCESSING_DIR / "scripts" / "warm_from_r4tun.py"

if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))


def _load_run_preprocessing():
    """Import 1_preprocessing.run_preprocessing despite the leading digit in the filename."""
    path = PREPROCESSING_DIR / "1_preprocessing.py"
    spec = importlib.util.spec_from_file_location("agents_1_preprocessing", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_preprocessing


run_preprocessing = _load_run_preprocessing()


def discover_active_tunnels() -> List[str]:
    out: List[str] = []
    for p in sorted(REFS_DIR.iterdir()):
        if not p.is_dir():
            continue
        if re.match(r"^\d+(?:-\d+)+$", p.name):
            params_dir = p / "parameters"
            if params_dir.is_dir():
                out.append(p.name)
    return out


def rings_for_tunnel(tunnel_id: str) -> List[int]:
    stem = tunnel_id.replace("-", "_")
    pattern = re.compile(rf"^{re.escape(stem)}_ring(\d+)\.txt$")
    rings = []
    for p in RINGS_DIR.glob(f"{stem}_ring*.txt"):
        m = pattern.match(p.name)
        if m:
            rings.append(int(m.group(1)))
    return sorted(rings)


def ensure_warm_params(tunnel_id: str, ring_ids: List[int]) -> Tuple[int, int]:
    """Make sure every ring has parameters_preprocessing.json.

    Returns (n_warmed_via_script, n_copied_from_first_ring).
    """
    if not ring_ids:
        return (0, 0)
    needed = []
    for rid in ring_ids:
        target = PARAMS_ROOT / tunnel_id / f"r{rid}" / "parameters_preprocessing.json"
        if not target.is_file():
            needed.append(rid)
    if not needed:
        return (0, 0)

    seed_rid = min(ring_ids)
    seed_path = PARAMS_ROOT / tunnel_id / f"r{seed_rid}" / "parameters_preprocessing.json"
    n_warmed = 0
    if not seed_path.is_file():
        ref_dir = REFS_DIR / tunnel_id / "parameters"
        if not ref_dir.is_dir():
            raise FileNotFoundError(f"r4tun reference parameters missing for {tunnel_id}: {ref_dir}")
        cmd = [
            str(PROJECT_ROOT / "venv" / "bin" / "python"),
            str(WARM_SCRIPT),
            tunnel_id,
            str(seed_rid),
            "--reference-dir",
            str(ref_dir),
        ]
        print(f"[warm] {tunnel_id}: generating seed at r{seed_rid}")
        subprocess.run(cmd, check=True)
        n_warmed = 1
        if not seed_path.is_file():
            raise RuntimeError(f"warm_from_r4tun did not produce {seed_path}")

    n_copied = 0
    for rid in needed:
        target = PARAMS_ROOT / tunnel_id / f"r{rid}" / "parameters_preprocessing.json"
        if rid == seed_rid:
            continue
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_path, target)
        n_copied += 1
    if n_copied:
        print(f"[warm] {tunnel_id}: copied seed JSON to {n_copied} other ring(s)")
    return (n_warmed, n_copied)


def compute_nan_ratio(npy_path: Path) -> float:
    if not npy_path.is_file():
        return float("nan")
    arr = np.load(npy_path)
    if arr.size == 0:
        return float("nan")
    return float(np.isnan(arr).sum()) / float(arr.size)


def append_log_row(row: Dict[str, str]) -> None:
    fields = [
        "timestamp_utc",
        "tunnel_id",
        "ring_id",
        "status",
        "elapsed_s",
        "n_points",
        "nan_ratio",
        "error",
    ]
    write_header = not LOG_PATH.is_file()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tunnels",
        nargs="*",
        default=None,
        help="Optional tunnel id allow-list (default: all 30 active tunnels).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run rings even if depth_map.npy already exists.",
    )
    args = parser.parse_args()

    active = discover_active_tunnels()
    if not active:
        print(f"[batch] no active tunnels under {REFS_DIR}", file=sys.stderr)
        return 1
    if args.tunnels:
        wanted: Set[str] = set(args.tunnels)
        unknown = wanted - set(active)
        if unknown:
            print(f"[batch] unknown tunnel ids: {sorted(unknown)}", file=sys.stderr)
            return 2
        active = [t for t in active if t in wanted]

    plan: List[Tuple[str, int]] = []
    for tid in active:
        rids = rings_for_tunnel(tid)
        if not rids:
            print(f"[batch] no rings on disk for {tid} (looked under {RINGS_DIR})")
            continue
        for rid in rids:
            plan.append((tid, rid))

    print(f"[batch] {len(plan)} (tunnel, ring) pairs across {len(active)} tunnels")

    seen_tunnels: Set[str] = set()
    skipped = 0
    ran = 0
    failed = 0

    for tid, rid in plan:
        if tid not in seen_tunnels:
            ensure_warm_params(tid, rings_for_tunnel(tid))
            seen_tunnels.add(tid)

        out_dir = DATA_ROOT / tid / f"r{rid}"
        depth_npy = out_dir / "depth_map.npy"
        if depth_npy.is_file() and not args.force:
            skipped += 1
            continue

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        t0 = time.time()
        try:
            run_preprocessing(tid, rid)
            elapsed = time.time() - t0
            n_points = "?"
            try:
                ring_txt = RINGS_DIR / f"{tid.replace('-', '_')}_ring{rid}.txt"
                if ring_txt.is_file():
                    with ring_txt.open() as fh:
                        n_points = sum(1 for _ in fh)
            except Exception:
                pass
            nan_ratio = compute_nan_ratio(depth_npy)
            ran += 1
            append_log_row({
                "timestamp_utc": ts,
                "tunnel_id": tid,
                "ring_id": str(rid),
                "status": "OK",
                "elapsed_s": f"{elapsed:.2f}",
                "n_points": str(n_points),
                "nan_ratio": f"{nan_ratio:.4f}" if not np.isnan(nan_ratio) else "",
                "error": "",
            })
            print(f"[batch] {tid}/r{rid}: OK in {elapsed:.1f}s nan_ratio={nan_ratio:.4f}")
        except Exception as e:  # noqa: BLE001
            elapsed = time.time() - t0
            failed += 1
            tb = traceback.format_exc()
            print(f"[batch] {tid}/r{rid}: ERROR after {elapsed:.1f}s: {e}\n{tb}", file=sys.stderr)
            append_log_row({
                "timestamp_utc": ts,
                "tunnel_id": tid,
                "ring_id": str(rid),
                "status": "ERROR",
                "elapsed_s": f"{elapsed:.2f}",
                "n_points": "",
                "nan_ratio": "",
                "error": repr(e),
            })

    print(f"[batch] done. ran={ran} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
