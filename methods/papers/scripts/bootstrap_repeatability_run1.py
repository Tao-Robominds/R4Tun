#!/usr/bin/env python3
"""Snapshot primary m+s+k parameters and mIoU as repeatability run 1 (no API/GPU)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from repeatability_common import (  # noqa: E402
    ABLATION_FOLDER,
    MODELS,
    PARAM_BASE,
    copy_performance,
    find_performance_md,
    get_tunnel_ids,
    load_flat_params,
    param_json_name,
    run1_dir,
    vendor_data_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def snapshot_combo(tunnel: str, model: str) -> bool:
    src_params = PARAM_BASE / tunnel
    dst = run1_dir(tunnel, model)
    dst_params = dst / "parameters"
    dst_params.mkdir(parents=True, exist_ok=True)

    copied = 0
    for stage in ("unfolding", "denoising", "enhancing", "detecting", "sam"):
        name = param_json_name(stage, model)
        src = src_params / name
        if src.exists():
            shutil.copy2(src, dst_params / name)
            copied += 1

    vendor = vendor_data_dir(tunnel, model)
    copy_performance(vendor, dst)
    if find_performance_md(dst) is None:
        copy_performance(REPO_ROOT / "data" / "ablation" / ABLATION_FOLDER / tunnel, dst)

    ok = copied >= 4 and bool(load_flat_params(dst_params, model))
    print(f"  {tunnel} {model}: {copied} param files -> {dst} ({'ok' if ok else 'WARN'})")
    return ok


def main() -> None:
    tunnels = get_tunnel_ids()
    if not tunnels:
        print(f"No tunnels under {PARAM_BASE}")
        sys.exit(1)

    n_ok = 0
    for tunnel in tunnels:
        for model in MODELS:
            if snapshot_combo(tunnel, model):
                n_ok += 1

    print(f"\nSnapshotted run1 for {n_ok}/{len(tunnels) * len(MODELS)} combos.")


if __name__ == "__main__":
    main()
