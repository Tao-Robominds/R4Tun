#!/usr/bin/env python3
"""Promote improved rerun results into standard vendor directories.

For each combo where new mIoU >= old mIoU:
  1. rsync data/  -> data/ablation_{vendor}/{condition}/{tid}/
  2. copy model-specific parameter JSONs -> agents/ablation/{condition}/parameters/{tid}/
"""
from __future__ import annotations

import glob
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

VENDOR_MAP = {"opus4.6": "anthropic", "gpt5.4": "gpt", "gemini3flash": "gemini"}
COND_MAP = {"m_s": "memory+state", "m_s_k": "memory+state+knowledge"}

BATCH_TS = "20260417_025752"
SMOKE_TS = "20260417_023519"

PROMOTE = [
    # (tunnel, cond_short, model, old_miou, new_miou)
    ("4-4", "m_s",   "opus4.6",      0.099, 0.152),
    ("5-3", "m_s",   "opus4.6",      0.123, 0.227),
    ("5-4", "m_s",   "opus4.6",      0.106, 0.195),
    ("5-3", "m_s",   "gpt5.4",       0.106, 0.110),
    ("5-4", "m_s",   "gpt5.4",       0.096, 0.103),
    ("5-3", "m_s",   "gemini3flash",  0.087, 0.104),
    ("4-4", "m_s_k", "opus4.6",      0.047, 0.245),
    ("5-3", "m_s_k", "opus4.6",      0.089, 0.178),
    ("5-4", "m_s_k", "opus4.6",      0.068, 0.207),
    ("5-3", "m_s_k", "gpt5.4",       0.116, 0.124),
    ("5-4", "m_s_k", "gpt5.4",       0.098, 0.098),
    ("4-4", "m_s_k", "gemini3flash",  0.072, 0.108),
    ("5-3", "m_s_k", "gemini3flash",  0.080, 0.143),
]


def source_base(tid: str, cond: str, model: str) -> Path:
    if tid == "4-4" and cond == "m_s_k" and model == "opus4.6":
        return REPO / "logs" / tid / f"rerun_{SMOKE_TS}" / cond / model
    return REPO / "logs" / tid / f"rerun_{BATCH_TS}" / cond / model


def main() -> None:
    for tid, cond, model, old, new in PROMOTE:
        vendor = VENDOR_MAP[model]
        cond_full = COND_MAP[cond]
        src = source_base(tid, cond, model)

        src_data = src / "data"
        dst_data = REPO / "data" / f"ablation_{vendor}" / cond_full / tid
        assert src_data.is_dir(), f"Missing source data: {src_data}"

        print(f"[data]  {tid}/{cond}/{model}: {old:.3f} -> {new:.3f}")
        subprocess.check_call([
            "rsync", "-a", "--delete",
            str(src_data) + "/",
            str(dst_data) + "/",
        ])

        src_params = src / "parameters"
        dst_params = REPO / "agents" / "ablation" / cond_full / "parameters" / tid
        if src_params.is_dir():
            pattern = str(src_params / f"parameters_*_{model}.json")
            files = glob.glob(pattern)
            if files:
                dst_params.mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copy2(f, dst_params / Path(f).name)
                print(f"[param] copied {len(files)} parameter files for {model}")
            else:
                print(f"[param] WARNING: no files matching {pattern}")

    print(f"\nDone — promoted {len(PROMOTE)} combos.")


if __name__ == "__main__":
    main()
