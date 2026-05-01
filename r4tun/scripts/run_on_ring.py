#!/usr/bin/env python3
"""Phase 0: run tunnel-scoped r4tun stages on a single-ring point cloud (sanity check).

Symlinks the ring txt as ``data/subsets/<fake_tunnel_id>.txt``, copies reference
parameters into the ablation parameter tree for that fake id, runs unfolding →
denoising → enhancing with ``--ablation m_s_k --model gpt5.4``.

Writes log to ``data/4-1/r110/_phase0/log.md`` (default paths below).

Usage (from repo root, project venv only)::

    ./venv/bin/python r4tun/scripts/run_on_ring.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = REPO_ROOT / "venv" / "bin" / "python"

FAKE_TUNNEL_ID = "phase0_4_1_r110"
RING_TXT = REPO_ROOT / "data" / "rings" / "4_1_ring110.txt"
REF_PARAMS_DIR = REPO_ROOT / "r4tun" / "references" / "data" / "4-1" / "parameters"
PARAM_DST_DIR = (
    REPO_ROOT
    / "r4tun"
    / "agents"
    / "ablation"
    / "memory+state+knowledge"
    / "parameters"
    / FAKE_TUNNEL_ID
)
SUBSETS_DIR = REPO_ROOT / "data" / "subsets"
SUBSET_LINK = SUBSETS_DIR / f"{FAKE_TUNNEL_ID}.txt"
LOG_DIR = REPO_ROOT / "data" / "4-1" / "r110" / "_phase0"
LOG_PATH = LOG_DIR / "log.md"


def main() -> int:
    if not VENV_PYTHON.is_file():
        print(f"Missing venv Python: {VENV_PYTHON}", file=sys.stderr)
        return 1
    if not RING_TXT.is_file():
        print(f"Missing ring file: {RING_TXT}", file=sys.stderr)
        return 1
    if not REF_PARAMS_DIR.is_dir():
        print(f"Missing reference params: {REF_PARAMS_DIR}", file=sys.stderr)
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SUBSETS_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Phase 0 — direct r4tun on single ring",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Ring: `{RING_TXT.relative_to(REPO_ROOT)}`",
        f"- Fake tunnel id: `{FAKE_TUNNEL_ID}`",
        f"- Reference params: `{REF_PARAMS_DIR.relative_to(REPO_ROOT)}`",
        "",
    ]

    # Symlink subset
    if SUBSET_LINK.exists() or SUBSET_LINK.is_symlink():
        SUBSET_LINK.unlink()
    os.symlink(os.path.relpath(RING_TXT, SUBSET_LINK.parent), SUBSET_LINK)
    lines.append(f"- Symlinked `{SUBSET_LINK.relative_to(REPO_ROOT)}` → ring txt.")
    lines.append("")

    # Copy parameters (same filenames as reference)
    shutil.rmtree(PARAM_DST_DIR, ignore_errors=True)
    shutil.copytree(REF_PARAMS_DIR, PARAM_DST_DIR)
    lines.append(
        f"- Copied parameters to `{PARAM_DST_DIR.relative_to(REPO_ROOT)}`."
    )
    lines.append("")

    env = os.environ.copy()
    env["R4TUN_PIPELINE_OUT_PREFIX"] = "data/_phase0"

    stages = [
        ("unfolding", REPO_ROOT / "r4tun" / "agents" / "unfolding.py"),
        ("denoising", REPO_ROOT / "r4tun" / "agents" / "denoising.py"),
        ("enhancing", REPO_ROOT / "r4tun" / "agents" / "enhancing.py"),
    ]

    for name, script in stages:
        if not script.is_file():
            lines.append(f"## {name}\n\n**ERROR**: script missing `{script}`\n")
            continue
        cmd = [
            str(VENV_PYTHON),
            str(script),
            FAKE_TUNNEL_ID,
            "--ablation",
            "m_s_k",
            "--model",
            "gpt5.4",
        ]
        lines.append(f"## {name}\n")
        lines.append(f"```\n$ {' '.join(cmd)}\n```\n")
        try:
            cp = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=7200,
            )
        except subprocess.TimeoutExpired:
            lines.append("**Result**: TIMEOUT (>2h)\n")
            continue
        lines.append(f"**Exit code**: `{cp.returncode}`\n")
        if cp.stdout:
            lines.append("### stdout\n```\n")
            lines.append(cp.stdout[-12000:])
            lines.append("\n```\n")
        if cp.stderr:
            lines.append("### stderr\n```\n")
            lines.append(cp.stderr[-12000:])
            lines.append("\n```\n")

    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {LOG_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
