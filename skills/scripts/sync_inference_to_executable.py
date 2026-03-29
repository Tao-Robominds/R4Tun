#!/usr/bin/env python3
"""
Copy ``parameters_*_m_opus4.6.json`` from the memory ablation archive into
``configurable/<tunnel_id>/parameters_*.json`` so ``run_agents.sh`` loads the
current inference output.

Use for **any** tunnel (regular-staggered, continuous, complex-staggered, etc.),
as long as the five inference files exist under
``configurable/ablation/memory/parameters/<tunnel_id>/``.

Repo root: this file is ``repo/skills/scripts/…`` → ``parents[2]``.

Examples (repo root)::

  ./venv/bin/python skills/scripts/sync_inference_to_executable.py --tunnel-id 4-1
  ./venv/bin/python skills/scripts/sync_inference_to_executable.py --all
  ./venv/bin/python skills/scripts/sync_inference_to_executable.py --all --verify-only
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "configurable" / "ablation" / "memory" / "parameters"

STAGES = [
    ("parameters_unfolding.json", "parameters_unfolding_m_opus4.6.json"),
    ("parameters_denoising.json", "parameters_denoising_m_opus4.6.json"),
    ("parameters_enhancing.json", "parameters_enhancing_m_opus4.6.json"),
    ("parameters_detecting.json", "parameters_detecting_m_opus4.6.json"),
    ("parameters_sam.json", "parameters_sam_m_opus4.6.json"),
]


def _has_all_opus(archive_dir: Path) -> bool:
    return all((archive_dir / opus).is_file() for _, opus in STAGES)


def _discover_tunnel_ids() -> list[str]:
    if not PARAMS.is_dir():
        return []
    out: list[str] = []
    for child in sorted(PARAMS.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith(".") or name in ("__pycache__",):
            continue
        if _has_all_opus(child):
            out.append(name)
    return out


def sync_one(tunnel_id: str, *, verify_only: bool = False) -> tuple[bool, list[str]]:
    """Returns (ok, messages)."""
    msgs: list[str] = []
    a = PARAMS / tunnel_id
    c = ROOT / "configurable" / tunnel_id
    if not a.is_dir():
        return False, [f"missing archive dir: {a}"]
    missing = [opus for _, opus in STAGES if not (a / opus).is_file()]
    if missing:
        return False, [f"{tunnel_id}: missing inference files: {', '.join(missing)}"]

    if verify_only:
        ok = True
        for base, opus in STAGES:
            src, dst = a / opus, c / base
            if not dst.is_file():
                msgs.append(f"{tunnel_id}: {base}: executable missing")
                ok = False
            elif not filecmp.cmp(src, dst, shallow=False):
                msgs.append(f"{tunnel_id}: {base}: MISMATCH (run without --verify-only to copy)")
                ok = False
        if ok:
            msgs.append(f"{tunnel_id}: verify OK (all five match)")
        return ok, msgs

    c.mkdir(parents=True, exist_ok=True)
    for base, opus in STAGES:
        shutil.copy2(a / opus, c / base)
        msgs.append(f"{tunnel_id}: {base} <- {opus}")
    return True, msgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tunnel-id",
        action="append",
        dest="tunnel_ids",
        default=[],
        metavar="ID",
        help="Tunnel id (repeatable).",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Every directory under the archive that contains all five *_m_opus4.6.json files.",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not copy; check executable JSON already matches inference (exit 1 if any differ).",
    )
    args = ap.parse_args()

    if args.all and args.tunnel_ids:
        print("Use only one of --all or --tunnel-id", file=sys.stderr)
        return 2
    if not args.all and not args.tunnel_ids:
        print("Specify --tunnel-id ID or --all", file=sys.stderr)
        return 2

    tids = _discover_tunnel_ids() if args.all else list(args.tunnel_ids)
    if not tids:
        print("No tunnel ids to process.", file=sys.stderr)
        return 1

    exit_bad = False
    for tid in tids:
        ok, msgs = sync_one(tid, verify_only=args.verify_only)
        for m in msgs:
            print(m)
        if not ok:
            exit_bad = True
    return 1 if exit_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
