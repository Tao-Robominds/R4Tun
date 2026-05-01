"""Clean up `data/rings/`: drop legacy PNGs and any TXTs not in the canonical subset set.

Canonical set = union of every (tunnel_id, ring_id) found in `data/subsets/*.txt`,
mapped to underscored stems used by `data/rings/<tid>_ring<rid>.txt`.

By default this is a dry run that prints a summary. Pass --apply to actually delete.

Always-deleted (when --apply):
  * Every `data/rings/*.png` (legacy, unused by the new ring-native workflow).

Conditionally-deleted (when --apply):
  * Any `data/rings/<tid>_ring<rid>.txt` whose (tid, rid) is NOT in the canonical set.
  * Any `data/rings/<name>.{txt,png}` whose tunnel id is not among the active 30
    referenced under `r4tun/references/data/`.

Refuses to run if any tunnel listed under `r4tun/references/data/` lacks a
matching `data/subsets/<tid>.txt` (would indicate a stale rebuild).

Per workspace rules: never touches anything under `data/baseline/`, `data/bo/`, or `r4tun/`.
Run only with the project venv:

    ./venv/bin/python agents/1_preprocessing/scripts/cleanup_rings_dir.py
    ./venv/bin/python agents/1_preprocessing/scripts/cleanup_rings_dir.py --apply
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RINGS_DIR = PROJECT_ROOT / "data" / "rings"
SUBSETS_DIR = PROJECT_ROOT / "data" / "subsets"
REFS_DIR = PROJECT_ROOT / "r4tun" / "references" / "data"

SUBSET_RE = re.compile(r"^(\d+(?:-\d+)+)\.txt$")
RING_FILE_RE = re.compile(r"^(\d+(?:_\d+)+)_ring(\d+)\.(txt|png)$")


def _stem_to_tid(stem: str) -> str:
    return stem.replace("_", "-")


def load_canonical_pairs() -> Tuple[Set[Tuple[str, int]], Set[str]]:
    """Return (canonical pairs, set of active tunnel ids)."""
    pairs: Set[Tuple[str, int]] = set()
    active: Set[str] = set()
    for p in sorted(SUBSETS_DIR.glob("*.txt")):
        m = SUBSET_RE.match(p.name)
        if not m:
            continue
        tid = m.group(1)
        active.add(tid)
        rings: Set[int] = set()
        with p.open() as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6:
                    rings.add(int(float(parts[5])))
        for rid in rings:
            pairs.add((tid, rid))
    return pairs, active


def load_reference_tids() -> Set[str]:
    if not REFS_DIR.exists():
        return set()
    out: Set[str] = set()
    for p in REFS_DIR.iterdir():
        if not p.is_dir():
            continue
        if SUBSET_RE.match(p.name + ".txt") or re.match(r"^\d+(?:-\d+)+$", p.name):
            out.add(p.name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually delete; default is dry-run.")
    args = parser.parse_args()

    if not RINGS_DIR.exists():
        print(f"[cleanup] {RINGS_DIR} does not exist", file=sys.stderr)
        return 1

    canonical, active = load_canonical_pairs()
    refs = load_reference_tids()
    missing_subset = sorted(refs - active)
    if missing_subset:
        print(
            f"[cleanup] refusing to run: tunnels in r4tun/references/data without a "
            f"matching data/subsets/<tid>.txt: {missing_subset}",
            file=sys.stderr,
        )
        return 2

    to_delete: list[tuple[Path, str]] = []
    txt_seen = 0
    png_seen = 0

    for p in sorted(RINGS_DIR.iterdir()):
        if p.is_dir():
            continue
        m = RING_FILE_RE.match(p.name)
        if not m:
            # leave non-ring files (summary.json, _coverage_check.md, etc.) alone
            continue
        tid = _stem_to_tid(m.group(1))
        rid = int(m.group(2))
        ext = m.group(3)
        if ext == "png":
            png_seen += 1
            to_delete.append((p, "legacy png (unused by workflow)"))
            continue
        # ext == 'txt'
        txt_seen += 1
        if tid not in active:
            to_delete.append((p, f"tunnel {tid} not in active pool"))
        elif (tid, rid) not in canonical:
            to_delete.append((p, f"({tid}, r{rid}) not in canonical subset"))

    print(f"[cleanup] inspecting {RINGS_DIR}")
    print(f"  canonical (tid, ring) pairs:        {len(canonical)}")
    print(f"  active tunnels:                     {len(active)}")
    print(f"  reference tunnels:                  {len(refs)}")
    print(f"  ring-named TXT files seen:          {txt_seen}")
    print(f"  ring-named PNG files seen:          {png_seen}")
    print(f"  total proposed deletions:           {len(to_delete)}")

    by_reason: dict[str, int] = {}
    for _, reason in to_delete:
        by_reason[reason] = by_reason.get(reason, 0) + 1
    if by_reason:
        print("  by reason:")
        for reason, n in sorted(by_reason.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {n}")

    if not args.apply:
        print("[cleanup] dry-run only. Re-run with --apply to delete.")
        if to_delete and len(to_delete) <= 10:
            print("  preview:")
            for p, reason in to_delete:
                print(f"    {p.relative_to(PROJECT_ROOT)}  ({reason})")
        return 0

    if not to_delete:
        print("[cleanup] nothing to delete.")
        return 0

    log_path = RINGS_DIR / "_cleanup_log.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_entries = [f"\n## Cleanup run {ts}", f"- canonical pairs: {len(canonical)}", f"- removed: {len(to_delete)}", "", "| path | reason |", "|------|--------|"]
    for p, reason in to_delete:
        new_entries.append(f"| `{p.relative_to(PROJECT_ROOT)}` | {reason} |")
    if log_path.exists():
        existing = log_path.read_text()
        log_path.write_text(existing.rstrip() + "\n" + "\n".join(new_entries) + "\n")
    else:
        header = "# data/rings cleanup log\n"
        log_path.write_text(header + "\n".join(new_entries) + "\n")

    removed = 0
    for p, _ in to_delete:
        try:
            p.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    print(f"[cleanup] removed {removed} files. Log appended to {log_path.relative_to(PROJECT_ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
