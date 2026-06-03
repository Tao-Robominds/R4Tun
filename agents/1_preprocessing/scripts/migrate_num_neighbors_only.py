#!/usr/bin/env python3
"""One-shot: keep only num_neighbors in parameters_preprocessing.json trees."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARAMS_ROOT = SCRIPT_DIR.parent / "parameters"
LEGACY = ("curvature_neighbors", "outlier_neighbors")
DEFAULT = 20


def _resolve_canonical(data: dict) -> tuple[int, bool, dict]:
    legacy = {k: data[k] for k in LEGACY if k in data}
    if "num_neighbors" in data:
        legacy["num_neighbors"] = data["num_neighbors"]
    canonical = int(
        data.get("num_neighbors")
        or data.get("curvature_neighbors")
        or data.get("outlier_neighbors")
        or DEFAULT
    )
    unique = {int(v) for v in legacy.values()} if legacy else {canonical}
    conflict = len(unique) > 1
    return canonical, conflict, legacy


def migrate_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    canonical, conflict, legacy = _resolve_canonical(data)
    changed = False
    if data.get("num_neighbors") != canonical:
        data["num_neighbors"] = canonical
        changed = True
    for key in LEGACY:
        if key in data:
            del data[key]
            changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {
        "path": str(path.relative_to(PARAMS_ROOT.parent.parent.parent)),
        "changed": changed,
        "num_neighbors": canonical,
        "conflict": conflict,
        "legacy_values": legacy,
    }


def main() -> int:
    files = sorted(PARAMS_ROOT.rglob("parameters_preprocessing.json"))
    if not files:
        print("No parameters_preprocessing.json found", file=sys.stderr)
        return 1
    changed_n = 0
    conflicts = []
    for path in files:
        row = migrate_file(path)
        if row["changed"]:
            changed_n += 1
        if row["conflict"]:
            conflicts.append(row)
    print(f"Migrated {changed_n}/{len(files)} files under {PARAMS_ROOT}")
    if conflicts:
        print(f"WARNING: {len(conflicts)} file(s) had disagreeing legacy keys:")
        for row in conflicts:
            print(f"  {row['path']}: {row['legacy_values']} -> num_neighbors={row['num_neighbors']}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
