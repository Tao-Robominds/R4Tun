#!/usr/bin/env python3
"""Confirm pre-defined ring site params for one or more rings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BO_DIR = Path(__file__).resolve().parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.ceiling_gate import REPO_ROOT  # noqa: E402
from lib.manifest import load_manifest_rings  # noqa: E402
from lib.ring_site_params import confirm_ring_site_params  # noqa: E402

DEFAULT_BO_SOURCE = REPO_ROOT / "data" / "bo_calibration"
DEFAULT_HELD_SOURCE = REPO_ROOT / "data" / "held-out"
DEFAULT_OUT = REPO_ROOT / "logs" / "ring_site_params_v1" / "gate.json"


def _source_for_corpus(corpus: str | None) -> Path:
    if corpus == "held-out":
        return DEFAULT_HELD_SOURCE
    if corpus == "bo_calibration":
        return DEFAULT_BO_SOURCE
    return DEFAULT_BO_SOURCE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring-keys", nargs="+", required=True)
    ap.add_argument("--source-dir", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "bo_calibration" / "MANIFEST.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    manifest_by_key = {e["ring_key"]: e for e in load_manifest_rings(args.manifest)}

    results = []
    for rk in args.ring_keys:
        manifest_entry = manifest_by_key.get(rk)
        source = args.source_dir
        if source is None:
            from lib.ring_site_params import load_registry, registry_entry

            reg = load_registry()
            source = _source_for_corpus(registry_entry(reg, rk).get("corpus"))
        rec = confirm_ring_site_params(
            rk,
            source_root=source.resolve(),
            manifest_entry=manifest_entry,
            write_dir=(args.out.parent / rk.replace("/", "_")),
        )
        results.append(rec)
        print(json.dumps(rec, indent=2))

    summary = {
        "n_rings": len(results),
        "all_passed": all(r["passed"] for r in results),
        "rings": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
