"""Manifest loading and panel summary helpers for layout BO."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from lib.ceiling_gate import REPO_ROOT


def parse_ring_key(ring_key: str) -> tuple[str, int]:
    tunnel_id, rr = ring_key.split("/")
    return tunnel_id, int(rr.replace("r", ""))


SPARSE_SLOTS = frozenset({"sparse_6", "sparse_7"})


def n_evals_for_ring_entry(entry: dict[str, Any], default: int = 60) -> int:
    """Step 3 budget: sparse slots 120, representative slots 60."""
    slot = entry.get("diversity_slot", "")
    if slot in SPARSE_SLOTS:
        return 120
    return default


def load_manifest_rings(
    manifest_path: Path,
    *,
    only_ring: str | None = None,
    skip: set[str] | None = None,
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rings = manifest.get("rings", [])
    if only_ring:
        rings = [r for r in rings if r["ring_key"] == only_ring]
    if skip:
        rings = [r for r in rings if r["ring_key"] not in skip]
    return rings


def write_experience_panel_summary(
    run_root: Path,
    manifest_path: Path,
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge per-ring trials and write panel_summary.csv + experience_summary.json."""
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    panel_df = pd.DataFrame(summaries)
    panel_df.to_csv(run_root / "panel_summary.csv", index=False)

    all_trials: list[pd.DataFrame] = []
    for entry in manifest.get("rings", []):
        tunnel_id, ring_id = parse_ring_key(entry["ring_key"])
        trials_path = run_root / tunnel_id / f"r{ring_id}" / "bo_trials.csv"
        if trials_path.exists():
            all_trials.append(pd.read_csv(trials_path))

    merged_df = pd.concat(all_trials, ignore_index=True) if all_trials else None
    if merged_df is not None:
        merged_df.to_csv(run_root / "bo_trials.csv", index=False)

    experience_summary = {
        "n_rings": len(manifest.get("rings", [])),
        "n_rings_run_this_session": len(summaries),
        "n_total_trials": int(len(merged_df)) if merged_df is not None else 0,
        "rings_passed_experience_gate": int(sum(1 for s in summaries if s["experience_gate_passed"])),
        "panel_summary": str((run_root / "panel_summary.csv").relative_to(REPO_ROOT)),
        "bo_trials": str((run_root / "bo_trials.csv").relative_to(REPO_ROOT)),
    }
    (run_root / "experience_summary.json").write_text(
        json.dumps(experience_summary, indent=2) + "\n", encoding="utf-8"
    )
    return experience_summary
