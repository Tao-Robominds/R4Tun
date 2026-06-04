#!/usr/bin/env python3
"""Enrich Proxy4Tun BO trial CSVs with v5/seg features via calib replay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_BO = Path(__file__).resolve().parent
REPO_ROOT = _BO.parent
if str(_BO) not in sys.path:
    sys.path.insert(0, str(_BO))

from lib.calib_trial_replay import DEFAULT_CORPUS, enrichment_gate, replay_ring_trials  # noqa: E402


def enrich_trials_csv(
    trials_path: Path,
    *,
    corpus_root: Path,
    out_csv: Path,
    replay_root: Path,
    case_ids: list[str] | None = None,
    max_trials_per_ring: int | None = None,
) -> pd.DataFrame:
    trials = pd.read_csv(trials_path, low_memory=False)
    ids = case_ids or sorted(trials["case_id"].unique())
    parts: list[pd.DataFrame] = []
    for case_id in ids:
        print(f"Replay {case_id} ({trials_path.name}) ...", flush=True)
        sub = trials.loc[trials["case_id"] == case_id]
        if max_trials_per_ring:
            sub = sub.head(max_trials_per_ring)
        parts.append(
            replay_ring_trials(sub, corpus_root=corpus_root, run_root=replay_root, case_id=case_id)
        )
    enriched = pd.concat(parts, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(enriched)} rows)")
    return enriched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--replay-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="Parent dir; sets replay-root and gate path")
    ap.add_argument("--gate-case", default="")
    ap.add_argument("--max-trials-per-ring", type=int, default=None)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or args.out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_root = args.replay_root or (out_dir / "replay")
    trials_path = args.trials.resolve()

    if args.gate_case:
        gate_trials = pd.read_csv(trials_path).loc[lambda d: d["case_id"] == args.gate_case]
        if args.max_trials_per_ring:
            gate_trials = gate_trials.head(args.max_trials_per_ring)
        gate_df = replay_ring_trials(
            gate_trials,
            corpus_root=args.corpus,
            run_root=replay_root / "gate",
            case_id=args.gate_case,
        )
        gate = enrichment_gate(gate_df)
        gate["case_id"] = args.gate_case
        gate["lineage"] = str(trials_path.relative_to(REPO_ROOT.resolve()))
        gate_path = out_dir / "single_instance_gate.json"
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        gate_df.to_csv(out_dir / f"gate_{args.gate_case.replace('/', '_')}.csv", index=False)
        print(json.dumps(gate, indent=2))
        if not gate["passed"]:
            return 1
        if args.gate_only:
            return 0

    enrich_trials_csv(
        trials_path,
        corpus_root=args.corpus,
        out_csv=args.out_csv.resolve(),
        replay_root=replay_root,
        case_ids=args.cases,
        max_trials_per_ring=args.max_trials_per_ring,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
