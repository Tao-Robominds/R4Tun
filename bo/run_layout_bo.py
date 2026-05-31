#!/usr/bin/env python3
"""Detection layout GP-BO — single entry point for all BO trial modes.

Modes:
  ceiling-push  Iterative push toward GT ceiling (manifest-driven, any N rings)
  experience    Fixed-budget trial collection (manifest panel or single ring)
  verify        Pre-flight GT encode/decode round-trip smoke test

All writes go to logs/<run_id>/; corpora under data/ are read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_BO_DIR = Path(__file__).resolve().parent
REPO_ROOT = _BO_DIR.parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.layout_bo import (  # noqa: E402
    REPO_ROOT as _REPO,
    run_iterative_ceiling_push,
    run_ring_bo,
    write_panel_ceiling_push_summary,
)
from lib.manifest import (  # noqa: E402
    load_manifest_rings,
    parse_ring_key,
    write_experience_panel_summary,
)
from lib.verify import verify_ring  # noqa: E402

DEFAULT_BO_MANIFEST = REPO_ROOT / "data" / "bo" / "MANIFEST.json"
DEFAULT_BO_SOURCE = REPO_ROOT / "data" / "bo"
DEFAULT_MINIMUM_MANIFEST = REPO_ROOT / "data" / "minimum" / "MANIFEST.json"
DEFAULT_MINIMUM_SOURCE = REPO_ROOT / "data" / "minimum"


def _add_corpus_args(ap: argparse.ArgumentParser, *, require_run_root: bool = True) -> None:
    ap.add_argument("--manifest", default=str(DEFAULT_BO_MANIFEST), help="Corpus MANIFEST.json")
    ap.add_argument("--source-dir", default=str(DEFAULT_BO_SOURCE), help="Read-only preprocessing root")
    if require_run_root:
        ap.add_argument("--run-root", required=True, help="Sandbox output root under logs/")
    else:
        ap.add_argument("--run-root", default=str(REPO_ROOT / "logs" / "_layout_bo_verify"), help="Verify scratch dir")
    ap.add_argument("--only-ring", default=None, help="Run single ring e.g. 1-4/r206")
    ap.add_argument("--skip", nargs="*", default=[], help="Ring keys to skip")
    ap.add_argument("--seed", type=int, default=7)


def cmd_ceiling_push(args: argparse.Namespace) -> int:
    source = Path(args.source_dir).resolve()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip)

    rings = load_manifest_rings(Path(args.manifest), only_ring=args.only_ring, skip=skip)
    if args.only_ring and not rings:
        print(f"Ring not in manifest: {args.only_ring}", file=sys.stderr)
        return 1

    ring_results = []
    for entry in rings:
        ring_key = entry["ring_key"]
        tunnel_id, ring_id = parse_ring_key(ring_key)
        branch = entry.get("order_branch_default", "plus")
        seg = entry.get("segment_count")
        print(f"\n{'=' * 60}\nCeiling-push: {ring_key} (branch={branch})\n{'=' * 60}")
        result = run_iterative_ceiling_push(
            tunnel_id,
            ring_id,
            source_root=source,
            run_root=run_root,
            segment_count=seg,
            order_branch=branch,
            eval_chunk=args.eval_chunk,
            max_total_evals=args.max_total_evals,
            target_regret=args.target_regret,
            min_improvement=args.min_improvement,
            seed=args.seed,
        )
        ring_results.append(result)
        report = result["report"]
        print(
            f"  stop={report['stop_reason']} best={report['best_bo_miou']} "
            f"regret={report['regret_vs_ceiling']} evals={report['total_evals']}"
        )

    if ring_results:
        write_panel_ceiling_push_summary(run_root, ring_results)

    if args.only_ring and ring_results:
        gate = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "case_id": ring_results[0]["ctx"].case_id,
            "report": ring_results[0]["report"],
            "pass_criterion": "loop_completed_without_agent_errors",
            "passed": ring_results[0]["report"]["stop_reason"] != "agent_errors",
        }
        (run_root / "single_instance_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(gate, indent=2))

    return 0


def cmd_experience(args: argparse.Namespace) -> int:
    source = Path(args.source_dir).resolve()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip)

    if args.tunnel_id and args.ring_id is not None:
        rings = [{
            "ring_key": f"{args.tunnel_id}/r{args.ring_id}",
            "order_branch_default": args.order_branch,
            "segment_count": args.segment_count,
        }]
    else:
        rings = load_manifest_rings(Path(args.manifest), only_ring=args.only_ring, skip=skip)
        if args.only_ring and not rings:
            print(f"Ring not in manifest: {args.only_ring}", file=sys.stderr)
            return 1

    summaries = []
    for entry in rings:
        ring_key = entry["ring_key"]
        tunnel_id, ring_id = parse_ring_key(ring_key)
        branch = entry.get("order_branch_default", args.order_branch)
        seg = entry.get("segment_count") or args.segment_count
        print(f"\n{'=' * 60}\nExperience BO: {ring_key} (branch={branch})\n{'=' * 60}")
        result = run_ring_bo(
            tunnel_id,
            ring_id,
            source_root=source,
            run_root=run_root,
            n_evals=args.n_evals,
            seed=args.seed,
            segment_count=seg,
            order_branch=branch,
        )
        gate = result["gate"]
        best = result["bo_result"]["best_payload"]
        summaries.append({
            "ring_key": ring_key,
            "n_trials": gate["n_evals"],
            "ceiling_reference": gate.get("ceiling_miou_reference"),
            "best_bo_miou": gate["best_bo_miou"],
            "regret_vs_ceiling": gate.get("regret_vs_ceiling"),
            "miou_std": gate["miou_std"],
            "experience_gate_passed": gate["passed"],
            "best_r_surface_min": best.get("best_r_surface_min"),
            "best_k_y": best.get("best_k_y"),
        })

    if len(rings) > 1 or not args.tunnel_id:
        summary = write_experience_panel_summary(run_root, Path(args.manifest), summaries)
        print(json.dumps(summary, indent=2))
    elif summaries:
        print(json.dumps(summaries[0], indent=2))

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    source = Path(args.source_dir).resolve()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    rings = load_manifest_rings(Path(args.manifest), only_ring=args.only_ring, skip=set(args.skip))
    if args.only_ring and not rings:
        print(f"Ring not in manifest: {args.only_ring}", file=sys.stderr)
        return 1

    results = []
    for entry in rings:
        tunnel_id, ring_id = parse_ring_key(entry["ring_key"])
        branch = entry.get("order_branch_default", "plus")
        seg = entry.get("segment_count")
        results.append(
            verify_ring(
                tunnel_id,
                ring_id,
                source_root=source,
                run_root=run_root,
                order_branch=branch,
                segment_count=seg,
            )
        )

    all_pass = all(r["passed"] for r in results)
    print(json.dumps({"all_passed": all_pass, "rings": results}, indent=2))
    return 0 if all_pass else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Detection layout GP-BO trials")
    sub = ap.add_subparsers(dest="mode", required=True)

    push = sub.add_parser("ceiling-push", help="Iterative push toward GT ceiling (manifest, N rings)")
    _add_corpus_args(push)
    push.add_argument("--eval-chunk", type=int, default=128)
    push.add_argument("--max-total-evals", type=int, default=1024)
    push.add_argument("--target-regret", type=float, default=0.05)
    push.add_argument("--min-improvement", type=float, default=0.005)
    push.set_defaults(func=cmd_ceiling_push)

    exp = sub.add_parser("experience", help="Fixed-budget trial collection")
    _add_corpus_args(exp)
    exp.add_argument("--n-evals", type=int, default=64)
    exp.add_argument("--tunnel-id", default=None, help="Single ring without manifest")
    exp.add_argument("--ring-id", type=int, default=None)
    exp.add_argument("--segment-count", type=int, default=None, choices=[6, 7])
    exp.add_argument("--order-branch", default="plus", choices=["plus", "minus"])
    exp.set_defaults(func=cmd_experience)

    ver = sub.add_parser("verify", help="GT layout encode/decode smoke test")
    _add_corpus_args(ver, require_run_root=False)
    ver.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
