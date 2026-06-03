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
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_BO_DIR = Path(__file__).resolve().parent
REPO_ROOT = _BO_DIR.parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.layout_bo import (  # noqa: E402
    REPO_ROOT as _REPO,
    run_iterative_ceiling_push,
    run_ring_bo,
    write_panel_ceiling_push_summary,
    write_ring_regular_manifest,
)
from lib.manifest import (  # noqa: E402
    load_manifest_rings,
    n_evals_for_ring_entry,
    parse_ring_key,
    write_experience_panel_summary,
)
from lib.verify import verify_ring  # noqa: E402

HONESTY_GATE = REPO_ROOT / "bo" / "check_experience_honesty_gate.py"
GT_EXPERIENCE_GATE = REPO_ROOT / "bo" / "check_gt_experience_gate.py"

DEFAULT_BO_MANIFEST = REPO_ROOT / "data" / "bo_calibration" / "MANIFEST.json"
DEFAULT_BO_SOURCE = REPO_ROOT / "data" / "bo_calibration"
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
            manifest_entry=entry,
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


def _run_subprocess_gate(script: Path, run_root: Path, extra_args: list[str] | None = None) -> dict:
    out_name = script.stem.replace("check_", "") + ".json"
    if script.name == "check_gt_experience_gate.py":
        out_name = "gt_experience_gate.json"
    elif script.name == "check_experience_honesty_gate.py":
        out_name = "honesty_gate.json"
    out = run_root / out_name
    cmd = [str(REPO_ROOT / "venv" / "bin" / "python"), str(script), "--run-root", str(run_root)]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if out.is_file():
        return json.loads(out.read_text(encoding="utf-8"))
    return {"passed": False, "error": proc.stderr or proc.stdout}


def _run_honesty_gate(run_root: Path, *, expected_n: int | None = None) -> dict:
    extra = [f"--expected-n={expected_n}"] if expected_n is not None else None
    return _run_subprocess_gate(HONESTY_GATE, run_root, extra)


def _run_gt_experience_gate(run_root: Path, *, expected_n: int | None = None) -> dict:
    extra = [f"--expected-n={expected_n}"] if expected_n is not None else None
    return _run_subprocess_gate(GT_EXPERIENCE_GATE, run_root, extra)


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
        n_evals = n_evals_for_ring_entry(entry, default=args.n_evals) if entry.get("diversity_slot") else args.n_evals
        mode_label = "GT-anchor" if args.warm_anchor == "gt_derived" else "Honest"
        stream_note = f", stream={args.stream}" if args.stream != "full" else ""
        print(f"\n{'=' * 60}\n{mode_label} experience BO: {ring_key} (n_evals={n_evals}{stream_note})\n{'=' * 60}")
        prior_root = None if args.warm_anchor == "gt_derived" else (
            Path(args.prior_root).resolve() if args.prior_root else None
        )
        layout_handoff = (
            Path(args.layout_handoff_root).resolve()
            if getattr(args, "layout_handoff_root", None)
            else None
        )
        k_handoff = (
            Path(args.k_handoff_root).resolve()
            if getattr(args, "k_handoff_root", None)
            else None
        )
        result = run_ring_bo(
            tunnel_id,
            ring_id,
            source_root=source,
            run_root=run_root,
            n_evals=n_evals,
            seed=args.seed,
            segment_count=seg,
            manifest_entry=entry,
            order_branch=branch,
            prior_root=prior_root,
            warm_anchor=args.warm_anchor,
            experience_stream=args.stream,
            layout_handoff_root=layout_handoff,
            k_handoff_root=k_handoff,
        )
        gate = result["gate"]
        best = result["bo_result"]["best_payload"]
        ctx = result["ctx"]
        summaries.append({
            "ring_key": ring_key,
            "ring_is_regular": bool(ctx.ring_is_regular),
            "n_trials": gate["n_evals"],
            "ceiling_reference": gate.get("ceiling_miou_reference"),
            "best_bo_miou": gate["best_bo_miou"],
            "regret_vs_ceiling": gate.get("regret_vs_ceiling"),
            "miou_std": gate["miou_std"],
            "experience_gate_passed": gate["passed"],
            "best_layout_params": best.get("best_layout_params"),
            "best_r_surface_min": best.get("best_r_surface_min"),
            "r_surface_min_ceiling_ref": best.get("r_surface_min_ceiling_ref"),
            "best_k_y": best.get("best_k_y"),
        })

    if args.stream == "k" and summaries:
        write_ring_regular_manifest(
            run_root,
            [{"case_id": s["ring_key"], "ring_is_regular": s["ring_is_regular"]} for s in summaries],
        )

    gt_gate_args: dict[str, int | None] = {}
    if args.only_ring and summaries:
        gt_gate_args["expected_n"] = summaries[0]["n_trials"]
    elif args.warm_anchor == "gt_derived" and len(rings) > 1:
        gt_gate_args["expected_n"] = 480

    if len(rings) > 1 or not args.tunnel_id:
        summary = write_experience_panel_summary(run_root, Path(args.manifest), summaries)
        print(json.dumps(summary, indent=2))
        if args.warm_anchor == "gt_derived":
            gt_gate = _run_gt_experience_gate(run_root, **gt_gate_args)
            print(f"== panel GT experience gate: passed={gt_gate.get('passed')} ==")
        else:
            honesty = _run_honesty_gate(run_root)
            print(f"== panel honesty gate: passed={honesty.get('passed')} ==")
    elif summaries:
        print(json.dumps(summaries[0], indent=2))

    if args.warm_anchor == "gt_derived":
        panel_gate = _run_gt_experience_gate(run_root, **gt_gate_args)
    else:
        panel_gate = _run_honesty_gate(run_root)

    if args.only_ring and summaries:
        gate = summaries[0]
        if args.warm_anchor == "gt_derived":
            pass_criterion = "experience_gate_passed and gt_experience_gate_passed"
            passed = bool(gate["experience_gate_passed"] and panel_gate.get("passed"))
            single = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "case_id": gate["ring_key"],
                "command": (
                    "bo/run_layout_bo.py experience --warm-anchor gt_derived "
                    f"--only-ring {gate['ring_key']} --run-root {run_root}"
                ),
                "warm_anchor": args.warm_anchor,
                "target_n_evals": gate["n_trials"],
                "experience_gate": gate,
                "gt_experience_gate": panel_gate,
                "pass_criterion": pass_criterion,
                "passed": passed,
                "evidence_path": str(run_root / "single_instance_gate.json"),
            }
        else:
            stream_k_extra: dict = {}
            stream_d_extra: dict = {}
            if args.stream == "d":
                t, r = gate["ring_key"].split("/")
                trials_path = run_root / t / r / "bo_trials.csv"
                stream_k_best = 0.691
                sk_path = (
                    Path(args.k_handoff_root).resolve()
                    / t
                    / r
                    / "k_best_for_stream_d.json"
                )
                if sk_path.is_file():
                    stream_k_best = float(json.loads(sk_path.read_text())["best_bo_miou"])
                twin_spread = 0.0
                if trials_path.is_file():
                    tdf = pd.read_csv(trials_path)
                    base = tdf[tdf["kind"] == "twin_baseline"]
                    if not base.empty and "gt_miou_plus" in base.columns:
                        row = base.iloc[0]
                        mp, mm = row.get("gt_miou_plus"), row.get("gt_miou_minus")
                        if pd.notna(mp) and pd.notna(mm):
                            twin_spread = abs(float(mp) - float(mm))
                    oracle_best = None
                    if "gt_miou_plus" in tdf.columns and "gt_miou_minus" in tdf.columns:
                        oracle_best = float(
                            tdf[["gt_miou_plus", "gt_miou_minus"]]
                            .apply(pd.to_numeric, errors="coerce")
                            .max(axis=1)
                            .max()
                        )
                stream_d_extra = {
                    "twin_miou_spread": twin_spread,
                    "stream_k_best_miou_ref": stream_k_best,
                    "oracle_branch_miou_max": oracle_best,
                    "best_bo_beats_stream_k": bool(
                        gate["best_bo_miou"] >= stream_k_best
                        or (oracle_best is not None and oracle_best >= stream_k_best)
                    ),
                }
            if args.stream == "k":
                trials_path = run_root / gate["ring_key"].split("/")[0] / (
                    gate["ring_key"].split("/")[1]
                ) / "bo_trials.csv"
                stream_l_best = 0.345
                sam_smoke = 0.083
                sl_best_path = (
                    Path(args.layout_handoff_root).resolve()
                    / gate["ring_key"].split("/")[0]
                    / gate["ring_key"].split("/")[1]
                    / "layout_best_for_stream_k.json"
                )
                if sl_best_path.is_file():
                    stream_l_best = float(
                        json.loads(sl_best_path.read_text())["best_bo_miou"]
                    )
                sam_path = (
                    Path(args.prior_root).resolve()
                    / gate["ring_key"].replace("/", "_")
                    / "sam4tun_prior.json"
                )
                if sam_path.is_file():
                    sam_smoke = float(json.loads(sam_path.read_text()).get("smoke_gt_miou", sam_smoke))
                k_std = gate.get("miou_std", 0.0)
                if trials_path.is_file():
                    tdf = pd.read_csv(trials_path)
                    if "k_y_frac" in tdf.columns:
                        k_std = float(tdf["k_y_frac"].std())
                stream_k_extra = {
                    "k_y_frac_std": k_std,
                    "stream_l_best_miou_ref": stream_l_best,
                    "sam_smoke_miou_ref": sam_smoke,
                    "best_bo_beats_stream_l": bool(gate["best_bo_miou"] > stream_l_best),
                    "best_bo_beats_sam_smoke": bool(gate["best_bo_miou"] > sam_smoke),
                }
            single = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "case_id": gate["ring_key"],
                "command": (
                    f"bo/run_layout_bo.py experience --stream {args.stream} "
                    f"--warm-anchor {args.warm_anchor} --only-ring {gate['ring_key']} "
                    f"--run-root {run_root}"
                ),
                "experience_stream": args.stream,
                "warm_anchor": args.warm_anchor,
                "target_n_evals": gate["n_trials"],
                "experience_gate": gate,
                "honesty_gate": panel_gate,
                "stream_k_checks": stream_k_extra,
                "stream_d_checks": stream_d_extra,
                "pass_criterion": "experience_gate_passed and honesty_gate_passed and zero gt_layout trials",
                "passed": bool(gate["experience_gate_passed"] and panel_gate.get("passed")),
                "evidence_path": str(run_root / "single_instance_gate.json"),
            }
            if args.stream == "d" and stream_d_extra:
                single["pass_criterion"] = (
                    "experience_gate_passed and honesty_gate_passed; "
                    "twin spread>=0.02; best or oracle-branch mIoU >= stream_k"
                )
                single["passed"] = bool(
                    single["passed"]
                    and stream_d_extra.get("twin_miou_spread", 0) >= 0.02
                    and stream_d_extra.get("best_bo_beats_stream_k")
                )
            if args.stream == "k" and stream_k_extra:
                single["pass_criterion"] = (
                    "experience_gate_passed and honesty_gate_passed; "
                    "k_y_frac_std>0.05; best>mIoU stream_l and sam smoke"
                )
                single["passed"] = bool(
                    single["passed"]
                    and stream_k_extra.get("k_y_frac_std", 0) > 0.05
                    and stream_k_extra.get("best_bo_beats_stream_l")
                    and stream_k_extra.get("best_bo_beats_sam_smoke")
                )
        (run_root / "single_instance_gate.json").write_text(json.dumps(single, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(single, indent=2))
        return 0 if single["passed"] else 1

    if len(rings) > 1 and not panel_gate.get("passed"):
        return 1

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
                manifest_entry=entry,
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
    exp.add_argument("--n-evals", type=int, default=60, help="Default evals for non-sparse rings (sparse=120 via manifest slot)")
    exp.add_argument("--tunnel-id", default=None, help="Single ring without manifest")
    exp.add_argument("--ring-id", type=int, default=None)
    exp.add_argument("--segment-count", type=int, default=None, choices=[6, 7])
    exp.add_argument("--order-branch", default="plus", choices=["plus", "minus"])
    exp.add_argument(
        "--prior-root",
        default=str(REPO_ROOT / "logs" / "proxy4tun" / "sam4tun_prior"),
        help="SAM4Tun prior JSON root from build_sam4tun_prior.py (ignored when --warm-anchor gt_derived)",
    )
    exp.add_argument(
        "--warm-anchor",
        default="sam4tun",
        choices=["sam4tun", "geometric", "gt_derived"],
        help="Warm-start policy: sam4tun (v4), geometric (v3), gt_derived (v5 GT-anchor)",
    )
    exp.add_argument(
        "--stream",
        default="full",
        choices=["full", "layout", "k", "d"],
        help="full: joint BO; layout: Stream L; k: Stream K; d: Stream D (order, frozen L+K)",
    )
    exp.add_argument(
        "--layout-handoff-root",
        default=str(REPO_ROOT / "logs" / "proxy4tun" / "stream_l"),
        help="Stream L handoff root (layout_best_for_stream_k.json per ring)",
    )
    exp.add_argument(
        "--k-handoff-root",
        default=str(REPO_ROOT / "logs" / "proxy4tun" / "stream_k"),
        help="Stream K handoff root (k_best_for_stream_d.json per ring)",
    )
    exp.set_defaults(func=cmd_experience)

    ver = sub.add_parser("verify", help="GT layout encode/decode smoke test")
    _add_corpus_args(ver, require_run_root=False)
    ver.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
