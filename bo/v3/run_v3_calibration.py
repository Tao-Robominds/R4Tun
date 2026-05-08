"""v3 BO calibration driver (Ax/BoTorch).

Usage::

    ./venv/bin/python -m bo.v3.run_v3_calibration baseline --rings 4-9/r365
    ./venv/bin/python -m bo.v3.run_v3_calibration bo \\
        --tunnel 4-9 --ring 365 --stage preprocessing --trials 10
    ./venv/bin/python -m bo.v3.run_v3_calibration bo \\
        --tunnel 4-9 --ring 365 --stage detection --trials 10 \\
        --frozen-pre logs/v3/bo_calibration/4-9/r365/baseline/sandbox/4-9/r365

Modes
-----
* ``baseline``: run the R4Tun-seeded pipeline once per ring and persist
  per-ring baseline mIoU + intrinsics under
  ``logs/v3/bo_calibration/<tunnel>/r<ring>/baseline/``.
* ``bo``: run an Ax experiment for one (ring, stage) pair, ``--trials``
  successful trials (failures are budgeted out and do not count). Outputs:

    logs/v3/bo_calibration/<tunnel>/r<ring>/<stage>/
        ax_experiment.json    (snapshot for resume / aggregation)
        trials/<trial_idx>/   (per-trial sandbox + trial_status.json)
        intrinsics.csv        (one row per trial, all intrinsic fields)
        best.json             (current best fixed-class mIoU)

Path immutability is enforced via :func:`bo.v3._paths.assert_writable` on
every directory the driver creates; any attempt to point ``--base-dir``
under a protected prefix raises before any work begins.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bo.v3._paths import assert_writable, REPO_ROOT  # noqa: E402
from bo.v3 import r4tun_seed  # noqa: E402
from bo.v3 import spaces  # noqa: E402
from bo.v3.objectives import (  # noqa: E402
    STAGE_BASELINE,
    STAGE_DETECTION,
    STAGE_PREPROCESSING,
    TrialResult,
    run_trial,
)

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger("bo.v3")


# ---------------------------------------------------------------------------
# Panel lookup
# ---------------------------------------------------------------------------

PANEL_PATH = REPO_ROOT / "data" / "v3" / "panels" / "bo" / "bo_calibration_panel_v3.json"


def _load_panel() -> dict[str, dict[str, Any]]:
    panel = json.loads(PANEL_PATH.read_text())
    return {r["ring_key"]: r for r in panel["rings"]}


def _ring_keys(panel: dict[str, dict[str, Any]], explicit: Optional[list[str]]) -> list[str]:
    if explicit:
        for rk in explicit:
            if rk not in panel:
                raise SystemExit(f"ring_key {rk!r} not in calibration panel")
        return explicit
    return list(panel.keys())


def _tunnel_diameter_for_panel(rinfo: dict[str, Any]) -> float:
    """Best-available tunnel diameter for the seed.

    Use the per-ring preprocessing parameter file if it exists (these are
    where v1/v2 stored the actual measured diameter); else fall back to
    the panel's family-based default (4.x → 7.5 m, 5.x → 7.5 m).
    """
    tid = rinfo["tunnel_id"]
    rid = rinfo["ring_id"]
    pre = REPO_ROOT / "agents" / "1_preprocessing" / "parameters" / tid / f"r{int(rid)}" / "parameters_preprocessing.json"
    if pre.exists():
        try:
            d = json.loads(pre.read_text())
            return float(d.get("tunnel_diameter", 7.5))
        except Exception:  # noqa: BLE001
            pass
    return 7.5


# ---------------------------------------------------------------------------
# Baseline mode
# ---------------------------------------------------------------------------

def cmd_baseline(args: argparse.Namespace) -> int:
    panel = _load_panel()
    rings = _ring_keys(panel, args.rings)
    base_root = REPO_ROOT / "logs" / "v3" / "bo_calibration"
    assert_writable(base_root)

    summary_rows: list[dict[str, Any]] = []
    for rk in rings:
        rinfo = panel[rk]
        tid = rinfo["tunnel_id"]
        rid = int(rinfo["ring_id"])
        ring_root = base_root / tid / f"r{rid}" / "baseline"
        sandbox = ring_root / "sandbox"
        if sandbox.exists() and args.fresh:
            shutil.rmtree(sandbox)
        sandbox.mkdir(parents=True, exist_ok=True)
        assert_writable(sandbox)
        diameter = _tunnel_diameter_for_panel(rinfo)
        seed_pre = r4tun_seed.load_r4tun_preprocessing(target_tunnel_diameter=diameter)
        seed_det = r4tun_seed.load_r4tun_detection(target_tunnel_diameter=diameter)
        # Baseline run with both seeds verbatim.
        result = run_trial(
            tunnel_id=tid,
            ring_id=rid,
            stage=STAGE_BASELINE,
            trial_dir=sandbox,
            seed_pre_params=seed_pre,
            seed_det_params=seed_det,
            trial_params={},
            timeout_sec=args.timeout,
            mem_cap_bytes=args.mem_cap_gb * (1024**3) if args.mem_cap_gb > 0 else None,
        )
        # Persist the baseline-specific summary alongside trial_status.json.
        baseline_payload = {
            "ring_key": rk,
            "regime_label": rinfo.get("regime_label"),
            "tunnel_diameter": diameter,
            "miou_fixed_class": result.miou_fixed_class,
            "miou_permutation": result.miou_permutation,
            "status": result.status,
            "failure_mode": result.failure_mode,
            "failure_detail": result.failure_detail,
            "elapsed_sec": result.elapsed_sec,
            "intrinsics": result.intrinsics,
            "trial_dir": result.trial_dir,
            "stage_logs": result.pre_logs,
        }
        (ring_root / "baseline_summary.json").write_text(
            json.dumps(baseline_payload, indent=2, default=str) + "\n"
        )
        summary_rows.append({
            "ring_key": rk,
            "regime_label": rinfo.get("regime_label"),
            "miou_fixed_class": result.miou_fixed_class,
            "miou_permutation": result.miou_permutation,
            "status": result.status,
            "failure_mode": result.failure_mode,
            "elapsed_sec": result.elapsed_sec,
        })
        logger.info(
            "baseline %s: status=%s mIoU(fixed)=%s mIoU(perm)=%s failure=%s elapsed=%.1fs",
            rk, result.status, result.miou_fixed_class, result.miou_permutation,
            result.failure_mode, result.elapsed_sec,
        )
    # Write a panel-level summary CSV.
    summary_path = base_root / "baseline_summary.csv"
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
    logger.info("baseline run complete; summary at %s", summary_path)
    return 0


# ---------------------------------------------------------------------------
# BO mode
# ---------------------------------------------------------------------------

def _ax_client(stage: str, *, seed_value: int = 12345):
    """Build an AxClient for one (ring, stage) experiment."""
    from ax.service.ax_client import AxClient, ObjectiveProperties

    if stage == STAGE_PREPROCESSING:
        params = spaces.preprocessing_space()
        constraints = spaces.preprocessing_constraints()
    elif stage == STAGE_DETECTION:
        params = spaces.detection_space(include_offsets=True, include_a4=True)
        constraints = spaces.detection_constraints()
    else:
        raise ValueError(f"BO mode does not support stage {stage!r}")

    ac = AxClient(verbose_logging=False, random_seed=int(seed_value))
    ac.create_experiment(
        name=f"v3_bo_{stage}",
        parameters=params,
        objectives={"miou_fixed": ObjectiveProperties(minimize=False)},
        tracking_metric_names=["miou_perm", "elapsed_sec"],
        parameter_constraints=constraints,
    )
    return ac


def _seed_initial_baseline(ax_client, *, seed_pre_params: dict[str, Any], seed_det_params: dict[str, Any], stage: str) -> None:
    """Inject the R4Tun seed as the first manually-attached trial.

    Ax will consume this as the initial ``data point`` so the GP starts
    from the regular-reference floor. The trial is not run here; we use
    ``attach_trial`` with the seed parameters and then immediately
    ``complete_trial`` once the baseline result is known by the caller.
    """
    if stage == STAGE_PREPROCESSING:
        keys = [p["name"] for p in spaces.preprocessing_space()]
        seed: dict[str, Any] = {}
        for k in keys:
            if k.startswith("target_distance_"):
                td = sorted([float(t) for t in seed_pre_params.get("target_distances", [0.06, 0.03, 0.015])], reverse=True)
                while len(td) < 3:
                    td.append(0.015)
                seed[k] = float(td[int(k.rsplit("_", 1)[-1]) - 1])
            elif k in seed_pre_params:
                seed[k] = seed_pre_params[k]
            else:
                # Not in seed; fall back to the bound midpoint.
                bounds = next(p["bounds"] for p in spaces.preprocessing_space() if p["name"] == k)
                seed[k] = (bounds[0] + bounds[1]) / 2.0
    else:
        keys = [p["name"] for p in spaces.detection_space(include_offsets=True, include_a4=True)]
        seed = {}
        for k in keys:
            if k.startswith("offset_"):
                blk = k[len("offset_"):]
                seed[k] = float(seed_det_params.get("per_ring_offsets", {}).get("0", {}).get(blk, 0.0))
            elif k in seed_det_params:
                seed[k] = seed_det_params[k]
            else:
                bounds = next(p["bounds"] for p in spaces.detection_space(include_offsets=True, include_a4=True) if p["name"] == k)
                seed[k] = (bounds[0] + bounds[1]) / 2.0
    # Clip to the declared bounds.
    for p in (
        spaces.preprocessing_space() if stage == STAGE_PREPROCESSING
        else spaces.detection_space(include_offsets=True, include_a4=True)
    ):
        nm, lo, hi = p["name"], p["bounds"][0], p["bounds"][1]
        if nm in seed:
            try:
                if p.get("value_type") == "int":
                    seed[nm] = int(round(max(lo, min(hi, float(seed[nm])))))
                else:
                    seed[nm] = float(max(lo, min(hi, float(seed[nm]))))
            except Exception:  # noqa: BLE001
                seed[nm] = lo
    ax_client.attach_trial(parameters=seed)


def cmd_bo(args: argparse.Namespace) -> int:
    panel = _load_panel()
    rk = f"{args.tunnel}/r{int(args.ring)}"
    if rk not in panel:
        raise SystemExit(f"ring_key {rk!r} not in calibration panel")
    rinfo = panel[rk]
    tid = rinfo["tunnel_id"]
    rid = int(rinfo["ring_id"])
    stage = args.stage
    if stage not in (STAGE_PREPROCESSING, STAGE_DETECTION):
        raise SystemExit(f"--stage must be one of {STAGE_PREPROCESSING}, {STAGE_DETECTION}")

    base_root = REPO_ROOT / "logs" / "v3" / "bo_calibration" / tid / f"r{rid}" / stage
    if args.label:
        base_root = base_root / str(args.label)
    assert_writable(base_root)
    base_root.mkdir(parents=True, exist_ok=True)
    trial_root = base_root / "trials"
    trial_root.mkdir(parents=True, exist_ok=True)

    diameter = _tunnel_diameter_for_panel(rinfo)
    seed_pre = r4tun_seed.load_r4tun_preprocessing(target_tunnel_diameter=diameter)
    seed_det = r4tun_seed.load_r4tun_detection(target_tunnel_diameter=diameter)

    frozen_pre_dir: Optional[Path] = None
    if stage == STAGE_DETECTION:
        if not args.frozen_pre:
            raise SystemExit("--frozen-pre is required for detection BO (path to a preprocessing winner ring dir)")
        frozen_pre_dir = Path(args.frozen_pre)
        if not frozen_pre_dir.exists():
            raise SystemExit(f"--frozen-pre dir does not exist: {frozen_pre_dir}")

    ax_client = _ax_client(stage, seed_value=int(args.seed))
    _seed_initial_baseline(ax_client, seed_pre_params=seed_pre, seed_det_params=seed_det, stage=stage)

    intrinsics_path = base_root / "intrinsics.csv"
    intrinsics_keys: list[str] = []
    intrinsics_rows: list[dict[str, Any]] = []
    best: dict[str, Any] = {"trial_index": None, "miou_fixed_class": float("-inf")}

    # Heavy artefacts (CSV intermediates / depth maps / point clouds) only
    # need to live on the running best-trial. Anything older or non-best is
    # pruned to keep BO from filling the disk over a 6-ring × 30-trial run.
    HEAVY_PATTERNS = (
        "unwrapped.csv", "denoised.csv", "enhanced.csv", "final.csv",
        "detected.csv", "all_segments.csv", "*.npy", "*.pkl", "*.png",
    )

    def _prune_heavy(d: Optional[Path | str]) -> None:
        if d is None:
            return
        p = Path(d)
        if not p.exists():
            return
        for pat in HEAVY_PATTERNS:
            for f in p.rglob(pat):
                try:
                    f.unlink()
                except OSError:
                    pass
        # Also drop the staged input ring file (only needed during the run).
        for f in p.rglob(f"{tid}_r{rid}.txt"):
            try:
                f.unlink()
            except OSError:
                pass

    started = time.time()
    successful = 0
    attempted = 0
    failure_counts: dict[str, int] = {}
    max_total = max(args.trials * 4, args.trials + 30)  # safety cap on total attempts
    while successful < args.trials and attempted < max_total:
        attempted += 1
        params, trial_idx = ax_client.get_next_trial()
        trial_dir = trial_root / f"trial_{trial_idx:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "params_suggested.json").write_text(json.dumps(params, indent=2, default=str) + "\n")
        logger.info("trial %d (stage=%s) launching with %d params", trial_idx, stage, len(params))
        result = run_trial(
            tunnel_id=tid,
            ring_id=rid,
            stage=stage,
            trial_dir=trial_dir,
            seed_pre_params=seed_pre,
            seed_det_params=seed_det,
            trial_params=params,
            frozen_pre_dir=frozen_pre_dir,
            timeout_sec=args.timeout,
            mem_cap_bytes=args.mem_cap_gb * (1024**3) if args.mem_cap_gb > 0 else None,
        )
        if result.status == "ok" and result.miou_fixed_class is not None:
            successful += 1
            ax_client.complete_trial(
                trial_idx,
                raw_data={
                    "miou_fixed": float(result.miou_fixed_class),
                    "miou_perm": float(result.miou_permutation) if result.miou_permutation is not None else float("nan"),
                    "elapsed_sec": float(result.elapsed_sec),
                },
            )
            if float(result.miou_fixed_class) > best["miou_fixed_class"]:
                # New best: prune the previous best's heavy artefacts.
                if best.get("trial_dir"):
                    _prune_heavy(best["trial_dir"])
                best = {
                    "trial_index": int(trial_idx),
                    "miou_fixed_class": float(result.miou_fixed_class),
                    "miou_permutation": (
                        float(result.miou_permutation) if result.miou_permutation is not None else None
                    ),
                    "trial_dir": result.trial_dir,
                    "parameters": params,
                }
                (base_root / "best.json").write_text(json.dumps(best, indent=2, default=str) + "\n")
            else:
                # Not the new best -> prune this trial's heavy artefacts now.
                _prune_heavy(trial_dir)
        else:
            ax_client.log_trial_failure(trial_index=trial_idx)
            mode = result.failure_mode or "unknown"
            failure_counts[mode] = failure_counts.get(mode, 0) + 1
            # Failed trials never serve as input to step 2b -> always prune.
            _prune_heavy(trial_dir)
        # Append intrinsics.csv row regardless of status.
        row = {
            "trial_index": int(trial_idx),
            "stage": stage,
            "status": result.status,
            "failure_mode": result.failure_mode,
            "miou_fixed_class": result.miou_fixed_class,
            "miou_permutation": result.miou_permutation,
            "elapsed_sec": result.elapsed_sec,
            **{f"param/{k}": v for k, v in params.items()},
            **{k: v for k, v in result.intrinsics.items() if not isinstance(v, list)},
        }
        for k in row:
            if k not in intrinsics_keys:
                intrinsics_keys.append(k)
        intrinsics_rows.append(row)
        # Flush after every trial so we never lose evidence to a crash.
        with open(intrinsics_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=intrinsics_keys)
            w.writeheader()
            for r in intrinsics_rows:
                w.writerow({k: r.get(k) for k in intrinsics_keys})
        logger.info(
            "trial %d → %s mIoU(fixed)=%s mIoU(perm)=%s elapsed=%.1fs (succ=%d/%d, fail=%s)",
            trial_idx, result.status, result.miou_fixed_class, result.miou_permutation,
            result.elapsed_sec, successful, args.trials, failure_counts,
        )
        if args.max_wall_clock and (time.time() - started) > args.max_wall_clock:
            logger.warning("reached --max-wall-clock; stopping early at %d successful trials", successful)
            break

    # Snapshot the Ax experiment (best-effort: schema differs across Ax 1.x).
    snap_path = base_root / "ax_experiment.json"
    try:
        ax_client.save_to_json_file(filepath=str(snap_path))
    except Exception as exc:  # noqa: BLE001
        snap_payload = {
            "warning": f"AxClient.save_to_json_file failed: {exc!r}",
            "best": best,
            "successful_trials": successful,
            "attempted_trials": attempted,
            "failure_counts": failure_counts,
        }
        snap_path.write_text(json.dumps(snap_payload, indent=2, default=str) + "\n")
    summary = {
        "tunnel_id": tid,
        "ring_id": rid,
        "stage": stage,
        "best": best,
        "successful_trials": successful,
        "attempted_trials": attempted,
        "failure_counts": failure_counts,
        "elapsed_total_sec": time.time() - started,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (base_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    logger.info("BO complete: %s", summary)
    return 0


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="v3 BO calibration driver (Ax/BoTorch)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("baseline", help="Run R4Tun-seeded baseline on each calibration ring")
    pb.add_argument("--rings", nargs="*", help="Optional explicit ring_keys (default: full panel)")
    pb.add_argument("--timeout", type=float, default=600.0)
    pb.add_argument("--mem-cap-gb", type=float, default=16.0)
    pb.add_argument("--fresh", action="store_true", help="Wipe existing baseline sandbox before running")
    pb.set_defaults(func=cmd_baseline)

    pb2 = sub.add_parser("bo", help="Run one Ax experiment for one (ring, stage)")
    pb2.add_argument("--tunnel", required=True)
    pb2.add_argument("--ring", required=True, type=int)
    pb2.add_argument("--stage", required=True, choices=[STAGE_PREPROCESSING, STAGE_DETECTION])
    pb2.add_argument("--trials", type=int, default=10, help="Number of *successful* trials (failures don't count)")
    pb2.add_argument("--seed", type=int, default=12345)
    pb2.add_argument("--timeout", type=float, default=600.0, help="Per-trial wall-clock cap (s)")
    pb2.add_argument("--mem-cap-gb", type=float, default=16.0)
    pb2.add_argument("--max-wall-clock", type=float, default=0.0, help="Total wall-clock cap (s); 0 = unlimited")
    pb2.add_argument("--frozen-pre", default=None, help="Path to a preprocessing winner ring dir (required for stage=detection)")
    pb2.add_argument("--label", default=None, help="Optional sub-label so multiple runs do not collide (e.g. pilot)")
    pb2.set_defaults(func=cmd_bo)

    ns = p.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
