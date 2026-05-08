"""v3 held-out evaluation harness.

Runs the deployment-time pipeline (preprocessing -> detection -> segmentation
-> evaluation) on each ring in
``data/v3/panels/heldout/heldout_panel_v3.json``, with ``gravity_anchor.enabled``
toggled by ``--arm``. No BO, no LLM — this is the deterministic Arm A / Arm B
harness for the paper's three-arm ablation.

Usage::

    ./venv/bin/python -m bo.v3.heldout_runner --arm a_unanchored
    ./venv/bin/python -m bo.v3.heldout_runner --arm b_anchored
    ./venv/bin/python -m bo.v3.heldout_runner --arm a_unanchored --rings 4-3/r177 4-4/r212

Outputs per ring under ``logs/v3/heldout/<arm>/<tunnel>/r<ring>/``:

* ``parameters_preprocessing.json`` (r4tun-seed with gravity_anchor.enabled toggled)
* ``parameters_detection.json``     (r4tun-seed verbatim)
* ``parameters_segmentation.json``  (default irregular)
* full pipeline outputs (depth_map.npy, final.csv, ...)
* ``intrinsics.json``        (the v3 diagnostic intrinsics)
* ``evaluation.json``        (fixed-class and permutation-invariant mIoU)
* ``ontology.json``          (verdict from :mod:`bo.v3.ontology`)
* ``trial_status.json``      (the structured TrialResult payload)

Aggregate scoreboard at ``logs/v3/heldout/<arm>/scoreboard_<arm>.csv``.

Path immutability is enforced by :mod:`bo.v3._paths`; outputs land under
``logs/v3/heldout`` only.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bo.v3._paths import assert_writable  # noqa: E402
from bo.v3 import r4tun_seed  # noqa: E402
from bo.v3.objectives import STAGE_BASELINE, run_trial  # noqa: E402
from bo.v3.ontology import evaluate_ontology  # noqa: E402

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger("bo.v3.heldout")

PANEL_PATH = REPO_ROOT / "data" / "v3" / "panels" / "heldout" / "heldout_panel_v3.json"
ARMS = ("a_unanchored", "b_anchored")


# ---------------------------------------------------------------------------
# Panel + diameter lookup
# ---------------------------------------------------------------------------

def _load_panel() -> list[dict[str, Any]]:
    panel = json.loads(PANEL_PATH.read_text())
    return list(panel["rings"])


def _ring_diameter(rinfo: dict[str, Any]) -> float:
    """Best-available tunnel diameter, mirroring the BO baseline runner."""
    tid = rinfo["tunnel_id"]
    rid = int(rinfo["ring_id"])
    pre = (
        REPO_ROOT
        / "agents"
        / "1_preprocessing"
        / "parameters"
        / tid
        / f"r{rid}"
        / "parameters_preprocessing.json"
    )
    if pre.exists():
        try:
            d = json.loads(pre.read_text())
            return float(d.get("tunnel_diameter", 7.5))
        except Exception:  # noqa: BLE001
            pass
    return 7.5


# ---------------------------------------------------------------------------
# Per-arm parameter override
# ---------------------------------------------------------------------------

def _apply_arm_to_seed(seed_pre: dict[str, Any], arm: str) -> dict[str, Any]:
    """Toggle gravity_anchor.enabled on the preprocessing seed.

    Other parameters stay at the r4tun reference. The seed mutation is
    confined to a copy so the caller's dict is never disturbed.
    """
    out = json.loads(json.dumps(seed_pre))  # deep-copy via JSON round-trip
    ga = out.get("gravity_anchor")
    if not isinstance(ga, dict):
        ga = {"enabled": False, "n_bins": 360}
        out["gravity_anchor"] = ga
    if arm == "a_unanchored":
        ga["enabled"] = False
    elif arm == "b_anchored":
        ga["enabled"] = True
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return out


# ---------------------------------------------------------------------------
# Heavy-artefact pruning (mirrors run_v3_calibration._prune_heavy)
# ---------------------------------------------------------------------------

HEAVY_PATTERNS = (
    "unwrapped.csv",
    "denoised.csv",
    "enhanced.csv",
    "*.npy",
    "*.png",
)


def _prune_heavy(d: Optional[Path]) -> None:
    """Drop heavy intermediates after we have collected per-ring artefacts.

    We KEEP ``final.csv``, ``boundaries_per_ring.json``, ``pixel_to_point.pkl``,
    and ``depth_map.png`` because Arm-C reflection (and ontology) need them.
    Everything else can go.
    """
    if d is None:
        return
    p = Path(d)
    if not p.exists():
        return
    keep = {"depth_map.png", "depth_map_outlier.npy"}  # ontology needs depth_map.png; keep outlier npy too (small)
    for pat in HEAVY_PATTERNS:
        for f in p.rglob(pat):
            if f.name in keep:
                continue
            try:
                f.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Per-ring driver
# ---------------------------------------------------------------------------

def _run_one_ring(
    *,
    rinfo: dict[str, Any],
    arm: str,
    arm_root: Path,
    timeout_sec: float,
    mem_cap_bytes: Optional[int],
    fresh: bool,
) -> dict[str, Any]:
    tid = rinfo["tunnel_id"]
    rid = int(rinfo["ring_id"])
    rk = rinfo["ring_key"]
    sandbox = arm_root / tid / f"r{rid}"
    if sandbox.exists() and fresh:
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)
    assert_writable(sandbox)

    diameter = _ring_diameter(rinfo)
    seed_pre = r4tun_seed.load_r4tun_preprocessing(target_tunnel_diameter=diameter)
    seed_det = r4tun_seed.load_r4tun_detection(target_tunnel_diameter=diameter)
    seed_pre = _apply_arm_to_seed(seed_pre, arm)

    # The trial-dir convention from `run_trial` is that the per-ring outputs
    # land under <trial_dir>/<tunnel>/r<ring>/. Pass the arm sandbox as the
    # parent so the per-ring layout matches logs/v3/heldout/<arm>/<tunnel>/r<ring>/.
    trial_dir = arm_root  # per-ring nesting is added by run_trial via tid/rk
    ring_dir = trial_dir / tid / f"r{rid}"
    ring_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    result = run_trial(
        tunnel_id=tid,
        ring_id=rid,
        stage=STAGE_BASELINE,
        trial_dir=trial_dir,
        seed_pre_params=seed_pre,
        seed_det_params=seed_det,
        trial_params={},
        timeout_sec=timeout_sec,
        mem_cap_bytes=mem_cap_bytes,
    )
    elapsed = time.time() - started

    # Persist intrinsics + evaluation snapshots in canonical filenames so
    # downstream Arm-C reflection and the aggregator can find them.
    intrinsics_payload = dict(result.intrinsics or {})
    intrinsics_payload["miou_fixed_class"] = result.miou_fixed_class
    intrinsics_payload["miou_permutation"] = result.miou_permutation
    (ring_dir / "intrinsics.json").write_text(
        json.dumps(intrinsics_payload, indent=2, default=str) + "\n"
    )
    eval_payload = {
        "ring_key": rk,
        "arm": arm,
        "miou_fixed_class": result.miou_fixed_class,
        "miou_permutation": result.miou_permutation,
        "status": result.status,
        "failure_mode": result.failure_mode,
        "failure_detail": result.failure_detail,
        "elapsed_sec": result.elapsed_sec,
    }
    (ring_dir / "evaluation.json").write_text(
        json.dumps(eval_payload, indent=2, default=str) + "\n"
    )

    # Run ontology only if segmentation produced final.csv. _early_fail
    # routes still emit a verdict (with hard_failures listing the missing
    # final.csv) so the scoreboard always has an ontology column.
    ontology_verdict = evaluate_ontology(ring_dir)
    (ring_dir / "ontology.json").write_text(
        json.dumps(ontology_verdict, indent=2, default=str) + "\n"
    )

    # Drop heavy intermediates we no longer need for Arm A/B aggregation.
    # Keep depth_map.png (ontology), final.csv (mIoU + reflection),
    # boundaries_per_ring.json (ontology), pixel_to_point.pkl (reflection).
    _prune_heavy(ring_dir)

    return {
        "ring_key": rk,
        "split": rinfo.get("split"),
        "regime_label": rinfo.get("regime_label"),
        "stress_case": bool(
            rk in {"4-4/r212", "4-3/r177", "4-6/r283"}
        ),
        "miou_fixed_class": result.miou_fixed_class,
        "miou_permutation": result.miou_permutation,
        "status": result.status,
        "failure_mode": result.failure_mode,
        "elapsed_sec": result.elapsed_sec,
        "wall_clock_sec": elapsed,
        "ontology_passed": bool(ontology_verdict.get("passed")),
        "ontology_hard_failures": ";".join(ontology_verdict.get("hard_failures") or []),
        "ontology_structural_score": float(ontology_verdict.get("structural_score") or 0.0),
        "diameter_used": diameter,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    arm = args.arm
    if arm not in ARMS:
        raise SystemExit(f"--arm must be one of {ARMS}, got {arm!r}")

    panel = _load_panel()
    explicit = set(args.rings) if args.rings else None
    rings = [r for r in panel if (explicit is None or r["ring_key"] in explicit)]
    if explicit:
        missing = explicit - {r["ring_key"] for r in rings}
        if missing:
            raise SystemExit(f"ring_keys not in held-out panel: {sorted(missing)}")

    arm_root = REPO_ROOT / "logs" / "v3" / "heldout" / arm
    assert_writable(arm_root)
    arm_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    started = time.time()
    for i, rinfo in enumerate(rings, 1):
        rk = rinfo["ring_key"]
        logger.info("[%s] %d/%d %s — start", arm, i, len(rings), rk)
        try:
            row = _run_one_ring(
                rinfo=rinfo,
                arm=arm,
                arm_root=arm_root,
                timeout_sec=args.timeout,
                mem_cap_bytes=args.mem_cap_gb * (1024**3) if args.mem_cap_gb > 0 else None,
                fresh=bool(args.fresh),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] %s — runner exception", arm, rk)
            row = {
                "ring_key": rk,
                "split": rinfo.get("split"),
                "regime_label": rinfo.get("regime_label"),
                "stress_case": False,
                "miou_fixed_class": None,
                "miou_permutation": None,
                "status": "failed",
                "failure_mode": "runner_exception",
                "elapsed_sec": 0.0,
                "wall_clock_sec": 0.0,
                "ontology_passed": False,
                "ontology_hard_failures": f"runner_exception:{exc!r}",
                "ontology_structural_score": 0.0,
                "diameter_used": None,
            }
        rows.append(row)
        logger.info(
            "[%s] %s done: status=%s mIoU(fixed)=%s ontology_passed=%s",
            arm, rk, row["status"], row["miou_fixed_class"], row["ontology_passed"],
        )

    # Write the per-arm scoreboard.
    scoreboard_path = arm_root / f"scoreboard_arm_{arm[0]}.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(scoreboard_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Aggregate-level summary.
    successful = [r for r in rows if r["status"] == "ok" and r["miou_fixed_class"] is not None]
    n_ok = len(successful)
    if successful:
        mean_fixed = float(sum(r["miou_fixed_class"] for r in successful) / n_ok)
        mean_perm = float(
            sum((r["miou_permutation"] or 0.0) for r in successful) / n_ok
        )
    else:
        mean_fixed = float("nan")
        mean_perm = float("nan")
    summary = {
        "arm": arm,
        "panel_size": len(rings),
        "n_status_ok": n_ok,
        "n_failed": len(rows) - n_ok,
        "mean_miou_fixed_class": mean_fixed,
        "mean_miou_permutation": mean_perm,
        "n_ontology_passed": sum(1 for r in rows if r["ontology_passed"]),
        "elapsed_total_sec": time.time() - started,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "panel_path": str(PANEL_PATH.relative_to(REPO_ROOT)),
        "scoreboard_path": str(scoreboard_path.relative_to(REPO_ROOT)),
    }
    (arm_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("[%s] complete: %s", arm, summary)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="v3 held-out runner (Arm A/B harness)")
    p.add_argument(
        "--arm",
        required=True,
        choices=list(ARMS),
        help="a_unanchored = gravity_anchor.enabled=false; b_anchored = true",
    )
    p.add_argument("--rings", nargs="*", help="Optional explicit ring_keys (default: all 40)")
    p.add_argument("--timeout", type=float, default=600.0, help="Per-ring wall-clock cap (s)")
    p.add_argument("--mem-cap-gb", type=float, default=16.0)
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Wipe the per-ring sandbox before running (re-run from scratch)",
    )
    p.set_defaults(func=cmd_run)
    ns = p.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
