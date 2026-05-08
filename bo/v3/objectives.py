"""Per-trial pipeline runner for v3 BO.

Each BO trial is a sandboxed subprocess (or in-process call) that runs:

    preprocessing  →  detection  →  segmentation  →  evaluation (intrinsics)

For BO Stage 2a (preprocessing-tuned trial), the trial parameters override
``parameters_preprocessing.json`` while ``parameters_detection.json`` stays
at the seed defaults. For BO Stage 2b (detection-tuned trial), the
preprocessing winner's depth maps + pixel mappings are *copied* into the
sandbox (so the same preprocessing artefacts are reused across all
detection trials) and the trial parameters override
``parameters_detection.json`` only.

Failure modes (timeout, OOM via :func:`resource.setrlimit`, missing
required artefacts, exceptions in the agent CLIs) are caught here, logged
under ``<trial_dir>/trial_status.json``, and reported to the driver so it
can call :meth:`AxClient.log_trial_failure`. The objective wrapper does
not raise on failure; it returns a structured dict with
``status="failed"`` and ``failure_mode`` set so the driver can map
infeasible regions in parameter space.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .intrinsics import collect_trial_intrinsics

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = REPO_ROOT / "venv" / "bin" / "python"
PREPROCESSING_CLI = REPO_ROOT / "agents" / "1_preprocessing" / "1_preprocessing.py"
DETECTION_CLI = REPO_ROOT / "agents" / "2_detection" / "2_detection.py"
SEGMENTATION_CLI = REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py"

# The agents pipeline expects depth_map.png as well; CV2/PIL writes it.
REQUIRED_PRE_ARTEFACTS = ("depth_map_outlier.npy", "depth_map.png", "pixel_to_point.pkl", "ring_count.txt")
REQUIRED_DET_ARTEFACTS = ("boundaries_per_ring.json", "all_segments.csv")
REQUIRED_SEG_ARTEFACTS = ("final.csv",)


# ---------------------------------------------------------------------------
# Stage classification
# ---------------------------------------------------------------------------

STAGE_PREPROCESSING = "preprocessing"
STAGE_DETECTION = "detection"
STAGE_BASELINE = "baseline"


@dataclass
class TrialResult:
    """Structured result of a single BO trial.

    Status is one of ``ok``, ``failed``. When ``failed``, ``failure_mode``
    is one of ``oom``, ``timeout``, ``missing_artefact_pre``,
    ``missing_artefact_det``, ``missing_artefact_seg``, ``missing_eval``,
    ``exception``. ``intrinsics`` always carries whatever the extractors
    could compute (even after a partial pipeline failure).
    """

    status: str
    miou_fixed_class: Optional[float]
    miou_permutation: Optional[float]
    intrinsics: dict[str, Any]
    elapsed_sec: float
    trial_dir: str
    stage: str
    parameters_used: dict[str, Any]
    failure_mode: Optional[str] = None
    failure_detail: Optional[str] = None
    pre_logs: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Parameter merging
# ---------------------------------------------------------------------------

def _flatten_offset_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Strip Ax-style ``offset_*`` parameters into a per-block dict.

    The agents pipeline reads per-block offsets from
    ``parameters_detection.json["per_ring_offsets"]["0"][<block>]``;
    Ax exposes them as flat ``offset_K``, ``offset_B1``, ... entries.
    Returns ``(remaining_params, offsets_for_ring_0)``.
    """
    offsets: dict[str, float] = {}
    out: dict[str, Any] = {}
    for k, v in params.items():
        if k.startswith("offset_"):
            block = k[len("offset_"):]
            offsets[block] = float(v)
        else:
            out[k] = v
    return out, offsets


def render_preprocessing_params(
    base: dict[str, Any], trial_params: dict[str, Any]
) -> dict[str, Any]:
    """Merge BO-suggested preprocessing parameters into the seed dict.

    Honours Stage-2a search-space contracts:
    - ``radius_max`` is clamped strictly above ``radius_min`` (the
      pipeline collapses if the gate inverts).
    - ``target_distances`` is rebuilt from the three trial scalars and
      sorted descending.
    """
    out = dict(base)
    for k, v in trial_params.items():
        if k.startswith("target_distance_"):
            continue  # handled below
        out[k] = v
    out["radius_min"] = float(trial_params.get("radius_min", out.get("radius_min", 2.3)))
    out["radius_max"] = float(
        max(
            trial_params.get("radius_max", out.get("radius_max", 3.0)),
            float(out["radius_min"]) + 0.05,
        )
    )
    td = sorted(
        [
            float(trial_params.get("target_distance_1", 0.06)),
            float(trial_params.get("target_distance_2", 0.03)),
            float(trial_params.get("target_distance_3", 0.015)),
        ],
        reverse=True,
    )
    out["target_distances"] = td
    # Mirror legacy aliases the agents pipeline still reads.
    if "outlier_interpolation_radius" in trial_params:
        out["inter_radius"] = float(trial_params["outlier_interpolation_radius"])
    if "outlier_num_interpolations" in trial_params:
        out["num_interpolations"] = int(trial_params["outlier_num_interpolations"])
    if "curvature_neighbors" in trial_params:
        out["num_neighbors"] = int(trial_params["curvature_neighbors"])
    return out


def render_detection_params(
    base: dict[str, Any], trial_params: dict[str, Any]
) -> dict[str, Any]:
    """Merge BO-suggested detection parameters into the seed dict.

    Honours Stage-2b search-space contracts:
    - Angle ranges enforce ``pos_min < pos_max`` and ``neg_min < neg_max``.
    - ``canny_low < canny_high`` is enforced.
    - ``offset_*`` flat keys are folded into ``per_ring_offsets["0"]``.
    """
    others, offsets = _flatten_offset_params(trial_params)
    out = dict(base)
    out.update(others)
    # Numeric cleanups.
    if "angle_pos_min" in out and "angle_pos_max" in out:
        if out["angle_pos_min"] >= out["angle_pos_max"]:
            out["angle_pos_max"] = out["angle_pos_min"] + 0.5
    if "angle_neg_min" in out and "angle_neg_max" in out:
        if out["angle_neg_min"] >= out["angle_neg_max"]:
            out["angle_neg_max"] = out["angle_neg_min"] + 0.5
    if "canny_low" in out and "canny_high" in out:
        if out["canny_low"] >= out["canny_high"]:
            out["canny_high"] = out["canny_low"] + 1
    if offsets:
        # Preserve any pre-existing offsets the seed had for blocks we are
        # not tuning (e.g. when detection_space is built without offset_A4).
        existing = dict(out.get("per_ring_offsets", {}).get("0", {}))
        existing.update({k: float(v) for k, v in offsets.items()})
        out.setdefault("per_ring_offsets", {})["0"] = existing
    return out


# ---------------------------------------------------------------------------
# Subprocess runner with timeout + memory cap
# ---------------------------------------------------------------------------

def _run_subprocess(
    cmd: list[str],
    *,
    timeout_sec: float,
    mem_cap_bytes: Optional[int],
    env_extra: Optional[dict[str, str]] = None,
    log_path: Path,
) -> dict[str, Any]:
    """Run a CLI subprocess with timeout + (best-effort) memory cap.

    On Linux the memory cap is enforced via :func:`resource.setrlimit` in
    the child process (preexec). Output is written to ``log_path``. Returns
    a dict with ``returncode``, ``elapsed_sec``, ``timed_out``,
    ``oom``, ``cmd``.
    """
    import resource

    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _preexec() -> None:
        if mem_cap_bytes is not None:
            try:
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (int(mem_cap_bytes), int(mem_cap_bytes)),
                )
            except Exception:  # noqa: BLE001
                pass
        # New process group so SIGKILL can wipe the whole subtree on timeout.
        os.setpgrp()

    env = os.environ.copy()
    # Pin the agents' parameter loaders to the trial sandbox so checked-in
    # per-ring overrides under agents/<stage>/parameters/<tunnel>/r<ring>/
    # do not shadow the BO trial's parameters.
    env.setdefault("INTRINSIC_PARAMS_BASE_DIR_ONLY", "1")
    if env_extra:
        env.update(env_extra)
    start = time.time()
    timed_out = False
    oom = False
    rc = -1
    detail = ""
    with open(log_path, "w") as logf:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                preexec_fn=_preexec,
                env=env,
            )
            try:
                rc = proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
                proc.wait(timeout=10)
                rc = -9
                detail = f"timeout after {timeout_sec:.0f}s"
        except Exception as exc:  # noqa: BLE001
            rc = -1
            detail = f"spawn failure: {exc!r}\n{traceback.format_exc()}"
    elapsed = time.time() - start
    # OOM detection: scan the log tail for MemoryError / Killed (137).
    try:
        tail = log_path.read_text()[-4096:]
    except Exception:  # noqa: BLE001
        tail = ""
    if not timed_out and (
        "MemoryError" in tail
        or "OOM" in tail
        or "Killed" in tail
        or "MemoryError(" in tail
    ):
        oom = True
        if not detail:
            detail = "OOM detected in subprocess output"
    if not timed_out and rc != 0 and not detail:
        detail = f"non-zero exit code {rc}; tail:\n{tail[-600:]}"
    return {
        "returncode": int(rc),
        "elapsed_sec": float(elapsed),
        "timed_out": bool(timed_out),
        "oom": bool(oom),
        "cmd": [str(c) for c in cmd],
        "log_path": str(log_path),
        "failure_detail": detail,
    }


def _check_artefacts(ring_dir: Path, names: tuple[str, ...]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for n in names:
        if not (ring_dir / n).exists():
            missing.append(n)
    return (len(missing) == 0), missing


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(
    *,
    tunnel_id: str,
    ring_id: int,
    stage: str,
    trial_dir: Path,
    seed_pre_params: dict[str, Any],
    seed_det_params: dict[str, Any],
    trial_params: dict[str, Any],
    frozen_pre_dir: Optional[Path] = None,
    timeout_sec: float = 600.0,
    mem_cap_bytes: Optional[int] = 16 * 1024**3,
) -> TrialResult:
    """Run one full pipeline trial and return a :class:`TrialResult`.

    Parameters
    ----------
    stage:
        One of :data:`STAGE_PREPROCESSING`, :data:`STAGE_DETECTION`,
        :data:`STAGE_BASELINE`. ``frozen_pre_dir`` MUST be set when
        ``stage == STAGE_DETECTION`` (the driver provides the
        preprocessing-winner's directory to copy from).
    trial_dir:
        Per-trial sandbox root; the agents pipeline will read/write under
        ``<trial_dir>/<tunnel_id>/r<ring_id>/``.
    seed_pre_params, seed_det_params:
        Flat-schema seed dicts produced by :mod:`bo.v3.r4tun_seed`.
    trial_params:
        The Ax-suggested parameters for the active stage. Passed through
        :func:`render_preprocessing_params` / :func:`render_detection_params`.
    """
    trial_dir = Path(trial_dir)
    ring_key = f"r{int(ring_id)}"
    ring_dir = trial_dir / tunnel_id / ring_key
    ring_dir.mkdir(parents=True, exist_ok=True)

    pre_params = (
        render_preprocessing_params(seed_pre_params, trial_params)
        if stage == STAGE_PREPROCESSING
        else dict(seed_pre_params)
    )
    det_params = (
        render_detection_params(seed_det_params, trial_params)
        if stage == STAGE_DETECTION
        else dict(seed_det_params)
    )
    if stage == STAGE_BASELINE:
        # Baseline uses both seeds verbatim.
        pass

    (ring_dir / "parameters_preprocessing.json").write_text(json.dumps(pre_params, indent=2) + "\n")
    (ring_dir / "parameters_detection.json").write_text(json.dumps(det_params, indent=2) + "\n")
    # Segmentation defaults — mirror agents/3_segmentation/parameters/_default_irregular.
    (ring_dir / "parameters_segmentation.json").write_text(json.dumps({"k_cap": 130, "ab_cap": 390}, indent=2) + "\n")

    pre_logs: list[dict[str, Any]] = []
    start = time.time()

    # ----- Stage A: preprocessing (run unless we are reusing frozen artefacts)
    if stage == STAGE_DETECTION and frozen_pre_dir is not None:
        # Copy the preprocessing winner's outputs into this trial's dir.
        for fname in (
            "unwrapped.csv",
            "denoised.csv",
            "enhanced.csv",
            "ring_count.txt",
            "depth_map.npy",
            "depth_map.png",
            "depth_map_outlier.npy",
            "pixel_to_point.pkl",
            "gravity_anchor_meta.json",
        ):
            src = frozen_pre_dir / fname
            if src.exists():
                shutil.copy2(src, ring_dir / fname)
        # Sanity check.
        ok, missing = _check_artefacts(ring_dir, REQUIRED_PRE_ARTEFACTS)
        if not ok:
            return _early_fail(
                trial_dir, stage, ring_dir, "missing_artefact_pre", pre_params, det_params,
                f"frozen preprocessing missing: {missing}", pre_logs, time.time() - start,
            )
    else:
        # Stage the input ring point cloud.
        src_ring = REPO_ROOT / "data" / "v3" / "panels" / "bo" / "rings" / f"{tunnel_id.replace('-', '_')}_ring{int(ring_id)}.txt"
        if not src_ring.exists():
            src_ring = REPO_ROOT / "data" / "v3" / "panels" / "heldout" / "rings" / f"{tunnel_id.replace('-', '_')}_ring{int(ring_id)}.txt"
        if not src_ring.exists():
            src_ring = REPO_ROOT / "data" / "rings" / f"{tunnel_id.replace('-', '_')}_ring{int(ring_id)}.txt"
        if not src_ring.exists():
            return _early_fail(
                trial_dir, stage, ring_dir, "missing_input", pre_params, det_params,
                f"input ring not found: {src_ring}", pre_logs, time.time() - start,
            )
        dst_ring = ring_dir / f"{tunnel_id}_r{int(ring_id)}.txt"
        if not dst_ring.exists():
            shutil.copy2(src_ring, dst_ring)

        log_pre = trial_dir / "logs" / "preprocessing.log"
        info = _run_subprocess(
            [
                str(VENV_PYTHON),
                str(PREPROCESSING_CLI),
                tunnel_id,
                str(int(ring_id)),
                "--data-dir",
                str(trial_dir),
            ],
            timeout_sec=float(timeout_sec),
            mem_cap_bytes=mem_cap_bytes,
            log_path=log_pre,
        )
        info["stage"] = "preprocessing"
        pre_logs.append(info)
        if info["timed_out"]:
            return _early_fail(trial_dir, stage, ring_dir, "timeout", pre_params, det_params, info["failure_detail"], pre_logs, time.time() - start)
        if info["oom"]:
            return _early_fail(trial_dir, stage, ring_dir, "oom", pre_params, det_params, info["failure_detail"], pre_logs, time.time() - start)
        if info["returncode"] != 0:
            return _early_fail(trial_dir, stage, ring_dir, "exception", pre_params, det_params, info["failure_detail"], pre_logs, time.time() - start)
        ok, missing = _check_artefacts(ring_dir, REQUIRED_PRE_ARTEFACTS)
        if not ok:
            return _early_fail(
                trial_dir, stage, ring_dir, "missing_artefact_pre", pre_params, det_params,
                f"missing after preprocessing: {missing}", pre_logs, time.time() - start,
            )

    # ----- Stage B: detection
    log_det = trial_dir / "logs" / "detection.log"
    info = _run_subprocess(
        [
            str(VENV_PYTHON),
            str(DETECTION_CLI),
            tunnel_id,
            str(int(ring_id)),
            "--data-dir",
            str(trial_dir),
        ],
        timeout_sec=float(timeout_sec),
        mem_cap_bytes=mem_cap_bytes,
        log_path=log_det,
    )
    info["stage"] = "detection"
    pre_logs.append(info)
    if info["timed_out"]:
        return _early_fail(trial_dir, stage, ring_dir, "timeout", pre_params, det_params, info["failure_detail"], pre_logs, time.time() - start)
    if info["oom"]:
        return _early_fail(trial_dir, stage, ring_dir, "oom", pre_params, det_params, info["failure_detail"], pre_logs, time.time() - start)
    if info["returncode"] != 0:
        return _early_fail(trial_dir, stage, ring_dir, "exception", pre_params, det_params, info["failure_detail"], pre_logs, time.time() - start)
    ok, missing = _check_artefacts(ring_dir, REQUIRED_DET_ARTEFACTS)
    if not ok:
        return _early_fail(
            trial_dir, stage, ring_dir, "missing_artefact_det", pre_params, det_params,
            f"missing after detection: {missing}", pre_logs, time.time() - start,
        )

    # ----- Stage C: segmentation
    log_seg = trial_dir / "logs" / "segmentation.log"
    info = _run_subprocess(
        [
            str(VENV_PYTHON),
            str(SEGMENTATION_CLI),
            tunnel_id,
            str(int(ring_id)),
            "--data-dir",
            str(trial_dir),
        ],
        timeout_sec=float(timeout_sec),
        mem_cap_bytes=mem_cap_bytes,
        log_path=log_seg,
    )
    info["stage"] = "segmentation"
    pre_logs.append(info)
    if info["timed_out"]:
        return _early_fail(trial_dir, stage, ring_dir, "timeout", pre_params, det_params, info["failure_detail"], pre_logs, time.time() - start)
    if info["oom"]:
        return _early_fail(trial_dir, stage, ring_dir, "oom", pre_params, det_params, info["failure_detail"], pre_logs, time.time() - start)
    if info["returncode"] != 0:
        return _early_fail(trial_dir, stage, ring_dir, "exception", pre_params, det_params, info["failure_detail"], pre_logs, time.time() - start)
    ok, missing = _check_artefacts(ring_dir, REQUIRED_SEG_ARTEFACTS)
    if not ok:
        return _early_fail(
            trial_dir, stage, ring_dir, "missing_artefact_seg", pre_params, det_params,
            f"missing after segmentation: {missing}", pre_logs, time.time() - start,
        )

    # ----- Intrinsics + mIoU
    intrinsics = collect_trial_intrinsics(ring_dir)
    miou_fixed = intrinsics.pop("miou_fixed_class", None)
    miou_perm = intrinsics.pop("miou_permutation", None)

    elapsed = time.time() - start
    result = TrialResult(
        status="ok",
        miou_fixed_class=miou_fixed,
        miou_permutation=miou_perm,
        intrinsics=intrinsics,
        elapsed_sec=elapsed,
        trial_dir=str(trial_dir),
        stage=stage,
        parameters_used={"preprocessing": pre_params, "detection": det_params},
        pre_logs=pre_logs,
    )
    _persist_trial_status(trial_dir, result)
    return result


def _early_fail(
    trial_dir: Path,
    stage: str,
    ring_dir: Path,
    failure_mode: str,
    pre_params: dict[str, Any],
    det_params: dict[str, Any],
    detail: str,
    pre_logs: list[dict[str, Any]],
    elapsed: float,
) -> TrialResult:
    """Build a failure :class:`TrialResult`, salvaging whatever intrinsics
    the partial pipeline left behind (e.g. preprocessing-only)."""
    salvage = collect_trial_intrinsics(ring_dir, include_segmentation=False)
    miou_fixed = salvage.pop("miou_fixed_class", None)
    miou_perm = salvage.pop("miou_permutation", None)
    result = TrialResult(
        status="failed",
        miou_fixed_class=miou_fixed,
        miou_permutation=miou_perm,
        intrinsics=salvage,
        elapsed_sec=elapsed,
        trial_dir=str(trial_dir),
        stage=stage,
        parameters_used={"preprocessing": pre_params, "detection": det_params},
        failure_mode=failure_mode,
        failure_detail=detail,
        pre_logs=pre_logs,
    )
    _persist_trial_status(trial_dir, result)
    return result


def _persist_trial_status(trial_dir: Path, result: TrialResult) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "trial_status.json").write_text(json.dumps(result.to_json(), indent=2, default=str) + "\n")
