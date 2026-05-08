"""Single-ring driver for Arm C interactive reflection.

Two subcommands cover one reflection iteration:

* ``snapshot`` builds the LLM-facing state for iteration ``k`` (parent
  parameters, intrinsics, ontology verdict, J_reflect_v3) and renders
  the prompt template at
  ``bo/v3/prompts/reflect_one_ring.md`` into a markdown file the
  Cursor session pastes to Cursor-Opus-4.7.
* ``apply`` consumes the LLM's JSON proposal, clips it to the frozen
  hard bounds, runs the pipeline once (preprocessing -> detection ->
  segmentation -> evaluation), and writes the iteration trace under
  ``logs/v3/heldout/c_reflection/<tunnel>/r<ring>/iters/i<k>/``.

Iteration chaining
------------------

* iter 1's parent is the per-ring Arm B output
  (``logs/v3/heldout/b_anchored/<tunnel>/r<ring>/``).
* iter ``k > 1``'s parent is ``iters/i<k-1>/``.

Per-iteration outputs
---------------------

* ``snapshot.md``       — the LLM prompt populated with current state
* ``proposal.json``     — the LLM's raw output (saved verbatim by the user)
* ``proposal_applied.json`` — the proposal after hard-bound clipping
* ``parameters_preprocessing.json`` etc — the resolved sandbox parameters
* ``intrinsics.json``, ``evaluation.json``, ``ontology.json`` — runtime
  artefacts (same schema as Arm A/B)
* ``j_reflect_v3.json`` — composite score breakdown for this iteration
* ``iter_trace.json``   — single-row summary suitable for the scoreboard
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bo.v3._paths import assert_writable  # noqa: E402
from bo.v3 import r4tun_seed  # noqa: E402
from bo.v3.objectives import (  # noqa: E402
    STAGE_BASELINE,
    REQUIRED_PRE_ARTEFACTS,
    REQUIRED_DET_ARTEFACTS,
    REQUIRED_SEG_ARTEFACTS,
    _run_subprocess,
    PREPROCESSING_CLI,
    DETECTION_CLI,
    SEGMENTATION_CLI,
    VENV_PYTHON,
)
from bo.v3.intrinsics import collect_trial_intrinsics  # noqa: E402
from bo.v3.ontology import compute_j_reflect_v3, evaluate_ontology  # noqa: E402

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger("bo.v3.reflect")

PANEL_PATH = REPO_ROOT / "data" / "v3" / "panels" / "heldout" / "heldout_panel_v3.json"
FROZEN_PATH = REPO_ROOT / "data" / "v3" / "calibration" / "llm_loop_frozen.json"
PROMPT_PATH = REPO_ROOT / "bo" / "v3" / "prompts" / "reflect_one_ring.md"
ARM_B_ROOT = REPO_ROOT / "logs" / "v3" / "heldout" / "b_anchored"
ARM_C_ROOT = REPO_ROOT / "logs" / "v3" / "heldout" / "c_reflection"

# Knobs that the agents pipeline reads as integers.
INT_KNOBS = ("outlier_neighbors", "interpolation_window")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_panel() -> dict[str, dict[str, Any]]:
    panel = json.loads(PANEL_PATH.read_text())
    return {r["ring_key"]: r for r in panel["rings"]}


def _load_frozen() -> dict[str, Any]:
    return json.loads(FROZEN_PATH.read_text())


def _ring_diameter(rinfo: dict[str, Any]) -> float:
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


def _ring_paths(ring_key: str) -> tuple[str, int]:
    if "/r" not in ring_key:
        raise SystemExit(f"ring_key must look like '4-3/r177', got {ring_key!r}")
    tid, rest = ring_key.split("/r", 1)
    return tid, int(rest)


def _parent_dir(ring_key: str, iter_k: int) -> Path:
    tid, rid = _ring_paths(ring_key)
    if iter_k == 1:
        parent = ARM_B_ROOT / tid / f"r{rid}"
    else:
        parent = ARM_C_ROOT / tid / f"r{rid}" / "iters" / f"i{iter_k - 1}"
    return parent


def _iter_dir(ring_key: str, iter_k: int) -> Path:
    tid, rid = _ring_paths(ring_key)
    return ARM_C_ROOT / tid / f"r{rid}" / "iters" / f"i{iter_k}"


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def _format_threshold_arrow(intr_name: str, value: Any, frozen: dict[str, Any]) -> str:
    """Return e.g. ``0.0008 (perm: >= 0.000475 ✗)`` for the prompt table."""
    direction = ">="
    perm: Optional[float] = None
    if intr_name == "miou_perm_minus_fixed_gap":
        direction = "<="
        perm = (
            frozen["guardrails"]["G_stability"]["thresholds_permissive"]
            .get("miou_perm_minus_fixed_gap")
        )
    else:
        diag = next(
            (d for d in frozen["diagnostic_intrinsics"] if d["name"] == intr_name),
            None,
        )
        if diag is None:
            return f"{value!s} (no calibrated threshold)"
        perm = diag.get("permissive_threshold_top_quartile_min")
    if value is None:
        return f"<missing> (perm: {direction} {perm}, ✗)"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{value!s} (perm: {direction} {perm}, ?)"
    if perm is None:
        return f"{v:.6g} (no calibrated threshold)"
    if direction == ">=":
        ok = v >= float(perm)
    else:
        ok = v <= float(perm)
    flag = "✓" if ok else "✗"
    return f"{v:.6g} (perm: {direction} {perm:.6g}, {flag})"


def _read_parent_state(ring_key: str, iter_k: int) -> dict[str, Any]:
    """Load parent intrinsics + evaluation + ontology + parameters."""
    parent = _parent_dir(ring_key, iter_k)
    if not parent.exists():
        raise SystemExit(
            f"parent directory missing for iter {iter_k}: {parent}\n"
            "Run Arm B first (or the previous iteration)."
        )
    state: dict[str, Any] = {"parent_dir": str(parent.relative_to(REPO_ROOT))}
    for fname in ("intrinsics.json", "evaluation.json", "ontology.json"):
        p = parent / fname
        if p.exists():
            state[fname.replace(".json", "")] = json.loads(p.read_text())
    pre = parent / "parameters_preprocessing.json"
    det = parent / "parameters_detection.json"
    state["parameters_preprocessing"] = (
        json.loads(pre.read_text()) if pre.exists() else None
    )
    state["parameters_detection"] = (
        json.loads(det.read_text()) if det.exists() else None
    )
    return state


def _bundle_pass(name: str, intrinsics: dict[str, Any], frozen: dict[str, Any]) -> bool:
    """Evaluate one guardrail bundle on the parent state's intrinsics."""
    bundle = frozen["guardrails"][name]
    rule = bundle["rule"]
    perm = bundle["thresholds_permissive"]
    for intr_name, t in perm.items():
        v = intrinsics.get(intr_name)
        if v is None and intr_name == "miou_perm_minus_fixed_gap":
            mf = intrinsics.get("miou_fixed_class")
            mp = intrinsics.get("miou_permutation")
            if mf is not None and mp is not None:
                v = float(mp) - float(mf)
        if v is None:
            return False
        try:
            v = float(v)
        except (TypeError, ValueError):
            return False
        if rule == "all_of_min" and v < float(t):
            return False
        if rule == "max_below" and v > float(t):
            return False
    return True


def _render_snapshot(
    *,
    ring_key: str,
    iter_k: int,
    state: dict[str, Any],
    frozen: dict[str, Any],
    rinfo: dict[str, Any],
) -> str:
    template = PROMPT_PATH.read_text()
    intrinsics = state.get("intrinsics") or {}
    evaluation = state.get("evaluation") or {}
    ontology = state.get("ontology") or {}

    # Tunable parameters table
    rows = ["| knob | default_deployable | soft_bounds | hard_bounds | spearman | sign |",
            "|------|--------------------|-------------|-------------|----------|------|"]
    pre_params = state.get("parameters_preprocessing") or {}
    for p in frozen["tunable_parameters"]:
        knob = p["name"]
        cur = pre_params.get(knob)
        if knob == "target_distance_1":
            tds = pre_params.get("target_distances") or [0.08, 0.04, 0.02]
            cur = float(tds[0]) if len(tds) >= 1 else None
        if knob == "target_distance_2":
            tds = pre_params.get("target_distances") or [0.08, 0.04, 0.02]
            cur = float(tds[1]) if len(tds) >= 2 else None
        sb = p["soft_bounds_p25_p75"]
        hb = p["hard_bounds_min_max"]
        s = p["pooled_spearman_vs_miou"]
        sign = "↑" if s > 0 else "↓"
        rows.append(
            f"| `{knob}` | {cur if cur is not None else p['default_deployable']} "
            f"(deploy={p['default_deployable']}) | "
            f"[{sb[0]:.4g}, {sb[1]:.4g}] | "
            f"[{hb[0]:.4g}, {hb[1]:.4g}] | "
            f"{s:+.3f} | {sign} |"
        )
    tunable_table = "\n".join(rows)

    # Guardrail bundles block
    bundle_lines = []
    for bname in ("G_pre", "G_layout", "G_stability"):
        b = frozen["guardrails"][bname]
        bundle_lines.append(f"### `{bname}` (rule = `{b['rule']}`)")
        bundle_lines.append(f"_{b['rationale']}_")
        bundle_lines.append("")
        bundle_lines.append("| intrinsic | strict | permissive |")
        bundle_lines.append("|-----------|--------|------------|")
        for intr in b["intrinsics"]:
            s = b["thresholds_strict"].get(intr)
            p = b["thresholds_permissive"].get(intr)
            bundle_lines.append(f"| `{intr}` | {s} | {p} |")
        passed = _bundle_pass(bname, intrinsics, frozen)
        bundle_lines.append(f"\nParent state: **{'PASS' if passed else 'FAIL'}**\n")
    guardrail_block = "\n".join(bundle_lines)

    # Intrinsics table
    intr_rows = ["| intrinsic | value (vs permissive threshold) |",
                 "|-----------|----------------------------------|"]
    for diag in frozen["diagnostic_intrinsics"]:
        nm = diag["name"]
        v = intrinsics.get(nm)
        intr_rows.append(f"| `{nm}` | {_format_threshold_arrow(nm, v, frozen)} |")
    mf = intrinsics.get("miou_fixed_class") or evaluation.get("miou_fixed_class")
    mp = intrinsics.get("miou_permutation") or evaluation.get("miou_permutation")
    if mf is not None and mp is not None:
        gap = float(mp) - float(mf)
        intr_rows.append(
            f"| `miou_perm_minus_fixed_gap` | "
            f"{_format_threshold_arrow('miou_perm_minus_fixed_gap', gap, frozen)} |"
        )
    intr_table = "\n".join(intr_rows)

    # Ontology table
    ont_rows = ["| check | passed | tag |", "|-------|--------|-----|"]
    HARD = {"O_block_set", "O_block_count", "O_no_duplicates"}
    for nm, payload in (ontology.get("breakdown") or {}).items():
        tag = "HARD" if nm in HARD else "SOFT"
        passed = payload.get("passed")
        ont_rows.append(f"| `{nm}` | {'✓' if passed else '✗'} | {tag} |")
    ont_table = "\n".join(ont_rows)
    hard_failures = ontology.get("hard_failures") or []
    hf_list = "_(none)_" if not hard_failures else ", ".join(f"`{h}`" for h in hard_failures)

    # J_reflect_v3 of parent state
    g_pre = _bundle_pass("G_pre", intrinsics, frozen)
    g_layout = _bundle_pass("G_layout", intrinsics, frozen)
    g_stab = _bundle_pass("G_stability", intrinsics, frozen)
    parent_j = compute_j_reflect_v3(
        ontology_verdict=ontology or {},
        g_pre_pass=g_pre,
        g_layout_pass=g_layout,
        g_stability_pass=g_stab,
    )

    populated = (
        template.replace("{{TUNABLE_PARAMETERS_TABLE}}", tunable_table)
        .replace("{{GUARDRAIL_BUNDLES}}", guardrail_block)
        .replace("{{ITER}}", str(iter_k))
        .replace("{{RING_KEY}}", ring_key)
        .replace("{{REGIME_LABEL}}", str(rinfo.get("regime_label")))
        .replace("{{SPLIT}}", str(rinfo.get("split")))
        .replace("{{PARENT_ITER}}", str(iter_k - 1) if iter_k > 1 else "Arm B")
        .replace("{{PARENT_MIOU_FIXED}}", f"{mf:.6g}" if mf is not None else "n/a")
        .replace("{{PARENT_MIOU_PERM}}", f"{mp:.6g}" if mp is not None else "n/a")
        .replace("{{PARENT_J_REFLECT_V3}}", f"{parent_j:.4f}")
        .replace("{{INTRINSICS_TABLE}}", intr_table)
        .replace("{{ONTOLOGY_VERDICT_TABLE}}", ont_table)
        .replace("{{HARD_FAILURES_LIST}}", hf_list)
    )
    return populated


def cmd_snapshot(args: argparse.Namespace) -> int:
    panel = _load_panel()
    if args.ring not in panel:
        raise SystemExit(f"ring_key {args.ring!r} not in held-out panel")
    rinfo = panel[args.ring]
    frozen = _load_frozen()
    state = _read_parent_state(args.ring, int(args.iter))
    snapshot_md = _render_snapshot(
        ring_key=args.ring, iter_k=int(args.iter), state=state, frozen=frozen, rinfo=rinfo,
    )
    iter_dir = _iter_dir(args.ring, int(args.iter))
    iter_dir.mkdir(parents=True, exist_ok=True)
    assert_writable(iter_dir)
    out = iter_dir / "snapshot.md"
    out.write_text(snapshot_md)
    logger.info("snapshot written to %s", out.relative_to(REPO_ROOT))
    if args.print:
        print(snapshot_md)
    return 0


# ---------------------------------------------------------------------------
# Apply (run pipeline once with the proposed parameters)
# ---------------------------------------------------------------------------

def _clip_proposal(proposal: dict[str, Any], frozen: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Clip values to hard_bounds_min_max; return (clipped, notes)."""
    notes: list[str] = []
    out: dict[str, Any] = {}
    knob_index = {p["name"]: p for p in frozen["tunable_parameters"]}
    for k, v in proposal.items():
        if k not in knob_index:
            notes.append(f"REJECTED unknown knob {k!r}")
            continue
        p = knob_index[k]
        lo, hi = p["hard_bounds_min_max"]
        try:
            vf = float(v)
        except (TypeError, ValueError):
            notes.append(f"REJECTED non-numeric value for {k}: {v!r}")
            continue
        original = vf
        vf = max(float(lo), min(float(hi), vf))
        if k in INT_KNOBS:
            vf = int(round(vf))
        if abs(float(original) - float(vf)) > 1e-9:
            notes.append(
                f"clipped {k} from {original!r} to {vf!r} (hard_bounds=[{lo}, {hi}])"
            )
        out[k] = vf
    return out, notes


def _apply_to_pre_params(base: dict[str, Any], clipped: dict[str, Any]) -> dict[str, Any]:
    """Merge clipped knob values into a preprocessing parameter dict.

    Mirrors :func:`bo.v3.objectives.render_preprocessing_params` for the 5
    surviving v3 knobs only; everything else stays at the parent values.
    """
    out = dict(base)
    if "radius_max" in clipped:
        out["radius_max"] = float(clipped["radius_max"])
        rmin = float(out.get("radius_min", 2.3))
        if out["radius_max"] <= rmin + 0.05:
            out["radius_max"] = rmin + 0.05
    if "outlier_neighbors" in clipped:
        out["outlier_neighbors"] = int(clipped["outlier_neighbors"])
    if "interpolation_window" in clipped:
        out["interpolation_window"] = int(clipped["interpolation_window"])
    if "target_distance_1" in clipped or "target_distance_2" in clipped:
        td = list(out.get("target_distances") or [0.08, 0.04, 0.02])
        while len(td) < 3:
            td.append(0.02)
        if "target_distance_1" in clipped:
            td[0] = float(clipped["target_distance_1"])
        if "target_distance_2" in clipped:
            td[1] = float(clipped["target_distance_2"])
        out["target_distances"] = sorted([float(x) for x in td[:3]], reverse=True)
    return out


def _stage_input_ring(tid: str, rid: int, sandbox_root: Path) -> Optional[Path]:
    """Copy the held-out point cloud into the iter sandbox if not present."""
    candidates = (
        REPO_ROOT / "data" / "v3" / "panels" / "heldout" / "rings" / f"{tid.replace('-', '_')}_ring{rid}.txt",
        REPO_ROOT / "data" / "rings" / f"{tid.replace('-', '_')}_ring{rid}.txt",
    )
    for cand in candidates:
        if cand.exists():
            ring_dir = sandbox_root / tid / f"r{rid}"
            ring_dir.mkdir(parents=True, exist_ok=True)
            dst = ring_dir / f"{tid}_r{rid}.txt"
            if not dst.exists():
                shutil.copy2(cand, dst)
            return dst
    return None


def _run_pipeline(*, tid: str, rid: int, sandbox_root: Path, timeout_sec: float, mem_cap_bytes: Optional[int]) -> dict[str, Any]:
    """Run preprocessing -> detection -> segmentation in the iter sandbox."""
    ring_dir = sandbox_root / tid / f"r{rid}"
    log_dir = sandbox_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, Any]] = []
    for cli, log_name, required in (
        (PREPROCESSING_CLI, "preprocessing.log", REQUIRED_PRE_ARTEFACTS),
        (DETECTION_CLI, "detection.log", REQUIRED_DET_ARTEFACTS),
        (SEGMENTATION_CLI, "segmentation.log", REQUIRED_SEG_ARTEFACTS),
    ):
        info = _run_subprocess(
            [str(VENV_PYTHON), str(cli), tid, str(rid), "--data-dir", str(sandbox_root)],
            timeout_sec=float(timeout_sec),
            mem_cap_bytes=mem_cap_bytes,
            log_path=log_dir / log_name,
        )
        info["stage"] = log_name.split(".")[0]
        logs.append(info)
        if info["timed_out"] or info["oom"] or info["returncode"] != 0:
            return {"status": "failed", "stage_logs": logs, "stage": info["stage"]}
        for r in required:
            if not (ring_dir / r).exists():
                return {
                    "status": "failed",
                    "stage_logs": logs,
                    "stage": info["stage"],
                    "missing": r,
                }
    return {"status": "ok", "stage_logs": logs}


def cmd_apply(args: argparse.Namespace) -> int:
    panel = _load_panel()
    if args.ring not in panel:
        raise SystemExit(f"ring_key {args.ring!r} not in held-out panel")
    rinfo = panel[args.ring]
    frozen = _load_frozen()
    iter_k = int(args.iter)
    iter_dir = _iter_dir(args.ring, iter_k)
    iter_dir.mkdir(parents=True, exist_ok=True)
    assert_writable(iter_dir)

    proposal_path = Path(args.proposal).resolve()
    proposal = json.loads(proposal_path.read_text())
    raw = proposal.get("proposal") if "proposal" in proposal else proposal
    rationale = (
        args.rationale or proposal.get("rationale") or "(no rationale provided)"
    )
    if not isinstance(raw, dict):
        raise SystemExit("proposal JSON must contain an object under 'proposal'")
    (iter_dir / "proposal.json").write_text(
        json.dumps({"proposal": raw, "rationale": rationale}, indent=2) + "\n"
    )

    # Build the parameter set: parent's params + clipped proposal patch
    parent = _read_parent_state(args.ring, iter_k)
    base_pre = parent.get("parameters_preprocessing")
    base_det = parent.get("parameters_detection")
    if base_pre is None or base_det is None:
        # Fall back to a fresh r4tun seed (anchored) if parent JSON missing.
        diameter = _ring_diameter(rinfo)
        base_pre = r4tun_seed.load_r4tun_preprocessing(target_tunnel_diameter=diameter)
        base_det = r4tun_seed.load_r4tun_detection(target_tunnel_diameter=diameter)
        ga = base_pre.setdefault("gravity_anchor", {})
        ga["enabled"] = True

    clipped, notes = _clip_proposal(raw, frozen)
    new_pre = _apply_to_pre_params(base_pre, clipped)
    (iter_dir / "proposal_applied.json").write_text(
        json.dumps({"clipped": clipped, "notes": notes}, indent=2) + "\n"
    )

    # Stage input + write parameter files into iter sandbox.
    tid = rinfo["tunnel_id"]
    rid = int(rinfo["ring_id"])
    ring_dir = iter_dir / tid / f"r{rid}"
    ring_dir.mkdir(parents=True, exist_ok=True)
    src_ring = _stage_input_ring(tid, rid, iter_dir)
    if src_ring is None:
        raise SystemExit(f"input point cloud not found for {args.ring}")
    (ring_dir / "parameters_preprocessing.json").write_text(json.dumps(new_pre, indent=2) + "\n")
    (ring_dir / "parameters_detection.json").write_text(json.dumps(base_det, indent=2) + "\n")
    (ring_dir / "parameters_segmentation.json").write_text(
        json.dumps({"k_cap": 130, "ab_cap": 390}, indent=2) + "\n"
    )

    # If the proposal is a no-op AND the user passed --short-circuit, skip
    # the run and just record a "plateaued" trace.
    if not clipped and args.short_circuit:
        trace = {
            "ring_key": args.ring,
            "iter": iter_k,
            "status": "plateaued",
            "miou_fixed_class": (parent.get("evaluation") or {}).get("miou_fixed_class"),
            "miou_permutation": (parent.get("evaluation") or {}).get("miou_permutation"),
            "j_reflect_v3": None,
            "rationale": rationale,
            "proposal_clipped": clipped,
        }
        (iter_dir / "iter_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        logger.info("plateaued at iter %d for %s", iter_k, args.ring)
        return 0

    # Run the pipeline once.
    started = time.time()
    run = _run_pipeline(
        tid=tid, rid=rid, sandbox_root=iter_dir,
        timeout_sec=float(args.timeout),
        mem_cap_bytes=args.mem_cap_gb * (1024**3) if args.mem_cap_gb > 0 else None,
    )
    elapsed = time.time() - started
    intrinsics_payload: dict[str, Any] = {}
    eval_payload: dict[str, Any] = {"ring_key": args.ring, "iter": iter_k, "elapsed_sec": elapsed}

    if run["status"] == "ok":
        intrinsics_full = collect_trial_intrinsics(ring_dir)
        miou_fixed = intrinsics_full.pop("miou_fixed_class", None)
        miou_perm = intrinsics_full.pop("miou_permutation", None)
        intrinsics_payload = dict(intrinsics_full)
        intrinsics_payload["miou_fixed_class"] = miou_fixed
        intrinsics_payload["miou_permutation"] = miou_perm
        eval_payload["miou_fixed_class"] = miou_fixed
        eval_payload["miou_permutation"] = miou_perm
        eval_payload["status"] = "ok"
    else:
        eval_payload["status"] = "failed"
        eval_payload["stage_failed"] = run.get("stage")

    (ring_dir / "intrinsics.json").write_text(json.dumps(intrinsics_payload, indent=2, default=str) + "\n")
    (ring_dir / "evaluation.json").write_text(json.dumps(eval_payload, indent=2, default=str) + "\n")

    ontology_verdict = evaluate_ontology(ring_dir)
    (ring_dir / "ontology.json").write_text(json.dumps(ontology_verdict, indent=2, default=str) + "\n")

    g_pre = _bundle_pass("G_pre", intrinsics_payload, frozen)
    g_layout = _bundle_pass("G_layout", intrinsics_payload, frozen)
    g_stab = _bundle_pass("G_stability", intrinsics_payload, frozen)
    j = compute_j_reflect_v3(
        ontology_verdict=ontology_verdict,
        g_pre_pass=g_pre,
        g_layout_pass=g_layout,
        g_stability_pass=g_stab,
    )
    (iter_dir / "j_reflect_v3.json").write_text(
        json.dumps({
            "j_reflect_v3": j,
            "g_pre_pass": g_pre,
            "g_layout_pass": g_layout,
            "g_stability_pass": g_stab,
            "structural_score": ontology_verdict.get("structural_score"),
        }, indent=2) + "\n"
    )

    trace = {
        "ring_key": args.ring,
        "iter": iter_k,
        "status": eval_payload["status"],
        "miou_fixed_class": eval_payload.get("miou_fixed_class"),
        "miou_permutation": eval_payload.get("miou_permutation"),
        "j_reflect_v3": j,
        "g_pre_pass": g_pre,
        "g_layout_pass": g_layout,
        "g_stability_pass": g_stab,
        "ontology_passed": bool(ontology_verdict.get("passed")),
        "ontology_hard_failures": ontology_verdict.get("hard_failures"),
        "rationale": rationale,
        "proposal_clipped": clipped,
        "elapsed_sec": elapsed,
    }
    (iter_dir / "iter_trace.json").write_text(json.dumps(trace, indent=2, default=str) + "\n")
    # Lift the per-iter artefacts to iter_dir/ root so the next iteration's
    # _read_parent_state finds them at the canonical flat layout (it looks
    # at iter_dir/, not iter_dir/<tunnel>/r<ring>/).
    for fname in (
        "parameters_preprocessing.json",
        "parameters_detection.json",
        "intrinsics.json",
        "evaluation.json",
        "ontology.json",
    ):
        src = ring_dir / fname
        if src.exists():
            shutil.copy2(src, iter_dir / fname)
    logger.info(
        "iter %d/%s done: mIoU(fixed)=%s J=%.4f hard_failures=%s",
        iter_k, args.ring, eval_payload.get("miou_fixed_class"), j,
        ontology_verdict.get("hard_failures"),
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Single-ring Arm-C reflection driver")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("snapshot", help="Render the LLM prompt for one iteration")
    ps.add_argument("--ring", required=True, help="ring_key, e.g. 4-3/r177")
    ps.add_argument("--iter", required=True, type=int, help="iteration index (1-based)")
    ps.add_argument("--print", action="store_true", help="also print the snapshot to stdout")
    ps.set_defaults(func=cmd_snapshot)

    pa = sub.add_parser("apply", help="Run the pipeline once with the proposed parameters")
    pa.add_argument("--ring", required=True)
    pa.add_argument("--iter", required=True, type=int)
    pa.add_argument("--proposal", required=True, help="Path to the LLM's JSON proposal")
    pa.add_argument("--rationale", default=None, help="Override rationale (defaults to JSON's value)")
    pa.add_argument("--timeout", type=float, default=900.0)
    pa.add_argument("--mem-cap-gb", type=float, default=16.0)
    pa.add_argument(
        "--short-circuit",
        action="store_true",
        help="If proposal is empty after clipping, mark the iter as plateaued without running.",
    )
    pa.set_defaults(func=cmd_apply)

    ns = p.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
