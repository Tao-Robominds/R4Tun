"""Run full deterministic stabilisation (K + direction) on held-out panel.

For each held-out ring:
1) run preprocessing + detection once
2) run segmentation for direction plus and minus hypotheses
3) score/select direction using intrinsic-only evidence
4) persist chosen artefacts as canonical outputs

Writes under:
    logs/v3_direction_stabilisation_v1/<tunnel>/r<ring>/
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bo.v3._paths import assert_writable
from bo.v3 import r4tun_seed
from bo.v3.objectives import (
    PREPROCESSING_CLI,
    DETECTION_CLI,
    SEGMENTATION_CLI,
    VENV_PYTHON,
    _run_subprocess,
    REQUIRED_PRE_ARTEFACTS,
    REQUIRED_DET_ARTEFACTS,
    REQUIRED_SEG_ARTEFACTS,
)
from bo.v3.intrinsics import collect_trial_intrinsics
from bo.v3.ontology import evaluate_ontology

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PANEL_PATH = REPO_ROOT / "data" / "v3" / "panels" / "heldout" / "heldout_panel_v3.json"
BASELINE_SCOREBOARD = REPO_ROOT / "logs" / "v3" / "heldout" / "scoreboard_yrank.csv"
RUN_ROOT = REPO_ROOT / "logs" / "v3_direction_stabilisation_v1"


@dataclass
class VariantResult:
    name: str
    status: str
    miou_fixed: float | None
    miou_perm: float | None
    intrinsics: dict[str, Any]
    ontology: dict[str, Any]
    segment_completion_meta: dict[str, Any]
    n_pred_segments: int
    score: float
    selector_terms: dict[str, Any]


def _load_panel() -> list[dict[str, Any]]:
    payload = json.loads(PANEL_PATH.read_text())
    return list(payload["rings"])


def _load_baseline_map() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(BASELINE_SCOREBOARD, newline="") as f:
        for row in csv.DictReader(f):
            if row["arm"] != "a_unanchored":
                continue
            out[row["ring_key"]] = {
                "bottom_baseline_miou": float(row["miou_fixed_yrank"]),
                "k_only_miou": float(row["miou_fixed_canonical"]),
            }
    return out


def _check_required(ring_dir: Path, names: tuple[str, ...]) -> list[str]:
    return [n for n in names if not (ring_dir / n).exists()]


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


def _stage_input_ring(ring_dir: Path, tunnel_id: str, ring_id: int) -> None:
    candidates = (
        REPO_ROOT / "data" / "v3" / "panels" / "heldout" / "rings" / f"{tunnel_id.replace('-', '_')}_ring{ring_id}.txt",
        REPO_ROOT / "data" / "rings" / f"{tunnel_id.replace('-', '_')}_ring{ring_id}.txt",
    )
    for cand in candidates:
        if cand.exists():
            dst = ring_dir / f"{tunnel_id}_r{ring_id}.txt"
            if not dst.exists():
                shutil.copy2(cand, dst)
            return
    raise FileNotFoundError(f"input point cloud not found for {tunnel_id}/r{ring_id}")


def _run_pre_and_det(
    ring_root: Path,
    tunnel_id: str,
    ring_id: int,
    timeout_sec: float,
    mem_cap_bytes: int | None,
) -> None:
    # preprocessing
    pre_log = ring_root / "logs" / "preprocessing.log"
    info = _run_subprocess(
        [str(VENV_PYTHON), str(PREPROCESSING_CLI), tunnel_id, str(ring_id), "--data-dir", str(RUN_ROOT)],
        timeout_sec=timeout_sec,
        mem_cap_bytes=mem_cap_bytes,
        log_path=pre_log,
    )
    if info["timed_out"] or info["oom"] or info["returncode"] != 0:
        raise RuntimeError(f"preprocessing failed: {info['failure_detail']}")
    miss = _check_required(ring_root, REQUIRED_PRE_ARTEFACTS)
    if miss:
        raise RuntimeError(f"preprocessing missing artefacts: {miss}")

    # detection
    det_log = ring_root / "logs" / "detection.log"
    info = _run_subprocess(
        [str(VENV_PYTHON), str(DETECTION_CLI), tunnel_id, str(ring_id), "--data-dir", str(RUN_ROOT)],
        timeout_sec=timeout_sec,
        mem_cap_bytes=mem_cap_bytes,
        log_path=det_log,
    )
    if info["timed_out"] or info["oom"] or info["returncode"] != 0:
        raise RuntimeError(f"detection failed: {info['failure_detail']}")
    miss = _check_required(ring_root, REQUIRED_DET_ARTEFACTS)
    if miss:
        raise RuntimeError(f"detection missing artefacts: {miss}")


def _segment_variant(
    *,
    ring_root: Path,
    tunnel_id: str,
    ring_id: int,
    variant: str,
    timeout_sec: float,
    mem_cap_bytes: int | None,
) -> VariantResult:
    seg_csv_name = f"all_segments_direction_{variant}.csv"
    bnd_name = f"boundaries_per_ring_direction_{variant}.json"
    seg_csv = ring_root / seg_csv_name
    bnd = ring_root / bnd_name
    if not seg_csv.exists() or not bnd.exists():
        raise FileNotFoundError(f"direction files missing for {variant}")

    # Overwrite canonical boundary file so segmentation uses the selected direction.
    shutil.copy2(bnd, ring_root / "boundaries_per_ring.json")
    seg_log = ring_root / "logs" / f"segmentation_{variant}.log"
    info = _run_subprocess(
        [
            str(VENV_PYTHON),
            str(SEGMENTATION_CLI),
            tunnel_id,
            str(ring_id),
            "--data-dir",
            str(RUN_ROOT),
            "--segments-file",
            seg_csv_name,
        ],
        timeout_sec=timeout_sec,
        mem_cap_bytes=mem_cap_bytes,
        log_path=seg_log,
    )
    if info["timed_out"] or info["oom"] or info["returncode"] != 0:
        return VariantResult(
            name=variant,
            status="failed",
            miou_fixed=None,
            miou_perm=None,
            intrinsics={},
            ontology={},
            segment_completion_meta={},
            n_pred_segments=0,
            score=-1e9,
            selector_terms={"failure": info["failure_detail"]},
        )

    miss = _check_required(ring_root, REQUIRED_SEG_ARTEFACTS)
    if miss:
        return VariantResult(
            name=variant,
            status="failed",
            miou_fixed=None,
            miou_perm=None,
            intrinsics={},
            ontology={},
            segment_completion_meta={},
            n_pred_segments=0,
            score=-1e9,
            selector_terms={"failure": f"missing seg artefacts: {miss}"},
        )

    intr = collect_trial_intrinsics(ring_root)
    miou_fixed = intr.pop("miou_fixed_class", None)
    miou_perm = intr.pop("miou_permutation", None)
    ontology = evaluate_ontology(ring_root)
    seg_meta_path = ring_root / "segment_completion_meta_segmentation.json"
    seg_meta = json.loads(seg_meta_path.read_text()) if seg_meta_path.exists() else {}
    n_pred = 0
    try:
        import pandas as pd

        fdf = pd.read_csv(ring_root / "final.csv", usecols=lambda c: c == "pred")
        n_pred = len({int(v) for v in fdf["pred"].unique() if 1 <= int(v) <= 7})
    except Exception:  # noqa: BLE001
        pass

    score, terms = _score_direction(ontology, intr, seg_meta)
    # Persist variant artefacts
    shutil.copy2(ring_root / "final.csv", ring_root / f"final_direction_{variant}.csv")
    (ring_root / f"intrinsics_direction_{variant}.json").write_text(
        json.dumps(intr | {"miou_fixed_class": miou_fixed, "miou_permutation": miou_perm}, indent=2, default=str) + "\n"
    )
    (ring_root / f"ontology_direction_{variant}.json").write_text(
        json.dumps(ontology, indent=2, default=str) + "\n"
    )
    (ring_root / f"evaluation_direction_{variant}.json").write_text(
        json.dumps(
            {
                "ring_key": f"{tunnel_id}/r{ring_id}",
                "direction": variant,
                "miou_fixed_class": miou_fixed,
                "miou_permutation": miou_perm,
                "status": "ok",
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    return VariantResult(
        name=variant,
        status="ok",
        miou_fixed=miou_fixed,
        miou_perm=miou_perm,
        intrinsics=intr,
        ontology=ontology,
        segment_completion_meta=seg_meta,
        n_pred_segments=n_pred,
        score=score,
        selector_terms=terms,
    )


def _score_direction(ontology: dict[str, Any], intr: dict[str, Any], seg_meta: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    bd = ontology.get("breakdown", {})
    hard_pass = all(
        bool((bd.get(k) or {}).get("passed"))
        for k in ("O_block_set", "O_block_count", "O_no_duplicates")
    )
    one_k_pass = bool((bd.get("O_one_K_unique") or {}).get("passed"))
    structural = float(ontology.get("structural_score") or 0.0)
    seg_type = 1.0 if bool(intr.get("seg_segment_type_completeness")) else 0.0
    ring_comp = float(intr.get("seg_ring_completeness_avg") or 0.0)
    mask_cov = float(intr.get("seg_mask_coverage_pct") or 0.0) / 100.0
    var_ratio = intr.get("seg_block_size_variance_ratio")
    if var_ratio is None:
        balance = 0.0
    else:
        vr = float(var_ratio)
        if 3.0 <= vr <= 20.0:
            balance = 1.0
        else:
            balance = max(0.0, 1.0 - min(abs(vr - 10.0) / 20.0, 1.0))
    repairs = 0
    try:
        repairs = len((seg_meta.get("completion_after_projection") or {}).get("reassigned_point_indices") or {})
    except Exception:  # noqa: BLE001
        repairs = 0
    repair_penalty = min(0.25, 0.05 * repairs)
    gate = 1.0 if (hard_pass and one_k_pass) else 0.0
    score = gate * (
        0.40 * structural
        + 0.20 * seg_type
        + 0.15 * ring_comp
        + 0.10 * mask_cov
        + 0.15 * balance
    ) - repair_penalty
    return score, {
        "hard_pass": hard_pass,
        "one_k_pass": one_k_pass,
        "structural_score": structural,
        "seg_type_completeness": seg_type,
        "ring_completeness_avg": ring_comp,
        "mask_coverage_norm": mask_cov,
        "balance_score": balance,
        "repairs": repairs,
        "repair_penalty": repair_penalty,
        "score": score,
    }


def _choose_direction(plus: VariantResult, minus: VariantResult) -> tuple[str, str, dict[str, Any]]:
    if plus.status != "ok" and minus.status == "ok":
        return "minus", "high", {"reason": "plus_failed"}
    if minus.status != "ok" and plus.status == "ok":
        return "plus", "high", {"reason": "minus_failed"}
    if plus.status != "ok" and minus.status != "ok":
        return "failed", "failed", {"reason": "both_failed"}
    diff = float(plus.score - minus.score)
    if abs(diff) < 0.02:
        return "tie_plus_default", "low", {"reason": "near_tie", "score_diff": diff}
    if diff > 0:
        conf = "high" if diff >= 0.10 else "medium"
        return "plus", conf, {"reason": "higher_score", "score_diff": diff}
    conf = "high" if diff <= -0.10 else "medium"
    return "minus", conf, {"reason": "higher_score", "score_diff": diff}


def _materialize_choice(ring_root: Path, choice: str) -> None:
    if choice in {"plus", "tie_plus_default"}:
        suffix = "plus"
    elif choice == "minus":
        suffix = "minus"
    else:
        return
    shutil.copy2(ring_root / f"final_direction_{suffix}.csv", ring_root / "final.csv")
    shutil.copy2(
        ring_root / f"boundaries_per_ring_direction_{suffix}.json",
        ring_root / "boundaries_per_ring.json",
    )
    shutil.copy2(
        ring_root / f"intrinsics_direction_{suffix}.json",
        ring_root / "intrinsics.json",
    )
    shutil.copy2(
        ring_root / f"ontology_direction_{suffix}.json",
        ring_root / "ontology.json",
    )
    shutil.copy2(
        ring_root / f"evaluation_direction_{suffix}.json",
        ring_root / "evaluation.json",
    )


def _run_one_ring(
    rinfo: dict[str, Any],
    timeout_sec: float,
    mem_cap_bytes: int | None,
    fresh: bool,
) -> dict[str, Any]:
    tid = rinfo["tunnel_id"]
    rid = int(rinfo["ring_id"])
    ring_key = rinfo["ring_key"]
    ring_root = RUN_ROOT / tid / f"r{rid}"
    if ring_root.exists() and fresh:
        shutil.rmtree(ring_root)
    ring_root.mkdir(parents=True, exist_ok=True)
    (ring_root / "logs").mkdir(parents=True, exist_ok=True)
    assert_writable(ring_root)

    # seed params (unanchored to match bottom vs K-only comparison)
    diameter = _ring_diameter(rinfo)
    seed_pre = r4tun_seed.load_r4tun_preprocessing(target_tunnel_diameter=diameter)
    ga = seed_pre.setdefault("gravity_anchor", {})
    ga["enabled"] = False
    seed_det = r4tun_seed.load_r4tun_detection(target_tunnel_diameter=diameter)
    (ring_root / "parameters_preprocessing.json").write_text(json.dumps(seed_pre, indent=2) + "\n")
    (ring_root / "parameters_detection.json").write_text(json.dumps(seed_det, indent=2) + "\n")
    (ring_root / "parameters_segmentation.json").write_text(json.dumps({"k_cap": 130, "ab_cap": 390}, indent=2) + "\n")
    _stage_input_ring(ring_root, tid, rid)

    _run_pre_and_det(ring_root, tid, rid, timeout_sec, mem_cap_bytes)
    plus = _segment_variant(
        ring_root=ring_root,
        tunnel_id=tid,
        ring_id=rid,
        variant="plus",
        timeout_sec=timeout_sec,
        mem_cap_bytes=mem_cap_bytes,
    )
    minus = _segment_variant(
        ring_root=ring_root,
        tunnel_id=tid,
        ring_id=rid,
        variant="minus",
        timeout_sec=timeout_sec,
        mem_cap_bytes=mem_cap_bytes,
    )

    choice, conf, reason = _choose_direction(plus, minus)
    _materialize_choice(ring_root, choice)
    chosen = plus if choice in {"plus", "tie_plus_default"} else minus
    decision = {
        "ring_key": ring_key,
        "chosen_direction": choice,
        "direction_confidence": conf,
        "plus_score": plus.score,
        "minus_score": minus.score,
        "selector_terms": {
            "plus": plus.selector_terms,
            "minus": minus.selector_terms,
            "choice_reason": reason,
        },
        "gt_used_for_selection": False,
    }
    (ring_root / "direction_decision.json").write_text(json.dumps(decision, indent=2, default=str) + "\n")
    return {
        "ring_key": ring_key,
        "split": rinfo.get("split"),
        "pattern_type": rinfo.get("pattern_type"),
        "chosen_direction": choice,
        "direction_confidence": conf,
        "k_plus_direction_miou": chosen.miou_fixed,
        "ontology_passed": bool(chosen.ontology.get("passed")) if chosen.ontology else False,
        "n_pred_segments": chosen.n_pred_segments,
        "plus_miou": plus.miou_fixed,
        "minus_miou": minus.miou_fixed,
        "plus_status": plus.status,
        "minus_status": minus.status,
    }


def _write_outputs(rows: list[dict[str, Any]], baseline_map: dict[str, dict[str, float]]) -> None:
    score_rows: list[dict[str, Any]] = []
    for r in rows:
        base = baseline_map.get(r["ring_key"], {})
        bottom = base.get("bottom_baseline_miou")
        k_only = base.get("k_only_miou")
        kd = r.get("k_plus_direction_miou")
        score_rows.append(
            {
                **r,
                "bottom_baseline_miou": bottom,
                "k_only_miou": k_only,
                "lift_k_only_minus_bottom": (None if bottom is None or k_only is None else (k_only - bottom)),
                "lift_direction_minus_bottom": (None if bottom is None or kd is None else (kd - bottom)),
                "lift_direction_minus_k_only": (None if k_only is None or kd is None else (kd - k_only)),
            }
        )

    score_path = RUN_ROOT / "scoreboard.csv"
    fieldnames = [
        "ring_key",
        "split",
        "pattern_type",
        "bottom_baseline_miou",
        "k_only_miou",
        "k_plus_direction_miou",
        "lift_k_only_minus_bottom",
        "lift_direction_minus_bottom",
        "lift_direction_minus_k_only",
        "chosen_direction",
        "direction_confidence",
        "ontology_passed",
        "n_pred_segments",
        "plus_miou",
        "minus_miou",
        "plus_status",
        "minus_status",
    ]
    with score_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(score_rows)

    no_lift = [r for r in score_rows if r["lift_direction_minus_bottom"] is not None and r["lift_direction_minus_bottom"] <= 0]
    with (RUN_ROOT / "no_lift_rings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(no_lift, key=lambda x: x["lift_direction_minus_bottom"]))

    with (RUN_ROOT / "direction_decisions.csv").open("w", newline="") as f:
        fields = ["ring_key", "chosen_direction", "direction_confidence", "plus_miou", "minus_miou", "plus_status", "minus_status"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in score_rows:
            w.writerow({k: r.get(k) for k in fields})

    def _mean(vals: list[float | None]) -> float | None:
        nums = [v for v in vals if v is not None]
        return (sum(nums) / len(nums)) if nums else None

    summary = {
        "n_total": len(score_rows),
        "bottom_baseline_mean": _mean([r["bottom_baseline_miou"] for r in score_rows]),
        "k_only_mean": _mean([r["k_only_miou"] for r in score_rows]),
        "k_plus_direction_mean": _mean([r["k_plus_direction_miou"] for r in score_rows]),
        "n_no_lift_k_only": sum(
            (r["lift_k_only_minus_bottom"] is not None and r["lift_k_only_minus_bottom"] <= 0)
            for r in score_rows
        ),
        "n_no_lift_k_plus_direction": sum(
            (r["lift_direction_minus_bottom"] is not None and r["lift_direction_minus_bottom"] <= 0)
            for r in score_rows
        ),
    }
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="All-40 deterministic direction stabilisation run")
    p.add_argument("--rings", nargs="*", default=None, help="optional subset ring keys")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--mem-cap-gb", type=float, default=16.0)
    p.add_argument("--fresh", action="store_true", help="remove existing per-ring run dirs first")
    ns = p.parse_args(argv)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    panel = _load_panel()
    baseline_map = _load_baseline_map()
    ring_filter = set(ns.rings or [])
    rows: list[dict[str, Any]] = []
    for r in panel:
        if ring_filter and r["ring_key"] not in ring_filter:
            continue
        row = _run_one_ring(
            r,
            timeout_sec=float(ns.timeout),
            mem_cap_bytes=(int(ns.mem_cap_gb * (1024**3)) if ns.mem_cap_gb > 0 else None),
            fresh=bool(ns.fresh),
        )
        rows.append(row)
        print(f"[ok] {r['ring_key']} choice={row['chosen_direction']} miou={row['k_plus_direction_miou']}")
    _write_outputs(rows, baseline_map)
    print(f"Wrote: {RUN_ROOT / 'scoreboard.csv'}")
    print(f"Wrote: {RUN_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

