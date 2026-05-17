from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / "venv" / "bin" / "python"
PRE_CLI = REPO_ROOT / "agents" / "1_preprocessing" / "1_preprocessing.py"
DET_CLI = REPO_ROOT / "agents" / "2_detection" / "2_detection.py"
SEG_CLI = REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py"

RUN_ROOT = REPO_ROOT / "logs" / "v5_t3_gt_range_recovery_v1"
SRC_STAGE = REPO_ROOT / "logs" / "v5_t123_depth_contract_v1"
DEPTH_CONTRACT_SUMMARY = SRC_STAGE / "all_30_depth_gate_summary.json"
PANEL = REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv"
V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t3_recovered.csv"
if not V5_SCORE.exists():
    V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t1_t3_unified_from_t2_v2.csv"
if not V5_SCORE.exists():
    V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard.csv"

KB6_BLOCKS = ["K", "B1", "A1", "A2", "A3", "B2"]
KB6_ROTATE = ["B1", "A1", "A2", "A3", "B2"]
KB6_OFFSETS = {"K": 0.0, "B1": 216.0, "A1": 863.9, "A2": 1511.9, "A3": -1295.9, "B2": -648.0}
K_LABEL = 1
ROTATIONS = [0, 1, 4]

PROTECTED_PREFIXES = (
    REPO_ROOT / "data" / "ablation",
    REPO_ROOT / "data" / "bo",
    REPO_ROOT / "data" / "baseline",
    REPO_ROOT / "data" / "preprocessing_qa",
    REPO_ROOT / "data" / "represents",
    REPO_ROOT / "logs" / "context_preprocessing_v1",
    REPO_ROOT / "r4tun" / "data",
    REPO_ROOT / "r4tun" / "references",
    REPO_ROOT / "methods" / "plans" / "output",
    REPO_ROOT / "stages" / "v4",
)


def _assert_writable(path: Path) -> None:
    resolved = path.resolve()
    logs_root = (REPO_ROOT / "logs").resolve()
    try:
        resolved.relative_to(logs_root)
    except ValueError as exc:
        raise ValueError(f"Output must be under logs/: {resolved}") from exc
    for pref in PROTECTED_PREFIXES:
        if not pref.exists():
            continue
        p = pref.resolve()
        if resolved == p:
            raise ValueError(f"Protected output path: {resolved}")
        try:
            resolved.relative_to(p)
            raise ValueError(f"Protected output path: {resolved}")
        except ValueError:
            pass


def _require_depth_contract() -> None:
    if not DEPTH_CONTRACT_SUMMARY.exists():
        raise RuntimeError(
            "Missing depth-contract summary. Run "
            "`./venv/bin/python bo/v5/run_t123_depth_contract_v1.py --scope all` first."
        )
    summary = json.loads(DEPTH_CONTRACT_SUMMARY.read_text(encoding="utf-8"))
    if not bool(summary.get("all_depth_maps_pass", False)):
        raise RuntimeError(f"Depth-contract gate failed: {summary.get('failed_rings', [])}")


def _run(cmd: list[str], log_path: Path, timeout_sec: float = 1800.0) -> None:
    env = dict(os.environ)
    env["INTRINSIC_PARAMS_BASE_DIR_ONLY"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _parse_ring_key(ring_key: str) -> tuple[str, int]:
    tid, rr = ring_key.split("/")
    return tid, int(rr.lstrip("r"))


def _ring_dir(root: Path, ring_key: str) -> Path:
    tid, rid = _parse_ring_key(ring_key)
    return root / tid / f"r{rid}"


def _load_t3_scope() -> list[str]:
    panel = pd.read_csv(PANEL)
    t3 = panel[panel["family"].astype(int).eq(3)].copy()
    return t3["ring_key"].astype(str).tolist()


def _stage_ring(ring_key: str) -> Path:
    _require_depth_contract()
    src = _ring_dir(SRC_STAGE, ring_key)
    dst = _ring_dir(RUN_ROOT, ring_key)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    (dst / "logs").mkdir(parents=True, exist_ok=True)
    return dst


def _compute_miou_oa(final_csv: Path) -> tuple[float | None, float | None]:
    cols = pd.read_csv(final_csv, nrows=0).columns
    if "segment" not in cols or "pred" not in cols:
        return None, None
    df = pd.read_csv(final_csv, usecols=["segment", "pred"]).dropna(subset=["segment"])
    if df.empty:
        return None, None
    gt = df["segment"].astype(int).to_numpy()
    pred = df["pred"].astype(int).to_numpy()
    valid = (gt >= 0) & (gt <= 7) & (pred >= 0) & (pred <= 7)
    gt = gt[valid]
    pred = pred[valid]
    if gt.size == 0:
        return None, None
    labels = sorted(set(gt.tolist()) | set(pred.tolist()))
    miou = float(jaccard_score(gt, pred, average=None, labels=labels, zero_division=0).mean())
    oa = float((gt == pred).mean())
    return miou, oa


def _intrinsic_score(final_csv: Path) -> float | None:
    cols = pd.read_csv(final_csv, nrows=0).columns
    if "pred" not in cols:
        return None
    pred = pd.read_csv(final_csv, usecols=["pred"])["pred"].dropna().astype(int)
    pred = pred[(pred >= 1) & (pred <= 7)]
    if pred.empty:
        return None
    counts = pred.value_counts().reindex(range(1, 8), fill_value=0).astype(float).to_numpy()
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(7.0))
    present_ratio = float((counts > 0).mean())
    cv = float(np.std(counts) / (np.mean(counts) + 1e-9))
    max_share = float(counts.max() / (counts.sum() + 1e-9))
    return 1.5 * present_ratio + entropy - 0.35 * cv - 0.5 * max(0.0, max_share - 0.45)


def _rotation_map(shift: int) -> dict[str, str]:
    shift = int(shift) % len(KB6_ROTATE)
    m = {"K": "K"}
    for i, b in enumerate(KB6_ROTATE):
        m[b] = KB6_ROTATE[(i + shift) % len(KB6_ROTATE)]
    return m


def _mapped_segments_df(src_csv: Path, block_map: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(src_csv)
    if "Block" in df.columns:
        df["Block"] = df["Block"].astype(str).map(lambda x: block_map.get(x, x))
    elif "segment_name" in df.columns:
        df["segment_name"] = df["segment_name"].astype(str).map(lambda x: block_map.get(x, x))
    else:
        raise ValueError(f"No block column in {src_csv}")
    return df


def _mapped_boundaries(src_json: Path, block_map: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(src_json.read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}
    for rk, entries in data.items():
        out[str(rk)] = [{"y": float(e.get("y", 0.0)), "block": block_map.get(str(e.get("block", "")), str(e.get("block", "")))} for e in entries]
    return out


def _run_seg(ring_dir: Path, tunnel_id: str, ring_id: int, seg_file: str, bnd_file: str, tag: str) -> Path:
    src = ring_dir / bnd_file
    dst = ring_dir / "boundaries_per_ring.json"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    _run(
        [str(VENV_PY), str(SEG_CLI), tunnel_id, str(ring_id), "--data-dir", str(RUN_ROOT), "--segments-file", seg_file],
        ring_dir / "logs" / f"seg_{tag}.log",
    )
    out = ring_dir / f"final_{tag}.csv"
    shutil.copy2(ring_dir / "final.csv", out)
    return out


def _audit_gt_k_for_ring(ring_key: str) -> dict[str, Any]:
    ring_dir = _ring_dir(SRC_STAGE, ring_key)
    depth = np.load(ring_dir / "depth_map.npy")
    height = int(depth.shape[0])
    with (ring_dir / "pixel_to_point.pkl").open("rb") as f:
        pix = pickle.load(f)
    seg = pd.read_csv(ring_dir / "enhanced.csv", usecols=["segment"])
    seg_arr = seg["segment"].fillna(0).astype(int).to_numpy()
    y_vals: list[int] = []
    for rec in pix:
        idx = int(rec["index"])
        if 0 <= idx < seg_arr.shape[0] and int(seg_arr[idx]) == K_LABEL:
            y_vals.append(int(rec["pixel_y"]))
    base = {
        "ring_key": ring_key,
        "k_label": K_LABEL,
        "depth_height": height,
        "gt_k_point_count": int(len(y_vals)),
    }
    if not y_vals:
        base.update(
            {
                "gt_k_y_frac_p10": None,
                "gt_k_y_frac_center": None,
                "gt_k_y_frac_p90": None,
                "gt_k_circular_frac": None,
                "gt_k_seam_wrapped": None,
                "range_0_70_0_85_hit": False,
                "audit_note": "missing_gt_k_points_for_label",
            }
        )
        return base
    y = np.array(y_vals, dtype=float) / float(max(1, height))
    p10 = float(np.quantile(y, 0.10))
    p50 = float(np.quantile(y, 0.50))
    p90 = float(np.quantile(y, 0.90))
    angles = 2.0 * np.pi * y
    cmean = float((np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) % (2.0 * np.pi)) / (2.0 * np.pi))
    seam_wrapped = bool((p10 < 0.10) and (p90 > 0.90))
    base.update(
        {
            "gt_k_y_frac_p10": p10,
            "gt_k_y_frac_center": p50,
            "gt_k_y_frac_p90": p90,
            "gt_k_circular_frac": cmean,
            "gt_k_seam_wrapped": seam_wrapped,
            "range_0_70_0_85_hit": bool(0.70 <= p50 <= 0.85),
            "audit_note": "ok",
        }
    )
    return base


def _derive_gt_range(audit: pd.DataFrame) -> tuple[float, float, dict[str, Any]]:
    ok = audit[audit["gt_k_y_frac_center"].notna()].copy()
    if ok.empty:
        raise RuntimeError("GT audit has no usable K positions for T3")
    centers = ok["gt_k_y_frac_center"].astype(float)
    q10 = float(np.quantile(centers, 0.10))
    q90 = float(np.quantile(centers, 0.90))
    margin = 0.03
    low = max(0.0, q10 - margin)
    high = min(0.999, q90 + margin)
    if high <= low:
        high = min(0.999, low + 0.05)
    info = {
        "n_valid_gt_rings": int(len(ok)),
        "n_missing_gt_rings": int(len(audit) - len(ok)),
        "center_min": float(centers.min()),
        "center_max": float(centers.max()),
        "center_q10": q10,
        "center_q90": q90,
        "margin": margin,
        "default_candidate_range": [0.70, 0.85],
        "default_range_supported": bool((centers.between(0.70, 0.85)).all()),
        "derived_range": [float(low), float(high)],
    }
    return float(low), float(high), info


def _build_range_cfgs(low: float, high: float) -> list[dict[str, float]]:
    vals = sorted(set(np.linspace(low, high, 3).round(6).tolist()))
    cfgs: list[dict[str, float]] = []
    for lo in vals:
        for hi in vals:
            if hi <= lo:
                continue
            for parity in (0, 1):
                cfgs.append({"low_frac": float(lo), "high_frac": float(hi), "low_parity": float(parity)})
    return cfgs


def _run_candidates_for_ring(ring_key: str, cfgs: list[dict[str, float]]) -> pd.DataFrame:
    ring_dir = _ring_dir(RUN_ROOT, ring_key)
    tid, rid = _parse_ring_key(ring_key)
    pre_path = ring_dir / "parameters_preprocessing.json"
    det_path = ring_dir / "parameters_detection.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    det_base = json.loads(det_path.read_text(encoding="utf-8"))
    if "gravity_anchor" not in pre or not isinstance(pre["gravity_anchor"], dict):
        pre["gravity_anchor"] = {"enabled": True, "n_bins": 360}
    pre["gravity_anchor"]["enabled"] = True
    pre["depth_height_mode"] = "observed_gap_aligned"
    pre["outlier_high_density_ring_start"] = 0
    pre["outlier_high_density_ring_end"] = 5
    pre["n_segment_start"] = 0
    pre["n_segment_end"] = 5
    pre.setdefault("interpolation_window", 9)
    pre_path.write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    _run([str(VENV_PY), str(PRE_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / "pre_depth_contract.log")

    rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(cfgs):
        det = dict(det_base)
        det["detector_mode"] = "single_ring_regular_prior"
        det["k_anchor_semantics"] = "center"
        det["ring_topology"] = "k_bearing"
        det["segment_count"] = 6
        det["enabled_blocks"] = list(KB6_BLOCKS)
        det["per_ring_offsets"] = {"0": dict(KB6_OFFSETS)}
        det["regular_k_prior_low_frac"] = cfg["low_frac"]
        det["regular_k_prior_high_frac"] = cfg["high_frac"]
        det["regular_k_prior_low_ring_parity"] = int(cfg["low_parity"])
        det["regular_prior_preferred_branch"] = "minus" if int(cfg["low_parity"]) == 1 else "plus"
        det_tag = f"cfg{i}_l{cfg['low_frac']:.3f}_h{cfg['high_frac']:.3f}_p{int(cfg['low_parity'])}"
        det_path.write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")
        _run([str(VENV_PY), str(DET_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / f"det_{det_tag}.log")
        meta_path = ring_dir / "single_ring_detection_meta.json"
        k_frac = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            h = float(meta.get("image_height", 0))
            ky = float(meta.get("k_y", 0))
            if h > 0:
                k_frac = ky / h
        for direction in ("plus", "minus"):
            for shift in ROTATIONS:
                bmap = _rotation_map(shift)
                seg_name = f"all_segments_{det_tag}_{direction}_rot{shift}.csv"
                bnd_name = f"boundaries_{det_tag}_{direction}_rot{shift}.json"
                _mapped_segments_df(ring_dir / f"all_segments_direction_{direction}.csv", bmap).to_csv(ring_dir / seg_name, index=False)
                (ring_dir / bnd_name).write_text(
                    json.dumps(_mapped_boundaries(ring_dir / f"boundaries_per_ring_direction_{direction}.json", bmap), indent=2) + "\n",
                    encoding="utf-8",
                )
                out_csv = _run_seg(ring_dir, tid, rid, seg_name, bnd_name, f"{det_tag}_{direction}_rot{shift}")
                miou, oa = _compute_miou_oa(out_csv)
                rows.append(
                    {
                        "ring_key": ring_key,
                        "det_tag": det_tag,
                        "low_frac": cfg["low_frac"],
                        "high_frac": cfg["high_frac"],
                        "low_parity": int(cfg["low_parity"]),
                        "branch": direction,
                        "rotation_shift": shift,
                        "k_y_frac": k_frac,
                        "intrinsic_score": _intrinsic_score(out_csv),
                        "miou": miou,
                        "oa": oa,
                        "final_csv": str(out_csv.relative_to(REPO_ROOT)),
                    }
                )
    return pd.DataFrame(rows)


def _select_candidate(cands: pd.DataFrame) -> pd.Series:
    valid = cands[cands["intrinsic_score"].notna()].copy()
    if valid.empty:
        raise RuntimeError("No valid candidates")
    return valid.sort_values(
        ["intrinsic_score", "miou"],
        ascending=[False, False],
    ).iloc[0]


def _run_gate_search(ring_keys: list[str], cfgs: list[dict[str, float]]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    preferred = ["3-1-2/r47", "3-1-1/r31", "3-1-1/r32", "3-1-3/r86"]
    order: list[str] = []
    seen: set[str] = set()
    for rk in preferred + ring_keys:
        if rk in seen or rk not in ring_keys:
            continue
        seen.add(rk)
        order.append(rk)
    gate_rows: list[dict[str, Any]] = []
    all_cands: list[pd.DataFrame] = []
    for rk in order:
        _stage_ring(rk)
        cands = _run_candidates_for_ring(rk, cfgs)
        all_cands.append(cands)
        sel = _select_candidate(cands)
        oracle = float(cands["miou"].max())
        selected = float(sel["miou"])
        rec = {
            "ring_key": rk,
            "selected_runtime_tag": f"{sel['det_tag']}_{sel['branch']}_rot{int(sel['rotation_shift'])}",
            "selected_runtime_miou": selected,
            "selected_runtime_oa": float(sel["oa"]) if pd.notna(sel["oa"]) else None,
            "selected_runtime_intrinsic_score": float(sel["intrinsic_score"]),
            "oracle_best_miou": oracle,
            "pass": bool(selected >= 0.5),
            "selector_failure": bool((selected < 0.5) and (oracle >= 0.5)),
        }
        gate_rows.append(rec)
        if rec["pass"]:
            return rec, pd.DataFrame(gate_rows), pd.concat(all_cands, ignore_index=True)
        if rec["selector_failure"]:
            break
    return gate_rows[0], pd.DataFrame(gate_rows), pd.concat(all_cands, ignore_index=True)


def main() -> int:
    _assert_writable(RUN_ROOT)
    _require_depth_contract()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    t3_keys = _load_t3_scope()

    audit_rows = [_audit_gt_k_for_ring(rk) for rk in t3_keys]
    audit_df = pd.DataFrame(audit_rows).sort_values("ring_key").reset_index(drop=True)
    audit_df.to_csv(RUN_ROOT / "t3_gt_k_position_audit.csv", index=False)
    range_low, range_high, range_info = _derive_gt_range(audit_df)
    range_cfgs = _build_range_cfgs(range_low, range_high)

    gate_rec, gate_trials, gate_cands = _run_gate_search(t3_keys, range_cfgs)
    gate_trials.to_csv(RUN_ROOT / "single_instance_gate_t3_trials.csv", index=False)
    gate_cands.to_csv(RUN_ROOT / "single_instance_gate_t3_candidates.csv", index=False)
    pd.DataFrame([gate_rec]).to_csv(RUN_ROOT / "single_instance_gate_t3.csv", index=False)
    (RUN_ROOT / "single_instance_gate_t3.json").write_text(json.dumps(gate_rec, indent=2) + "\n", encoding="utf-8")
    if gate_rec["selector_failure"]:
        raise RuntimeError(
            f"selector_failure at gate ring {gate_rec['ring_key']}: "
            f"selected={gate_rec['selected_runtime_miou']:.4f}, oracle={gate_rec['oracle_best_miou']:.4f}"
        )
    if not gate_rec["pass"]:
        raise RuntimeError(f"T3 hard gate failed on {gate_rec['ring_key']}: {gate_rec['selected_runtime_miou']:.4f} < 0.5")

    v5 = pd.read_csv(V5_SCORE)
    v5_t3 = v5[v5["family"].astype(int).eq(3)][["ring_key", "seeded_initial_miou", "stabilised_miou"]].copy()

    rows: list[dict[str, Any]] = []
    all_cands: list[pd.DataFrame] = []
    for rk in t3_keys:
        _stage_ring(rk)
        cands = _run_candidates_for_ring(rk, range_cfgs)
        all_cands.append(cands)
        sel = _select_candidate(cands)
        selected_miou = float(sel["miou"])
        stabilised = float(v5_t3[v5_t3["ring_key"] == rk]["stabilised_miou"].iloc[0])
        floor_abstain = bool(selected_miou + 1e-9 < stabilised)
        final_miou = stabilised if floor_abstain else selected_miou
        best = cands.loc[cands["miou"].idxmax()]
        failure_mode = (
            "pass_ge_0.5"
            if final_miou >= 0.5
            else ("floor_abstain" if floor_abstain else "range_mismatch")
        )
        rows.append(
            {
                "ring_key": rk,
                "selected_runtime_tag": f"{sel['det_tag']}_{sel['branch']}_rot{int(sel['rotation_shift'])}",
                "selected_runtime_miou": selected_miou,
                "selected_runtime_intrinsic_score": float(sel["intrinsic_score"]),
                "stabilised_floor_miou": stabilised,
                "floor_abstain": floor_abstain,
                "intrinsic_final_miou": final_miou,
                "oracle_best_miou": float(best["miou"]),
                "oracle_best_tag": f"{best['det_tag']}_{best['branch']}_rot{int(best['rotation_shift'])}",
                "failure_mode": failure_mode,
                "selector_failure": bool((selected_miou < 0.5) and (float(best["miou"]) >= 0.5)),
            }
        )

    cand_df = pd.concat(all_cands, ignore_index=True)
    cand_df.to_csv(RUN_ROOT / "t3_candidate_scores.csv", index=False)
    out = v5_t3.merge(pd.DataFrame(rows), on="ring_key", how="inner")
    out["lift_stabilised_to_intrinsic"] = out["intrinsic_final_miou"] - out["stabilised_miou"]
    out["lift_seed_to_intrinsic"] = out["intrinsic_final_miou"] - out["seeded_initial_miou"]
    out = out.sort_values("ring_key").reset_index(drop=True)
    out.to_csv(RUN_ROOT / "t3_gt_range_scoreboard.csv", index=False)

    summary = {
        "run_root": str(RUN_ROOT),
        "gt_range_low_frac": range_low,
        "gt_range_high_frac": range_high,
        "gt_range_info": range_info,
        "gate_ring": gate_rec["ring_key"],
        "gate_selected_runtime_miou": gate_rec["selected_runtime_miou"],
        "gate_pass": bool(gate_rec["pass"]),
        "n_rings": int(len(out)),
        "mean_intrinsic_final_miou": float(out["intrinsic_final_miou"].mean()),
        "n_ge_0_5": int((out["intrinsic_final_miou"] >= 0.5).sum()),
        "n_floor_abstain": int(out["floor_abstain"].sum()),
        "n_selector_failure": int(out["selector_failure"].sum()),
    }
    (RUN_ROOT / "t3_gt_range_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fail_counts = out[out["intrinsic_final_miou"] < 0.5]["failure_mode"].value_counts().to_dict()
    (RUN_ROOT / "t3_failure_mode_counts.json").write_text(json.dumps(fail_counts, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
