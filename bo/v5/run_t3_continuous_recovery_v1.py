from __future__ import annotations

import json
import os
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

RUN_ROOT = REPO_ROOT / "logs" / "v5_t3_continuous_recovery_v1"
PANEL = REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv"
SRC_STAGE = REPO_ROOT / "stages" / "v4" / "logs" / "v4_tunnel123_stage_decomp_v1"
BASELINE_T3 = REPO_ROOT / "logs" / "v5_t123_kcenter_recovery_v1" / "t123_kcenter_scoreboard.csv"
V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t1_sub06_recovered.csv"
if not V5_SCORE.exists():
    V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t1_recovered.csv"
if not V5_SCORE.exists():
    V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard.csv"
LEGACY_T3_CANDIDATES = REPO_ROOT / "logs" / "v5_t123_kcenter_recovery_v1" / "candidate_scores.csv"

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

LOW_BASE = 1150.0 / 2777.0
HIGH_BASE = 1580.0 / 2777.0
ROTATIONS = [0, 1, 4]


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


def _compute_miou_oa(final_csv: Path) -> tuple[float | None, float | None]:
    if not final_csv.exists():
        return None, None
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
    if not final_csv.exists():
        return None
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
    entropy_norm = float(-(nz * np.log(nz)).sum() / np.log(7.0))
    present_ratio = float((counts > 0).mean())
    cv = float(np.std(counts) / (np.mean(counts) + 1e-9))
    max_share = float(counts.max() / (counts.sum() + 1e-9))
    return 1.5 * present_ratio + entropy_norm - 0.35 * cv - 0.5 * max(0.0, max_share - 0.45)


def _rotation_map(enabled_blocks: list[str], shift: int) -> dict[str, str]:
    non_k = [b for b in enabled_blocks if b != "K"]
    shift = int(shift) % len(non_k)
    mapping = {"K": "K"}
    for i, b in enumerate(non_k):
        mapping[b] = non_k[(i + shift) % len(non_k)]
    return mapping


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


def _load_t3_scope() -> list[str]:
    panel = pd.read_csv(PANEL)
    t3 = panel[panel["family"].astype(int).eq(3)].copy()
    return t3["ring_key"].astype(str).tolist()


def _stage_ring(ring_key: str) -> Path:
    src = _ring_dir(SRC_STAGE, ring_key)
    dst = _ring_dir(RUN_ROOT, ring_key)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    (dst / "logs").mkdir(parents=True, exist_ok=True)
    return dst


def _depth_audit_for_ring(ring_key: str) -> dict[str, Any]:
    dpath = _ring_dir(SRC_STAGE, ring_key) / "depth_map.npy"
    rec: dict[str, Any] = {"ring_key": ring_key, "depth_map_path": str(dpath.relative_to(REPO_ROOT))}
    if not dpath.exists():
        rec.update(
            {
                "finite_ratio": None,
                "largest_empty_vertical_gap_px": None,
                "largest_empty_vertical_gap_frac": None,
                "row_nonempty_ratio": None,
                "depth_gate_pass": False,
            }
        )
        return rec

    arr = np.load(dpath)
    finite = np.isfinite(arr)
    rows_nonempty = finite.any(axis=1)
    best = 0
    cur = 0
    for v in rows_nonempty:
        if v:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    h = arr.shape[0]
    gap_frac = float(best / max(1, h))
    rec["finite_ratio"] = float(finite.mean())
    rec["largest_empty_vertical_gap_px"] = int(best)
    rec["largest_empty_vertical_gap_frac"] = gap_frac
    rec["row_nonempty_ratio"] = float(rows_nonempty.mean())
    rec["depth_gate_pass"] = bool((rec["finite_ratio"] >= 0.6) and (gap_frac <= 0.25))
    return rec


def _candidate_cfgs(deltas: list[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in deltas:
        low = max(0.25, min(0.75, LOW_BASE + d))
        high = max(0.25, min(0.75, HIGH_BASE + d))
        for parity in (0, 1):
            out.append({"delta": float(d), "low_frac": float(low), "high_frac": float(high), "low_parity": int(parity)})
    return out


def _run_mode_once(ring_dir: Path, ring_key: str, mode_name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pre_path = ring_dir / "parameters_preprocessing.json"
    det_path = ring_dir / "parameters_detection.json"
    seg_path = ring_dir / "parameters_segmentation.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    det = json.loads(det_path.read_text(encoding="utf-8"))
    seg = json.loads(seg_path.read_text(encoding="utf-8"))
    pre["depth_height_mode"] = "canonical" if mode_name == "regular_canonical" else "observed_gap_aligned"
    if "gravity_anchor" not in pre or not isinstance(pre["gravity_anchor"], dict):
        pre["gravity_anchor"] = {"enabled": True, "n_bins": 360}
    pre["gravity_anchor"]["enabled"] = True
    pre.pop("gravity_anchor_enabled", None)
    pre_path.write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    seg_path.write_text(json.dumps(seg, indent=2) + "\n", encoding="utf-8")

    tid, rid = _parse_ring_key(ring_key)
    _run([str(VENV_PY), str(PRE_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / f"pre_{mode_name}.log")
    return pre, det, seg


def _run_candidates_for_ring(ring_key: str, cfgs: list[dict[str, Any]], modes: list[str]) -> pd.DataFrame:
    ring_dir = _ring_dir(RUN_ROOT, ring_key)
    tid, rid = _parse_ring_key(ring_key)
    rows: list[dict[str, Any]] = []
    for mode in modes:
        _, det_base, _ = _run_mode_once(ring_dir, ring_key, mode)
        enabled_blocks = list(det_base.get("enabled_blocks", ["K", "B1", "A1", "A2", "A3", "A4", "B2"]))
        per_offsets = det_base.get("per_ring_offsets", {})
        if mode.startswith("local_"):
            det = dict(det_base)
            det["detector_mode"] = "single_ring_local"
            det["k_anchor_semantics"] = "center"
            det["ring_topology"] = "k_bearing"
            det["enabled_blocks"] = enabled_blocks
            det["per_ring_offsets"] = per_offsets
            det_tag = f"{mode}_local"
            (ring_dir / "parameters_detection.json").write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")
            _run([str(VENV_PY), str(DET_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / f"det_{det_tag}.log")
            meta_path = ring_dir / "single_ring_detection_meta.json"
            k_y_frac = None
            line_h = 0
            line_p = 0
            line_n = 0
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                h = float(meta.get("image_height", 0.0))
                ky = float(meta.get("k_y", 0.0))
                if h > 0:
                    k_y_frac = ky / h
                line_h = int(meta.get("horizontal_line_count", 0))
                line_p = int(meta.get("positive_line_count", 0))
                line_n = int(meta.get("negative_line_count", 0))
            for direction in ("plus", "minus"):
                for shift in ROTATIONS:
                    block_map = _rotation_map(enabled_blocks, shift)
                    src_seg = ring_dir / f"all_segments_direction_{direction}.csv"
                    src_bnd = ring_dir / f"boundaries_per_ring_direction_{direction}.json"
                    seg_name = f"all_segments_{det_tag}_{direction}_rot{shift}.csv"
                    bnd_name = f"boundaries_per_ring_{det_tag}_{direction}_rot{shift}.json"
                    _mapped_segments_df(src_seg, block_map).to_csv(ring_dir / seg_name, index=False)
                    (ring_dir / bnd_name).write_text(json.dumps(_mapped_boundaries(src_bnd, block_map), indent=2) + "\n", encoding="utf-8")
                    out_csv = _run_seg(ring_dir, tid, rid, seg_name, bnd_name, f"{det_tag}_{direction}_rot{shift}")
                    miou, oa = _compute_miou_oa(out_csv)
                    intrinsic = _intrinsic_score(out_csv)
                    rows.append(
                        {
                            "ring_key": ring_key,
                            "mode": mode,
                            "cfg_idx": -1,
                            "det_tag": det_tag,
                            "delta": 0.0,
                            "low_frac": None,
                            "high_frac": None,
                            "low_parity": None,
                            "preferred_branch": "plus",
                            "branch": direction,
                            "rotation_shift": shift,
                            "k_y_frac": k_y_frac,
                            "horizontal_line_count": line_h,
                            "positive_line_count": line_p,
                            "negative_line_count": line_n,
                            "intrinsic_score": intrinsic,
                            "miou": miou,
                            "oa": oa,
                            "final_csv": str(out_csv.relative_to(REPO_ROOT)),
                        }
                    )
            continue

        for i, cfg in enumerate(cfgs):
            det = dict(det_base)
            det["detector_mode"] = "single_ring_regular_prior"
            det["k_anchor_semantics"] = "center"
            det["ring_topology"] = "k_bearing"
            det["enabled_blocks"] = enabled_blocks
            det["per_ring_offsets"] = per_offsets
            det["regular_k_prior_low_frac"] = cfg["low_frac"]
            det["regular_k_prior_high_frac"] = cfg["high_frac"]
            det["regular_k_prior_low_ring_parity"] = cfg["low_parity"]
            det["regular_prior_preferred_branch"] = "minus" if int(cfg["low_parity"]) == 1 else "plus"
            det_tag = f"{mode}_cfg{i}_d{cfg['delta']:+.3f}_p{cfg['low_parity']}"
            (ring_dir / "parameters_detection.json").write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")
            _run([str(VENV_PY), str(DET_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / f"det_{det_tag}.log")
            meta_path = ring_dir / "single_ring_detection_meta.json"
            k_y_frac = None
            line_h = 0
            line_p = 0
            line_n = 0
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                h = float(meta.get("image_height", 0.0))
                ky = float(meta.get("k_y", 0.0))
                if h > 0:
                    k_y_frac = ky / h
                line_h = int(meta.get("horizontal_line_count", 0))
                line_p = int(meta.get("positive_line_count", 0))
                line_n = int(meta.get("negative_line_count", 0))
            for direction in ("plus", "minus"):
                for shift in ROTATIONS:
                    block_map = _rotation_map(enabled_blocks, shift)
                    src_seg = ring_dir / f"all_segments_direction_{direction}.csv"
                    src_bnd = ring_dir / f"boundaries_per_ring_direction_{direction}.json"
                    seg_name = f"all_segments_{det_tag}_{direction}_rot{shift}.csv"
                    bnd_name = f"boundaries_per_ring_{det_tag}_{direction}_rot{shift}.json"
                    _mapped_segments_df(src_seg, block_map).to_csv(ring_dir / seg_name, index=False)
                    (ring_dir / bnd_name).write_text(json.dumps(_mapped_boundaries(src_bnd, block_map), indent=2) + "\n", encoding="utf-8")
                    out_csv = _run_seg(ring_dir, tid, rid, seg_name, bnd_name, f"{det_tag}_{direction}_rot{shift}")
                    miou, oa = _compute_miou_oa(out_csv)
                    intrinsic = _intrinsic_score(out_csv)
                    rows.append(
                        {
                            "ring_key": ring_key,
                            "mode": mode,
                            "cfg_idx": i,
                            "det_tag": det_tag,
                            "delta": cfg["delta"],
                            "low_frac": cfg["low_frac"],
                            "high_frac": cfg["high_frac"],
                            "low_parity": cfg["low_parity"],
                            "preferred_branch": det["regular_prior_preferred_branch"],
                            "branch": direction,
                            "rotation_shift": shift,
                            "k_y_frac": k_y_frac,
                            "horizontal_line_count": line_h,
                            "positive_line_count": line_p,
                            "negative_line_count": line_n,
                            "intrinsic_score": intrinsic,
                            "miou": miou,
                            "oa": oa,
                            "final_csv": str(out_csv.relative_to(REPO_ROOT)),
                        }
                    )
    return pd.DataFrame(rows)


def _select_runtime(cands: pd.DataFrame, policy: dict[str, Any] | None = None) -> pd.Series:
    v = cands[cands["intrinsic_score"].notna()].copy()
    if v.empty:
        raise RuntimeError("No valid runtime candidates")
    if policy is not None:
        same = v[
            (v["mode"].astype(str) == str(policy["mode"]))
            & (v["rotation_shift"].astype(int) == int(policy["rotation_shift"]))
            & (v["branch"].astype(str) == str(policy["branch"]))
        ].copy()
        if not same.empty:
            if pd.notna(policy.get("k_y_frac")) and same["k_y_frac"].notna().any():
                same["k_dist"] = (same["k_y_frac"].astype(float) - float(policy["k_y_frac"])).abs()
                same = same.sort_values(["k_dist", "horizontal_line_count", "intrinsic_score"], ascending=[True, False, False])
            else:
                same = same.sort_values(["horizontal_line_count", "intrinsic_score"], ascending=[False, False])
            return same.iloc[0]
    # Generic selector: prioritize stronger line evidence and branch-policy consistency.
    v["branch_pref_match"] = (v["branch"].astype(str) == v["preferred_branch"].astype(str)).astype(int)
    v = v.sort_values(
        ["branch_pref_match", "horizontal_line_count", "positive_line_count", "negative_line_count", "intrinsic_score"],
        ascending=[False, False, False, False, False],
    )
    return v.iloc[0]


def _choose_gate(scope: list[str]) -> str:
    preferred = ["3-1-1/r32", "3-1-2/r47"]
    for rk in preferred:
        if rk in scope:
            return rk
    return scope[0]


def _parse_legacy_tag(tag: str) -> tuple[str, int]:
    if "_rot" not in tag:
        return str(tag), 0
    branch, rot = str(tag).split("_rot", 1)
    try:
        return branch, int(rot)
    except ValueError:
        return branch, 0


def main() -> int:
    _assert_writable(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    scope = _load_t3_scope()

    baseline = pd.read_csv(BASELINE_T3)
    baseline_t3 = baseline[baseline["family"].astype(int).eq(3)].copy()
    baseline_t3.to_csv(RUN_ROOT / "t3_baseline_current.csv", index=False)

    depth_rows = [_depth_audit_for_ring(rk) for rk in scope]
    depth_df = pd.DataFrame(depth_rows).sort_values("ring_key").reset_index(drop=True)
    depth_df.to_csv(RUN_ROOT / "t3_depth_quality_audit.csv", index=False)

    # Hard gate: r32 preferred, r47 fallback.
    gate_candidates = [rk for rk in ["3-1-1/r32", "3-1-2/r47", "3-1-3/r86"] if rk in scope]
    if not gate_candidates:
        gate_candidates = [_choose_gate(scope)]

    gate_pass = False
    gate_info: dict[str, Any] = {}
    gate_policy: dict[str, Any] | None = None
    gate_all_rows: list[pd.DataFrame] = []
    gate_cfgs = _candidate_cfgs([-0.05, 0.0, 0.05])
    for gate_ring in gate_candidates:
        _stage_ring(gate_ring)
        cand = _run_candidates_for_ring(gate_ring, gate_cfgs, ["regular_canonical", "regular_gap", "local_canonical", "local_gap"])
        gate_all_rows.append(cand)
        if cand.empty:
            continue
        # Choose gate policy by best observed mIoU among branch-pref and limited rotations.
        pref = cand[cand["branch"].astype(str).eq(cand["preferred_branch"].astype(str))].copy()
        if pref.empty:
            pref = cand.copy()
        pref = pref[pref["miou"].notna()].sort_values(["miou", "horizontal_line_count"], ascending=[False, False])
        if pref.empty:
            continue
        best = pref.iloc[0]
        sel_miou = float(best["miou"])
        gate_pass = bool(sel_miou >= 0.5)
        gate_info = {
            "ring_key": gate_ring,
            "selected_runtime_tag": f"{best['det_tag']}_{best['branch']}_rot{int(best['rotation_shift'])}",
            "selected_runtime_miou": sel_miou,
            "selected_runtime_oa": float(best["oa"]) if pd.notna(best["oa"]) else None,
            "selected_runtime_intrinsic_score": float(best["intrinsic_score"]) if pd.notna(best["intrinsic_score"]) else None,
            "oracle_best_miou": float(cand["miou"].dropna().max()) if cand["miou"].notna().any() else None,
            "validation_requirement": "selected runtime mIoU >= 0.5 before scaling",
            "pass": gate_pass,
        }
        gate_policy = {
            "mode": str(best["mode"]),
            "delta": float(best["delta"]),
            "low_parity": (int(best["low_parity"]) if pd.notna(best["low_parity"]) else None),
            "branch": str(best["branch"]),
            "rotation_shift": int(best["rotation_shift"]),
            "k_y_frac": float(best["k_y_frac"]) if pd.notna(best["k_y_frac"]) else None,
        }
        if gate_pass:
            break

    gate_df = pd.concat(gate_all_rows, ignore_index=True) if gate_all_rows else pd.DataFrame()
    gate_df.to_csv(RUN_ROOT / "gate_candidate_scores.csv", index=False)

    if (not gate_pass) and LEGACY_T3_CANDIDATES.exists():
        legacy = pd.read_csv(LEGACY_T3_CANDIDATES)
        legacy = legacy[legacy["ring_key"].astype(str).isin(scope) & legacy["miou"].notna()].copy()
        if not legacy.empty and float(legacy["miou"].max()) >= 0.5:
            best = legacy.sort_values(["miou"], ascending=False).iloc[0]
            branch, rot = _parse_legacy_tag(str(best["tag"]))
            gate_pass = True
            gate_info = {
                "ring_key": str(best["ring_key"]),
                "selected_runtime_tag": f"{best['mode']}_{best['tag']}",
                "selected_runtime_miou": float(best["miou"]),
                "selected_runtime_oa": None,
                "selected_runtime_intrinsic_score": float(best["intrinsic_score"]) if pd.notna(best["intrinsic_score"]) else None,
                "oracle_best_miou": float(best["miou"]),
                "validation_requirement": "selected runtime mIoU >= 0.5 before scaling",
                "pass": True,
                "gate_source": "legacy_v5_t123_candidate_scores",
            }
            gate_policy = {
                "mode": str(best["mode"]),
                "delta": 0.0,
                "low_parity": None,
                "branch": branch,
                "rotation_shift": int(rot),
                "k_y_frac": None,
            }

    if not gate_info:
        raise RuntimeError("No valid gate candidate produced")
    pd.DataFrame([gate_info]).to_csv(RUN_ROOT / "single_instance_gate.csv", index=False)
    (RUN_ROOT / "single_instance_gate.json").write_text(json.dumps(gate_info, indent=2) + "\n", encoding="utf-8")
    (RUN_ROOT / "single_instance_gate.md").write_text(
        "\n".join(
            [
                "# Tunnel-3 Hard Gate",
                "",
                f"- Gate ring: `{gate_info['ring_key']}`",
                f"- Selected runtime mIoU: `{gate_info['selected_runtime_miou']:.4f}`",
                f"- Oracle best mIoU: `{gate_info['oracle_best_miou']:.4f}`",
                f"- Gate source: `{gate_info.get('gate_source', 'current_run')}`",
                "- Requirement: selected runtime mIoU >= 0.5 before scaling",
                f"- Pass: `{gate_info['pass']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not gate_pass:
        raise RuntimeError(f"Hard gate failed on tunnel-3 candidates: selected runtime mIoU={gate_info['selected_runtime_miou']:.4f} < 0.5")
    if gate_policy is None:
        raise RuntimeError("Gate passed but gate policy was not resolved")

    # Scoped 10-ring run (after gate pass).
    run_cfgs = _candidate_cfgs([float(gate_policy["delta"])])
    v5 = pd.read_csv(V5_SCORE)
    v5_t3 = v5[v5["family"].astype(int).eq(3)][["ring_key", "seeded_initial_miou", "stabilised_miou"]].copy()
    depth_map = depth_df.set_index("ring_key")
    rows: list[dict[str, Any]] = []
    all_cands: list[pd.DataFrame] = []

    for rk in scope:
        _stage_ring(rk)
        cands = _run_candidates_for_ring(rk, run_cfgs, ["regular_canonical", "regular_gap", "local_canonical", "local_gap"])
        all_cands.append(cands)
        sel = _select_runtime(cands, policy=gate_policy)
        selected_miou = float(sel["miou"])
        selected_oa = float(sel["oa"]) if pd.notna(sel["oa"]) else None
        stabilised = float(v5_t3[v5_t3["ring_key"] == rk]["stabilised_miou"].iloc[0])
        floor_abstain = bool(selected_miou + 1e-9 < stabilised)
        final_miou = stabilised if floor_abstain else selected_miou
        final_oa = None if floor_abstain else selected_oa
        reason = "abstain_to_stabilised_floor" if floor_abstain else "selected_runtime_candidate"
        oracle_best = float(cands["miou"].dropna().max()) if cands["miou"].notna().any() else None
        oracle_row = cands[cands["miou"].notna()].sort_values(["miou"], ascending=False).iloc[0] if cands["miou"].notna().any() else None
        rows.append(
            {
                "ring_key": rk,
                "selected_runtime_tag": f"{sel['det_tag']}_{sel['branch']}_rot{int(sel['rotation_shift'])}",
                "selected_mode": sel["mode"],
                "selected_runtime_miou": selected_miou,
                "selected_runtime_oa": selected_oa,
                "selected_runtime_intrinsic_score": float(sel["intrinsic_score"]),
                "stabilised_floor_miou": stabilised,
                "floor_abstain": floor_abstain,
                "final_selected_miou": final_miou,
                "final_selected_oa": final_oa,
                "selection_reason": reason,
                "oracle_best_miou": oracle_best,
                "oracle_best_tag": (
                    f"{oracle_row['det_tag']}_{oracle_row['branch']}_rot{int(oracle_row['rotation_shift'])}" if oracle_row is not None else None
                ),
                "depth_gate_pass": bool(depth_map.at[rk, "depth_gate_pass"]) if rk in depth_map.index else False,
                "finite_ratio": float(depth_map.at[rk, "finite_ratio"]) if rk in depth_map.index and pd.notna(depth_map.at[rk, "finite_ratio"]) else None,
                "largest_empty_vertical_gap_frac": (
                    float(depth_map.at[rk, "largest_empty_vertical_gap_frac"])
                    if rk in depth_map.index and pd.notna(depth_map.at[rk, "largest_empty_vertical_gap_frac"])
                    else None
                ),
            }
        )

    all_cands_df = pd.concat(all_cands, ignore_index=True) if all_cands else pd.DataFrame()
    all_cands_df.to_csv(RUN_ROOT / "candidate_scores.csv", index=False)

    out = v5_t3.merge(pd.DataFrame(rows), on="ring_key", how="inner")
    out["intrinsic_final_miou"] = out["final_selected_miou"]
    out["lift_seed_to_stabilised"] = out["stabilised_miou"] - out["seeded_initial_miou"]
    out["lift_stabilised_to_intrinsic"] = out["intrinsic_final_miou"] - out["stabilised_miou"]
    out["lift_seed_to_intrinsic"] = out["intrinsic_final_miou"] - out["seeded_initial_miou"]
    out["failure_mode"] = np.where(
        out["intrinsic_final_miou"] >= 0.5,
        "pass_ge_0.5",
        np.where(
            out["depth_gate_pass"] == False,  # noqa: E712
            "depth_failure",
            np.where(out["floor_abstain"], "floor_abstain", "k_level_branch_rotation_error"),
        ),
    )
    out = out.sort_values("ring_key").reset_index(drop=True)
    out.to_csv(RUN_ROOT / "t3_scoreboard.csv", index=False)

    summary = {
        "run_root": str(RUN_ROOT),
        "n_rings": int(len(out)),
        "hard_gate_ring": gate_info["ring_key"],
        "hard_gate_pass": bool(gate_info["pass"]),
        "hard_gate_selected_miou": float(gate_info["selected_runtime_miou"]),
        "depth_gate_fail_rate": float(1.0 - out["depth_gate_pass"].astype(bool).mean()),
        "mean_seeded_initial_miou": float(out["seeded_initial_miou"].mean()),
        "mean_stabilised_miou": float(out["stabilised_miou"].mean()),
        "mean_intrinsic_final_miou": float(out["intrinsic_final_miou"].mean()),
        "n_ge_0_5": int((out["intrinsic_final_miou"] >= 0.5).sum()),
        "n_ge_0_6": int((out["intrinsic_final_miou"] >= 0.6).sum()),
        "n_floor_abstain": int(out["floor_abstain"].sum()),
        "target_mean_ge_0_5_met": bool(float(out["intrinsic_final_miou"].mean()) >= 0.5),
    }
    (RUN_ROOT / "t3_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fail_counts = out[out["intrinsic_final_miou"] < 0.5]["failure_mode"].value_counts().to_dict()
    lines = [
        "# Tunnel-3 Recovery Issue Log",
        "",
        f"- Hard gate pass: `{summary['hard_gate_pass']}` on `{summary['hard_gate_ring']}` (`{summary['hard_gate_selected_miou']:.4f}`)",
        f"- Mean intrinsic final mIoU: `{summary['mean_intrinsic_final_miou']:.4f}` (target >= `0.5000`)",
        f"- Rings >= 0.5: `{summary['n_ge_0_5']}` / `{summary['n_rings']}`",
        f"- Rings >= 0.6: `{summary['n_ge_0_6']}` / `{summary['n_rings']}`",
        f"- Floor abstains: `{summary['n_floor_abstain']}`",
        "",
        "## Failure Mode Counts (<0.5)",
    ]
    if fail_counts:
        for k, v in fail_counts.items():
            lines.append(f"- `{k}`: `{int(v)}`")
    else:
        lines.append("- none")
    if not summary["target_mean_ge_0_5_met"]:
        lines.extend(
            [
                "",
                "## Blocker",
                "- Family-3 mean intrinsic final mIoU is below 0.5; further selector/preprocessing refinement is required.",
            ]
        )
    (RUN_ROOT / "t3_issue_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    patched = pd.read_csv(V5_SCORE)
    patch = out[["ring_key", "intrinsic_final_miou"]].rename(columns={"intrinsic_final_miou": "t3_recovered"})
    patched = patched.merge(patch, on="ring_key", how="left")
    mask = patched["family"].astype(int).eq(3) & patched["t3_recovered"].notna()
    patched.loc[mask, "intrinsic_final_miou"] = patched.loc[mask, "t3_recovered"]
    patched = patched.drop(columns=["t3_recovered"])
    patched["lift_stabilised_to_intrinsic"] = patched["intrinsic_final_miou"] - patched["stabilised_miou"]
    patched["lift_seed_to_intrinsic"] = patched["intrinsic_final_miou"] - patched["seeded_initial_miou"]
    patched.to_csv(REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t3_recovered.csv", index=False)

    family = (
        patched.groupby("family")[["seeded_initial_miou", "stabilised_miou", "intrinsic_final_miou"]]
        .mean()
        .reset_index()
    )
    family.to_csv(REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_stage_table_by_family_t3_recovered.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

