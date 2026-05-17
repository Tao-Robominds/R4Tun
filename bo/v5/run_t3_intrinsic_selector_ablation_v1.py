from __future__ import annotations

import argparse
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

RUN_ROOT = REPO_ROOT / "logs" / "v5_t3_intrinsic_selector_ablation_v1"
SRC_STAGE = REPO_ROOT / "logs" / "v5_t123_depth_contract_v1"
DEPTH_CONTRACT_SUMMARY = SRC_STAGE / "all_30_depth_gate_summary.json"
PANEL = REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv"
V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t3_recovered.csv"
if not V5_SCORE.exists():
    V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard.csv"

KB6_BLOCKS = ["K", "B1", "A1", "A2", "A3", "B2"]
KB6_ROTATE = ["B1", "A1", "A2", "A3", "B2"]
KB6_OFFSETS = {"K": 0.0, "B1": 216.0, "A1": 863.9, "A2": 1511.9, "A3": -1295.9, "B2": -648.0}
EXPECTED_T3_CYCLIC_ORDER = ["K", "B2", "A3", "A2", "A1", "B1"]
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


def _circular_dist(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 1.0 - d)


def _segment_balance(pred_values: pd.Series) -> tuple[float, dict[str, float]]:
    pred = pred_values[(pred_values >= 1) & (pred_values <= 7)]
    if pred.empty:
        return 0.0, {"present_ratio": 0.0, "entropy": 0.0, "cv": 1.0, "max_share": 1.0, "balance_raw": 0.0}
    counts = pred.value_counts().reindex(range(1, 8), fill_value=0).astype(float).to_numpy()
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(7.0))
    present_ratio = float((counts > 0).mean())
    cv = float(np.std(counts) / (np.mean(counts) + 1e-9))
    max_share = float(counts.max() / (counts.sum() + 1e-9))
    balance_raw = 1.5 * present_ratio + entropy - 0.35 * cv - 0.5 * max(0.0, max_share - 0.45)
    balance_norm = float(np.clip(balance_raw / 2.5, 0.0, 1.0))
    return balance_norm, {
        "present_ratio": present_ratio,
        "entropy": entropy,
        "cv": cv,
        "max_share": max_share,
        "balance_raw": balance_raw,
    }


def _cyclic_order_consistency(seg_csv: Path) -> float:
    seg = pd.read_csv(seg_csv)
    if "Ring" not in seg.columns or "Block" not in seg.columns or "Y" not in seg.columns:
        return 0.0
    ring = seg[seg["Ring"].astype(int).eq(0)].copy()
    if ring.empty:
        return 0.0
    ring["Block"] = ring["Block"].astype(str)
    ring = ring.sort_values("Y")
    observed: list[str] = []
    for b in ring["Block"].tolist():
        if b not in observed:
            observed.append(b)
    n = min(len(observed), len(EXPECTED_T3_CYCLIC_ORDER))
    if n < 3:
        return 0.0
    best = 0.0
    for shift in range(n):
        match = sum(1 for i in range(n) if observed[i] == EXPECTED_T3_CYCLIC_ORDER[(i + shift) % n]) / float(n)
        best = max(best, float(match))
    return float(np.clip(best, 0.0, 1.0))


def _boundary_spacing(boundary_json: Path, ring_height: int) -> float:
    data = json.loads(boundary_json.read_text(encoding="utf-8"))
    entries = data.get("0") or data.get(0) or []
    if not entries:
        return 0.0
    ys = sorted(float(e.get("y", 0.0)) for e in entries)
    if len(ys) < 3:
        return 0.0
    diffs = [(ys[(i + 1) % len(ys)] - ys[i]) % ring_height for i in range(len(ys))]
    arr = np.array(diffs, dtype=float)
    cv = float(np.std(arr) / (np.mean(arr) + 1e-9))
    return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))


def _k_anchor_consistency(seg_csv: Path, ring_height: int, k_y_frac: float | None) -> float:
    if k_y_frac is None:
        return 0.0
    seg = pd.read_csv(seg_csv)
    ring = seg[seg["Ring"].astype(int).eq(0)].copy()
    if ring.empty:
        return 0.0
    k_rows = ring[ring["Block"].astype(str).eq("K")]
    if k_rows.empty:
        return 0.0
    k_frac_seg = float(k_rows.iloc[0]["Y"] % ring_height) / float(ring_height)
    dist = _circular_dist(k_frac_seg, float(k_y_frac))
    return float(np.clip(1.0 - (dist / 0.5), 0.0, 1.0))


def _intrinsic_components(
    *,
    final_csv: Path,
    seg_csv: Path,
    bnd_json: Path,
    ring_height: int,
    k_y_frac: float | None,
) -> dict[str, float]:
    pred = pd.read_csv(final_csv, usecols=["pred"])["pred"].dropna().astype(int)
    balance_norm, balance_meta = _segment_balance(pred)
    cyclic_score = _cyclic_order_consistency(seg_csv)
    spacing_score = _boundary_spacing(bnd_json, ring_height=ring_height)
    k_anchor_score = _k_anchor_consistency(seg_csv, ring_height=ring_height, k_y_frac=k_y_frac)
    total = 0.25 * balance_norm + 0.50 * cyclic_score + 0.15 * spacing_score + 0.10 * k_anchor_score
    return {
        "intrinsic_total": float(total),
        "balance_norm": float(balance_norm),
        "cyclic_order_score": float(cyclic_score),
        "spacing_score": float(spacing_score),
        "k_anchor_score": float(k_anchor_score),
        **balance_meta,
    }


def _select_candidate(cands: pd.DataFrame) -> pd.Series:
    valid = cands[cands["intrinsic_score"].notna()].copy()
    if valid.empty:
        raise RuntimeError("No valid candidates")
    return valid.sort_values(["intrinsic_score", "miou"], ascending=[False, False]).iloc[0]


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

    ring_height = int(np.load(ring_dir / "depth_map.npy").shape[0])
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
        det.pop("regular_prior_preferred_branch", None)
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
                seg_path = ring_dir / seg_name
                bnd_path = ring_dir / bnd_name
                _mapped_segments_df(ring_dir / f"all_segments_direction_{direction}.csv", bmap).to_csv(seg_path, index=False)
                bnd_path.write_text(
                    json.dumps(_mapped_boundaries(ring_dir / f"boundaries_per_ring_direction_{direction}.json", bmap), indent=2) + "\n",
                    encoding="utf-8",
                )
                out_csv = _run_seg(ring_dir, tid, rid, seg_name, bnd_name, f"{det_tag}_{direction}_rot{shift}")
                miou, oa = _compute_miou_oa(out_csv)
                score_parts = _intrinsic_components(
                    final_csv=out_csv,
                    seg_csv=seg_path,
                    bnd_json=bnd_path,
                    ring_height=ring_height,
                    k_y_frac=k_frac,
                )
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
                        "intrinsic_score": score_parts["intrinsic_total"],
                        "miou": miou,
                        "oa": oa,
                        "final_csv": str(out_csv.relative_to(REPO_ROOT)),
                        "segments_csv": str(seg_path.relative_to(REPO_ROOT)),
                        "boundaries_json": str(bnd_path.relative_to(REPO_ROOT)),
                        **score_parts,
                    }
                )
    return pd.DataFrame(rows)


def _build_range_cfgs() -> list[dict[str, float]]:
    vals = [0.446, 0.484, 0.523]
    cfgs: list[dict[str, float]] = []
    for lo in vals:
        for hi in vals:
            if hi <= lo:
                continue
            for parity in (0, 1):
                cfgs.append({"low_frac": float(lo), "high_frac": float(hi), "low_parity": float(parity)})
    return cfgs


def _score_ring(ring_key: str, cfgs: list[dict[str, float],]) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    _stage_ring(ring_key)
    cands = _run_candidates_for_ring(ring_key, cfgs)
    selected = _select_candidate(cands)
    oracle = cands.loc[cands["miou"].idxmax()]
    return selected, oracle, cands


def main() -> int:
    parser = argparse.ArgumentParser(description="T3 intrinsic-only selector ablation.")
    parser.add_argument("--probe-ring", default="3-1-1/r28", help="First ring to validate.")
    parser.add_argument("--probe-success-threshold", type=float, default=0.75, help="Required selected mIoU for scale-up.")
    parser.add_argument("--scale-if-pass", action="store_true", help="Scale to all T3 rings only if probe passes.")
    args = parser.parse_args()

    _assert_writable(RUN_ROOT)
    _require_depth_contract()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    cfgs = _build_range_cfgs()

    probe_sel, probe_oracle, probe_cands = _score_ring(args.probe_ring, cfgs)
    probe_cands.to_csv(RUN_ROOT / "probe_candidate_scores.csv", index=False)
    probe = {
        "ring_key": args.probe_ring,
        "selected_runtime_tag": f"{probe_sel['det_tag']}_{probe_sel['branch']}_rot{int(probe_sel['rotation_shift'])}",
        "selected_runtime_miou": float(probe_sel["miou"]),
        "selected_runtime_intrinsic_score": float(probe_sel["intrinsic_score"]),
        "oracle_best_tag": f"{probe_oracle['det_tag']}_{probe_oracle['branch']}_rot{int(probe_oracle['rotation_shift'])}",
        "oracle_best_miou": float(probe_oracle["miou"]),
        "oracle_gap": float(probe_oracle["miou"] - probe_sel["miou"]),
        "pass": bool(float(probe_sel["miou"]) >= float(args.probe_success_threshold)),
        "success_threshold": float(args.probe_success_threshold),
    }
    pd.DataFrame([probe]).to_csv(RUN_ROOT / "probe_summary.csv", index=False)
    (RUN_ROOT / "probe_summary.json").write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")

    if not (probe["pass"] and args.scale_if_pass):
        return 0

    t3_keys = _load_t3_scope()
    v5 = pd.read_csv(V5_SCORE)
    v5_t3 = v5[v5["family"].astype(int).eq(3)][["ring_key", "seeded_initial_miou", "stabilised_miou"]].copy()

    rows: list[dict[str, Any]] = []
    all_cands: list[pd.DataFrame] = []
    for rk in t3_keys:
        sel, ora, cands = _score_ring(rk, cfgs)
        all_cands.append(cands)
        stabilised = float(v5_t3[v5_t3["ring_key"] == rk]["stabilised_miou"].iloc[0])
        selected_miou = float(sel["miou"])
        final_miou = stabilised if selected_miou < stabilised else selected_miou
        rows.append(
            {
                "ring_key": rk,
                "selected_runtime_tag": f"{sel['det_tag']}_{sel['branch']}_rot{int(sel['rotation_shift'])}",
                "selected_runtime_miou": selected_miou,
                "selected_runtime_intrinsic_score": float(sel["intrinsic_score"]),
                "stabilised_floor_miou": stabilised,
                "floor_abstain": bool(selected_miou < stabilised),
                "intrinsic_final_miou": final_miou,
                "oracle_best_miou": float(ora["miou"]),
                "oracle_best_tag": f"{ora['det_tag']}_{ora['branch']}_rot{int(ora['rotation_shift'])}",
            }
        )

    cand_df = pd.concat(all_cands, ignore_index=True)
    cand_df.to_csv(RUN_ROOT / "t3_candidate_scores.csv", index=False)
    out = v5_t3.merge(pd.DataFrame(rows), on="ring_key", how="inner")
    out["lift_stabilised_to_intrinsic"] = out["intrinsic_final_miou"] - out["stabilised_miou"]
    out["lift_seed_to_intrinsic"] = out["intrinsic_final_miou"] - out["seeded_initial_miou"]
    out = out.sort_values("ring_key").reset_index(drop=True)
    out.to_csv(RUN_ROOT / "t3_intrinsic_selector_scoreboard.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
