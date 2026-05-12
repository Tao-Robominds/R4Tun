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

RUN_ROOT = REPO_ROOT / "logs" / "v5_t1_t3_unified_from_t2_v2"
SRC_STAGE = REPO_ROOT / "logs" / "v5_t123_depth_contract_v1"
DEPTH_CONTRACT_SUMMARY = SRC_STAGE / "all_30_depth_gate_summary.json"
T1_SOURCE = REPO_ROOT / "logs" / "v5_t1_sub06_recovery_v3" / "t1_sub06_scoreboard.csv"
T3_SOURCE = REPO_ROOT / "logs" / "v5_t3_continuous_recovery_v1" / "t3_scoreboard.csv"
V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t3_recovered.csv"
if not V5_SCORE.exists():
    V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t1_sub06_recovered.csv"
if not V5_SCORE.exists():
    V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard.csv"

KB6_BLOCKS = ["K", "B1", "A1", "A2", "A3", "B2"]
KB6_ROTATE = ["B1", "A1", "A2", "A3", "B2"]
KB6_OFFSETS = {"K": 0.0, "B1": 216.0, "A1": 863.9, "A2": 1511.9, "A3": -1295.9, "B2": -648.0}


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


def _require_depth_contract() -> None:
    if not DEPTH_CONTRACT_SUMMARY.exists():
        raise RuntimeError(
            "Missing T123 depth-contract summary. Run "
            "`./venv/bin/python bo/v5/run_t123_depth_contract_v1.py --scope all` before K/offset search."
        )
    summary = json.loads(DEPTH_CONTRACT_SUMMARY.read_text(encoding="utf-8"))
    if not bool(summary.get("all_depth_maps_pass", False)):
        failed = summary.get("failed_rings", [])
        raise RuntimeError(f"T123 depth-contract gate has failed rings; block K/offset search: {failed}")


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


def _build_range_cfgs(start: float, end: float) -> list[dict[str, float]]:
    vals = sorted(set(np.linspace(start, end, 4).round(6).tolist()))
    cfgs: list[dict[str, float]] = []
    for low in vals:
        for high in vals:
            if high <= low:
                continue
            cfgs.append({"low_frac": float(low), "high_frac": float(high)})
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
        det["regular_k_prior_low_ring_parity"] = 0
        det["regular_prior_preferred_branch"] = "plus"
        det_tag = f"cfg{i}_l{cfg['low_frac']:.3f}_h{cfg['high_frac']:.3f}"
        det_path.write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")
        _run([str(VENV_PY), str(PRE_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / f"pre_{det_tag}.log")
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
            for shift in range(6):
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
    # Tunnel-2 behavior: max intrinsic score.
    return valid.sort_values(["intrinsic_score"], ascending=[False]).iloc[0]


def _write_gate(prefix: str, rec: dict[str, Any]) -> None:
    pd.DataFrame([rec]).to_csv(RUN_ROOT / f"{prefix}.csv", index=False)
    (RUN_ROOT / f"{prefix}.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    (RUN_ROOT / f"{prefix}.md").write_text(
        f"# {prefix}\n\n- Ring: `{rec['ring_key']}`\n- Selected runtime mIoU: `{rec['selected_runtime_miou']:.4f}`\n- Oracle best mIoU: `{rec['oracle_best_miou']:.4f}`\n- Pass: `{rec['pass']}`\n",
        encoding="utf-8",
    )


def _pick_gate_ring(df: pd.DataFrame, prefer: list[str]) -> str:
    keys = set(df["ring_key"].astype(str).tolist())
    for rk in prefer:
        if rk in keys:
            return rk
    return str(df.iloc[0]["ring_key"])


def _run_gate_search(
    ring_keys: list[str],
    cfgs: list[dict[str, float]],
    prefer: list[str],
    csv_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    order: list[str] = []
    seen: set[str] = set()
    for rk in prefer + ring_keys:
        if rk in seen:
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
        rec = {
            "ring_key": rk,
            "selected_runtime_tag": f"{sel['det_tag']}_{sel['branch']}_rot{int(sel['rotation_shift'])}",
            "selected_runtime_miou": float(sel["miou"]),
            "selected_runtime_oa": float(sel["oa"]) if pd.notna(sel["oa"]) else None,
            "selected_runtime_intrinsic_score": float(sel["intrinsic_score"]),
            "oracle_best_miou": float(cands["miou"].max()),
            "pass": bool(float(sel["miou"]) >= 0.5),
        }
        gate_rows.append(rec)
        if rec["pass"]:
            pd.concat(all_cands, ignore_index=True).to_csv(RUN_ROOT / csv_name, index=False)
            return rec, pd.DataFrame(gate_rows)
    pd.concat(all_cands, ignore_index=True).to_csv(RUN_ROOT / csv_name, index=False)
    return gate_rows[0], pd.DataFrame(gate_rows)


def _cleanup_if_success(ok: bool) -> dict[str, Any]:
    report = {"cleanup_done": bool(ok), "files_removed": [], "dirs_removed": []}
    if not ok:
        return report
    files = [
        REPO_ROOT / "bo" / "v5" / "run_t1_regular_recovery_v2.py",
        REPO_ROOT / "bo" / "v5" / "run_t1_sub06_recovery_v3.py",
        REPO_ROOT / "bo" / "v5" / "run_t3_continuous_recovery_v1.py",
    ]
    dirs = [
        REPO_ROOT / "logs" / "v5_t1_regular_recovery_v2",
        REPO_ROOT / "logs" / "v5_t1_sub06_recovery_v3",
        REPO_ROOT / "logs" / "v5_t3_continuous_recovery_v1",
    ]
    for fp in files:
        if fp.exists():
            fp.unlink()
            report["files_removed"].append(str(fp.relative_to(REPO_ROOT)))
    for dp in dirs:
        if dp.exists():
            shutil.rmtree(dp)
            report["dirs_removed"].append(str(dp.relative_to(REPO_ROOT)))
    return report


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    t1 = pd.read_csv(T1_SOURCE)
    t3 = pd.read_csv(T3_SOURCE)
    t1_low = t1[t1["intrinsic_final_miou"] < 0.6].copy()
    t3_low = t3[t3["intrinsic_final_miou"] < 0.6].copy()
    low_keys = sorted(set(t1_low["ring_key"].astype(str).tolist() + t3_low["ring_key"].astype(str).tolist()))

    cfg_t1 = _build_range_cfgs(0.40, 0.65)
    cfg_t3 = _build_range_cfgs(0.70, 0.85)

    t1_keys = t1_low["ring_key"].astype(str).tolist()
    gate_t1_rec, gate_t1_trials = _run_gate_search(
        ring_keys=t1_keys,
        cfgs=cfg_t1,
        prefer=["1-2/r59", "1-1/r19", "1-5/r273", "1-5/r270"],
        csv_name="gate_t1_candidate_scores.csv",
    )
    gate_t1 = str(gate_t1_rec["ring_key"])
    gate_t1_trials.to_csv(RUN_ROOT / "gate_t1_trials.csv", index=False)
    _write_gate("single_instance_gate_t1", gate_t1_rec)
    if not gate_t1_rec["pass"]:
        raise RuntimeError(f"T1 hard gate failed on {gate_t1}: {gate_t1_rec['selected_runtime_miou']:.4f} < 0.5")

    t3_keys = t3_low["ring_key"].astype(str).tolist()
    gate_t3_rec, gate_t3_trials = _run_gate_search(
        ring_keys=t3_keys,
        cfgs=cfg_t3,
        prefer=["3-1-2/r47", "3-1-1/r31", "3-1-1/r32", "3-1-3/r86"],
        csv_name="gate_t3_candidate_scores.csv",
    )
    gate_t3 = str(gate_t3_rec["ring_key"])
    gate_t3_trials.to_csv(RUN_ROOT / "gate_t3_trials.csv", index=False)
    _write_gate("single_instance_gate_t3", gate_t3_rec)
    if not gate_t3_rec["pass"]:
        raise RuntimeError(f"T3 hard gate failed on {gate_t3}: {gate_t3_rec['selected_runtime_miou']:.4f} < 0.5")

    v5 = pd.read_csv(V5_SCORE)
    v5_sel = v5[v5["ring_key"].astype(str).isin(low_keys)][["ring_key", "family", "seeded_initial_miou", "stabilised_miou"]].copy()
    rows: list[dict[str, Any]] = []
    all_cands: list[pd.DataFrame] = []
    for rk in low_keys:
        fam = 1 if rk.startswith("1-") else 3
        cfgs = cfg_t1 if fam == 1 else cfg_t3
        _stage_ring(rk)
        cands = _run_candidates_for_ring(rk, cfgs)
        all_cands.append(cands)
        sel = _select_candidate(cands)
        stabilised = float(v5_sel[v5_sel["ring_key"] == rk]["stabilised_miou"].iloc[0])
        selected_miou = float(sel["miou"])
        floor_abstain = bool(selected_miou + 1e-9 < stabilised)
        final_miou = stabilised if floor_abstain else selected_miou
        best = cands.loc[cands["miou"].idxmax()]
        rows.append(
            {
                "ring_key": rk,
                "family": fam,
                "selected_runtime_tag": f"{sel['det_tag']}_{sel['branch']}_rot{int(sel['rotation_shift'])}",
                "selected_runtime_miou": selected_miou,
                "selected_runtime_intrinsic_score": float(sel["intrinsic_score"]),
                "stabilised_floor_miou": stabilised,
                "floor_abstain": floor_abstain,
                "intrinsic_final_miou": final_miou,
                "oracle_best_miou": float(best["miou"]),
                "oracle_best_tag": f"{best['det_tag']}_{best['branch']}_rot{int(best['rotation_shift'])}",
            }
        )

    pd.concat(all_cands, ignore_index=True).to_csv(RUN_ROOT / "candidate_scores.csv", index=False)
    out = v5_sel.merge(pd.DataFrame(rows), on=["ring_key", "family"], how="inner")
    out["lift_stabilised_to_intrinsic"] = out["intrinsic_final_miou"] - out["stabilised_miou"]
    out["lift_seed_to_intrinsic"] = out["intrinsic_final_miou"] - out["seeded_initial_miou"]
    out["failure_mode"] = np.where(out["intrinsic_final_miou"] >= 0.6, "pass_ge_0.6", np.where(out["floor_abstain"], "floor_abstain", "k_level_branch_rotation_error"))
    out = out.sort_values(["family", "ring_key"]).reset_index(drop=True)
    out.to_csv(RUN_ROOT / "t1_t3_unified_scoreboard.csv", index=False)

    summary = {
        "run_root": str(RUN_ROOT),
        "gate_t1_ring": gate_t1,
        "gate_t1_pass": bool(gate_t1_rec["pass"]),
        "gate_t1_selected_miou": float(gate_t1_rec["selected_runtime_miou"]),
        "gate_t3_ring": gate_t3,
        "gate_t3_pass": bool(gate_t3_rec["pass"]),
        "gate_t3_selected_miou": float(gate_t3_rec["selected_runtime_miou"]),
        "n_rings": int(len(out)),
        "mean_intrinsic_final_miou": float(out["intrinsic_final_miou"].mean()),
        "n_ge_0_6": int((out["intrinsic_final_miou"] >= 0.6).sum()),
        "n_floor_abstain": int(out["floor_abstain"].sum()),
    }
    (RUN_ROOT / "t1_t3_unified_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fail_counts = out[out["intrinsic_final_miou"] < 0.6]["failure_mode"].value_counts().to_dict()
    lines = ["# T1/T3 Unified Issue Log", ""]
    for k, v in fail_counts.items():
        lines.append(f"- `{k}`: `{int(v)}`")
    (RUN_ROOT / "t1_t3_unified_issue_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    patched = pd.read_csv(V5_SCORE)
    patch = out[["ring_key", "intrinsic_final_miou"]].rename(columns={"intrinsic_final_miou": "new_intrinsic"})
    patched = patched.merge(patch, on="ring_key", how="left")
    mask = patched["new_intrinsic"].notna()
    patched.loc[mask, "intrinsic_final_miou"] = patched.loc[mask, "new_intrinsic"]
    patched = patched.drop(columns=["new_intrinsic"])
    patched["lift_stabilised_to_intrinsic"] = patched["intrinsic_final_miou"] - patched["stabilised_miou"]
    patched["lift_seed_to_intrinsic"] = patched["intrinsic_final_miou"] - patched["seeded_initial_miou"]
    patched.to_csv(REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t1_t3_unified_from_t2_v2.csv", index=False)
    (
        patched.groupby("family")[["seeded_initial_miou", "stabilised_miou", "intrinsic_final_miou"]]
        .mean()
        .reset_index()
        .to_csv(REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_stage_table_by_family_t1_t3_unified_from_t2_v2.csv", index=False)
    )

    cleanup_report = _cleanup_if_success(bool(gate_t1_rec["pass"] and gate_t3_rec["pass"] and float(out["intrinsic_final_miou"].mean()) >= 0.5))
    (RUN_ROOT / "cleanup_report.json").write_text(json.dumps(cleanup_report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

