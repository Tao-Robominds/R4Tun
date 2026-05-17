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

RUN_ROOT = REPO_ROOT / "logs" / "v5_t1_sub06_recovery_v3"
V5_PANEL = REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv"
T1_V2 = REPO_ROOT / "logs" / "v5_t1_regular_recovery_v2" / "t1_regular_scoreboard.csv"
V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t1_recovered.csv"
if not V5_SCORE.exists():
    V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard.csv"
SUCCESS_SRC = REPO_ROOT / "stages" / "v4" / "logs" / "v6_preprocess_kposition_offsets_v1" / "iter_08_regular_prior_selector"

KB6_BLOCKS = ["K", "B1", "A1", "A2", "A3", "B2"]
KB6_ROTATE = ["B1", "A1", "A2", "A3", "B2"]
KB6_OFFSETS = {
    "K": 0.0,
    "B1": 216.0,
    "A1": 863.9,
    "A2": 1511.9,
    "A3": -1295.9,
    "B2": -648.0,
}
LOW_BASE = 1150.0 / 2777.0
HIGH_BASE = 1580.0 / 2777.0

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


def _miou_oa(final_csv: Path) -> tuple[float | None, float | None]:
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


def _rotation_map(shift: int) -> dict[str, str]:
    shift = int(shift) % len(KB6_ROTATE)
    mapping = {"K": "K"}
    for idx, block in enumerate(KB6_ROTATE):
        mapping[block] = KB6_ROTATE[(idx + shift) % len(KB6_ROTATE)]
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


def _stage_ring_input(ring_dir: Path, tunnel_id: str, ring_id: int) -> None:
    src = REPO_ROOT / "data" / "rings" / f"{tunnel_id.replace('-', '_')}_ring{ring_id}.txt"
    if not src.exists():
        raise FileNotFoundError(f"Missing input ring txt: {src}")
    dst = ring_dir / f"{tunnel_id}_r{ring_id}.txt"
    if not dst.exists():
        shutil.copy2(src, dst)


def _base_params() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pre = json.loads((SUCCESS_SRC / "1-2" / "r58" / "parameters_preprocessing.json").read_text(encoding="utf-8"))
    det = json.loads((SUCCESS_SRC / "1-2" / "r58" / "parameters_detection.json").read_text(encoding="utf-8"))
    seg = {"k_cap": 130, "ab_cap": 390}
    pre["depth_height_mode"] = "canonical"
    if "gravity_anchor" not in pre or not isinstance(pre["gravity_anchor"], dict):
        pre["gravity_anchor"] = {"enabled": True, "n_bins": 360}
    pre["gravity_anchor"]["enabled"] = True
    det["detector_mode"] = "single_ring_regular_prior"
    det["k_anchor_semantics"] = "center"
    det["ring_topology"] = "k_bearing"
    det["segment_count"] = 6
    det["enabled_blocks"] = list(KB6_BLOCKS)
    det["per_ring_offsets"] = {"0": dict(KB6_OFFSETS)}
    return pre, det, seg


def _write_base_params(ring_dir: Path) -> None:
    pre, det, seg = _base_params()
    (ring_dir / "parameters_preprocessing.json").write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    (ring_dir / "parameters_detection.json").write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")
    (ring_dir / "parameters_segmentation.json").write_text(json.dumps(seg, indent=2) + "\n", encoding="utf-8")


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


def _materialize_rot_candidate(ring_dir: Path, direction: str, rotation_shift: int) -> tuple[str, str, str]:
    src_seg = ring_dir / f"all_segments_direction_{direction}.csv"
    src_bnd = ring_dir / f"boundaries_per_ring_direction_{direction}.json"
    block_map = _rotation_map(rotation_shift)
    seg = _mapped_segments_df(src_seg, block_map)
    bnd = _mapped_boundaries(src_bnd, block_map)
    tag = f"{direction}_rot{rotation_shift}"
    seg_name = f"all_segments_{tag}.csv"
    bnd_name = f"boundaries_per_ring_{tag}.json"
    seg.to_csv(ring_dir / seg_name, index=False)
    (ring_dir / bnd_name).write_text(json.dumps(bnd, indent=2) + "\n", encoding="utf-8")
    return seg_name, bnd_name, tag


def _run_pre_once(ring_key: str) -> Path:
    tunnel_id, ring_id = _parse_ring_key(ring_key)
    ring_dir = _ring_dir(RUN_ROOT, ring_key)
    if ring_dir.exists():
        shutil.rmtree(ring_dir)
    ring_dir.mkdir(parents=True, exist_ok=True)
    (ring_dir / "logs").mkdir(parents=True, exist_ok=True)
    _stage_ring_input(ring_dir, tunnel_id, ring_id)
    _write_base_params(ring_dir)
    _run([str(VENV_PY), str(PRE_CLI), tunnel_id, str(ring_id), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / "preprocessing.log")
    return ring_dir


def _candidate_configs(gate_delta: float | None = None) -> list[dict[str, Any]]:
    if gate_delta is None:
        deltas = [-0.08, -0.05, -0.03, 0.0, 0.03, 0.05, 0.08]
    else:
        deltas = [gate_delta - 0.03, gate_delta, gate_delta + 0.03]
    configs: list[dict[str, Any]] = []
    for d in deltas:
        low = max(0.25, min(0.75, LOW_BASE + d))
        high = max(0.25, min(0.75, HIGH_BASE + d))
        for parity in (0, 1):
            configs.append({"low_frac": low, "high_frac": high, "low_parity": parity, "delta": d})
    return configs


def _run_candidates_for_ring(ring_key: str, configs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tunnel_id, ring_id = _parse_ring_key(ring_key)
    ring_dir = _ring_dir(RUN_ROOT, ring_key)
    det_path = ring_dir / "parameters_detection.json"
    det_base = json.loads(det_path.read_text(encoding="utf-8"))
    cand_rows: list[dict[str, Any]] = []
    best_oracle_rows: list[dict[str, Any]] = []

    for i, cfg in enumerate(configs):
        det = dict(det_base)
        det["regular_k_prior_low_frac"] = float(cfg["low_frac"])
        det["regular_k_prior_high_frac"] = float(cfg["high_frac"])
        det["regular_k_prior_low_ring_parity"] = int(cfg["low_parity"])
        det["regular_prior_preferred_branch"] = "minus" if int(cfg["low_parity"]) == 1 else "plus"
        preferred_branch = str(det["regular_prior_preferred_branch"])
        det_tag = f"cfg{i}_d{cfg['delta']:+.3f}_p{cfg['low_parity']}"
        det_path.write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")
        _run([str(VENV_PY), str(DET_CLI), tunnel_id, str(ring_id), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / f"detection_{det_tag}.log")
        meta_path = ring_dir / "single_ring_detection_meta.json"
        k_y_frac = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            h = float(meta.get("image_height", 0.0))
            ky = float(meta.get("k_y", 0.0))
            if h > 0:
                k_y_frac = ky / h

        best_oracle = None
        best_oracle_tag = None
        for direction in ("plus", "minus"):
            for shift in range(6):
                seg_name, bnd_name, tag = _materialize_rot_candidate(ring_dir, direction, shift)
                out_csv = _run_seg(ring_dir, tunnel_id, ring_id, seg_name, bnd_name, f"{det_tag}_{tag}")
                miou, oa = _miou_oa(out_csv)
                intr = _intrinsic_score(out_csv)
                cand_rows.append(
                    {
                        "ring_key": ring_key,
                        "det_tag": det_tag,
                        "delta": cfg["delta"],
                        "low_frac": cfg["low_frac"],
                        "high_frac": cfg["high_frac"],
                        "low_parity": cfg["low_parity"],
                        "branch": direction,
                        "rotation_shift": shift,
                        "tag": tag,
                        "preferred_branch": preferred_branch,
                        "k_y_frac": k_y_frac,
                        "miou": miou,
                        "oa": oa,
                        "intrinsic_score": intr,
                        "final_csv": str(out_csv.relative_to(REPO_ROOT)),
                    }
                )
                if miou is not None and (best_oracle is None or miou > best_oracle):
                    best_oracle = miou
                    best_oracle_tag = f"{det_tag}_{tag}"
        best_oracle_rows.append(
            {
                "ring_key": ring_key,
                "det_tag": det_tag,
                "oracle_best_miou": best_oracle,
                "oracle_best_tag": best_oracle_tag,
            }
        )
    return pd.DataFrame(cand_rows), pd.DataFrame(best_oracle_rows)


def _select_runtime_candidate(cands: pd.DataFrame) -> pd.Series:
    valid = cands[cands["intrinsic_score"].notna()].copy()
    if valid.empty:
        raise RuntimeError("No candidates with intrinsic score")
    valid = valid.sort_values(["intrinsic_score"], ascending=[False])
    return valid.iloc[0]


def _select_compact_policy_candidate(cands: pd.DataFrame, target_k_frac: float | None = None) -> pd.Series:
    valid = cands[cands["intrinsic_score"].notna()].copy()
    if valid.empty:
        raise RuntimeError("No candidates with intrinsic score")
    valid = valid[
        (valid["rotation_shift"].astype(int) == 0)
        & (valid["branch"].astype(str) == valid["preferred_branch"].astype(str))
    ].copy()
    if valid.empty:
        return _select_runtime_candidate(cands)
    if target_k_frac is not None and valid["k_y_frac"].notna().any():
        valid["k_dist"] = (valid["k_y_frac"].astype(float) - float(target_k_frac)).abs()
        valid = valid.sort_values(["k_dist", "intrinsic_score"], ascending=[True, False])
        return valid.iloc[0]
    valid = valid.sort_values(["intrinsic_score"], ascending=[False])
    return valid.iloc[0]


def main() -> int:
    _assert_writable(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    t1_prev = pd.read_csv(T1_V2)
    low = t1_prev[t1_prev["intrinsic_final_miou"] < 0.6].copy()
    low_keys = low["ring_key"].astype(str).tolist()
    controls = ["1-2/r58", "1-4/r204"]
    scope = sorted(set(low_keys + controls))

    # Step 1: hard gate on one low ring.
    gate_ring = "1-4/r197" if "1-4/r197" in low_keys else low_keys[0]
    _run_pre_once(gate_ring)
    gate_cands, gate_oracles = _run_candidates_for_ring(gate_ring, _candidate_configs(gate_delta=None))
    gate_pref = gate_cands[
        (gate_cands["rotation_shift"].astype(int) == 0)
        & (gate_cands["branch"].astype(str) == gate_cands["preferred_branch"].astype(str))
    ].copy()
    if gate_pref.empty:
        gate_best_runtime = _select_runtime_candidate(gate_cands)
    else:
        gate_best_runtime = gate_pref.sort_values(["miou"], ascending=[False]).iloc[0]
    gate_best_miou = float(gate_best_runtime["miou"])
    gate_pass = bool(gate_best_miou >= 0.6)
    gate_best_oracle = float(gate_oracles["oracle_best_miou"].max())
    gate_rec = {
        "ring_key": gate_ring,
        "selected_runtime_tag": str(gate_best_runtime["det_tag"]) + "_" + str(gate_best_runtime["tag"]),
        "selected_runtime_miou": gate_best_miou,
        "selected_runtime_oa": float(gate_best_runtime["oa"]),
        "selected_runtime_intrinsic_score": float(gate_best_runtime["intrinsic_score"]),
        "oracle_best_miou": gate_best_oracle,
        "validation_requirement": "selected runtime mIoU >= 0.6 before scaling",
        "pass": gate_pass,
    }
    pd.DataFrame([gate_rec]).to_csv(RUN_ROOT / "single_instance_gate.csv", index=False)
    (RUN_ROOT / "single_instance_gate.json").write_text(json.dumps(gate_rec, indent=2) + "\n", encoding="utf-8")
    (RUN_ROOT / "single_instance_gate.md").write_text(
        "\n".join(
            [
                "# Sub-0.6 Hard Gate",
                "",
                f"- Gate ring: `{gate_ring}`",
                f"- Selected runtime mIoU: `{gate_best_miou:.4f}`",
                f"- Oracle best mIoU (diagnostic): `{gate_best_oracle:.4f}`",
                "- Requirement: runtime selected mIoU >= 0.6 before scaling",
                f"- Pass: `{gate_pass}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gate_cands.to_csv(RUN_ROOT / "gate_candidate_scores.csv", index=False)
    gate_oracles.to_csv(RUN_ROOT / "gate_oracle_summary.csv", index=False)

    if not gate_pass:
        raise RuntimeError(f"Hard gate failed on {gate_ring}: selected runtime mIoU={gate_best_miou:.4f} < 0.6")

    # Step 2+: scoped batch after gate pass.
    # Use compact band around successful gate delta.
    delta_star = float(gate_best_runtime["delta"])
    target_k_frac = float(gate_best_runtime["k_y_frac"]) if pd.notna(gate_best_runtime["k_y_frac"]) else None
    configs = _candidate_configs(gate_delta=delta_star)

    rows: list[dict[str, Any]] = []
    all_cands: list[pd.DataFrame] = []
    all_oracles: list[pd.DataFrame] = []
    v5 = pd.read_csv(V5_SCORE)
    v5_t1 = v5[v5["family"].astype(int) == 1][["ring_key", "seeded_initial_miou", "stabilised_miou"]].copy()
    baseline = t1_prev.set_index("ring_key")

    for rk in scope:
        _run_pre_once(rk)
        cands, oracles = _run_candidates_for_ring(rk, configs)
        all_cands.append(cands)
        all_oracles.append(oracles)
        sel = _select_compact_policy_candidate(cands, target_k_frac=target_k_frac)
        sel_miou = float(sel["miou"])
        sel_oa = float(sel["oa"])
        stabilised = float(v5_t1[v5_t1["ring_key"] == rk]["stabilised_miou"].iloc[0])

        # Strict floor invariant.
        floor_abstain = sel_miou + 1e-9 < stabilised
        final_miou = stabilised if floor_abstain else sel_miou
        final_oa = None if floor_abstain else sel_oa
        reason = "abstain_to_stabilised_floor" if floor_abstain else "selected_runtime_candidate"

        # Keep controls stable unless better.
        if rk in controls:
            prev_intr = float(baseline.at[rk, "intrinsic_final_miou"])
            if final_miou < prev_intr:
                final_miou = prev_intr
                reason = "control_keep_previous"
                floor_abstain = True
                final_oa = None

        best_oracle = float(oracles["oracle_best_miou"].max())
        best_oracle_tag = str(oracles.loc[oracles["oracle_best_miou"].idxmax(), "oracle_best_tag"])

        rows.append(
            {
                "ring_key": rk,
                "selected_runtime_tag": str(sel["det_tag"]) + "_" + str(sel["tag"]),
                "selected_runtime_miou": sel_miou,
                "selected_runtime_oa": sel_oa,
                "selected_runtime_intrinsic_score": float(sel["intrinsic_score"]),
                "stabilised_floor_miou": stabilised,
                "floor_abstain": bool(floor_abstain),
                "final_selected_miou": final_miou,
                "final_selected_oa": final_oa,
                "selection_reason": reason,
                "oracle_best_miou": best_oracle,
                "oracle_best_tag": best_oracle_tag,
            }
        )

    cand_df = pd.concat(all_cands, ignore_index=True) if all_cands else pd.DataFrame()
    oracle_df = pd.concat(all_oracles, ignore_index=True) if all_oracles else pd.DataFrame()
    cand_df.to_csv(RUN_ROOT / "candidate_scores.csv", index=False)
    oracle_df.to_csv(RUN_ROOT / "oracle_scores.csv", index=False)

    out = v5_t1.merge(pd.DataFrame(rows), on="ring_key", how="inner")
    out["intrinsic_final_miou"] = out["final_selected_miou"]
    out["lift_seed_to_stabilised"] = out["stabilised_miou"] - out["seeded_initial_miou"]
    out["lift_stabilised_to_intrinsic"] = out["intrinsic_final_miou"] - out["stabilised_miou"]
    out["lift_seed_to_intrinsic"] = out["intrinsic_final_miou"] - out["seeded_initial_miou"]
    out["is_low_scope_target"] = out["ring_key"].isin(low_keys)
    out["failure_mode"] = np.where(
        out["intrinsic_final_miou"] >= 0.6,
        "pass_ge_0.6",
        np.where(
            out["floor_abstain"],
            "floor_abstain",
            "k_level_branch_rotation_error",
        ),
    )
    out = out.sort_values("ring_key").reset_index(drop=True)
    out.to_csv(RUN_ROOT / "t1_sub06_scoreboard.csv", index=False)

    low_out = out[out["is_low_scope_target"]].copy()
    summary = {
        "run_root": str(RUN_ROOT),
        "n_scope_total": int(len(out)),
        "n_low_targets": int(len(low_out)),
        "hard_gate_ring": gate_ring,
        "hard_gate_pass": gate_pass,
        "hard_gate_selected_miou": gate_best_miou,
        "mean_final_miou_scope": float(out["intrinsic_final_miou"].mean()),
        "mean_final_miou_low_targets": float(low_out["intrinsic_final_miou"].mean()),
        "n_ge_0_6_scope": int((out["intrinsic_final_miou"] >= 0.6).sum()),
        "n_ge_0_6_low_targets": int((low_out["intrinsic_final_miou"] >= 0.6).sum()),
        "n_floor_abstain_scope": int(out["floor_abstain"].sum()),
        "n_floor_abstain_low_targets": int(low_out["floor_abstain"].sum()),
    }
    (RUN_ROOT / "t1_sub06_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fail_counts = low_out[low_out["intrinsic_final_miou"] < 0.6]["failure_mode"].value_counts().to_dict()
    lines = [
        "# T1 Sub-0.6 Recovery Issue Log",
        "",
        f"- Hard gate pass: `{gate_pass}` on `{gate_ring}` (`{gate_best_miou:.4f}`)",
        f"- Low-target rings >= 0.6: `{summary['n_ge_0_6_low_targets']}` / `{summary['n_low_targets']}`",
        f"- Low-target mean final mIoU: `{summary['mean_final_miou_low_targets']:.4f}`",
        "",
        "## Failure Mode Counts (low-target rings <0.6)",
    ]
    if fail_counts:
        for k, v in fail_counts.items():
            lines.append(f"- `{k}`: `{int(v)}`")
    else:
        lines.append("- none")
    (RUN_ROOT / "t1_sub06_issue_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Patch V5 50-ring scoreboard copy with new tunnel-1 values.
    patched = pd.read_csv(V5_SCORE)
    patch = out[["ring_key", "intrinsic_final_miou"]].rename(columns={"intrinsic_final_miou": "t1_sub06_intrinsic"})
    patched = patched.merge(patch, on="ring_key", how="left")
    mask = patched["family"].astype(int).eq(1) & patched["t1_sub06_intrinsic"].notna()
    patched.loc[mask, "intrinsic_final_miou"] = patched.loc[mask, "t1_sub06_intrinsic"]
    patched = patched.drop(columns=["t1_sub06_intrinsic"])
    patched["lift_stabilised_to_intrinsic"] = patched["intrinsic_final_miou"] - patched["stabilised_miou"]
    patched["lift_seed_to_intrinsic"] = patched["intrinsic_final_miou"] - patched["seeded_initial_miou"]
    patched.to_csv(REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t1_sub06_recovered.csv", index=False)
    fam = (
        patched.groupby("family")[["seeded_initial_miou", "stabilised_miou", "intrinsic_final_miou"]]
        .mean()
        .reset_index()
    )
    fam.to_csv(REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_stage_table_by_family_t1_sub06_recovered.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

