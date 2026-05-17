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

RUN_ROOT = REPO_ROOT / "logs" / "v5_t1_regular_recovery_v2"
V5_PANEL = REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv"
V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard.csv"
T123_RECOVERED = REPO_ROOT / "logs" / "v5_t123_kcenter_recovery_v1" / "t123_kcenter_scoreboard.csv"
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


def _parse_ring_key(ring_key: str) -> tuple[str, int]:
    tid, rr = ring_key.split("/")
    return tid, int(rr.lstrip("r"))


def _ring_dir(root: Path, ring_key: str) -> Path:
    tid, rid = _parse_ring_key(ring_key)
    return root / tid / f"r{rid}"


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
    # Clone successful family-1 setup from proven r58 run.
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
    det["regular_k_prior_low_ring_parity"] = 1
    det["regular_prior_preferred_branch"] = "minus"
    return pre, det, seg


def _write_params(ring_dir: Path) -> None:
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


def _run_one_ring(ring_key: str) -> dict[str, Any]:
    tunnel_id, ring_id = _parse_ring_key(ring_key)
    ring_dir = _ring_dir(RUN_ROOT, ring_key)
    if ring_dir.exists():
        shutil.rmtree(ring_dir)
    ring_dir.mkdir(parents=True, exist_ok=True)
    (ring_dir / "logs").mkdir(parents=True, exist_ok=True)
    _stage_ring_input(ring_dir, tunnel_id, ring_id)
    _write_params(ring_dir)

    _run([str(VENV_PY), str(PRE_CLI), tunnel_id, str(ring_id), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / "preprocessing.log")
    _run([str(VENV_PY), str(DET_CLI), tunnel_id, str(ring_id), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / "detection.log")

    plus_csv = _run_seg(ring_dir, tunnel_id, ring_id, "all_segments_direction_plus.csv", "boundaries_per_ring_direction_plus.json", "direction_plus")
    minus_csv = _run_seg(ring_dir, tunnel_id, ring_id, "all_segments_direction_minus.csv", "boundaries_per_ring_direction_minus.json", "direction_minus")
    plus_miou, plus_oa = _miou_oa(plus_csv)
    minus_miou, minus_oa = _miou_oa(minus_csv)

    # Family-level regular rule from successful gate.
    # For tunnel-1 regular pattern we use even->minus, odd->plus (matches r58/r23 evidence).
    selected_branch = "minus" if (ring_id % 2 == 0) else "plus"
    selected_csv = minus_csv if selected_branch == "minus" else plus_csv
    selected_miou, selected_oa = _miou_oa(selected_csv)

    oracle_best = None
    oracle_best_tag = None
    for direction in ("plus", "minus"):
        for shift in range(6):
            seg_name, bnd_name, tag = _materialize_rot_candidate(ring_dir, direction, shift)
            out_csv = _run_seg(ring_dir, tunnel_id, ring_id, seg_name, bnd_name, f"oracle_{tag}")
            mm, _ = _miou_oa(out_csv)
            if mm is not None and (oracle_best is None or mm > oracle_best):
                oracle_best = mm
                oracle_best_tag = tag

    return {
        "ring_key": ring_key,
        "tunnel_id": tunnel_id,
        "ring_id": ring_id,
        "selected_branch": selected_branch,
        "plus_miou": plus_miou,
        "minus_miou": minus_miou,
        "selected_miou": selected_miou,
        "plus_oa": plus_oa,
        "minus_oa": minus_oa,
        "selected_oa": selected_oa,
        "oracle_best_miou": oracle_best,
        "oracle_best_tag": oracle_best_tag,
    }


def main() -> int:
    _assert_writable(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(V5_PANEL)
    t1 = panel[panel["family"].astype(int) == 1].copy().sort_values(["tunnel_id", "ring_id"]).reset_index(drop=True)
    t1.to_csv(RUN_ROOT / "panel_t1_from_v5.csv", index=False)

    # Step 1: gate ring (must be in panel).
    gate_key = "1-2/r58"
    if gate_key not in set(t1["ring_key"].astype(str)):
        raise RuntimeError(f"Gate ring not in t1 panel: {gate_key}")
    gate = _run_one_ring(gate_key)
    gate_pass = bool((gate["selected_miou"] or 0.0) >= 0.6)
    gate_rec = {
        **gate,
        "validation_requirement": "selected mIoU >= 0.6 before scaling to all tunnel-1 rings",
        "pass": gate_pass,
    }
    pd.DataFrame([gate_rec]).to_csv(RUN_ROOT / "single_instance_gate.csv", index=False)
    (RUN_ROOT / "single_instance_gate.json").write_text(json.dumps(gate_rec, indent=2) + "\n", encoding="utf-8")
    (RUN_ROOT / "single_instance_gate.md").write_text(
        "\n".join(
            [
                "# Tunnel-1 Gate Before Scaling",
                "",
                f"- Ring: `{gate_key}`",
                f"- Selected branch: `{gate['selected_branch']}`",
                f"- Selected mIoU: `{(gate['selected_miou'] or float('nan')):.4f}`",
                f"- Requirement: `>= 0.6`",
                f"- Pass: `{gate_pass}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not gate_pass:
        raise RuntimeError(f"Gate failed for {gate_key}: selected mIoU={(gate['selected_miou'] or float('nan')):.4f} < 0.6")

    # Step 2+: apply to all tunnel-1 rings.
    rows: list[dict[str, Any]] = []
    for rk in t1["ring_key"].astype(str).tolist():
        rows.append(_run_one_ring(rk))
    raw = pd.DataFrame(rows)

    v5 = pd.read_csv(V5_SCORE)
    t1_v5 = v5[v5["family"].astype(int) == 1][["ring_key", "seeded_initial_miou", "stabilised_miou"]].copy()
    out = t1_v5.merge(raw, on="ring_key", how="left")
    out["intrinsic_final_miou"] = out["selected_miou"].astype(float)
    out["lift_seed_to_stabilised"] = out["stabilised_miou"] - out["seeded_initial_miou"]
    out["lift_stabilised_to_intrinsic"] = out["intrinsic_final_miou"] - out["stabilised_miou"]
    out["lift_seed_to_intrinsic"] = out["intrinsic_final_miou"] - out["seeded_initial_miou"]
    out["selector_failure_vs_floor"] = out["intrinsic_final_miou"] + 1e-9 < out["stabilised_miou"]

    out["failure_mode"] = np.where(
        out["intrinsic_final_miou"] >= 0.6,
        "pass_ge_0.6",
        np.where(
            out["selector_failure_vs_floor"],
            "selector_failure_vs_stabilised_floor",
            "k_branch_rotation_error",
        ),
    )
    out = out.sort_values(["tunnel_id", "ring_id"]).reset_index(drop=True)
    out.to_csv(RUN_ROOT / "t1_regular_scoreboard.csv", index=False)

    summary = {
        "run_root": str(RUN_ROOT),
        "n_rings": int(len(out)),
        "mean_seeded_initial_miou": float(out["seeded_initial_miou"].mean()),
        "mean_stabilised_miou": float(out["stabilised_miou"].mean()),
        "mean_intrinsic_final_miou": float(out["intrinsic_final_miou"].mean()),
        "n_ge_0_6": int((out["intrinsic_final_miou"] >= 0.6).sum()),
        "n_selector_failures_vs_floor": int(out["selector_failure_vs_floor"].sum()),
        "mean_oracle_best_miou": float(out["oracle_best_miou"].mean()),
    }
    (RUN_ROOT / "t1_regular_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fail_counts = out[out["intrinsic_final_miou"] < 0.6]["failure_mode"].value_counts().to_dict()
    lines = [
        "# T1 Regular Recovery Issue Log",
        "",
        f"- Rings >= 0.6: `{summary['n_ge_0_6']}` / `{summary['n_rings']}`",
        f"- Mean selected mIoU: `{summary['mean_intrinsic_final_miou']:.4f}`",
        f"- Mean oracle best mIoU: `{summary['mean_oracle_best_miou']:.4f}`",
        "",
        "## Failure Mode Counts (<0.6)",
    ]
    if fail_counts:
        for k, v in fail_counts.items():
            lines.append(f"- `{k}`: `{int(v)}`")
    else:
        lines.append("- none")
    (RUN_ROOT / "t1_issue_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # patch V5 scoreboard copy for tunnel-1 only
    patched = pd.read_csv(V5_SCORE)
    patch = out[["ring_key", "intrinsic_final_miou"]].rename(columns={"intrinsic_final_miou": "t1_intrinsic_recovered"})
    patched = patched.merge(patch, on="ring_key", how="left")
    mask = patched["family"].astype(int).eq(1) & patched["t1_intrinsic_recovered"].notna()
    patched.loc[mask, "intrinsic_final_miou"] = patched.loc[mask, "t1_intrinsic_recovered"]
    patched = patched.drop(columns=["t1_intrinsic_recovered"])
    patched["lift_stabilised_to_intrinsic"] = patched["intrinsic_final_miou"] - patched["stabilised_miou"]
    patched["lift_seed_to_intrinsic"] = patched["intrinsic_final_miou"] - patched["seeded_initial_miou"]
    patched.to_csv(REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t1_recovered.csv", index=False)

    # patch current T123-recovered copy to include new T1.
    if T123_RECOVERED.exists():
        t123 = pd.read_csv(T123_RECOVERED)
        t123 = t123.drop(columns=["intrinsic_final_miou"], errors="ignore").merge(
            out[["ring_key", "intrinsic_final_miou"]],
            on="ring_key",
            how="left",
        )
        t123.to_csv(REPO_ROOT / "logs" / "v5_t123_kcenter_recovery_v1" / "t123_kcenter_scoreboard_t1_recovered.csv", index=False)

    fam = (
        patched.groupby("family")[["seeded_initial_miou", "stabilised_miou", "intrinsic_final_miou"]]
        .mean()
        .reset_index()
    )
    fam.to_csv(REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_stage_table_by_family_t1_recovered.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

