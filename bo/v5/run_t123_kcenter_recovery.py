from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
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

RUN_ROOT = REPO_ROOT / "logs" / "v5_t123_kcenter_recovery_v1"
SRC_STAGE = REPO_ROOT / "logs" / "v5_t123_depth_contract_v1"
DEPTH_CONTRACT_SUMMARY = SRC_STAGE / "all_30_depth_gate_summary.json"
V5_PANEL = REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv"
V5_SCORE = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard.csv"

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


@dataclass
class Candidate:
    ring_key: str
    mode: str
    tag: str
    final_csv: Path
    miou: float | None
    intrinsic_score: float | None


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
    env = dict(**{k: v for k, v in os.environ.items()})
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


def _compute_miou(final_csv: Path) -> float | None:
    if not final_csv.exists():
        return None
    cols = pd.read_csv(final_csv, nrows=0).columns
    if "segment" not in cols or "pred" not in cols:
        return None
    df = pd.read_csv(final_csv, usecols=["segment", "pred"]).dropna(subset=["segment"])
    if df.empty:
        return None
    gt = df["segment"].astype(int).to_numpy()
    pred = df["pred"].astype(int).to_numpy()
    valid = (gt >= 0) & (gt <= 7) & (pred >= 0) & (pred <= 7)
    if not valid.any():
        return None
    labels = sorted(set(gt[valid].tolist()) | set(pred[valid].tolist()))
    ious = jaccard_score(gt[valid], pred[valid], average=None, labels=labels, zero_division=0)
    return float(np.mean(ious))


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
    entropy = float(-(nz * np.log(nz)).sum())
    entropy_norm = entropy / math.log(7.0)
    present_ratio = float((counts > 0).mean())
    cv = float(np.std(counts) / (np.mean(counts) + 1e-9))
    max_share = float(counts.max() / (counts.sum() + 1e-9))
    # Prefer complete and balanced segmentation footprints.
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


def _run_seg(mode_root: Path, ring_dir: Path, tunnel_id: str, ring_id: int, seg_file: str, bnd_file: str, tag: str) -> Path:
    src = ring_dir / bnd_file
    dst = ring_dir / "boundaries_per_ring.json"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    _run(
        [str(VENV_PY), str(SEG_CLI), tunnel_id, str(ring_id), "--data-dir", str(mode_root), "--segments-file", seg_file],
        ring_dir / "logs" / f"seg_{tag}.log",
    )
    out = ring_dir / f"final_{tag}.csv"
    shutil.copy2(ring_dir / "final.csv", out)
    return out


def _load_panel_t123() -> pd.DataFrame:
    panel = pd.read_csv(V5_PANEL)
    panel = panel[panel["family"].astype(int).isin([1, 2, 3])].copy()
    return panel.sort_values(["family", "tunnel_id", "ring_id"]).reset_index(drop=True)


def _stage_ring_from_v4(ring_key: str, dst_dir: Path) -> None:
    _require_depth_contract()
    src = _ring_dir(SRC_STAGE, ring_key)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src, dst_dir)
    (dst_dir / "logs").mkdir(parents=True, exist_ok=True)


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


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _depth_quality_gate(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in panel.itertuples(index=False):
        ring_key = str(r.ring_key)
        dpath = _ring_dir(SRC_STAGE, ring_key) / "depth_map.npy"
        rec: dict[str, Any] = {"ring_key": ring_key, "depth_map_path": str(dpath.relative_to(REPO_ROOT))}
        if dpath.exists():
            arr = np.load(dpath)
            finite = np.isfinite(arr)
            rows_nonempty = finite.any(axis=1)
            # longest False run
            best = 0
            cur = 0
            for v in rows_nonempty:
                if v:
                    cur = 0
                else:
                    cur += 1
                    best = max(best, cur)
            h = arr.shape[0]
            rec["finite_ratio"] = float(finite.mean())
            rec["largest_empty_vertical_gap_px"] = int(best)
            rec["largest_empty_vertical_gap_frac"] = float(best / max(1, h))
            rec["row_nonempty_ratio"] = float(rows_nonempty.mean())
            rec["gate_pass"] = bool((rec["finite_ratio"] >= 0.6) and (rec["largest_empty_vertical_gap_frac"] <= 0.25))
        else:
            rec["finite_ratio"] = None
            rec["largest_empty_vertical_gap_px"] = None
            rec["largest_empty_vertical_gap_frac"] = None
            rec["row_nonempty_ratio"] = None
            rec["gate_pass"] = False
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(RUN_ROOT / "depth_quality_gate.csv", index=False)
    return df


def _run_detection_for_mode(
    ring_key: str,
    mode_name: str,
    pre_overrides: dict[str, Any],
    det_overrides: dict[str, Any],
) -> list[Candidate]:
    tunnel_id, ring_id = _parse_ring_key(ring_key)
    base_ring = _ring_dir(RUN_ROOT, ring_key)
    mode_root = RUN_ROOT / "_mode_runs" / mode_name
    mode_dir = _ring_dir(mode_root, ring_key)
    if mode_dir.exists():
        shutil.rmtree(mode_dir)
    mode_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_ring, mode_dir)
    (mode_dir / "logs").mkdir(parents=True, exist_ok=True)

    pre_path = mode_dir / "parameters_preprocessing.json"
    det_path = mode_dir / "parameters_detection.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    det = json.loads(det_path.read_text(encoding="utf-8"))
    pre.update(pre_overrides)
    # Keep compatibility with schema used by preprocessing agent.
    if "gravity_anchor" not in pre or not isinstance(pre["gravity_anchor"], dict):
        pre["gravity_anchor"] = {"enabled": True, "n_bins": 360}
    pre["gravity_anchor"]["enabled"] = bool(pre_overrides.get("gravity_anchor_enabled", True))
    # Remove legacy flat key to avoid ambiguity.
    pre.pop("gravity_anchor_enabled", None)
    det.update(det_overrides)
    det["ring_topology"] = "k_bearing"
    det["segment_count"] = 6
    det["k_anchor_semantics"] = "center"
    det["enabled_blocks"] = list(KB6_BLOCKS)
    det["per_ring_offsets"] = {"0": dict(KB6_OFFSETS)}
    _write_json(pre_path, pre)
    _write_json(det_path, det)

    if pre_overrides:
        _run(
            [str(VENV_PY), str(PRE_CLI), tunnel_id, str(ring_id), "--data-dir", str(mode_root)],
            mode_dir / "logs" / "preprocessing.log",
        )
    _run(
        [str(VENV_PY), str(DET_CLI), tunnel_id, str(ring_id), "--data-dir", str(mode_root)],
        mode_dir / "logs" / "detection.log",
    )

    cands: list[Candidate] = []
    for direction in ("plus", "minus"):
        for shift in range(6):
            seg_name, bnd_name, tag = _materialize_rot_candidate(mode_dir, direction, shift)
            out_csv = _run_seg(mode_root, mode_dir, tunnel_id, ring_id, seg_name, bnd_name, f"{mode_name}_{tag}")
            cands.append(
                Candidate(
                    ring_key=ring_key,
                    mode=mode_name,
                    tag=tag,
                    final_csv=out_csv,
                    miou=_compute_miou(out_csv),
                    intrinsic_score=_intrinsic_score(out_csv),
                )
            )
    return cands


def main() -> int:
    _assert_writable(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    panel = _load_panel_t123()
    panel.to_csv(RUN_ROOT / "panel_t123_from_v5.csv", index=False)

    # Stage source ring dirs from archived v4 run.
    for r in panel.itertuples(index=False):
        _stage_ring_from_v4(str(r.ring_key), _ring_dir(RUN_ROOT, str(r.ring_key)))

    gate = _depth_quality_gate(panel)
    gate_fail_rate = 1.0 - float(gate["gate_pass"].mean()) if not gate.empty else 1.0

    records: list[dict[str, Any]] = []
    cand_rows: list[dict[str, Any]] = []

    for r in panel.itertuples(index=False):
        ring_key = str(r.ring_key)
        fam = int(r.family)
        candidates: list[Candidate] = []

        if fam in (1, 2):
            candidates.extend(
                _run_detection_for_mode(
                    ring_key,
                    mode_name="regular_prior_gap",
                    pre_overrides={
                        "gravity_anchor_enabled": True,
                        "depth_height_mode": "observed_gap_aligned",
                        "outlier_high_density_ring_start": 0,
                        "outlier_high_density_ring_end": 5,
                    },
                    det_overrides={
                        "detector_mode": "single_ring_regular_prior",
                        "regular_k_prior_low_frac": 1150.0 / 2777.0,
                        "regular_k_prior_high_frac": 1580.0 / 2777.0,
                        "regular_k_prior_low_ring_parity": 0,
                        "regular_prior_preferred_branch": "plus",
                    },
                )
            )
        else:
            # Tunnel 3: compare local canonical, local gap-aligned, and regular prior canonical.
            candidates.extend(
                _run_detection_for_mode(
                    ring_key,
                    mode_name="local_canonical",
                    pre_overrides={"gravity_anchor_enabled": True, "depth_height_mode": "canonical"},
                    det_overrides={"detector_mode": "single_ring_local"},
                )
            )
            candidates.extend(
                _run_detection_for_mode(
                    ring_key,
                    mode_name="local_gap",
                    pre_overrides={"gravity_anchor_enabled": True, "depth_height_mode": "observed_gap_aligned"},
                    det_overrides={"detector_mode": "single_ring_local"},
                )
            )
            candidates.extend(
                _run_detection_for_mode(
                    ring_key,
                    mode_name="regular_prior_canonical",
                    pre_overrides={"gravity_anchor_enabled": True, "depth_height_mode": "canonical"},
                    det_overrides={
                        "detector_mode": "single_ring_regular_prior",
                        "regular_k_prior_low_frac": 1150.0 / 2777.0,
                        "regular_k_prior_high_frac": 1580.0 / 2777.0,
                        "regular_k_prior_low_ring_parity": 0,
                    },
                )
            )

        for c in candidates:
            cand_rows.append(
                {
                    "ring_key": c.ring_key,
                    "mode": c.mode,
                    "tag": c.tag,
                    "intrinsic_score": c.intrinsic_score,
                    "miou": c.miou,
                    "final_csv": str(c.final_csv.relative_to(REPO_ROOT)),
                }
            )
        valid = [c for c in candidates if c.intrinsic_score is not None]
        if not valid:
            raise RuntimeError(f"No valid candidate for {ring_key}")
        sel = sorted(valid, key=lambda x: float(x.intrinsic_score), reverse=True)[0]

        records.append(
            {
                "ring_key": ring_key,
                "family": fam,
                "selected_mode": sel.mode,
                "selected_tag": sel.tag,
                "selected_intrinsic_score": sel.intrinsic_score,
                "selected_miou": sel.miou,
                "selected_final_csv": str(sel.final_csv.relative_to(REPO_ROOT)),
            }
        )

    pd.DataFrame(cand_rows).to_csv(RUN_ROOT / "candidate_scores.csv", index=False)
    selected = pd.DataFrame(records).sort_values(["family", "ring_key"]).reset_index(drop=True)

    v5 = pd.read_csv(V5_SCORE)
    v5_t123 = v5[v5["family"].astype(int).isin([1, 2, 3])][["ring_key", "seeded_initial_miou", "stabilised_miou"]].copy()
    out = v5_t123.merge(selected, on="ring_key", how="left")
    out["intrinsic_final_miou"] = out["selected_miou"].astype(float)
    out["lift_seed_to_stabilised"] = out["stabilised_miou"] - out["seeded_initial_miou"]
    out["lift_stabilised_to_intrinsic"] = out["intrinsic_final_miou"] - out["stabilised_miou"]
    out["lift_seed_to_intrinsic"] = out["intrinsic_final_miou"] - out["seeded_initial_miou"]

    # Failure modes.
    gate_m = gate[["ring_key", "gate_pass", "finite_ratio", "largest_empty_vertical_gap_frac"]]
    out = out.merge(gate_m, on="ring_key", how="left")
    out["failure_mode"] = np.where(
        out["intrinsic_final_miou"] >= 0.6,
        "pass_ge_0.6",
        np.where(
            out["gate_pass"] == False,  # noqa: E712
            "depth_failure",
            np.where(
                out["lift_stabilised_to_intrinsic"] <= 0.0,
                "guardrail_or_selector_abstain",
                "k_or_branch_or_rotation_error",
            ),
        ),
    )

    out.to_csv(RUN_ROOT / "t123_kcenter_scoreboard.csv", index=False)

    fam_means = (
        out.groupby("family")[["seeded_initial_miou", "stabilised_miou", "intrinsic_final_miou"]]
        .mean()
        .reset_index()
    )
    fam_means.to_csv(RUN_ROOT / "t123_family_stage_means.csv", index=False)

    summary = {
        "run_root": str(RUN_ROOT),
        "n_rings": int(len(out)),
        "depth_gate_fail_rate": float(gate_fail_rate),
        "mean_seeded_initial_miou": float(out["seeded_initial_miou"].mean()),
        "mean_stabilised_miou": float(out["stabilised_miou"].mean()),
        "mean_intrinsic_final_miou": float(out["intrinsic_final_miou"].mean()),
        "family_means": {
            str(int(r.family)): {
                "seeded_initial_miou": float(r.seeded_initial_miou),
                "stabilised_miou": float(r.stabilised_miou),
                "intrinsic_final_miou": float(r.intrinsic_final_miou),
            }
            for r in fam_means.itertuples(index=False)
        },
        "n_ge_0.6": int((out["intrinsic_final_miou"] >= 0.6).sum()),
        "n_below_0.6": int((out["intrinsic_final_miou"] < 0.6).sum()),
    }
    (RUN_ROOT / "t123_kcenter_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    issue_lines = [
        "# T123 K-Center Recovery Issue Log",
        "",
        f"- Depth gate fail rate: `{summary['depth_gate_fail_rate']:.4f}`",
        f"- Rings >= 0.6 after recovery: `{summary['n_ge_0.6']}` / `{summary['n_rings']}`",
        "",
        "## Failure Mode Counts (<0.6)",
    ]
    fcounts = (
        out[out["intrinsic_final_miou"] < 0.6]["failure_mode"]
        .value_counts()
        .to_dict()
    )
    if not fcounts:
        issue_lines.append("- none")
    else:
        for k, v in fcounts.items():
            issue_lines.append(f"- `{k}`: `{int(v)}`")
    (RUN_ROOT / "t123_issue_log.md").write_text("\n".join(issue_lines) + "\n", encoding="utf-8")

    # Patched V5 scoreboard copy.
    patched = pd.read_csv(V5_SCORE)
    patch_cols = out[["ring_key", "intrinsic_final_miou"]].rename(columns={"intrinsic_final_miou": "intrinsic_final_miou_recovered"})
    patched = patched.merge(patch_cols, on="ring_key", how="left")
    mask = patched["family"].astype(int).isin([1, 2, 3]) & patched["intrinsic_final_miou_recovered"].notna()
    patched.loc[mask, "intrinsic_final_miou"] = patched.loc[mask, "intrinsic_final_miou_recovered"]
    patched = patched.drop(columns=["intrinsic_final_miou_recovered"])
    patched["lift_stabilised_to_intrinsic"] = patched["intrinsic_final_miou"] - patched["stabilised_miou"]
    patched["lift_seed_to_intrinsic"] = patched["intrinsic_final_miou"] - patched["seeded_initial_miou"]
    patched.to_csv(REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t123_recovered.csv", index=False)

    # patched stage summary for quick readout
    pmeans = (
        patched.groupby("family")[["seeded_initial_miou", "stabilised_miou", "intrinsic_final_miou"]]
        .mean()
        .reset_index()
    )
    pmeans.to_csv(REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_stage_table_by_family_t123_recovered.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

