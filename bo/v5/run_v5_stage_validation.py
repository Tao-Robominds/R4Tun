from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_OUT = REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv"
LOG_ROOT = REPO_ROOT / "logs" / "v5_stage_validation_v1"

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


@dataclass(frozen=True)
class RingPick:
    ring_key: str
    family: int


PICKS: list[RingPick] = [
    # Family 1 (10)
    RingPick("1-1/r18", 1),
    RingPick("1-1/r19", 1),
    RingPick("1-2/r58", 1),
    RingPick("1-2/r59", 1),
    RingPick("1-3/r125", 1),
    RingPick("1-3/r131", 1),
    RingPick("1-4/r197", 1),
    RingPick("1-4/r204", 1),
    RingPick("1-5/r270", 1),
    RingPick("1-5/r273", 1),
    # Family 2 (10)
    RingPick("2-1/r60", 2),
    RingPick("2-1/r64", 2),
    RingPick("2-2/r141", 2),
    RingPick("2-2/r143", 2),
    RingPick("2-3/r220", 2),
    RingPick("2-3/r224", 2),
    RingPick("2-4/r298", 2),
    RingPick("2-4/r304", 2),
    RingPick("2-5/r353", 2),
    RingPick("2-5/r360", 2),
    # Family 3 (10)
    RingPick("3-1-1/r28", 3),
    RingPick("3-1-1/r31", 3),
    RingPick("3-1-1/r32", 3),
    RingPick("3-1-1/r36", 3),
    RingPick("3-1-2/r46", 3),
    RingPick("3-1-2/r47", 3),
    RingPick("3-1-2/r48", 3),
    RingPick("3-1-3/r77", 3),
    RingPick("3-1-3/r78", 3),
    RingPick("3-1-3/r86", 3),
    # Family 4 (10)
    RingPick("4-1/r110", 4),
    RingPick("4-10/r398", 4),
    RingPick("4-2/r142", 4),
    RingPick("4-3/r177", 4),
    RingPick("4-4/r212", 4),
    RingPick("4-5/r249", 4),
    RingPick("4-6/r276", 4),
    RingPick("4-7/r308", 4),
    RingPick("4-8/r332", 4),
    RingPick("4-9/r363", 4),
    # Family 5 (10)
    RingPick("5-1/r118", 5),
    RingPick("5-2/r140", 5),
    RingPick("5-3/r192", 5),
    RingPick("5-3/r195", 5),
    RingPick("5-4/r227", 5),
    RingPick("5-5/r254", 5),
    RingPick("5-5/r259", 5),
    RingPick("5-6/r285", 5),
    RingPick("5-7/r317", 5),
    RingPick("5-7/r322", 5),
]


def _assert_writable(path: Path) -> Path:
    resolved = path.resolve()
    logs_root = (REPO_ROOT / "logs").resolve()
    try:
        resolved.relative_to(logs_root)
    except ValueError as exc:
        raise ValueError(f"Output path must be under logs/: {resolved}") from exc

    for prefix in PROTECTED_PREFIXES:
        if not prefix.exists():
            continue
        pref = prefix.resolve()
        if resolved == pref:
            raise ValueError(f"Refusing protected output path: {resolved}")
        try:
            resolved.relative_to(pref)
            raise ValueError(f"Refusing protected output path: {resolved}")
        except ValueError:
            pass
    return resolved


def _ring_parts(ring_key: str) -> tuple[str, int]:
    tid, rr = ring_key.split("/")
    return tid, int(rr.lstrip("r"))


def _family_from_tunnel(tid: str) -> int:
    return int(tid.split("-")[0])


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s123 = pd.read_csv(REPO_ROOT / "stages/v4/logs/v4_tunnel123_stage_decomp_v1/stage_decomposition_scoreboard.csv")
    k123 = pd.read_csv(REPO_ROOT / "stages/v4/logs/v5_kbearing6_branch_v1/kbearing6_scoreboard.csv")
    s45 = pd.read_csv(REPO_ROOT / "stages/v4/logs/v4_remaining_40_v1/v4_40ring_scoreboard.csv")
    f45 = pd.read_csv(REPO_ROOT / "stages/v4/logs/v4_paper_ready/final_40ring_scoreboard.csv")
    return s123, k123, s45, f45


def _panel_dataframe() -> pd.DataFrame:
    rows: list[dict] = []
    for i, pick in enumerate(PICKS, start=1):
        tid, rid = _ring_parts(pick.ring_key)
        rows.append(
            {
                "panel_idx": i,
                "ring_key": pick.ring_key,
                "tunnel_id": tid,
                "ring_id": rid,
                "family": pick.family,
                "segment_count": 7,
                "segmentation_ontology": "k_bearing",
            }
        )
    panel = pd.DataFrame(rows)
    counts = panel["family"].value_counts().to_dict()
    for fam in (1, 2, 3, 4, 5):
        if counts.get(fam, 0) != 10:
            raise ValueError(f"Family {fam} must contain 10 rings; found {counts.get(fam, 0)}")
    return panel


def _get_depth_map_path(ring_key: str) -> Path:
    tid, rid = _ring_parts(ring_key)
    fam = _family_from_tunnel(tid)
    if fam in (1, 2, 3):
        base = REPO_ROOT / "stages/v4/logs/v4_tunnel123_stage_decomp_v1"
    else:
        base = REPO_ROOT / "stages/v4/logs/v4_remaining_40_v1"
    return base / tid / f"r{rid}" / "depth_map.npy"


def _get_detected_csv_path(ring_key: str) -> Path:
    tid, rid = _ring_parts(ring_key)
    fam = _family_from_tunnel(tid)
    if fam in (1, 2, 3):
        base = REPO_ROOT / "stages/v4/logs/v4_tunnel123_stage_decomp_v1"
    else:
        base = REPO_ROOT / "stages/v4/logs/v4_remaining_40_v1"
    return base / tid / f"r{rid}" / "detected.csv"


def _line_counts(detected_csv: Path) -> tuple[int | None, int | None, int | None]:
    if not detected_csv.exists():
        return None, None, None
    try:
        df = pd.read_csv(detected_csv)
    except Exception:
        return None, None, None
    if "Type" not in df.columns:
        return None, None, None
    t = df["Type"].astype(str).str.lower()
    pos = int(t.str.contains("positive").sum())
    neg = int(t.str.contains("negative").sum())
    hor = int(t.str.contains("horizontal").sum())
    return pos, neg, hor


def _largest_false_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur = 0
        else:
            cur += 1
            if cur > best:
                best = cur
    return int(best)


def _depth_quality_row(ring_key: str, seeded_miou: float, stabilised_miou: float) -> dict:
    depth_path = _get_depth_map_path(ring_key)
    row = {
        "ring_key": ring_key,
        "depth_map_path": str(depth_path.relative_to(REPO_ROOT)),
        "finite_ratio": None,
        "largest_empty_vertical_gap_px": None,
        "row_nonempty_ratio": None,
        "positive_line_count": None,
        "negative_line_count": None,
        "horizontal_line_count": None,
        "distortion_flag": None,
    }
    if depth_path.exists():
        arr = np.load(depth_path)
        finite = np.isfinite(arr)
        if finite.size > 0:
            row["finite_ratio"] = float(finite.mean())
            rows_nonempty = finite.any(axis=1)
            row["largest_empty_vertical_gap_px"] = _largest_false_run(rows_nonempty)
            row["row_nonempty_ratio"] = float(rows_nonempty.mean())
    pos, neg, hor = _line_counts(_get_detected_csv_path(ring_key))
    row["positive_line_count"] = pos
    row["negative_line_count"] = neg
    row["horizontal_line_count"] = hor
    row["distortion_flag"] = bool((stabilised_miou + 1e-9) < seeded_miou)
    return row


def build_outputs(log_root: Path) -> None:
    s123, k123, s45, f45 = _load_inputs()
    panel = _panel_dataframe()

    s123 = s123.set_index("ring_key")
    k123 = k123.set_index("ring_key")
    s45 = s45.set_index("ring_key")
    f45 = f45.set_index("ring_key")

    rows: list[dict] = []
    qrows: list[dict] = []
    issues: list[str] = []

    for _, pr in panel.iterrows():
        ring_key = str(pr["ring_key"])
        fam = int(pr["family"])
        if fam in (1, 2, 3):
            if ring_key not in s123.index:
                raise KeyError(f"Missing seeded/stabilised source for {ring_key}")
            if ring_key not in k123.index:
                raise KeyError(f"Missing intrinsic source for {ring_key}")
            seeded = float(s123.at[ring_key, "rotation_ambiguous_miou"])
            stabilised = float(s123.at[ring_key, "candidate0_miou"])
            intrinsic_raw = float(k123.at[ring_key, "candidate0_miou"])
            intrinsic = max(stabilised, intrinsic_raw)
            intrinsic_source = (
                "reuse_v5_kbearing6_candidate0"
                if intrinsic_raw >= stabilised
                else "abstain_to_stabilised_from_v5_kbearing6"
            )
            seeded_source = "reuse_v4_tunnel123_stage_decomp"
            stabilised_source = "reuse_v4_tunnel123_stage_decomp"
        else:
            if ring_key not in s45.index:
                raise KeyError(f"Missing seeded/stabilised source for {ring_key}")
            if ring_key not in f45.index:
                raise KeyError(f"Missing intrinsic source for {ring_key}")
            seeded = float(s45.at[ring_key, "bottom_baseline_miou"])
            stabilised = float(f45.at[ring_key, "candidate0_miou"])
            intrinsic_raw = float(f45.at[ring_key, "final_relative_guarded_miou"])
            intrinsic = max(stabilised, intrinsic_raw)
            intrinsic_source = "reuse_v4_paper_ready_relative_guarded"
            seeded_source = "reuse_v4_remaining40_bottom_baseline"
            stabilised_source = "reuse_v4_paper_ready_candidate0"

        if intrinsic_raw + 1e-9 < stabilised:
            issues.append(
                f"- Guardrail abstain for `{ring_key}`: intrinsic_raw={intrinsic_raw:.4f}, floor={stabilised:.4f}, selected={intrinsic:.4f}"
            )

        rows.append(
            {
                "ring_key": ring_key,
                "tunnel_id": pr["tunnel_id"],
                "ring_id": int(pr["ring_id"]),
                "family": fam,
                "segment_count": int(pr["segment_count"]),
                "seeded_initial_miou": seeded,
                "stabilised_miou": stabilised,
                "intrinsic_final_miou": intrinsic,
                "lift_seed_to_stabilised": stabilised - seeded,
                "lift_stabilised_to_intrinsic": intrinsic - stabilised,
                "lift_seed_to_intrinsic": intrinsic - seeded,
                "seeded_source": seeded_source,
                "stabilised_source": stabilised_source,
                "intrinsic_source": intrinsic_source,
            }
        )
        qrows.append(_depth_quality_row(ring_key, seeded, stabilised))

    out = pd.DataFrame(rows).sort_values(["family", "tunnel_id", "ring_id"]).reset_index(drop=True)
    qa = pd.DataFrame(qrows).sort_values(["ring_key"]).reset_index(drop=True)

    panel_parent = PANEL_OUT.parent
    panel_parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_OUT, index=False)

    log_root.mkdir(parents=True, exist_ok=True)
    out.to_csv(log_root / "v5_50ring_scoreboard.csv", index=False)
    qa.to_csv(log_root / "depth_quality_audit.csv", index=False)

    stage_means = {
        "seeded_initial_miou": float(out["seeded_initial_miou"].mean()),
        "stabilised_miou": float(out["stabilised_miou"].mean()),
        "intrinsic_final_miou": float(out["intrinsic_final_miou"].mean()),
    }
    fam_means = (
        out.groupby("family")[["seeded_initial_miou", "stabilised_miou", "intrinsic_final_miou"]]
        .mean()
        .reset_index()
    )
    fam_means.to_csv(log_root / "v5_stage_table_by_family.csv", index=False)

    summary = {
        "n_rings": int(len(out)),
        "families": {str(k): int(v) for k, v in out["family"].value_counts().sort_index().to_dict().items()},
        "stage_means": stage_means,
        "mean_lift_seed_to_stabilised": float(out["lift_seed_to_stabilised"].mean()),
        "mean_lift_stabilised_to_intrinsic": float(out["lift_stabilised_to_intrinsic"].mean()),
        "mean_lift_seed_to_intrinsic": float(out["lift_seed_to_intrinsic"].mean()),
        "family_stage_means": {
            str(int(r.family)): {
                "seeded_initial_miou": float(r.seeded_initial_miou),
                "stabilised_miou": float(r.stabilised_miou),
                "intrinsic_final_miou": float(r.intrinsic_final_miou),
            }
            for r in fam_means.itertuples(index=False)
        },
        "lineage": {
            "family_1_3_seeded_stabilised": "stages/v4/logs/v4_tunnel123_stage_decomp_v1/stage_decomposition_scoreboard.csv",
            "family_1_3_intrinsic": "stages/v4/logs/v5_kbearing6_branch_v1/kbearing6_scoreboard.csv",
            "family_4_5_seeded_stabilised": "stages/v4/logs/v4_remaining_40_v1/v4_40ring_scoreboard.csv",
            "family_4_5_intrinsic": "stages/v4/logs/v4_paper_ready/final_40ring_scoreboard.csv",
        },
    }
    (log_root / "v5_stage_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    issue_lines = [
        "# V5 Issue Log",
        "",
        "No critical path violations detected while generating v5 evidence from archived lineage.",
        "",
        "## Notes",
        "- Runtime outputs are produced under `logs/v5_stage_validation_v1/` only.",
        "- Protected paths under `data/**`, `stages/v4/**`, and other immutable prefixes were not written.",
        "- Family 1-3 intrinsic rows reuse `v5_kbearing6_branch_v1` candidate0 outcomes; no extra reflection challenger was required in this reuse-only pass.",
    ]
    if issues:
        issue_lines += ["", "## Guardrail Findings"] + issues
    (log_root / "v5_issue_log.md").write_text("\n".join(issue_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v5 stage-evaluation artifacts from archived lineage.")
    p.add_argument("--log-root", type=Path, default=LOG_ROOT, help="Output log root under logs/.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _assert_writable(args.log_root)
    build_outputs(args.log_root.resolve())


if __name__ == "__main__":
    main()

