from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from t45_depth_contract import (
    GATE_RINGS,
    RUN_ROOT,
    assert_writable,
    audit_many,
    audit_many_source,
    load_t45_panel,
    preprocess_ring,
    write_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
T123_AUDIT = REPO_ROOT / "logs" / "v5_t123_depth_contract_v1" / "all_30_depth_gate_depth_quality_audit.csv"
PAPER_AUDIT_ROOT = REPO_ROOT / "logs" / "v5_depth_contract_paper_audit_v1"


def _run_depth_contract(ring_keys: list[str], *, label: str) -> None:
    for ring_key in ring_keys:
        preprocess_ring(ring_key)
    audit = audit_many(ring_keys).sort_values("ring_key").reset_index(drop=True)
    audit.to_csv(RUN_ROOT / f"{label}_depth_quality_audit.csv", index=False)
    write_summary(label, audit)


def _make_visual_review(audit_t45: pd.DataFrame) -> pd.DataFrame:
    out = audit_t45.copy()
    out["flag_missing_depth_map"] = out["depth_gate_reason"].astype(str).str.contains("missing_depth_map")
    out["flag_low_finite_ratio"] = out["depth_gate_reason"].astype(str).str.contains("finite_ratio_low")
    out["flag_low_row_coverage"] = out["depth_gate_reason"].astype(str).str.contains("row_nonempty_ratio_low")
    out["flag_large_empty_band"] = out["depth_gate_reason"].astype(str).str.contains("large_empty_vertical_gap")
    out["manual_review_required"] = ~out["depth_gate_pass"].astype(bool)
    cols = [
        "ring_key",
        "depth_map_png",
        "depth_gate_pass",
        "depth_gate_reason",
        "finite_ratio",
        "row_nonempty_ratio",
        "largest_empty_vertical_gap_frac",
        "flag_missing_depth_map",
        "flag_low_finite_ratio",
        "flag_low_row_coverage",
        "flag_large_empty_band",
        "manual_review_required",
    ]
    return out[cols].sort_values(["manual_review_required", "ring_key"], ascending=[False, True]).reset_index(drop=True)


def _write_contact_sheet(audit_t45: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        note = (
            "Contact sheet not generated because matplotlib/numpy imports failed.\n"
            "Use depth_map_png paths in all_20_depth_gate_depth_quality_audit.csv for manual review.\n"
        )
        (RUN_ROOT / "t45_depth_map_contact_sheet.txt").write_text(note, encoding="utf-8")
        return

    rows = audit_t45.sort_values("ring_key").reset_index(drop=True)
    n = len(rows)
    cols = 5
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows=nrows, ncols=cols, figsize=(4.0 * cols, 2.8 * nrows))
    if nrows == 1:
        axes = [axes]
    flat_axes = []
    for row_axes in axes:
        if hasattr(row_axes, "__iter__"):
            flat_axes.extend(list(row_axes))
        else:
            flat_axes.append(row_axes)

    for ax_idx, ax in enumerate(flat_axes):
        if ax_idx >= n:
            ax.axis("off")
            continue
        rec = rows.iloc[ax_idx]
        png = REPO_ROOT / str(rec["depth_map_png"])
        title = f"{rec['ring_key']} | pass={bool(rec['depth_gate_pass'])}"
        if png.exists():
            img = plt.imread(png)
            if img.ndim == 2:
                ax.imshow(img, cmap="viridis")
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=9)
        else:
            ax.text(0.5, 0.5, f"{rec['ring_key']}\nmissing png", ha="center", va="center", fontsize=9)
            ax.set_title("missing", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RUN_ROOT / "t45_depth_map_contact_sheet.png", dpi=180)
    plt.close(fig)


def _merge_all50_audit(t45_audit: pd.DataFrame) -> pd.DataFrame:
    if not T123_AUDIT.exists():
        raise FileNotFoundError(f"Missing T123 audit: {T123_AUDIT}")
    t123 = pd.read_csv(T123_AUDIT)
    panel = pd.read_csv(REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv")[["ring_key", "family"]]
    all50 = pd.concat([t123, t45_audit], ignore_index=True)
    all50 = all50.merge(panel, on="ring_key", how="left")
    cols = [
        "ring_key",
        "family",
        "depth_map_path",
        "depth_map_png",
        "selected_interpolation_window",
        "height_px",
        "width_px",
        "finite_ratio",
        "row_nonempty_ratio",
        "largest_empty_vertical_gap_px",
        "largest_empty_vertical_gap_frac",
        "depth_gate_pass",
        "depth_gate_reason",
    ]
    for c in cols:
        if c not in all50.columns:
            all50[c] = None
    return all50[cols].sort_values(["family", "ring_key"]).reset_index(drop=True)


def _write_paper_summary(all50: pd.DataFrame, t45_before: pd.DataFrame, t45_after: pd.DataFrame) -> None:
    PAPER_AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    all50.to_csv(PAPER_AUDIT_ROOT / "all_50_depth_quality_audit.csv", index=False)

    t45_fail_before = t45_before[~t45_before["depth_gate_pass"].astype(bool)]
    t45_fail_after = t45_after[~t45_after["depth_gate_pass"].astype(bool)]
    summary = {
        "all_50_rows": int(len(all50)),
        "all_50_pass": int(all50["depth_gate_pass"].fillna(False).astype(bool).sum()),
        "all_50_fail": int((~all50["depth_gate_pass"].fillna(False).astype(bool)).sum()),
        "t45_rows": int(len(t45_after)),
        "t45_fail_before_count": int(len(t45_fail_before)),
        "t45_fail_after_count": int(len(t45_fail_after)),
        "t45_failed_rings_after": t45_fail_after["ring_key"].astype(str).tolist(),
    }
    (PAPER_AUDIT_ROOT / "all_50_depth_quality_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# v5 Depth QA (All 50 Rings)",
        "",
        "Method:",
        "- T1/T2/T3: existing hard-gated audit reused from `v5_t123_depth_contract_v1`.",
        "- T4/T5: rerun with r4tun-style depth contract (`gravity_anchor`, `observed_gap_aligned`, interpolation sweep).",
        "",
        "Thresholds:",
        "- finite_ratio >= 0.60",
        "- row_nonempty_ratio >= 0.90",
        "- largest_empty_vertical_gap_frac <= 0.08",
        "",
        f"Results: pass={summary['all_50_pass']}, fail={summary['all_50_fail']}, total={summary['all_50_rows']}.",
        f"T4/T5 fails before={summary['t45_fail_before_count']}, after={summary['t45_fail_after_count']}.",
    ]
    if summary["t45_failed_rings_after"]:
        lines.append("Unresolved T4/T5 rings after correction:")
        for ring in summary["t45_failed_rings_after"]:
            lines.append(f"- {ring}")
    else:
        lines.append("No unresolved T4/T5 rings after correction.")
    (PAPER_AUDIT_ROOT / "all_50_depth_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hard-gated r4tun-style depth-map contract for T4/T5.")
    parser.add_argument(
        "--scope",
        choices=["gate", "all", "audit"],
        default="all",
        help="Run only gate rings, all T4/T5 rings, or audit existing outputs.",
    )
    args = parser.parse_args()

    assert_writable(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    assert_writable(PAPER_AUDIT_ROOT)

    panel = load_t45_panel()
    panel.to_csv(RUN_ROOT / "panel_t45_segment6.csv", index=False)
    all_keys = panel["ring_key"].astype(str).tolist()

    if args.scope == "audit":
        t45_after = audit_many(all_keys).sort_values("ring_key").reset_index(drop=True)
        t45_after.to_csv(RUN_ROOT / "all_20_depth_gate_depth_quality_audit.csv", index=False)
        write_summary("all_20_depth_gate", t45_after)
        t45_before = audit_many_source(all_keys).sort_values("ring_key").reset_index(drop=True)
        t45_before.to_csv(RUN_ROOT / "all_20_source_depth_quality_audit.csv", index=False)
        visual = _make_visual_review(t45_after)
        visual.to_csv(RUN_ROOT / "t45_manual_review.csv", index=False)
        _write_contact_sheet(t45_after)
        all50 = _merge_all50_audit(t45_after)
        _write_paper_summary(all50, t45_before, t45_after)
        return 0

    # Always capture source-state audit before correction for traceability.
    t45_before = audit_many_source(all_keys).sort_values("ring_key").reset_index(drop=True)
    t45_before.to_csv(RUN_ROOT / "all_20_source_depth_quality_audit.csv", index=False)
    write_summary("all_20_source_depth_gate", t45_before)

    _run_depth_contract(GATE_RINGS, label="single_instance_depth_gate")
    if args.scope == "gate":
        return 0

    _run_depth_contract(all_keys, label="all_20_depth_gate")
    t45_after = audit_many(all_keys).sort_values("ring_key").reset_index(drop=True)
    visual = _make_visual_review(t45_after)
    visual.to_csv(RUN_ROOT / "t45_manual_review.csv", index=False)
    _write_contact_sheet(t45_after)
    all50 = _merge_all50_audit(t45_after)
    _write_paper_summary(all50, t45_before, t45_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
