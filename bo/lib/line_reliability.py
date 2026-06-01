"""Deployment-time line evidence reliability (rho_K, rho_AB)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.layout_bo import RingContext


@dataclass
class LineEvidence:
    oblique_pos: int = 0
    oblique_neg: int = 0
    horizontal: int = 0
    k_confidence: float = 0.0
    ab_order_consistency: float = 0.0
    oblique_angle_consistency: float = 0.0
    horizontal_spacing_consistency: float = 0.0
    finite_ratio: float = 0.0
    row_nonempty_ratio: float = 0.0
    rho_K: float = 0.0
    rho_AB: float = 0.0
    k_y: float | None = None
    k_type: str | None = None
    horizontal_y: list[float] = field(default_factory=list)
    log_path: str | None = None
    valid_line_anchor: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "oblique_pos": self.oblique_pos,
            "oblique_neg": self.oblique_neg,
            "horizontal": self.horizontal,
            "line_detection_confidence_K": self.k_confidence,
            "line_detection_confidence_AB": self.ab_order_consistency,
            "line_oblique_angle_consistency": self.oblique_angle_consistency,
            "line_horizontal_spacing_consistency": self.horizontal_spacing_consistency,
            "finite_ratio": self.finite_ratio,
            "row_nonempty_ratio": self.row_nonempty_ratio,
            "rho_K": self.rho_K,
            "rho_AB": self.rho_AB,
            "k_y": self.k_y,
            "k_type": self.k_type,
            "horizontal_y": self.horizontal_y,
            "valid_line_anchor": self.valid_line_anchor,
            "log_path": self.log_path,
        }


def _parse_log_counts(log_text: str) -> dict[str, int]:
    counts = {"oblique_pos": 0, "oblique_neg": 0, "horizontal": 0}
    m = re.search(r"Lines: \+(\d+) -(\d+) H(\d+)", log_text or "")
    if m:
        counts["oblique_pos"] = int(m.group(1))
        counts["oblique_neg"] = int(m.group(2))
        counts["horizontal"] = int(m.group(3))
    return counts


def _load_depth_qa(ctx: RingContext) -> tuple[float, float]:
    contract = ctx.src_ring / "depth_contract_selected.json"
    if contract.is_file():
        audit = json.loads(contract.read_text(encoding="utf-8")).get("audit", {})
        return float(audit.get("finite_ratio", 0.0)), float(audit.get("row_nonempty_ratio", 0.0))
    depth = ctx.sandbox_ring / "depth_map.npy"
    if depth.is_file():
        arr = np.load(depth)
        finite = np.isfinite(arr)
        return float(finite.sum() / max(arr.size, 1)), float((finite.sum(axis=1) > 0).mean())
    return 0.0, 0.0


def _horizontal_ys_from_detected(ctx: RingContext) -> list[float]:
    det = ctx.sandbox_ring / "detected.csv"
    if not det.is_file():
        return []
    df = pd.read_csv(det)
    ys = []
    for col in ("Y", "y"):
        if col in df.columns:
            ys.extend(pd.to_numeric(df[col], errors="coerce").dropna().tolist())
    return sorted(set(float(y) for y in ys))


def compute_line_evidence(
    ctx: RingContext,
    *,
    k_y: float | None,
    k_type: str | None,
    line_counts: dict[str, int],
    log_path: Path | None = None,
) -> LineEvidence:
    finite, row_ne = _load_depth_qa(ctx)
    ob_pos = int(line_counts.get("oblique_pos", 0))
    ob_neg = int(line_counts.get("oblique_neg", 0))
    horiz = int(line_counts.get("horizontal", 0))
    ob_total = ob_pos + ob_neg
    n_blocks = ctx.segment_count
    expected_horiz = max(n_blocks - 2, 1)

    k_conf = 0.0
    det_csv = ctx.sandbox_ring / "detected.csv"
    if det_csv.is_file():
        df = pd.read_csv(det_csv)
        if "Confidence" in df.columns and not df.empty:
            k_conf = float(pd.to_numeric(df["Confidence"], errors="coerce").mean())

    ev = LineEvidence(
        oblique_pos=ob_pos,
        oblique_neg=ob_neg,
        horizontal=horiz,
        k_confidence=k_conf,
        finite_ratio=finite,
        row_nonempty_ratio=row_ne,
        k_y=k_y,
        k_type=k_type,
        horizontal_y=_horizontal_ys_from_detected(ctx),
        log_path=str(log_path) if log_path else None,
    )

    rho_k = 0.0
    if 1 <= ob_total <= 3:
        rho_k += 0.35
    elif ob_total == 0:
        rho_k -= 0.5
    elif ob_total > 3:
        rho_k -= 0.25
    if k_conf >= 0.7:
        rho_k += 0.25
    elif k_conf > 0:
        rho_k += 0.1 * k_conf
    if ob_total > 0:
        rho_k += 0.20
    if finite >= 0.60 and row_ne >= 0.90:
        rho_k += 0.20
    ev.rho_K = float(np.clip(rho_k, 0.0, 1.0))

    rho_ab = 0.0
    if horiz > 0 and abs(horiz - expected_horiz) <= 2:
        rho_ab += 0.40
    elif horiz == 0:
        rho_ab -= 0.40
    if horiz > expected_horiz + 3:
        rho_ab -= 0.20
    if len(ev.horizontal_y) >= 2:
        gaps = np.diff(sorted(ev.horizontal_y))
        cv = float(np.std(gaps) / max(np.mean(gaps), 1e-6)) if len(gaps) else 1.0
        ev.horizontal_spacing_consistency = float(max(0.0, 1.0 - cv))
        if cv < 0.3:
            rho_ab += 0.30
    if finite >= 0.60:
        rho_ab += 0.30
    ev.rho_AB = float(np.clip(rho_ab, 0.0, 1.0))
    ev.oblique_angle_consistency = 0.8 if ob_total in (1, 2) else (0.3 if ob_total else 0.0)
    ev.ab_order_consistency = ev.horizontal_spacing_consistency
    ev.valid_line_anchor = ev.rho_K >= 0.2 and k_y is not None
    return ev
