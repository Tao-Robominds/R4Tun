"""Ontology layer for v3 held-out reflection.

Ports the structural-alignment checks from
``stages/v2/bo/structural_alignment_metrics.py`` and
``stages/v2/bo/run_detection_boundary_bo.py:_compute_meaningful_layout_metrics``
to the v3 pipeline (which writes ``final.csv`` + ``boundaries_per_ring.json``
but no ``detection/labelmap.npy``). The labelmap is reconstructed from
``pixel_to_point.pkl + final.csv + depth_map.png``, the same way
``agents/3_segmentation/scripts/extract_intrinsics.py`` already does.

Each ontology check is tagged hard-veto or soft-penalty:

* ``O_block_set`` (HARD): pred labels {1..7} = {K, B1, A1, A2, A3, A4, B2}
  must all be present in the ring (= seg_segment_type_completeness).
* ``O_block_count`` (HARD): boundaries_per_ring ring 0 must hold exactly 7
  components after unwrap-seam dedupe (with the v2 seam tolerance).
* ``O_no_duplicates`` (HARD): no expected block type appears more than once
  in boundaries_per_ring ring 0 after seam dedupe.
* ``O_one_K_unique`` (SOFT): K-class connected components <= 2 (allowing
  one wrap of the seam). Significant CCs only (>= 5% of K mass, >= 50 px).
* ``O_k_centrality`` (SOFT): K's largest CC spans >= 50% of image width.

Verdict schema::

    {
      "passed": bool,                 # all_of_min(hard checks) AND no fatal soft floors
      "hard_failures": [str, ...],    # names of failed hard checks
      "soft_score": float,            # in [0, 1], product of soft factors
      "structural_score": float,      # in [0, 1], used by J_reflect_v3
      "breakdown": {
        "O_block_set": {"passed": bool, "details": ...},
        ...
      }
    }
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Class IDs for K-bearing irregular linings (matches the v3 pipeline's
# seg pred encoding: K=1, B1=2, A1=3, A2=4, A3=5, A4=6, B2=7, BG=0).
EXPECTED_BLOCK_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
ID_TO_BLOCK = {1: "K", 2: "B1", 3: "A1", 4: "A2", 5: "A3", 6: "A4", 7: "B2"}
K_BLOCK_ID = 1
K_MIN_CC_MASS_FRAC = 0.05
K_MIN_CC_PIXELS = 50
K_CENTRALITY_MIN_FRAC = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _build_label_map(ring_dir: Path) -> Optional[np.ndarray]:
    """Reconstruct a (H, W) int label map from pixel_to_point + final.csv.

    Mirrors the helper in ``agents/3_segmentation/scripts/extract_intrinsics.py``
    so the v3 ontology can run without the v2 ``detection/labelmap.npy``.
    """
    p2p_path = ring_dir / "pixel_to_point.pkl"
    final_path = ring_dir / "final.csv"
    depth_path = ring_dir / "depth_map.png"
    if not (p2p_path.exists() and final_path.exists() and depth_path.exists()):
        return None
    try:
        import cv2  # noqa: WPS433  (heavy import gated by callers)
        img = cv2.imread(str(depth_path))
        if img is None:
            return None
        height, width = img.shape[:2]
        with open(p2p_path, "rb") as f:
            p2p = pickle.load(f)
        df = pd.read_csv(final_path, usecols=lambda c: c == "pred")
        if "pred" not in df.columns:
            return None
        pred = df["pred"].to_numpy()
        lm = np.zeros((height, width), dtype=np.int32)
        for entry in p2p:
            col = entry.get("pixel_x", entry.get("col", entry.get("pixel_col")))
            row = entry.get("pixel_y", entry.get("row", entry.get("pixel_row")))
            idx = entry.get("index", entry.get("point_index", entry.get("idx")))
            if row is None or col is None or idx is None:
                continue
            row = int(row); col = int(col); idx = int(idx)
            if 0 <= row < height and 0 <= col < width and 0 <= idx < len(pred):
                lm[row, col] = int(pred[idx])
        return lm
    except Exception:  # noqa: BLE001
        return None


def _densify_mask(mask: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Morphological closing on a sparse pixel-projected mask.

    The v3 pipeline stores labels per *point*, not per *pixel*; rasterising
    those labels onto the depth-map grid (via ``pixel_to_point.pkl``) leaves
    most pixels at 0 even when the underlying point set is contiguous. A
    short closing (3x3 or 5x5) restores the spatial extent of each block
    so 8-connected component analysis matches v2's pixel-rasterised
    labelmap behaviour. Iterations stay small to avoid bridging genuinely
    separate components.
    """
    if not mask.any():
        return mask
    try:
        import cv2

        k = max(1, int(kernel))
        struct = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        m = mask.astype(np.uint8)
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, struct, iterations=1)
        return closed.astype(bool)
    except Exception:  # noqa: BLE001
        return mask


def _connected_components_8(mask: np.ndarray) -> list[np.ndarray]:
    """Return list of binary masks, one per 8-connected component."""
    if not mask.any():
        return []
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        comp = np.zeros_like(mask, dtype=bool)
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= h or x < 0 or x >= w:
                continue
            if visited[y, x] or not mask[y, x]:
                continue
            visited[y, x] = True
            comp[y, x] = True
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    stack.append((ny, nx))
        comps.append(comp)
    return comps


def _seam_dedupe(blocks_sorted_by_y: list[str]) -> list[str]:
    """Drop the duplicate at the seam (top==bottom is one circular block)."""
    if len(blocks_sorted_by_y) >= 2 and blocks_sorted_by_y[0] == blocks_sorted_by_y[-1]:
        return blocks_sorted_by_y[:-1]
    return blocks_sorted_by_y


# ---------------------------------------------------------------------------
# Per-check implementations
# ---------------------------------------------------------------------------

def _check_block_set(ring_dir: Path) -> dict[str, Any]:
    """O_block_set (HARD): all 7 expected block types present in pred."""
    final_path = ring_dir / "final.csv"
    if not final_path.exists():
        return {"passed": False, "details": "final.csv missing"}
    try:
        df = pd.read_csv(final_path, usecols=lambda c: c == "pred")
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "details": f"read_csv failed: {exc!r}"}
    if "pred" not in df.columns:
        return {"passed": False, "details": "pred column missing"}
    actual = {int(v) for v in df["pred"].unique() if 0 < int(v) < 8}
    expected = set(EXPECTED_BLOCK_IDS)
    missing = sorted(expected - actual)
    return {
        "passed": len(missing) == 0,
        "details": {
            "actual_ids": sorted(actual),
            "missing_ids": missing,
            "missing_blocks": [ID_TO_BLOCK[m] for m in missing],
        },
    }


def _check_block_count_and_duplicates(ring_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """O_block_count (HARD): exactly 7 blocks per ring after seam dedupe.
    O_no_duplicates (HARD): no expected block type appears > 1 time.

    Both checks read ``boundaries_per_ring.json`` ring "0" and apply the
    v2 seam dedupe rule (top==bottom is one circular component split by
    the unwrap seam).
    """
    bpr = ring_dir / "boundaries_per_ring.json"
    if not bpr.exists():
        return (
            {"passed": False, "details": "boundaries_per_ring.json missing"},
            {"passed": False, "details": "boundaries_per_ring.json missing"},
        )
    try:
        data = json.loads(bpr.read_text())
    except Exception as exc:  # noqa: BLE001
        return (
            {"passed": False, "details": f"json parse failed: {exc!r}"},
            {"passed": False, "details": f"json parse failed: {exc!r}"},
        )
    entries = data.get("0", [])
    if not entries:
        return (
            {"passed": False, "details": "ring 0 has no boundary entries"},
            {"passed": False, "details": "ring 0 has no boundary entries"},
        )
    ordered = sorted(entries, key=lambda v: float(v.get("y", 0.0)))
    blocks = [str(e.get("block", "")) for e in ordered]
    blocks = _seam_dedupe(blocks)
    expected_count = len(EXPECTED_BLOCK_IDS)
    counts = {b: blocks.count(b) for b in sorted(set(blocks))}
    duplicates = {b: c for b, c in counts.items() if c > 1}
    block_count = {
        "passed": len(blocks) == expected_count,
        "details": {
            "n_blocks_after_seam_dedupe": len(blocks),
            "expected": expected_count,
            "blocks": blocks,
        },
    }
    no_dup = {
        "passed": len(duplicates) == 0,
        "details": {
            "duplicates": duplicates,
            "duplicate_count": int(sum(duplicates.values()) - len(duplicates)) if duplicates else 0,
        },
    }
    return block_count, no_dup


def _check_one_K_and_centrality(ring_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """O_one_K_unique (SOFT) + O_k_centrality (SOFT)."""
    lm = _build_label_map(ring_dir)
    if lm is None:
        return (
            {"passed": False, "details": "labelmap unavailable"},
            {"passed": False, "details": "labelmap unavailable"},
        )
    h, w = lm.shape
    k_mask_raw = (lm == K_BLOCK_ID)
    k_pixels_raw = int(k_mask_raw.sum())
    if k_pixels_raw == 0:
        return (
            {"passed": False, "details": "no K pixels in ring"},
            {"passed": False, "details": "no K pixels in ring"},
        )
    k_mask = _densify_mask(k_mask_raw, kernel=5)
    k_pixels = int(k_mask.sum())
    components = _connected_components_8(k_mask)
    sizes = sorted((int(c.sum()) for c in components), reverse=True)
    cc_count_total = len(components)
    min_size = max(K_MIN_CC_PIXELS, int(K_MIN_CC_MASS_FRAC * k_pixels))
    significant = [c for c in components if int(c.sum()) >= min_size]
    n_significant = len(significant)
    one_k = {
        "passed": 1 <= n_significant <= 2,
        "details": {
            "k_pixels_raw": k_pixels_raw,
            "k_pixels_densified": k_pixels,
            "n_components_total": cc_count_total,
            "n_components_significant": n_significant,
            "min_significant_size_px": min_size,
            "top_component_sizes": sizes[:5],
        },
    }
    if significant:
        largest = max(significant, key=lambda c: int(c.sum()))
        cols_with_k = largest.any(axis=0)
        col_frac = float(cols_with_k.sum()) / float(w) if w else 0.0
    else:
        col_frac = 0.0
    centrality = {
        "passed": col_frac >= K_CENTRALITY_MIN_FRAC,
        "details": {
            "largest_cc_col_fraction": col_frac,
            "min_required_fraction": K_CENTRALITY_MIN_FRAC,
        },
    }
    return one_k, centrality


# ---------------------------------------------------------------------------
# Public verdict + structural score
# ---------------------------------------------------------------------------

HARD_CHECKS = ("O_block_set", "O_block_count", "O_no_duplicates")
SOFT_CHECKS = ("O_one_K_unique", "O_k_centrality")


def evaluate_ontology(ring_dir: Path) -> dict[str, Any]:
    """Run all ontology checks and return the verdict.

    The returned dict is JSON-serialisable. ``passed`` is True iff every
    HARD check passes (SOFT checks contribute to ``soft_score`` but do
    not gate). ``structural_score`` is the J_reflect_v3 ingredient and
    falls in [0, 1] regardless of pass/fail.
    """
    block_set = _check_block_set(ring_dir)
    block_count, no_dup = _check_block_count_and_duplicates(ring_dir)
    one_k, centrality = _check_one_K_and_centrality(ring_dir)

    breakdown: dict[str, Any] = {
        "O_block_set": block_set,
        "O_block_count": block_count,
        "O_no_duplicates": no_dup,
        "O_one_K_unique": one_k,
        "O_k_centrality": centrality,
    }
    hard_failures = [name for name in HARD_CHECKS if not breakdown[name]["passed"]]

    one_k_factor = 1.0 if one_k["passed"] else 0.5
    cc_count = (one_k.get("details") or {}).get("n_components_significant")
    if isinstance(cc_count, int):
        if cc_count == 0 or cc_count >= 4:
            one_k_factor = 0.0
    centrality_factor = (centrality.get("details") or {}).get("largest_cc_col_fraction") or 0.0
    centrality_factor = _safe_clip(float(centrality_factor) / max(K_CENTRALITY_MIN_FRAC, 1e-9))

    dup_count = (no_dup.get("details") or {}).get("duplicate_count") or 0
    duplicate_penalty = _safe_clip(1.0 - float(dup_count) / float(len(EXPECTED_BLOCK_IDS)))

    soft_score = float(one_k_factor * centrality_factor)
    structural_score = _safe_clip(soft_score * duplicate_penalty)

    return {
        "passed": len(hard_failures) == 0,
        "hard_failures": hard_failures,
        "soft_score": soft_score,
        "structural_score": structural_score,
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# J_reflect_v3 composite
# ---------------------------------------------------------------------------

def compute_j_reflect_v3(
    *,
    ontology_verdict: dict[str, Any],
    g_pre_pass: bool,
    g_layout_pass: bool,
    g_stability_pass: bool,
    structural_floor: float = 0.05,
) -> float:
    """Reflection score the LLM optimises ascendingly.

    ``J_reflect_v3 = G_pre_pass * G_layout_pass * G_stability_pass *
    (structural_floor + (1 - structural_floor) * O_structural_score)``

    Mirrors the v2 ``J_reflect_v4`` floor (0.05) so a structurally
    invalid candidate is never zeroed out completely; the LLM still gets
    a small gradient to work against. Hard ontology vetoes are applied
    by the caller before this is read.
    """
    g_factor = float(bool(g_pre_pass) and bool(g_layout_pass) and bool(g_stability_pass))
    s = float(ontology_verdict.get("structural_score", 0.0))
    return float(g_factor * (structural_floor + (1.0 - structural_floor) * s))
