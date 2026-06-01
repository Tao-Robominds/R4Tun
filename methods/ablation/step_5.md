# Step 5 — Define proxy feature groups

Mechanism-based groups mapped to Step 4 calibration records. Same **480-trial** candidate pool for all proxy ablation levels (A2–A5); only the scoring feature set changes.

**Source:** `logs/bo_experience_v1/calibration_records.csv`  
**Spec:** `logs/bo_experience_v1/proxy_feature_groups.json`

---

## Group definitions

### Group 1 — Intrinsic observability (6 columns)

| Mechanism | Calibration columns |
|-----------|---------------------|
| Depth coverage | `feat_intrinsic_det_y_coverage_pct`, `feat_intrinsic_n_reclassified_by_r_filter` |
| Missing segment IDs | `feat_intrinsic_det_block_count_per_ring` |
| Boundary spacing geometry | `feat_intrinsic_det_min_y_gap_px`, `feat_intrinsic_arc_width_entropy`, `param_k_y_frac` |

### Group 2 — Boundary evidence (5 columns)

| Mechanism | Calibration columns |
|-----------|---------------------|
| K detection confidence | `feat_boundary_det_k_confidence_avg` |
| Oblique / horizontal Hough thresholds | `param_hough_oblique_threshold`, `param_hough_horizontal_threshold` |
| Line merge distance and snap tolerance | `param_line_merge_distance`, `param_line_snap_tolerance_px` |

Line-evidence thresholds are **trial parameters** (deployment-observable chosen layout knobs).

### Group 3 — Segment-design validity (6 mechanisms → 6 unique columns)

| Mechanism | Calibration columns |
|-----------|---------------------|
| A/B minimum visible span | `feat_intrinsic_arc_width_entropy`, `feat_intrinsic_det_min_y_gap_px` |
| Cyclic-order validity | `feat_design_det_y_order_consistency` |
| Segment connectivity | `feat_intrinsic_det_y_coverage_pct`, `feat_intrinsic_det_block_count_per_ring` |
| Fragmentation / island penalty | `feat_intrinsic_n_reclassified_by_r_filter` |

*New vs A3:* primarily `feat_design_det_y_order_consistency` (others overlap G1).

### Group 4 — Hard guardrails (5 columns)

| Mechanism | Calibration columns |
|-----------|---------------------|
| Exactly one K | `feat_design_det_k_count_match` |
| Valid 6- or 7-segment count | `feat_intrinsic_det_block_count_per_ring`, `feat_design_det_k_count_match` |
| Valid cyclic order | `feat_design_det_y_order_consistency` |
| Guardrail violations | `feat_design_det_guardrail_violation_count`, `feat_design_det_ready_for_segmentation` |

**Deployment-only (not in calibration pool):** safety floor vs Arm B baseline — applied at runtime in A5 evaluation, not logged per BO trial.

---

## Cumulative ablation levels (proxy input columns)

| Level | Adds | n columns |
|-------|------|----------:|
| **A2** | Group 1 only | 6 |
| **A3** | + Group 2 | 11 |
| **A4** | + Group 3 (new: order consistency) | 12 |
| **A5** | + Group 4 (K count, guardrails, ready flag) | 15 |

Full column lists: `proxy_feature_groups.json` → `cumulative_ablation_levels`.

**Not in numeric proxy vector:** `param_per_ring_offsets` (JSON); arc layout summarised via `param_k_y_frac` + `feat_intrinsic_arc_width_entropy`.

---

## Status — complete

| Check | Result |
|-------|--------|
| Records validated | **480** |
| All group columns present | pass |
| All columns populated | pass |
| Gate | `proxy_feature_groups_gate.json` — **passed** |

---

## Command

```bash
./venv/bin/python bo/build_proxy_feature_groups.py --run-root logs/bo_experience_v1
```

Requires Step 4 `calibration_records.csv` first.

---

## Next (Step 6)

Train regularised proxy (e.g. Ridge) per cumulative level A2→A5 on the same 480 records; target `response_gt_miou`. Validate groups by ablation lift, not individual correlation only.
