# Step 5d — Revise mechanism groups (v2)

**Source:** `logs/bo_feature_enrichment_v1/calibration_records_v2.csv`  
**Spec:** `logs/bo_feature_enrichment_v1/proxy_feature_groups_v2.json`

---

## v2 group mapping

| Group | Mechanisms | Key columns |
|-------|------------|-------------|
| G1 Intrinsic | depth 3a, PRE7 retention, det spacing | `feat_pre_finite_ratio`, `feat_pre_row_nonempty_ratio`, `feat_intrinsic_det_min_y_gap_px`, `param_k_y_frac` |
| G2 Boundary | Hough/merge/snap params | `param_hough_oblique_threshold`, `param_line_merge_distance`, … |
| G3 Design | seg completeness (replay), order | `feat_seg_ring_completeness_avg`, `feat_seg_mask_coverage_pct`, `feat_design_det_y_order_consistency` |
| G4 Guardrails | K count, violation count, ready flags | `feat_design_det_guardrail_violation_count`, `feat_seg_ready_for_evaluation` |

**Dropped v1 constants:** `feat_intrinsic_det_y_coverage_pct`, `feat_boundary_det_k_confidence_avg`, `feat_intrinsic_det_block_count_per_ring`.

---

## Cumulative levels (v2)

| Level | n columns |
|-------|----------:|
| A2 | 11 |
| A3 | 15 |
| A4 | 21 |
| A5 | 25 |

Selection metrics should use **`bo_regime_only`** (exclude `gt_layout` from top-1 regret eval).

---

## Status — complete

| Check | Result |
|-------|--------|
| All v2 columns present | pass |
| Gate | `proxy_feature_groups_v2_gate.json` — **passed** |

---

## Command

```bash
./venv/bin/python bo/build_proxy_feature_groups_v2.py --run-root logs/bo_feature_enrichment_v1
```
