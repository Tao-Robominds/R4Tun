# Step 5b — Stages-informed catalog + audit

**Sandbox:** `logs/bo_feature_enrichment_v1/`  
**Inputs:** `logs/bo_experience_v1/calibration_records.csv`, stages v5/v6/v7 hints

---

## Deliverables

| Artifact | Path |
|----------|------|
| Feature catalog | `logs/bo_feature_enrichment_v1/feature_catalog.json` |
| Audit report | `logs/bo_feature_enrichment_v1/feature_audit.json` |
| Audit CSV | `logs/bo_feature_enrichment_v1/feature_audit.csv` |

---

## Findings (v1 pool)

- **3 constant columns** under layout BO: `feat_intrinsic_det_y_coverage_pct`, `feat_boundary_det_k_confidence_avg`, `feat_intrinsic_det_block_count_per_ring`
- Best pooled Spearman |ρ| ≈ **0.19** (`feat_intrinsic_det_min_y_gap_px`)
- Regime split confirms dilution: gt/perturb trials dominate high-mIoU band

## High-priority `needs_replay` (stages evidence)

| Feature | stages ρ |
|---------|----------:|
| `seg_segment_type_completeness` | 0.731 |
| `seg_ring_completeness_avg` | ~0.63 |
| `seg_mask_coverage_pct` | ~0.65 |

PRE7 ring-constant features (`finite_ratio`, `row_nonempty_ratio`, …) join from `data/bo_calibration/`.

---

## Pass gate

| Check | Result |
|-------|--------|
| Four mechanism groups covered | pass |
| ≥3 high-priority needs_replay | **3** |
| Gate | `step_5b_gate.json` — **passed** |

---

## Command

```bash
./venv/bin/python bo/audit_proxy_features.py
```
