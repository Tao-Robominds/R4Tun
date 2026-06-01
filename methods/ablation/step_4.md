# Step 4 — Log each BO trial

**Calibration records:** parameters, proxy features, offline GT mIoU.

**Source trials:** `logs/bo_experience_v1/bo_trials.csv` (480 rows, Step 3)  
**Sandbox output:** `logs/bo_experience_v1/calibration_records.csv`

---

## Record schema

Each row is one candidate layout trial with three blocks:

| Block | Prefix | Contents |
|-------|--------|----------|
| **Parameters** | `param_*` | K position, A/B offsets, Hough thresholds, merge/snap distance, padding, fixed `r_surface_min` |
| **Proxy features** | `feat_intrinsic_*`, `feat_boundary_*`, `feat_design_*` | Deployment-observable signals at trial time |
| **Offline response** | `response_*` | GT mIoU, regret vs ceiling |

Meta: `trial_id`, `case_id`, `kind` (warm / perturb / GP-BO phase).

Full column map: `logs/bo_experience_v1/calibration_records_schema.json`

---

## Proxy feature groups (logged)

| Group | Columns |
|-------|---------|
| Intrinsic observability | `det_y_coverage_pct`, `det_block_count_per_ring`, `det_min_y_gap_px`, `arc_width_entropy`, `n_reclassified_by_r_filter`, `r_surface_min_otsu_ref` |
| Boundary evidence | `det_k_confidence_avg` + line-evidence params (`hough_*`, `line_merge_distance`, `line_snap_tolerance_px`) |
| Design regularizer | `det_k_count_match`, `det_y_order_consistency`, `det_ready_for_segmentation`, `det_guardrail_violation_count` |

Mechanism-based group definitions for proxy training are formalised in Step 5.

---

## Status — complete

| Check | Result |
|-------|--------|
| Records | **480 / 480** |
| Parameters logged | pass |
| Intrinsic features | pass |
| Boundary evidence | pass |
| Design regularizer | pass |
| Offline GT mIoU | pass |

**Gate:** `logs/bo_experience_v1/calibration_records_gate.json` — **passed**

| Stat | Value |
|------|------:|
| Mean GT mIoU | 0.289 |
| Min GT mIoU | 0.072 |
| Max GT mIoU | 0.917 |
| Rings | 6 |

---

## Command

```bash
./venv/bin/python bo/build_calibration_records.py \
  --run-root logs/bo_experience_v1 \
  --expected-n 480
```

---

## Outputs

- `calibration_records.csv` — structured 480-row calibration table
- `calibration_records_schema.json` — column ontology
- `calibration_records_gate.json` — completeness gate

Raw trial artifacts (per ring): `logs/bo_experience_v1/<tunnel>/r<N>/bo_trials.csv`
