# Step 5f — Lock A3+v5 proxies for held-out Stage A

**Sandbox:** `logs/bo_v5_proxy_v1/`  
**Master manifest:** `logs/bo_v5_proxy_v1/PROXY_A3_V5_HELD_OUT_MANIFEST.json`  
**Lock CLI:** `./venv/bin/python bo/lock_proxy_a3_v5_held_out.py`

Three frozen Ridge proxies for held-out within-pool selection. Trained on BO-calib only (`calibration_records_v2` + v5 columns); **do not retrain on held-out**.

## Variants

| ID | Name | Features | LORO mean selected mIoU | Per-ring (BO+perturb) |
|----|------|---------:|------------------------:|----------------------|
| **p11** | A3_v5_p11 | 11 | **0.792** | 0.855, 0.463, 0.877, 0.903, 0.827, 0.828 |
| **p10** | A3_v5_p10 | 10 | **0.767** | 0.855, 0.315, 0.877, 0.903, 0.827, 0.828 |
| **p9** | A3_v5_p9 | 9 | **0.693** | 0.855, 0.481, 0.877, 0.903, 0.827, 0.214 |

Reference: A3-slim LORO **0.726**.

## Feature sets

**Shared A3 (6):** `row_nonempty_ratio`, `valid_pixels`, `n_reclassified`, `arc_width_entropy`, `k_y_frac`, `hough_oblique`

**p11 adds (5):** `v5_balance_norm`, `v5_geom_boundary_gap_cv`, `v5_S_boundary`, `seg_k_size_ratio`, `seg_groove_score`

**p10 drops from p11:** `v5_geom_boundary_gap_cv`

**p9 drops from p10:** `seg_k_size_ratio`

**Redundant (all variants):** `v5_S_layout_coverage`, `v5_S_spacing`

## Artifacts per variant

| Variant | Manifest | Model | Equation |
|---------|----------|-------|----------|
| p11 | `PROXY_P11_MANIFEST.json` | `proxy_p11.json` | `proxy_p11_equation.json` |
| p10 | `PROXY_P10_MANIFEST.json` | `proxy_p10.json` | `proxy_p10_equation.json` |
| p9 | `PROXY_P9_MANIFEST.json` | `proxy_p9.json` | `proxy_p9_equation.json` |

## Selection rule

```text
pred_mIoU = Ridge(A3 + v5/seg features)
pick = argmax pred_mIoU over kind in {bo, perturb_*}
exclude gt_layout
requires seg replay at runtime
```

## Held-out usage

Stage A (`stage_a_proxy_select`, 25 rings): score candidate pools with frozen p11/p10/p9 — compare transfer vs A3-slim baseline. No proxy retrain on held-out.
