# Step 6 — Train the proxy

Mechanism-based **Ridge regression** on Step 5 cumulative feature groups (A2–A5). Same 480-trial pool; only the input feature set changes per level.

**Input:** `logs/bo_experience_v1/calibration_records.csv`  
**Models:** `logs/bo_experience_v1/proxy_models/proxy_{a2,a3,a4,a5}.json`  
**Summary:** `logs/bo_experience_v1/proxy_training_summary.json`

---

## Method

| Item | Choice |
|------|--------|
| Model | Ridge (L2), `alpha=1.0` |
| Preprocessing | StandardScaler per fit |
| Target | `response_gt_miou` |
| Feature selection | **None** — fixed cumulative groups from Step 5 |
| Constant columns | Dropped at fit time (layout BO invariants) |

Frozen coefficients + scaler stats saved as JSON for deployment / Step 7 LORO.

---

## Group ablation (calibration fit — design-time)

Ring-level proxy top-1 vs oracle-best mIoU on the **same 480 candidates**:

| Level | Features fit | Spearman ρ | MAE | Mean top-1 mIoU | Mean regret vs oracle | Lift vs prev |
|-------|-------------:|-----------:|----:|----------------:|----------------------:|-------------:|
| A2 | 5 | 0.299 | 0.134 | 0.160 | 0.708 | — |
| A3 | 9 | 0.294 | 0.134 | 0.160 | 0.708 | 0.000 |
| A4 | 10 | 0.308 | 0.132 | 0.224 | 0.644 | **0.065** |
| A5 | 12 | 0.308 | 0.132 | 0.224 | 0.644 | 0.000 |

**Dropped constant (all levels):** `feat_intrinsic_det_y_coverage_pct`  
**Also dropped A3+:** `feat_boundary_det_k_confidence_avg`  
**Also dropped A5:** `feat_design_det_k_count_match`

### Interpretation (calibration pool)

- **A3 vs A2:** boundary line-evidence params did not change ring-level top-1 selection on this fit (regret unchanged).
- **A4 vs A3:** adding design-regularizer features (`det_y_order_consistency`) improved mean proxy top-1 mIoU (+0.065) and reduced regret.
- **A5 vs A4:** guardrail features did not change selection on this fit (A4 and A5 identical metrics).

Pooled Spearman ρ ≈ 0.29–0.31 — moderate rank tracking on 480 trials. **Step 7 LORO** is the proper generalisation check (no p-values on pooled trials per ablation protocol).

---

## Failure-mode diagnosis (A5)

All 6 rings show proxy top-1 regret > 0.05 vs ring oracle on calibration fit — proxy selects plausible but sub-oracle layouts. Detail: `proxy_training_summary.json` → `failure_modes_A5`.

---

## Status — complete

| Check | Result |
|-------|--------|
| Models A2–A5 trained | pass |
| Group comparison logged | pass |
| Gate | `proxy_training_gate.json` — **passed** |

---

## Command

```bash
./venv/bin/python bo/train_proxy.py --run-root logs/bo_experience_v1 --alpha 1.0
```

Requires Step 4 + Step 5 artifacts.

---

## Outputs

| File | Content |
|------|---------|
| `proxy_models/proxy_*.json` | Frozen Ridge weights + scaler |
| `proxy_level_comparison.csv` | Cumulative level metrics |
| `proxy_ring_selection.csv` | Per-ring top-1 / regret |
| `proxy_calibration_predictions.csv` | All trial predictions by level |
| `proxy_training_gate.json` | Pass gate |

---

## Next (Step 7)

Leave-one-ring-out validation across 6 calibration rings — Spearman, top-1 mIoU, top-3 recall, regret, failure cases (no pooled p-values).
