# Step 6 (revised) — Retrain on enriched records

**Sandbox:** `logs/bo_feature_enrichment_v1/`  
**Records:** `calibration_records_v2.csv` (480 trials, seg + PRE7 enriched)  
**Models:** `proxy_v2_models/`

---

## v1 → v2 comparison (A5)

| Metric | v1 | v2 | Criterion |
|--------|---:|---:|-----------|
| Pooled Spearman ρ | 0.308 | **0.356** | > 0.35 ✓ |
| Mean ring regret (full pool) | ~0.64 | **0.514** | < 0.50 ✗ |
| A4 lift on bo-regime regret | — | **−0.274** (A4 worse than A3) | > 0 ✗ |

---

## Cumulative ablation (v2)

| Level | ρ | mean regret (full) | mean regret (bo+perturb) |
|-------|--:|-------------------:|-------------------------:|
| A2 | 0.311 | 0.708 | 0.451 |
| A3 | 0.310 | 0.708 | **0.114** |
| A4 | 0.356 | 0.514 | 0.388 |
| A5 | 0.356 | 0.514 | 0.388 |

Seg completeness features (A4) improve **pooled ρ** but hurt **bo-regime selection** vs A3 (overfit / regime mismatch). Full-pool regret remains dominated by non-`bo` trials (gt_layout oracle at ~0.87 mIoU).

---

## Success gate

`proxy_v2_success_gate.json` — **not passed** (1/3 criteria). **Step 7 LORO blocked** per plan.

Next iteration (not run here): Tier C v6 descriptors or narrow A4 to top seg columns; do not rerun BO.

---

## Canonical proxy — A3-slim (locked)

| Item | Value |
|------|-------|
| Features | 8 (4 PRE7 + 3 det + `hough_oblique`) |
| Seg replay | **No** |
| Selection regime | `bo` + `perturb_*` only |
| Pooled ρ | 0.303 |
| BO-regime regret | **0.141** |
| Full-pool regret | 0.409 |
| Manifest | `logs/bo_feature_enrichment_v1/PROXY_A3_SLIM_MANIFEST.json` |
| Model | `logs/bo_feature_enrichment_v1/proxy_v2_models/proxy_a3_slim.json` |

```bash
./venv/bin/python bo/lock_proxy_a3_slim.py
```

---

## Commands

```bash
./venv/bin/python bo/enrich_calibration_features.py --skip-gate
./venv/bin/python bo/train_proxy.py --run-root logs/bo_feature_enrichment_v1 --v2
```
