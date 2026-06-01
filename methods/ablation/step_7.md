# Step 7 — LORO (enriched proxy)

**Planned sandbox:** `logs/bo_feature_enrichment_v1/`

---

## Status — skipped

Step 7 LORO runs **only after** revised Step 6 success criteria pass. Enriched proxy gate:

| Criterion | Result |
|-----------|--------|
| Pooled ρ > 0.35 | **0.356** — pass |
| Mean regret < 0.50 | **0.514** — fail |
| A4 lift on bo-regime | **−0.274** — fail |

Evidence: `logs/bo_feature_enrichment_v1/proxy_v2_success_gate.json`

```bash
./venv/bin/python bo/run_loro.py --run-root logs/bo_feature_enrichment_v1
# → "Step 7 LORO skipped" (loro_eligible: false)
```

No LORO folds were run on the weak v1 baseline or the partial-pass v2 proxy.

---

## Note

Bo-regime-only selection at **A3** reaches mean regret **0.114** — suggesting enriched features help ranking within the BO trial subset, but full-pool top-1 selection and A4+ group stacking still fail the paper-grade gate.
