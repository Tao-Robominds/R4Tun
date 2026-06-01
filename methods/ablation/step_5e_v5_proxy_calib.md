# Step 5e — V5 proxy features on BO calibration (pre held-out)

**Sandbox:** `logs/bo_v5_proxy_v1/`  
**Corpus (read-only):** `data/bo_calibration/`  
**Trials:** `logs/bo_experience_v1/bo_trials.csv` (480 trials, 6 rings)

## What was added

| Module | Role |
|--------|------|
| `bo/lib/v5_proxy_features.py` | `balance_norm`, boundary geometry, class distribution, `S_boundary` composite |
| `bo/lib/v5_relative_proxy.py` | candidate0-relative deltas, `J_reflect`, within-pool selection metrics |
| `bo/enrich_v5_proxy_features.py` | Replay trials → extract v5 columns |
| `bo/eval_v5_relative_proxy.py` | Compare v5 variants vs A3-slim on calib panel |

## V5 feature set (per trial)

- **Class distribution:** `balance_norm`, `entropy`, `present_ratio`, `cv`, `max_share`, `struct_missing_ids_before_n`
- **Boundary geometry:** `geom_boundary_gap_cv`, min/max/mean gap fractions
- **Composite:** `S_boundary = S_continuity × S_K × S_spacing × S_layout_coverage`
- **Seg block-balance (replay):** `seg_k_size_ratio`, `seg_block_size_variance_ratio`, `seg_groove_score`

## Relative proxy design

- **candidate0** = `geometric_0` trial per ring (v5 static baseline analogue)
- **Relative score** = mean directed Δ vs candidate0 over v5 + seg features
- **J_reflect** = `v5_S_boundary × G_pre × G_layout` (simplified v3 guardrails)
- Selection excludes `gt_layout` trials

## Single-instance gate (passed)

- **Ring:** `1-4/r206`, 15 trials (smoke subset)
- **Evidence:** `logs/bo_v5_proxy_v1/single_instance_gate.json`
- v5 columns vary within pool (`balance_norm`, `geom_boundary_gap_cv`, `S_boundary`, seg features)

## Commands

```bash
# Gate (single ring)
./venv/bin/python bo/enrich_v5_proxy_features.py --skip-full --gate-case 1-4/r206 --max-trials-per-ring 15

# Full 480-trial enrichment
./venv/bin/python bo/enrich_v5_proxy_features.py --skip-gate

# Evaluate relative proxy on calib records
./venv/bin/python bo/eval_v5_relative_proxy.py
```

## Gate-ring smoke (1-4/r206, n=15, not panel conclusion)

| Variant | within-pool ρ | regret vs oracle |
|---------|--------------:|-----------------:|
| A3-slim | 0.10 | 0.080 |
| rel_feature_mean | **0.35** | 0.149 |
| J_reflect | 0.06 | 0.080 |

Full 6-ring panel eval pending `calibration_records_v5.csv` completion.
