# 01 Ring-Regime Discovery

## Goal
Create practical ring regimes for representative BO sampling and later held-out generalisation checks.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/01_ring_regime_discovery/`

## Inputs
- Seg2Tunnel ring data and ring ids
- Preprocessing outputs (`enhanced.csv`, `depth_map.png`)

## Actions
1. Compute per-ring descriptors: density, occlusion/missing area, boundary clarity, radius stats, pattern class, ring complexity.
2. Cluster or rule-group rings into regime labels.
3. Build representative-ring panel with fair coverage across regimes.

## Outputs
- `ring_descriptors.csv`
- `ring_regimes.csv`
- `regime_sampling_panel.json`

## Verify Prompt
`Are all rings described, regime-labeled, and represented in a fair BO sampling panel?`
