# Steps

Use step files in numeric order.  
Current paper workflow is:

- `00_methodology_chain.md`
- `01_ring_regime_discovery.md`
- `02_bo_calibration.md`
- `03_tuning_memory.md`
- `04_intrinsics_and_ontology.md`
- `05_proxy_and_calibration.md`
- `06_reflection_ablation.md`
- `07_generalisation_test.md`

The current plan validates fixed stage proxies as reflection triggers. The main evidence is Spearman correlation plus threshold-trigger validation against final mIoU. Ridge regression, Platt calibration, and leave-one-out ablation are optional appendix analyses only.

Design-time history is preserved separately in `methods/plans/preparation/`.
