# Steps

Use step files in numeric order.  
Current paper workflow is:

- `00_methodology_chain.md`
- `01_ring_regime_discovery.md`
- `02_bo_calibration.md`
- `03_tuning_memory.md`
- `04_intrinsics_and_ontology.md`
- `05_proxy_and_calibration.md`
- `06_diversity_expansion.md`
- `07_generalisation_test.md`

The current plan validates intrinsic proxies as empirical rewards for ring-level self-improvement. The main evidence is not LLM reflection. It is whether a proxy can predict or select mIoU-improving candidates, quantify its own confidence, expose out-of-distribution cases, and improve its confidence as BO experience becomes more diverse.

Design-time history is preserved separately in `methods/plans/preparation/`. Deprecated LLM/reflection material is treated as historical context only unless explicitly reintroduced.
