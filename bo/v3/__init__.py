"""bo/v3 — Ax/BoTorch BO calibration for the v3 plan.

This package contains the v3 calibration driver that:

- runs the gravity-anchored agents pipeline on each calibration ring,
- treats fixed-class canonical ring mIoU as the primary objective,
- logs permutation-invariant mIoU + curated intrinsic vector per trial,
- enforces ``logs/v3/`` output paths (immutability check) so it can never
  write under any of the protected ``data/...`` prefixes.

Key public entry points live in :mod:`bo.v3.run_v3_calibration` (driver)
and :mod:`bo.v3.aggregate_calibration` (Step 3 cross-ring analysis).
"""
