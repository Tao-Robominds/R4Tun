# v5 Depth QA (All 50 Rings)

Method:
- T1/T2/T3: existing hard-gated audit reused from `v5_t123_depth_contract_v1`.
- T4/T5: rerun with r4tun-style depth contract (`gravity_anchor`, `observed_gap_aligned`, interpolation sweep).

Thresholds:
- finite_ratio >= 0.60
- row_nonempty_ratio >= 0.90
- largest_empty_vertical_gap_frac <= 0.08

Results: pass=48, fail=2, total=50.
T4/T5 fails before=20, after=2.
Unresolved T4/T5 rings after correction:
- 4-6/r276
- 5-6/r285
