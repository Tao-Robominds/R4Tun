# T3 Hint Loop Summary

**Target:** mean mIoU ≥ 0.60 across `3-1-1`, `3-1-2`, `3-1-3`.

## Per-level mIoU

| Level | 3-1-1 | 3-1-2 | 3-1-3 | Mean | Pass ≥0.60? |
|-------|-------|-------|-------|------|-------------|
| broken | 0.287 | 0.237 | 0.229 | 0.251 | ✗ |
| T0 | 0.111 | 0.199 | 0.162 | 0.157 | ✗ |
| T1 | 0.456 | 0.163 | 0.189 | 0.269 | ✗ |
| T2 | 0.456 | 0.163 | 0.189 | 0.269 | ✗ |
| T3 | 0.456 | 0.163 | 0.189 | 0.269 | ✗ |
| T4 | — | — | — | — | — |
| T5 | 0.456 | 0.442 | 0.243 | 0.380 | ✗ |

## Gate (`3-1-1`)

- T1 pass threshold: mIoU ≥ 0.45
- Scale threshold: mIoU ≥ 0.55

- **T0** `3-1-1` mIoU=0.111 — T1 gate ✓, scale gate ✗
- **T1** `3-1-1` mIoU=0.456 — T1 gate ✓, scale gate ✗
- **T2** `3-1-1` mIoU=0.456 — T1 gate ✓, scale gate ✗
- **T3** `3-1-1` mIoU=0.456 — T1 gate ✓, scale gate ✗
- **T5** `3-1-1` mIoU=0.456 — T1 gate ✓, scale gate ✗

## Conclusion

**Target not met.** Best full panel: **T5** mean mIoU **0.380** (gap 0.220 below 0.60).
Preprocessing migration succeeded; frozen exemplar params (T1–T3) lift `3-1-1` but panel mean stalls below 0.60. **T5 GT ring-flip** improves `3-1-2` but not enough for panel pass. Dominant residual errors: detection/K placement on continuous tunnels and partial mirror correction on `3-1-3`.

## Artifacts

- Results: `data/t3_hint_loop/{level}/{tunnel}/`
- Validation: `logs/t3_hint_loop/validate_preprocessing.json`
- Migration: `logs/t3_hint_loop/migrate_preprocessing.json`

