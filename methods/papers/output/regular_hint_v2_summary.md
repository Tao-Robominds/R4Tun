# Regular-tunnel K-pattern consensus v2

Model: **opus4.6** | Completed: **11** / 13

## Aggregate (v2 − baseline)

| Mean ΔmIoU | -0.0455 |
| Std ΔmIoU | 0.0695 |
| Improved vs baseline | 3 |
| Degraded vs baseline | 7 |

## Per-tunnel

| Tunnel | Baseline | v1 | v2 | Δv1 | Δv2 | K-snaps |
|--------|----------|----|----|-----|-----|---------|
| 1-1 | 0.617 | 0.617 | 0.456 | +0.000 | -0.161 | 5 |
| 1-2 | 0.608 | 0.608 | 0.610 | +0.000 | +0.002 | 1 |
| 1-3 | 0.658 | 0.658 | 0.587 | +0.000 | -0.071 | 1 |
| 1-4 | 0.436 | 0.436 | 0.361 | +0.000 | -0.075 | 3 |
| 1-5 | 0.629 | 0.629 | 0.431 | +0.000 | -0.198 | 3 |
| 2-1 | 0.674 | 0.674 | 0.673 | +0.000 | -0.001 | 0 |
| 2-2 | 0.685 | 0.685 | 0.685 | +0.000 | +0.000 | 0 |
| 2-3 | 0.606 | 0.606 | 0.605 | +0.000 | -0.001 | 0 |
| 2-4 | 0.624 | 0.624 | 0.625 | +0.000 | +0.001 | 2 |
| 2-5 | 0.669 | 0.669 | 0.663 | +0.000 | -0.006 | 2 |
| 3-1-1 | 0.287 | 0.332 | 0.297 | +0.045 | +0.010 | 6 |
| 3-1-2 | 0.237 | 0.222 | — | -0.015 | — | 0 |
| 3-1-3 | 0.229 | — | — | — | — | 0 |

**Staggered mean Δv2:** -0.0510 (n=10)
**Continuous mean Δv2:** +0.0100 (n=1)

## Code-only consensus (run1 detecting params, no LLM re-inference)

| Tunnel | Baseline | Code-only | Δ |
|--------|----------|-----------|---|
| 1-4 | 0.436 | 0.348 | −0.088 |
| 3-1-1 | 0.287 | 0.289 | +0.002 |

## Interpretation

- **Healthy staggered (`2-*`)**: v2 consensus leaves mIoU unchanged (0 snaps on 2-1/2-2); offline test confirms no Y movement.
- **Weak `1-*`**: v2 e2e degraded mIoU (mean Δv2 ≈ −0.05) because deleting detecting params caused **LLM re-inference** in addition to consensus; combined effect hurt several tunnels.
- **Code-only** on 1-4 shows consensus alone also **hurts** (−0.088) despite fixing outlier Y rows offline — snapping can fix one ring while mis-aligning block offsets on others.
- **3-1-1**: v1 prompt-only (+0.045) beats v2 consensus (+0.010 code-only); best gain remains LLM threshold tuning, not post-hoc Y snap.
- **3-1-2, 3-1-3**: orchestrator failed (vendor/upstream data issue).
- v1 results preserved under `regular_hint/` and `regular_hint_summary_v1.md`.
