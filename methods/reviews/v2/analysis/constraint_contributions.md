# Constraint contributions — Opus-4.6 m+s+k vs SAM4Tun baseline (n=30)

Quantification of the four structural constraints of the fixed SAM4Tun positional
labelling rule. All numbers are computed by `methods/scripts/analyze_constraints.py`
from `final.csv` files (GT `segment` vs `pred`), split Regular (n=13) / Complex (n=17).
Source: `constraint_contributions.csv` (per tunnel), `per_ring.csv`, `aggregate.json`.

"Accuracy" below is GT-block class accuracy (fraction of ground-truth block points
whose predicted class is correct), not mIoU. It is the quantity the labelling rule
directly controls.

## Headline (block accuracy, baseline -> Opus m+s+k)
- Regular:  0.354 -> 0.672
- Complex:  0.000 -> 0.233

Adaptation recovers a large share of the regular ceiling and lifts complex tunnels
off zero, but the four constraints below cap the absolute level.

## Error composition (FP / FN / swap, fraction of GT points)
| Category | Method | Correct | FN | FP | Swap |
|---|---|---|---|---|---|
| Regular | SAM4Tun | 51% | 21% | 3% | 26% |
| Regular | m+s+k | 71% | 2% | 5% | 21% |
| Complex | SAM4Tun | 34% | 66% | 0% | 0% |
| Complex | m+s+k | 43% | 17% | 6% | 34% |

- FN = block -> background (under-segmentation); FP = background -> block; swap = wrong block class.
- Regular: adaptation removes FN (21->2%); residual is class swap (21%) = labelling rotation (C2/C3/C4).
- Complex: baseline is pure under-segmentation (66% FN, ~0% swap); adaptation recovers blocks
  (FN 66->17%) but the labelling rule then mislabels them (swap 0->34%).
- FP stays small (<=6%) everywhere: the residual is a labelling problem, not a noise/detection problem.

## C1 — Non-uniform point density
| | Regular | Complex |
|---|---|---|
| per-ring count max/min (within tunnel) | 19.2x | 38.3x |
| per-ring density CV | 0.77 | 1.07 |
| corr(density, per-ring accuracy) | +0.34 | +0.10 |

A single tunnel-level denoise/mask configuration must serve rings whose point count
varies up to ~19x (regular) and ~38x (complex). Denser rings segment better
(positive correlation), so sparse end rings are systematically penalised. The spread
is far wider than any single density-dependent parameter setting can match.

## C2 — Moving K-anchor (dominant on complex)
| | Regular | Complex |
|---|---|---|
| GT K-theta ring-to-ring span | 2.77 | 16.78 |
| mean \|K shift\| between rings | 1.86 | 8.78 |
| K-mislocated rings | 12% | 64% |
| accuracy on K-aligned rings | 0.638 | 0.281 |
| accuracy on K-mislocated rings | 0.238 | 0.143 |

Once K is mislocated the whole ring's projected labels rotate together. K-mislocation
jumps from 12% (regular) to 64% (complex), and mislocated rings collapse to
~0.14-0.24 accuracy versus ~0.28-0.64 when K is correct. This is the single largest
mechanism behind the low complex-tunnel ceiling.

## C3 — Fixed segment-offset template (residual, K-aligned rings only)
Weighted recall by sector-distance from K (0 = K, 3 = farthest block):

| distance | 0 (K) | 1 | 2 | 3 (far) |
|---|---|---|---|---|
| Regular | 0.71 | 0.66 | 0.69 | 0.69 |
| Complex | 0.76 | 0.23 | 0.27 | 0.32 |

On regular tunnels the 6-segment template fits, so K-aligned recall is roughly uniform
across distance (template drift is a small residual; representative tunnel 1-1 shows
the farthest block dropping to 0.68 vs 0.85-0.86 nearer K). On complex 7-segment
tunnels the fixed 6-step template mismatches from the first offset: even when K is
correctly placed, the immediately adjacent block (distance 1) collapses to 0.23.
The template is the residual limiter on regular tunnels and a structural mismatch on
complex ones.

## C4 — Hard-coded walk direction / handedness
Per-ring ordering outcome (counts across all rings in the category):

| | rotation-only | direction FLIP | other / mixed |
|---|---|---|---|
| Regular (130 rings) | 107 | 17 | 6 |
| Complex (166 rings) | 103 | 28 | 35 |
| accuracy on FLIP rings | ~0.08 | | |

Most rings are pure rotations (consistent with C2), but a hard-coded walk direction
produces full mirror-image flips on 17 regular and 28 complex rings, where accuracy
collapses to ~0.08-0.10. Flips concentrate in continuous-regular tunnels (3-1-2: 6/10
rings, 3-1-3: 8/10 rings) and reversed-handedness complex tunnels, which parameter
tuning alone cannot correct. The high "other" count on complex rings (35) reflects
under-segmentation that destroys the ordering entirely.

## Reading
- Regular ceiling is set mainly by C2 on the minority of K-mislocated rings, C4 flips
  on continuous tunnels, and C1 on sparse end rings; C3 is a small residual.
- Complex ceiling is set primarily by C2 (64% K-mislocated) and the C3 template
  mismatch (7-segment geometry vs 6-step template), compounded by C1 and C4.
- None of these are parameter values: they are properties of the detect-K-then-fixed-
  template labelling rule, so bounded parameter adaptation can raise accuracy toward
  the ceiling but cannot remove the ceiling.
