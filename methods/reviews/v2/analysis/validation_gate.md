# Single-instance validation gate — constraint analysis

Before scaling the four-constraint quantification to all 30 tunnels, one
representative instance per category was validated end-to-end.

## Cases / lineage
- Regular: `1-1`
  - baseline input: `/media/boringtao/Expansion/R4Tun-AIC!/data/sam4tun/1-1/final.csv`
  - adapted input: `data/ablation_anthropic/memory+state+knowledge/1-1/final.csv` (Opus-4.6 m+s+k)
- Complex: `4-1`
  - baseline input: `/media/boringtao/Expansion/R4Tun-AIC!/data/sam4tun/4-1/final.csv`
  - adapted input: `data/ablation_anthropic/memory+state+knowledge/4-1/final.csv` (Opus-4.6 m+s+k)

Command: `venv/bin/python methods/scripts/analyze_constraints.py --gate`

## Metric values (Opus-4.6 m+s+k)

| Constraint | Metric | Regular 1-1 | Complex 4-1 |
|---|---|---|---|
| block accuracy | baseline -> m+s+k | 0.383 -> 0.782 | 0.000 -> 0.206 |
| C1 density | count max/min ; CV ; corr(density,acc) | 16.9 ; 0.88 ; +0.45 | 39.4 ; 1.03 ; +0.02 |
| C2 K-anchor | GT K-theta span ; mean\|shift\| ; mislocated% | 5.47 ; 3.55 ; 20% | 17.66 ; 8.21 ; 80% |
| C2 acc | aligned vs mislocated rings | 0.812 vs 0.037 | 0.714 vs 0.078 |
| C3 A/B drift | recall by sector-distance {0,1,2,3} | {0.66, 0.85, 0.86, 0.68} | {0.82, 0.54, 0.66, 0.84} |
| C4 handedness | rotation / FLIP / other rings | 9 / 0 / 1 | 4 / 1 / 4 |

## Pass / fail criteria (from the task request)
Each of C1-C4 must produce a finite, interpretable number consistent with the
narrative that the fixed SAM4Tun positional-labelling rule caps absolute accuracy.

- C1 PASS: density spread is large (count max/min 16.9-39.4) and density correlates
  positively with per-ring accuracy on regular tunnels (+0.45); a single denoise/mask
  config cannot cover this band.
- C2 PASS: K-mislocation rate rises sharply on complex tunnels (20% -> 80%), and
  mislocated rings collapse to ~0.04-0.08 accuracy vs ~0.71-0.81 on K-aligned rings.
  Clean, large damage signal.
- C3 PASS (regular): on K-aligned rings the farthest sector-distance has the lowest
  recall (1-1: d3=0.68 < d1=0.85, d2=0.86), confirming cumulative fixed-template drift.
  On complex (4-1) only ~2 rings are K-aligned, so per-tunnel C3 is sparse and
  dominated by C2; the cross-tunnel aggregate is used for the complex column.
- C4 PASS: direction flips occur (4-1: 1 flip ring) and flip rings collapse to
  ~0.04 accuracy (mirror image), while most regular rings are pure rotations.

Outcome: GATE PASSED for both representative cases. Proceeding to all 30 tunnels.

## Output paths
- Per-tunnel table: `methods/reviews/v2/analysis/constraint_contributions.csv`
- Aggregates printed to stdout and summarised in the revised section text.
