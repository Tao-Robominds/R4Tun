# Arm C Reflection Pilot Report

## Aggregate
- rings: 7
- mean Arm B baseline GT mIoU: 0.5342
- mean best Arm C GT mIoU: 0.5393
- mean GT lift vs Arm B baseline: +0.0051
- proxy/GT best-iteration alignment rate: 0.857

## Ring-Level
- 4-2/r142: baseline=0.6095, best=0.6095, lift=+0.0000, best_iter=0, group=proxy_helped
- 4-3/r177: baseline=0.5649, best=0.5649, lift=+0.0000, best_iter=2, group=high_risk
- 4-4/r212: baseline=0.1147, best=0.1147, lift=+0.0000, best_iter=0, group=high_risk
- 4-4/r215: baseline=0.0740, best=0.1095, lift=+0.0355, best_iter=5, group=neutral
- 4-7/r308: baseline=0.8112, best=0.8112, lift=+0.0000, best_iter=0, group=high_risk
- 5-4/r227: baseline=0.7796, best=0.7796, lift=+0.0000, best_iter=0, group=proxy_helped
- 5-5/r251: baseline=0.7854, best=0.7854, lift=+0.0000, best_iter=0, group=proxy_helped

## Limitations
- Proxy score is a learned estimate and can mis-rank iterations on some rings.
- Pilot evidence is limited to a small subset; generalization is suggestive, not conclusive.
- Arm C currently tunes preprocessing only; detection/segmentation adaptation is out of scope.

## Confidence
- Medium confidence in per-ring qualitative behavior (explore/exploit schedule is auditable).
- Low-to-medium confidence for full 40-ring rollout until proxy/guardrail agreement improves.

## Rollout Decision
- ready_for_rollout: True
- decision: proceed pilot->broader rollout
