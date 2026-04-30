# Methodology Chain: BO -> Features -> Proxy -> Reflection -> Generalisation

This is the single overview of the paper workflow in `methods/plans/steps/01` to `07`.
Design-time reverse-engineering history lives in `methods/plans/preparation/`.

## Ordered chain

1. **Ring-regime discovery** (step 01) - build descriptors and regime labels for fair BO sampling.
2. **BO calibration** (step 02) - ring-wise BO with full trial logs and GT outcomes.
3. **Tuning memory** (step 03) - distill informative BO cases into reusable memory entries.
4. **Feature construction** (step 04) - compute two GT-free feature blocks:
   - `x_P`: pipeline intrinsics (quality signals from stage artifacts)
   - `x_O`: ontology/structure signals (plausibility constraints)
5. **Proxy + calibration** (step 05) - ridge regression `y_hat = f(x)` with `x = [x_P; x_O]`, then Platt calibration for `p_good`.
6. **Reflection ablation** (step 06) - fixed-rule reflection only (no LLM/RL routing), evaluated over a 3x4 feature/trigger grid.
7. **Generalisation test** (step 07) - held-out rings/tunnels with strict split isolation.

## Definitions

| Symbol | Meaning |
|--------|---------|
| `tau` | mIoU success threshold (e.g. 0.60). |
| `G` | Success event `G = 1[mIoU >= tau]` (GT available at calibration/eval time). |
| `y_hat` | Proxy mIoU prediction from feature vector `x`. |
| `s` | Proxy margin `s = y_hat - tau`. |
| `p_good` | Calibrated success probability `p_good = sigma(a*s + c)`. |
| `p_min` | Minimum accepted `p_good`. |
| `tau_reflect` | Reflection trigger threshold on `p_good` when guardrails pass. |

Acceptance rule: accept only when `y_hat >= tau` **and** `p_good >= p_min`.

## Reflection action map (fixed)

- poor ring boundary quality -> rerun boundary detection
- poor oblique line quality -> adjust K-line detection
- invalid segment count -> adjust geometry segmentation
- high spacing irregularity -> rerun ring boundary detection

## Ablation grid

- Feature blocks: `P only`, `O only`, `P union O`
- Triggers: `none`, `guardrails only`, `p_good only`, `guardrails + p_good`

Total cells: 12.
