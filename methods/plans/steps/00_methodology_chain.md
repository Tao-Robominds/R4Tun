# Methodology Chain: Stage BO -> Fixed Proxies -> Triggers -> Generalisation

This is the single overview of the paper workflow in `methods/plans/steps/01` to `07`.
Design-time reverse-engineering history lives in `methods/plans/preparation/`.

## Ordered chain

1. **Stage panel discovery** (step 01) - select representative and held-out rings separately for preprocessing and detection/boundary.
2. **Stage-wise BO calibration** (step 02) - run preprocessing BO and detection/boundary BO separately with full trial logs and GT outcomes.
3. **Trial dataset + tuning memory** (step 03) - collect selected/trial outputs, final mIoU after reprojection, and concise tuning memory.
4. **Fixed intrinsic proxies** (step 04) - compute GT-free combined scores:
   - `S_depth = S_coverage * S_empty`
   - `S_boundary = S_continuity * S_K * S_spacing * S_layout`
5. **Spearman + thresholds** (step 05) - validate each combined proxy against final mIoU and choose `T_depth` / `T_boundary`.
6. **Trigger validation** (step 06) - test whether low proxy values catch bad final outputs under fixed reflection rules.
7. **Generalisation test** (step 07) - held-out rings/tunnels with strict split isolation.

## Definitions

| Symbol | Meaning |
|--------|---------|
| `tau` | mIoU success threshold (e.g. 0.60). |
| `G_bad` | Bad final result `G_bad = 1[mIoU < tau]` (GT available only for validation). |
| `S_depth` | Preprocessing proxy in `[0, 1]`; low values trigger preprocessing reflection. |
| `S_boundary` | Detection/boundary proxy in `[0, 1]`; low values trigger boundary reflection. |
| `T_depth` | Learned threshold for `S_depth`. |
| `T_boundary` | Learned threshold for `S_boundary`. |

Deployment rule:

- trigger preprocessing reflection when `S_depth < T_depth`;
- trigger detection/boundary reflection when `S_boundary < T_boundary`;
- accept when neither trigger fires.

## Reflection action map (fixed)

- low `S_depth` -> rerun preprocessing reflection.
- low `S_boundary` -> rerun boundary/detection reflection.
- optional low-component tags (`S_K`, `S_spacing`, etc.) explain the trigger but do not define extra learned policies.

## Main validation

- Spearman correlation: `S_depth` vs final mIoU, `S_boundary` vs final mIoU.
- Threshold-trigger validation: bad-case recall, trigger precision, false-negative rate, accepted-case mIoU.

Leave-one-out ablation, Ridge regression, and logistic/Platt calibration are optional appendix analyses only.
