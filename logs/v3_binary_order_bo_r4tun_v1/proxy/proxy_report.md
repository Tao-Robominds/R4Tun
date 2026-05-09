# Branch mIoU Proxy Search

## Question
Can we learn an intrinsic proxy for branch quality, estimate `plus_mIoU` and `minus_mIoU` separately, and choose the order with the higher proxy score?

## Method
- Build two rows per ring: one `plus` branch and one `minus` branch.
- Train/evaluate by leave-section-out.
- Decision rule: choose `minus` iff `proxy_minus_miou > proxy_plus_miou`; otherwise keep `plus`.
- Readout labels `plus_miou` / `minus_miou` are used only for training and evaluation, not as decision-time features.

## Results
- S0 keep-plus mean: `0.2554`
- Oracle order mean: `0.4312`

| Strategy | Mean mIoU | Lift vs S0 | Oracle Recovery | Degrades | Minus Picks | Branch Proxy RMSE | Branch Proxy Corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `proxy_ridge` | `0.2533` | `-0.0021` | `-0.0121` | `13` | `25` | `0.2458` | `0.1412` |
| `proxy_rf` | `0.3341` | `+0.0787` | `0.4478` | `8` | `23` | `0.2116` | `0.3738` |
| `proxy_gbdt` | `0.3318` | `+0.0763` | `0.4342` | `7` | `23` | `0.2215` | `0.3720` |
| `proxy_bo` | `0.2786` | `+0.0232` | `0.1318` | `0` | `3` | `0.3948` | `0.2805` |

## Interpretation
- The branch-quality proxy formulation is stronger than the earlier binary classifier/selector framing.
- `proxy_rf` is the best lift so far:
  - earlier BO selector: `0.3152`
  - branch proxy RF: `0.3341`
  - extra gain over BO selector: `+0.0189`
- The proxy captures a real signal, but it is not yet a safe final selector because it still damages strong plus rings.

## Main Proxy RF Failure Rings
- `4-7/r308`: `0.8112 -> 0.1734` (`-0.6378`)
- `4-4/r212`: `0.6317 -> 0.1147` (`-0.5170`)
- `4-3/r177`: `0.5649 -> 0.1262` (`-0.4387`)
- `4-1/r110`: `0.4608 -> 0.0970` (`-0.3637`)
- `4-7/r305`: `0.3921 -> 0.0714` (`-0.3208`)

## Recommendation
This is the right direction for the research framing: learn an intrinsic branch mIoU proxy, then choose the branch with higher predicted quality.

Current best research candidate:
- Use `proxy_rf` / `proxy_gbdt` as the main evidence that proxy learning recovers substantial oracle headroom.
- Use `proxy_bo` as the safe abstaining variant: lower lift, but zero observed degradations.

Next improvement should focus on making the high-lift proxy safe:
- add a strong-plus guard trained to catch the `4-7/r308`, `4-4/r212`, `4-3/r177` failure family;
- or use a two-stage policy: branch proxy proposes flips, guardrail vetoes flips on high-confidence plus rings.
