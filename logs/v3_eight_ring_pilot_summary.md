# V3 Eight-Ring Pilot Summary

Source artifacts:

- `stages/v3/benchmarks/eight_ring_v3_comparison.csv`
- `stages/v3/logs/v3_arm_b_proxy_stabilisation_v1/arm_b_final_scoreboard.csv`
- `stages/v3/logs/v3_arm_b_proxy_stabilisation_v1/arm_b_final_summary.json`
- `stages/v3/logs/v3_arm_c_intrinsic_rule_selector_v1/aggregate_summary.csv`
- `stages/v3/logs/v3_cyclic_stab_reflect_v1/focused3/aggregate_summary.csv`

## Best Selected mIoU On 8 Pilot Rings

| Ring | Best selected mIoU | Source | Decision |
|---|---:|---|---|
| `4-1/r110` | 0.0970 | Arm B | Binary order `minus`; reflection abstained / failed to recover oracle `0.4608`. |
| `4-10/r398` | 0.8298 | Arm B | Binary order `minus`; reflection abstained. |
| `4-2/r142` | 0.6095 | Arm B | Binary order `minus`; reflection abstained. |
| `4-3/r170` | 0.8118 | Arm B | Binary order `plus`; reflection abstained. |
| `4-4/r215` | 0.1530 | Arm C KAB calibration | Selected reflection candidate `a_02`, branch `minus`. |
| `4-5/r249` | 0.4115 | Arm B | Binary order `minus`; reflection abstained. |
| `4-9/r367` | 0.4125 | Arm B | Binary order `minus`; reflection abstained. |
| `5-5/r254` | 0.1815 | Arm B | Binary order `plus`; unsafe reflection candidate rejected by safety floor. |

Mean best selected mIoU over the 8 rings: `0.4383`.

## Parameters Used

Arm B used the frozen v3 baseline parameters:

- Preprocessing: `tunnel_diameter=7.5`, `gravity_anchor.enabled=false`, `vertical_filter_window=6.8`, `radius_min=3.0`, `radius_max=4.5`, `target_distances=[0.08, 0.04, 0.02]`.
- Detection: `detector_mode=single_ring_local`, `k_expected_height_px=294.5243`, `eps=0.07`, enabled blocks `K,B1,A1,A2,A3,A4,B2`, and fixed `per_ring_offsets`.

The only best selected reflection improvement was `4-4/r215` candidate `a_02`:

- `k_y_delta_px=-60.31`
- `k_expected_height_scale=0.7089`
- `b_height_scale=1.2838`
- `a_height_scale=0.7084`
- `ab_height_shift_px=51.85`

## Stabilisation Decision

The intrinsic direction stabiliser itself mostly returned `tie_plus_default` with low confidence, so the final Arm B result relied on the proxy RF binary switch between `plus` and `minus`.

On the archived 40-ring Arm B run, the proxy switch lifted mean mIoU from `0.2554` for K-only `plus` to `0.3341`, with oracle binary order at `0.4312`.

## Reflection Decision

Reflection was kept Arm-B-safe: candidate `0` is the Arm B selected output, and a reflected candidate should only be deployed if it is intrinsically better; otherwise the selector abstains to Arm B.

In the eight-ring pilot evidence, most rings abstained to Arm B. The safety floor blocked below-Arm-B selections on cyclic/reflection attempts, including `5-5/r254`. The only deployed improvement in the best-selected summary was `4-4/r215`, where KAB calibration selected candidate `a_02` and lifted mIoU from Arm B `0.0795` to `0.1530`.
