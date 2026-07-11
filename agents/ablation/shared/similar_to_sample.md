# SIMILAR_TO_SAMPLE regime (T1/T2 regular tunnels)

**Trigger:** tunnel prefix `1-*` or `2-*` AND raw characteristics within ~25% of sample on diameter (~5.5 m), 6 segments, and comparable point density / theta span.

For `3-*` continuous-joint tunnels, use **T3_CONTINUOUS** block instead (not this regime).

**Performance target:** mIoU **≥ 0.70** on 1-1 and 2-1 before adapting harder tunnels. Reference tunnel `sample` reaches **≈ 0.88** with frozen `sam4tun/agents/parameters/sample/`.

## Adaptation policy

- **Default: retain sample parameters** unless stage state shows a **named failure mode**.
- Do **not** apply T4/T5 scaling, geometric SAM fallback, or aggressive percentile mask widening for T1/T2.
- Answer this checklist per stage before changing any value:
  1. Unfolding — does sample `diameter=5.5` / RANSAC defaults fail? (ellipse fit error in state?)
  2. Denoising — do `r_percentiles` show wall points outside sample mask `[2.7, 2.8]`?
  3. Enhancing — is depth map clipped or upsampling insufficient in state?
  4. Detecting — are rings sparse/missed in detected characteristics?
  5. SAM — is crop misalignment systematic in state? (Keep 6-seg; no A4.)

## Stage defaults for T1/T2 (when checklist = no)

| Stage | Keep from sample |
|-------|------------------|
| Unfolding | `diameter=5.5`, all RANSAC / polynomial defaults |
| Denoising | `mask_r_low=2.7`, `mask_r_high=2.8`, `y_step=0.5`, `z_step=0.001`, `grad_threshold=0.2` |
| Denoising (only if wall outside mask) | rules formula: `r=d/2`, `mask_r_low=r-0.15`, `mask_r_high=r+0.15`, `default_cutoff_z=mask_r_high+0.05` |
| Enhancing | upsampling `0.08/0.04/0.02`, `n_segment_end=5` |
| Detecting | `ring_spacing_constant=1.2`; keep Hough defaults unless sparse detection |
| SAM | `segment_per_ring=6`, sample `segment_order` and `prompt_points` |

## Complex tunnels (4-*, 5-*)

Not SIMILAR_TO_SAMPLE. T4/T5 use 7.5 m diameter, 7 segments, `ring_spacing=1.8`, SAM geometric fallback (prompt_points secondary).
