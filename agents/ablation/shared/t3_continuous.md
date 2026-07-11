# T3_CONTINUOUS regime (3-* continuous-joint tunnels)

**Trigger:** tunnel prefix `3-*` (T3 family: 5.5 m, 6 segments, **continuous joints**, often multi-station registration).

**Not SIMILAR_TO_SAMPLE:** same nominal diameter as T1/T2, but **radial distribution is wider** — sample mask `[2.7, 2.8]` typically excludes most wall points.

## Known failure modes (derive fixes from state — do not copy external parameter sets)

| Symptom | Likely cause | Fix stage |
|---------|--------------|-----------|
| `wall_pct < 15%` | Sample mask too narrow for p50(r) | Denoising — widen from percentiles |
| Retention OK but depth map >20% white | Sparse projection / weak gap-fill | Enhancing — coverage gate |
| Central seam brighter than edges | Multi-station θ discontinuity | Enhancing `window_size` + outlier fill |
| Edge/corner white bands | Peripheral point scarcity | `depth_threshold_low`, upsampling |

## Adaptation policy

### Denoising (highest priority)

1. Compute `wall_pct` = % of unfolded points with `mask_r_low <= r <= mask_r_high`.
2. **Never** keep sample `[2.7, 2.8]` if `wall_pct < 15%` OR `p50(r) > mask_r_high` OR `p10(r) < mask_r_low`.
3. **Rules minimum** `[2.6, 2.9]` for 5.5 m — **only when `p50 ≤ 2.9`**. If `p50 > 2.9`, use percentile mask `[p10 − 0.02, p99 + 0.02]`.
4. **After `wall_pct ≥ 50%`:** optional noise trim — raise `mask_r_low` only; keep `mask_r_high ≥ p99 + 0.02`.
5. Sync `default_cutoff_z = mask_r_high + 0.02`.

### Enhancing — depth-map coverage (when denoise retention ≥ 50%)

White space = NaN pixels after projection at `resolution=0.005`. This is **not** fixed by denoising once retention ≥ 50%.

**COVERAGE_FAILURE** when estimated `point_density = valid_denoised_points / grid_cells < 0.08` OR state implies large angular gaps.

Tune in order (document each):
1. `window_size` — neighborhood fill for remaining NaNs (9 → 11 → 13).
2. Upsampling stages — stage1 ≈ `0.85 × median_NN`; halve for stage2/stage3.
3. `depth_threshold_low` / `depth_threshold_high` — lower to add outlier gap-fill points.
4. `inter_radius`, `num_interpolations` — more joint interpolation when gap-fill count is low.
5. `curvature_threshold` — relax if upsampling adds too few midpoints.
6. `n_segment_end` — use **`ring_count − 1`** from unfolded state when `ring_count > segment_per_ring`.

### Detecting — K uniformity (`3-*`)

Continuous joints: all K-blocks share **one Y** (horizontal joint line). **One K knows all** — one reliable detection defines **Y\*** and K block shape for every ring; only **X** varies (ring column). Staggered assume fallback is wrong for T3.

**Checklist (answer with numbers):**
1. What is `Y_std` across ring prompts? **K_UNIFORM_FAILURE** if > 10 px post-snap.
2. What is `max |Y − Y*|`? Target **0 px** after uniform snap.
3. What is `assume` + `default` rate? Target < 10% pre-snap.
4. How many anchor detections? Pipeline snaps **all** rings to median Y* from ≥1 anchor (not only `assume` rows).
5. If pre-snap failure: lower `hough_threshold_horizontal` (35–45), widen `maxLineGap_horizontal` (20–30).

- `ring_spacing_constant=1.2` for T3.
- Hough may miss verticals — expect synthetic vertical fallback (X columns OK).
- See **K_UNIFORM_GATE** (alias **K_ALIGNMENT_GATE**) in detecting cot.md.

### SAM — K template uniformity (`3-*`)

- `segment_per_ring=6`, sample `segment_order` — no A4.
- K prompt centre **Y identical** every ring; tune `K_height`, `angle`, `crop_margin`, `y_bounds` **once** — not per-ring.
- See **K_TEMPLATE_UNIFORM** in segmenting cot.md.

## Checklist (answer with numbers before JSON)

1. Denoising — `wall_pct`? p5/p50/p99? Proposed mask?
2. Enhancing — `median_NN`? Estimated `point_density`? Planned `window_size` and upsampling?
3. Detecting — `Y_std`? `max |Y − Y*|`? `assume` rate? Anchor count? Uniform snap expected?
4. SAM — K Y uniform? Planned `K_height` / `angle` / `y_bounds` (tune once)?
5. Enhancing — if peripheral white > central white, bias toward `depth_threshold_low` and larger `window_size`.
