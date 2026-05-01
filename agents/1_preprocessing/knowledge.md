# LLM Reflection Guideline For Preprocessing

Use this guideline to decide whether an LLM should trigger reflection for a preprocessing result, what to inspect, and which preprocessing stage/parameters may need retuning.

## Reflection Goal

Reflection should answer:

1. Is the current preprocessing result acceptable?
2. If not, which stage most likely caused the problem?
3. Which small set of parameters should be retuned?
4. What evidence would prove the retune helped?

Do not trigger reflection just because a metric changed. Trigger reflection only when the change suggests a real preprocessing failure or a meaningful opportunity to improve depth-map support without damaging map structure.

## When To Trigger Reflection

Trigger reflection when one or more of these conditions holds:

- Target depth map is visually broken: large blank bands, severe sparsity, obvious overfill, smeared structures, unstable geometry, or discontinuous coverage.
- Valid support collapses compared with the current baseline.
- Foreground recall or target support is near zero.
- Empty-band ratio becomes large enough to affect downstream use.
- BO or a parameter sweep improves one diagnostic metric but worsens map structure.
- A ring behaves very differently from similar rings in density, radius spread, angular coverage, or map shape.
- A new tunnel/ring regime appears outside the empirical range of existing parameters.

Do not trigger reflection when:

- The result is visually acceptable and guardrail metrics are stable.
- Differences are tiny and do not affect depth-map usability.
- The failure is caused by missing inputs, missing mapping files, corrupted output, or known data absence. In that case, inspect data plumbing first.
- The desired improvement belongs to detection or segmentation rather than preprocessing.

## Evidence To Inspect

Use only preprocessing-level evidence:

- Depth map quality: valid coverage, largest empty row/column bands, visible holes, overfill, smearing, shape changes.
- Denoised support: how many points survive filtering, whether valid surface support is concentrated or missing.
- Enhanced support: whether interpolation fills gaps or hallucinates false support.
- Coordinate stability: whether unwrapped coordinates are smooth and consistent.
- Point cloud characteristics: density, angular coverage, largest angular gaps, radius spread, local sparsity.
- Guarded reward components: target foreground recall, coverage guard, empty-band guard.
- Diagnostic-only metrics: foreground IoU, precision, valid ratio, empty-band ratio.

Never use K position, segment order, walking order, block balance, detection-line quality, segmentation mIoU, or any downstream task metric to justify preprocessing reflection.

## Reflection Criteria

Prefer a guarded score for decision-making:

```text
guarded_score = target_foreground_recall * coverage_factor * empty_factor
```

Interpretation:

- `target_foreground_recall`: valid pixels should cover foreground support.
- `coverage_factor`: prevents sparse/empty maps from being rewarded.
- `empty_factor`: penalizes large structural gaps.
- `foreground_mask_iou`: diagnostic only; do not optimize it alone because it may reward sparse maps by reducing false positives.

A candidate is better only if:

- guarded score improves,
- valid coverage does not collapse,
- empty-band structure does not worsen,
- visual map quality remains acceptable,
- improvement is attributable to preprocessing, not a downstream artifact.

If metrics disagree with visual map quality, prefer the conservative interpretation and keep the stable baseline.

## Stage Diagnosis

### 1. Unfolding Reflection

Unfolding converts raw 3D points into a stable unwrapped coordinate system.

Trigger unfolding reflection when:

- Geometry is visibly distorted.
- Theta/radius coordinates are discontinuous.
- The same ring has large shape changes under small parameter changes.
- Empty bands align with coordinate discontinuities rather than filtering.
- Cropping/coverage appears wrong even before denoising and enhancement.

Likely parameters:

- `tunnel_diameter`: expected tunnel size/radius scale.
- `vertical_filter_window`: vertical support used for fitting.
- `ransac_threshold`: fitting tolerance.
- `ransac_inlier_ratio`: expected clean inlier fraction.
- `ransac_inlier_threshold_multiplier`: adaptive inlier strictness.
- `ransac_initial_iterations`: fitting robustness budget.

General tuning direction:

- If fitting is too strict and loses real wall points: increase tolerance or lower expected inlier ratio.
- If fitting follows clutter/noise: reduce tolerance, tighten inlier rules, or narrow the vertical support.
- If geometry is stable, do not retune unfolding first.

Empirical ranges:

- `vertical_filter_window`: about `4.0-7.0`
- `ransac_threshold`: about `0.5-1.5`
- `ransac_inlier_ratio`: about `0.5-0.85`
- `ransac_inlier_threshold_multiplier`: about `0.6-1.2`

### 2. Denoising Reflection

Denoising decides which unwrapped points remain valid surface support.

Trigger denoising reflection when:

- Valid coverage is too low before enhancement.
- Foreground recall is near zero.
- The map is empty or banded because support was removed.
- Denoised points are concentrated in a narrow radius/theta region.
- Many background/clutter points survive as valid surface support.

Likely parameters:

- `radius_min`: lower radius cutoff.
- `radius_max`: upper radius cutoff.
- `gradient_threshold`: sensitivity to abrupt radial/depth changes.
- `smoothing_window_size`: smoothing span.
- `smoothing_offset`: strictness/permissiveness bias.
- `default_cutoff_z`: fallback depth/radius cutoff.

General tuning direction:

- If too much true surface is removed: decrease `radius_min`, increase `radius_max`, increase `gradient_threshold`, or move `smoothing_offset` toward zero/positive.
- If too much clutter/background survives: increase `radius_min`, decrease `radius_max`, decrease `gradient_threshold`, or make `smoothing_offset` more negative.
- If empty bands appear after denoising: first loosen radius/gradient filtering before changing enhancement.
- If denoised support is already good, do not retune denoising first.

Empirical ranges:

- `radius_min`: `1.8-3.8`
- `radius_max`: `2.0-4.2`, always greater than `radius_min`
- `gradient_threshold`: `0.03-0.40`
- `smoothing_offset`: `-0.02-0.02`

### 3. Enhancing Reflection

Enhancing densifies valid support and interpolates missing surface regions.

Trigger enhancing reflection when:

- Denoised support is reasonable but depth map still has holes.
- Valid coverage is low because interpolation is weak.
- Empty bands are small/local rather than caused by missing whole coordinate regions.
- Map is overfilled or smeared after interpolation.
- Precision drops while recall rises.

Likely parameters:

- `target_distances`: three-stage surface upsampling distances.
- `curvature_neighbors` / `num_neighbors`: local neighborhood size.
- `interpolation_window`: projection/interpolation window.
- `outlier_interpolation_radius` / `inter_radius`: outlier interpolation radius.
- `outlier_num_interpolations` / `num_interpolations`: interpolation amount.
- `outlier_depth_map_window`: diagnostic outlier projection window.
- `outlier_neighbors`: outlier interpolation neighborhood.

General tuning direction:

- If holes remain with good support: lower target distances, increase interpolation window, increase interpolation radius slightly, or add interpolation passes.
- If overfilled/smeared: increase target distances, reduce interpolation window/radius, reduce interpolation passes, or increase neighbors for smoother local consensus.
- If high-density artifacts appear: reduce outlier radius/passes and inspect outlier map separately.
- If support is missing before enhancement, retune denoising first instead.

Empirical ranges:

- `curvature_neighbors` / `num_neighbors`: `8-40`
- `interpolation_window`: `1-15`
- `target_distance_1`: `0.03-0.12`
- `target_distance_2`: `0.015-0.06`
- `target_distance_3`: `0.008-0.04`
- `outlier_interpolation_radius`: `0.01-0.08`
- `outlier_num_interpolations`: `1-5`
- `outlier_depth_map_window`: `1-9`
- `outlier_neighbors`: `8-40`

## Final Reflection Decision

Choose exactly one final action:

- `keep_baseline`: current result is acceptable, or candidate gains are not reliable.
- `inspect_data_mapping`: metrics are impossible or near zero because files/mappings/support are missing or inconsistent.
- `retune_unfolding`: geometry/coordinate instability is the primary failure.
- `retune_denoising`: valid surface support is removed or clutter survives filtering.
- `retune_enhancing`: support exists but interpolation/upsampling is insufficient or excessive.
- `run_small_bo`: parameter uncertainty is localized to BO-supported denoising/enhancing ranges.
- `run_targeted_sweep`: a small hand sweep is safer than BO, especially for unfolding or one uncertain parameter.

Prefer the smallest action that can test the diagnosis.

## Required Reflection Output

The LLM reflection must use this format:

```markdown
## Trigger
- reflection_needed: yes/no
- trigger_reason:

## Evidence
- visual_depth_map:
- support_metrics:
- guardrail_metrics:
- data_integrity:

## Stage Diagnosis
- likely_stage: unfolding | denoising | enhancing | none | data_mapping
- why:
- ruled_out_stages:

## Parameter Decision
| parameter | stage | current_value | proposed_direction_or_range | reason |
|---|---|---:|---|---|

## Evaluation Plan
- primary_score:
- required_guardrails:
- diagnostic_metrics:
- stop_condition:

## Final Decision
- action:
- expected_effect:
- output_sandbox:
```

## Conservative Rules

- Retune only one stage at a time unless evidence clearly crosses stages.
- Prefer denoising before enhancing when support is missing.
- Prefer enhancing before denoising when support is present but sparse/holed.
- Prefer unfolding only for coordinate/geometry failures.
- Never accept a candidate that improves IoU but visibly worsens coverage or empty-band structure.
- Never tune preprocessing using downstream detection or segmentation outcomes.
# Preprocessing Reflection Prompt

Use this prompt when an LLM is asked to reflect on preprocessing results and decide whether to retune unfolding, denoising, enhancing, or no stage. The goal is not to blindly run BO. The goal is to diagnose the depth-map failure mode, identify the stage most likely responsible, and recommend a small parameter set/range for the next tuning pass.

## Context

The current official ring-level preprocessing path is B+C+D:

- B: target-ring depth maps are cropped to observed theta coverage, not forced to a full-circumference canvas.
- C: use tunnel-global `h/theta/r` coordinates from the r4tun whole-tunnel unwrapped reference when available.
- D: denoise/enhance with neighbor-ring context, but score and report only the target ring.

Main files:

- Official context runner: `agents/1_preprocessing/context_preprocessing.py`
- CLI wrapper: `agents/1_preprocessing/scripts/run_context_ring_trial.py`
- BO runner: `bo/run_preprocessing_iou_bo.py`
- Guarded metrics: `bo/preprocessing_iou_metrics.py`
- Fixed baseline reference: `logs/context_preprocessing_v1/<tunnel>/r<ring>/`
- BO trial archive: `data/bo/preprocessing/<tunnel>/r<ring>/...`

Do not write into protected baselines such as `data/ablation/**`, `data/baseline/**`, `data/preprocessing_qa/**`, `data/represents/**`, `r4tun/data/**`, `r4tun/references/**`, or `logs/context_preprocessing_v1/**`. For new experiments, write only into an approved sandbox and promote manually.

## Reflection Inputs

Before deciding what to retune, inspect:

1. `trial_meta.json`: context rings, coordinate source, output summary.
2. `depth_map.npy/png`: target-ring cropped depth map.
3. `context_depth_map.npy/png`: context/canonical diagnostic depth map.
4. `denoised.csv`: target-ring denoised subset for metric compatibility.
5. `context_denoised.csv` and `context_enhanced.csv`: whether context support was retained or over-filtered.
6. `pixel_to_point.pkl`: mapping needed for GT-derived diagnostics.
7. Metrics: `guarded_score`, `target_foreground_recall`, `foreground_mask_iou` diagnostic, `valid_ratio`, `empty_row_band_ratio`, `coverage_ok`, `empty_band_ok`.
8. Visual comparison against fixed baseline under `logs/context_preprocessing_v1/<tunnel>/r<ring>/`.

## Reflection Criteria

Primary BO/reflection score:

```text
guarded_score = target_foreground_recall * coverage_factor * empty_factor
```

Interpretation:

- `target_foreground_recall`: primary signal; valid depth pixels should cover GT foreground support.
- `coverage_factor`: prevents solutions that delete most valid pixels relative to the fixed baseline.
- `empty_factor`: penalizes large empty row bands that make the map structurally unusable.
- `foreground_mask_iou`: diagnostic only; do not optimize it alone because it can reward sparse/empty maps by reducing false positives.

Decision rule:

- Select BO best only when `guarded_score` improves and coverage/empty-band guardrails remain acceptable.
- If BO does not improve or produces visually worse/sparser maps, keep the fixed B+C+D baseline.
- If both baseline and BO score near zero, treat it as a support/projection or extreme sparsity problem; do not assume parameter tuning alone can recover the ring.

Observed representative BO results:

- `bo_best` selected: `5-7/r315`, `4-4/r215`, `5-1/r116`, `5-1/r113`.
- `fixed_baseline` selected: `4-6/r283`, `1-1/r25`, `5-1/r114`, `5-6/r285`.
- Strongest absolute gain observed on `5-1/r116`; large gain also on `4-4/r215`.
- Zero-gain cases should be treated conservatively: keep baseline unless a new diagnostic indicates a different stage failure.

## Stage 1: Unfolding Parameters

Unfolding creates `h/theta/r` coordinates. In the B+C+D path, prefer r4tun tunnel-global coordinates, so retune unfolding only when global coordinates are unavailable, wrong, or visually inconsistent.

Parameters:

- `tunnel_diameter`: nominal tunnel diameter used for radius interpretation and canonical theta height. Tune only when known geometry or radius distribution is clearly mismatched.
- `vertical_filter_window`: vertical cropping/window for robust ring support before fitting. Too small can discard real wall points; too large can admit clutter.
- `ransac_threshold`: distance threshold for center/geometry model fitting. Lower is stricter; higher tolerates noisy rings but can fit clutter.
- `ransac_probability`: desired RANSAC success probability. Usually keep high and fixed.
- `ransac_inlier_ratio`: expected inlier ratio. Lower for noisy/sparse rings; higher for clean rings.
- `ransac_sample_size`: sample size per RANSAC hypothesis. Usually keep fixed unless fitting is unstable.
- `ransac_initial_iterations`: initial RANSAC iteration budget. Increase only when fitting fails or highly sparse rings are unstable.
- `ransac_inlier_threshold_multiplier`: multiplier for adaptive inlier thresholding. Lower is stricter; higher accepts more points.

Empirical ranges/guidelines:

- `vertical_filter_window`: roughly `4.0-7.0`, driven by tunnel family/diameter and vertical spread.
- `ransac_threshold`: roughly `0.5-1.5`; use lower for clean dense rings, higher for noisy/sparse or partial rings.
- `ransac_inlier_ratio`: roughly `0.5-0.85`; lower when density is uneven or many points are clutter.
- `ransac_sample_size`: usually `5`; avoid tuning unless there is fitting evidence.
- `ransac_initial_iterations`: usually high (`~999`); increasing may help only fitting instability, not denoising/enhancement failures.
- `ransac_inlier_threshold_multiplier`: roughly `0.6-1.2`.

Retune unfolding when:

- `context_unwrapped.csv` has wrong or discontinuous theta/radius structure.
- Target and context maps show systematic geometric distortion, not just sparse valid pixels.
- Empty bands align with coordinate discontinuities or observed-theta cropping is still wrong.
- r4tun global unwrapped is missing and local PCA is used.

Do not retune unfolding when:

- The map geometry is stable but foreground support is filtered away during denoising.
- The main issue is sparse valid pixels after enhancing or outlier interpolation.

## Stage 2: Denoising Parameters

Denoising decides which unwrapped points remain as surface support (`pred != 0`). It has the largest effect on valid support before enhancement.

Parameters:

- `radius_min`: lower radius mask. Increasing removes inner/groove/clutter points but can delete true surface on smaller-radius rings.
- `radius_max`: upper radius mask. Decreasing removes outer clutter/noise but can delete true surface on larger-radius rings.
- `y_step`: theta/scan step for profile evaluation. Larger smooths/aggregates more; smaller is more local/noisy.
- `z_step`: radial/depth discretization step. Smaller is more sensitive; larger is smoother.
- `gradient_threshold`: threshold for detecting abrupt radial changes/outliers. Lower is stricter and may remove more; higher keeps more noisy support.
- `smoothing_window_size`: smoothing window along profile. Larger suppresses local noise but can wash out small structures.
- `smoothing_offset`: bias after smoothing/cutoff. More negative is stricter; more positive is permissive.
- `default_cutoff_z`: fallback radial/depth cutoff when local evidence is weak.
- `double_zero_cutoff`: keep as a structural switch unless there is clear evidence.

BO-tuned empirical ranges:

- `radius_min`: `1.8-3.8`
- `radius_max`: `2.0-4.2`, always enforce `radius_max >= radius_min + 0.05`
- `gradient_threshold`: `0.03-0.40`
- `smoothing_offset`: `-0.02-0.02`

Common warm-start values:

- `y_step`: `0.4`
- `z_step`: `0.005`
- `smoothing_window_size`: `5`
- `smoothing_offset`: around `-0.002`

Tune guidelines by observed change:

- Valid ratio too low, recall near zero, and `context_denoised.csv` has few `pred != 0` points: loosen denoising. Decrease `radius_min`, increase `radius_max`, increase `gradient_threshold`, move `smoothing_offset` toward zero/positive.
- Many false valid pixels or overfilled background: tighten denoising. Increase `radius_min`, decrease `radius_max`, decrease `gradient_threshold`, make `smoothing_offset` more negative.
- Empty horizontal/row bands after denoising: first check whether whole theta ranges were filtered out. If yes, widen radius mask or loosen gradient/smoothing.
- Large valid coverage but low precision: tighten radius bounds before changing enhancement.
- Large map differences between target and context: verify ring ownership and context composition before retuning denoising.

Retune denoising when:

- `valid_ratio` collapses relative to baseline.
- `target_foreground_recall` is near zero and denoised support is sparse.
- The depth map is structurally empty before enhancement can help.

Do not retune denoising first when:

- Denoised support exists but final map has holes because interpolation/upsampling is weak.

## Stage 3: Enhancing Parameters

Enhancing densifies retained support and interpolates surface/outlier regions. It can improve coverage and reduce holes, but it can also hallucinate background if too aggressive.

Parameters:

- `target_distances`: three-stage upsampling distances. Larger distances add fewer points; smaller distances densify more. The list is sorted high-to-low during BO.
- `curvature_neighbors` / `num_neighbors`: neighbors for curvature and local interpolation support. Lower preserves local detail but can be noisy; higher smooths but can bridge gaps incorrectly.
- `interpolation_window`: depth map interpolation/projection window. Larger fills gaps; too large can smear/overfill.
- `curvature_threshold_enh`: curvature cutoff controlling where surface upsampling is allowed.
- `depth_threshold_low` / `depth_threshold_high`: depth thresholds for outlier detection/interpolation.
- `inter_radius` / `outlier_interpolation_radius`: spatial radius for outlier interpolation. Larger bridges gaps; too large can cross unrelated structures.
- `duplicate_threshold` / `outlier_duplicate_threshold`: suppresses duplicate synthetic points.
- `num_interpolations` / `outlier_num_interpolations`: number of interpolation passes/points. More fills gaps but risks overfill.
- `outlier_depth_map_window`: window for outlier depth map diagnostic.
- `outlier_neighbors`: neighbor count for outlier interpolation.
- `max_outlier_points`: safety cap; avoid increasing unless subsampling is clearly limiting recovery.
- `outlier_bidirectional`: keep disabled unless a controlled experiment shows benefit.
- `n_segment_start` / `n_segment_end`, `outlier_high_density_ring_start` / `outlier_high_density_ring_end`: high-density special-case gates. `-1` disables the high-density segment logic.

BO-tuned empirical ranges:

- `curvature_neighbors` / `num_neighbors`: `8-40`
- `interpolation_window`: `1-15`
- `target_distance_1`: `0.03-0.12`
- `target_distance_2`: `0.015-0.06`
- `target_distance_3`: `0.008-0.04`
- `outlier_interpolation_radius` / `inter_radius`: `0.01-0.08`
- `outlier_num_interpolations` / `num_interpolations`: `1-5`
- `outlier_depth_map_window`: `1-9`
- `outlier_neighbors`: `8-40`

Common warm-start values:

- `target_distances`: `[0.06, 0.03, 0.015]`
- `curvature_neighbors` / `num_neighbors`: `20`
- `interpolation_window`: `9`
- `curvature_threshold_enh`: `0.005`
- `outlier_interpolation_radius`: `0.03`
- `outlier_num_interpolations`: `2`
- `outlier_depth_map_window`: `1`

Tune guidelines by observed change:

- Denoised support is good but valid ratio is low or there are many small holes: densify. Lower target distances, increase `interpolation_window`, modestly increase `outlier_interpolation_radius` or `outlier_num_interpolations`.
- Large empty row bands persist even with support: increase interpolation/window/radius carefully, but check whether the gaps are true missing theta coverage first.
- Overfilled background or visually smeared map: reduce `interpolation_window`, increase target distances, reduce `outlier_interpolation_radius`, reduce interpolation passes, or increase neighbor count to avoid local noise.
- Precision drops while recall improves: enhancement is likely too aggressive. Prefer smaller interpolation radius/window over tightening denoising if denoised support looked correct.
- Sparse high-density artifacts or outlier spikes: reduce outlier radius/passes and inspect `depth_map_outlier.npy`.

Retune enhancing when:

- `context_denoised.csv` retains enough support, but `depth_map.npy` still has holes or poor valid coverage.
- `target_foreground_recall` can improve without violating `coverage_ok` or `empty_band_ok`.
- Visual issues are interpolation holes, not coordinate distortion.

Do not retune enhancing first when:

- Denoising already removed most target foreground.
- Unwrapped coordinates are geometrically wrong.

## Stage Selection Heuristics

Use these heuristics to decide which stage(s) to retune:

1. Unfolding first:
   - wrong coordinate source, distorted theta/radius, discontinuity, or full map geometry mismatch.
   - Retune only a small set: `vertical_filter_window`, `ransac_threshold`, `ransac_inlier_ratio`, `ransac_inlier_threshold_multiplier`.

2. Denoising first:
   - support collapse, valid ratio near zero, target recall near zero, or empty bands caused before enhancement.
   - Retune: `radius_min`, `radius_max`, `gradient_threshold`, `smoothing_offset`; optionally `smoothing_window_size`.

3. Enhancing first:
   - denoised support is present but map has holes, sparse valid coverage, or poor interpolation.
   - Retune: `target_distances`, `interpolation_window`, `outlier_interpolation_radius`, `outlier_num_interpolations`, `outlier_neighbors`.

4. No retune:
   - BO best does not improve guarded score.
   - Visual result worsens even if one diagnostic improves.
   - Metrics are zero because GT-to-pixel mapping/support is missing or impossible.

5. Multi-stage retune:
   - Only recommend if evidence crosses stages. Example: coordinate distortion plus support collapse. Keep parameter set small and staged: unfold first, rerun; denoise/enhance second.

## Reflection Output Format

Return exactly these sections:

```markdown
## Diagnosis
- Ring/tunnel:
- Observed failure mode:
- Evidence from files/metrics:

## Stage Decision
- Retune stage(s): unfolding | denoising | enhancing | none
- Why this stage:
- Why not the other stages:

## Parameter Retune Proposal
| stage | parameter | current | proposed range | direction | rationale |
|---|---:|---:|---:|---|---|

## Reward And Guardrails
- Primary score to compare:
- Minimum acceptable coverage behavior:
- Maximum acceptable empty-band behavior:
- Diagnostic metrics to log:

## Final Reflection Decision
- selected action: keep_fixed_baseline | run_small_bo | run_targeted_sweep | inspect_data_mapping
- expected improvement:
- stop condition:
- output sandbox:
```

## Conservative Defaults

- Prefer `keep_fixed_baseline` if evidence is weak.
- Prefer `run_small_bo` only for denoising/enhancing parameter sets already covered by empirical BO ranges.
- Prefer `run_targeted_sweep` for unfolding because RANSAC/geometric parameters are less directly optimized by the guarded BO runner.
- Never optimize downstream detection/segmentation metrics in preprocessing reflection.
- Never use K position, segment sequence, walking order, or block balance as preprocessing BO criteria.
