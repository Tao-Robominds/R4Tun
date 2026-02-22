# Preprocessing Intrinsic Metrics Analysis

This report identifies the critical intrinsic metrics for preprocessing and recommends a proxy objective for Bayesian optimization when ground-truth mIoU is unavailable (e.g. new tunnels). It aligns with the project methodology: use intrinsic metrics as proxy ground truth, then correlate with mIoU and build predictors/guardrails.

## 1. Objective

Preprocessing is a **feasibility gate**, not a direct mIoU optimizer. Project evidence (see [tuning.md](tuning.md)) shows preprocessing contributes only ~0.1% mIoU once detection and SAM are tuned. Therefore:

- The main goal of preprocessing BO without GT is to **maximise the probability that output is ready for detection** (binary feasibility).
- Among feasible configurations, we use **margin-based tie-breaking** so BO does not chase noisy micro-gains.

The intrinsic metrics defined here serve as proxy ground truth for this stage and are already implemented in [scripts/extract_intrinsics.py](../scripts/extract_intrinsics.py). See also [intrinsics.md](intrinsics.md) for threshold definitions.

## 2. Three Critical Metrics

| Metric | Role | Good range | Failure mode (out of range) |
|--------|------|------------|-----------------------------|
| `pre_point_retention_pct` | Denoising sanity | 70–98% | &lt;70%: over-denoisied; &gt;98%: denoising ineffective |
| `pre_depth_map_valid_pixels` | Depth map density | 8k–35k | &lt;8k: too sparse → missed lines; &gt;35k: over-interpolation → false K-points |
| `pre_theta_coverage_pct` | Unfolding completeness | 98–102% (strict: 99.5–100.5%) | &lt;98%: incomplete coverage; &gt;102%: wraparound/duplicate segments |

### Formulas and sources

**1. pre_point_retention_pct** (target: 70–98%)

- **Source:** `denoised.csv` and `unwrapped.csv`
- **Formula:** `(N_denoised_valid / N_unwrapped) × 100`
- **Implementation:** If `pred` exists in denoised, valid = `(pred != 0).sum()`; else `len(denoised)`. See [extract_intrinsics.py](../scripts/extract_intrinsics.py) `_extract_point_retention()`.
- **Constants:** `POINT_RETENTION_PCT_MIN = 70.0`, `POINT_RETENTION_PCT_MAX = 98.0`

**2. pre_depth_map_valid_pixels** (target: 8k–35k)

- **Source:** `depth_map_outlier.npy` (the file detection consumes)
- **Formula:** `sum(1[~isnan(depth)])` → `int(np.sum(~np.isnan(depth)))`
- **Implementation:** [extract_intrinsics.py](../scripts/extract_intrinsics.py) `_extract_depth_map_valid_pixels()`
- **Constants:** `DEPTH_MAP_VALID_PIXELS_MIN = 8_000`, `DEPTH_MAP_VALID_PIXELS_MAX = 35_000`
- **Out of range:** Too sparse (&lt;8k) → detection misses structures; over-filled (&gt;35k, especially &gt;100k) → spurious structure / false K-points.

**3. pre_theta_coverage_pct** (near 100%; guardrail 98–102%)

- **Source:** `unwrapped.csv` theta column
- **Formula:** Theta is stored as `angle_deg * (π * diameter / 360)`. Convert back: `scale = π * diameter / 360`, then `θ_deg_min = t_min / scale`, `θ_deg_max = t_max / scale`, and `pre_theta_coverage_pct = (θ_deg_max − θ_deg_min) / 360 × 100`.
- **Implementation:** [extract_intrinsics.py](../scripts/extract_intrinsics.py) `_extract_theta_coverage()`
- **Constants:** `THETA_COVERAGE_PCT_MIN = 98.0`, `THETA_COVERAGE_PCT_MAX = 102.0` (strict band 99.5–100.5% for tie-break)

## 3. Weight Justification from Project Evidence

Weights for the margin-based tie-break (retention 0.5, depth 0.4, theta 0.1) are heuristic and chosen from project evidence:

- **Retention (0.5):** In preprocessing memory logs (e.g. [1-4 success](memory/20260203_233451_1-4_success_tunnel_diameter_radius_min_radius_max.md)), success/failure flips were strongly tied to `pre_point_retention_pct` crossing the guardrail (e.g. 1.5% → 72.9% after fixing `tunnel_diameter`). This was a major failure mode when preprocessing was bad.
- **Depth valid pixels (0.4):** Same recovery run showed a large move (3736 → 15198). This metric is directly tied to “too sparse vs over-interpolated” and downstream detectability in the intrinsic spec.
- **Theta coverage (0.1):** In the same success case, theta stayed unchanged (99.99% → 99.99%) while the run flipped to ready. So it behaves more like a **guardrail/validity check** than a fine-grained optimization signal for ranking feasible configs.

Preprocessing has minimal impact on final mIoU once detection/SAM are tuned; the score is intended as a **stability/feasibility tie-break**, not a precise mIoU predictor.

## 4. Recommended Proxy Objective

Use a two-stage intrinsic score for preprocessing BO (maximise this score):

**Stage A (hard gate):**  
`score = 0` if `pre_ready_for_detection == false`. Optionally, a small shaping term can still guide search toward feasibility (see below).

**Stage B (tie-break among feasible points):**  
`score = 1 + 0.5 * retention_margin + 0.4 * depth_margin + 0.1 * theta_margin`, where each margin is a normalised distance-from-bad-bounds in [0, 1].

### Drop-in scoring function

```python
def preprocessing_intrinsic_score(intrinsics: dict) -> float:
    """
    Objective for BO (maximize):
    1) Hard feasibility gate via pre_ready_for_detection
    2) Margin-based tie-break among feasible configs
    """
    ready = bool(intrinsics.get("pre_ready_for_detection", False))
    theta = float(intrinsics.get("pre_theta_coverage_pct", 0.0))
    retention = float(intrinsics.get("pre_point_retention_pct", 0.0))
    depth_valid = float(intrinsics.get("pre_depth_map_valid_pixels", 0.0))

    def range_margin(x, lo, hi):
        """Normalized in-range margin [0, 1]; 0 at/outside bounds, max near center."""
        if x <= lo or x >= hi:
            return 0.0
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        return max(0.0, 1.0 - abs(x - center) / half)

    if not ready:
        # Small shaping term to guide search toward feasibility
        fail_hint = (
            0.5 * range_margin(retention, 70.0, 98.0) +
            0.4 * range_margin(depth_valid, 8000.0, 35000.0) +
            0.1 * range_margin(theta, 99.5, 100.5)
        )
        return 0.1 * fail_hint  # stays < feasible scores

    retention_margin = range_margin(retention, 70.0, 98.0)
    depth_margin = range_margin(depth_valid, 8000.0, 35000.0)
    theta_margin = range_margin(theta, 99.5, 100.5)
    score = 1.0 + 0.5 * retention_margin + 0.4 * depth_margin + 0.1 * theta_margin
    return score
```

## 5. Guardrail Thresholds for Reflections

Use these as signals to trigger reflection (e.g. when violated, prompt for parameter or pipeline checks):

| Metric | Guardrail | Interpretation |
|--------|-----------|----------------|
| `pre_point_retention_pct` | min 0.4 (40%) | Below this, denoising/geometry likely wrong |
| `pre_real_detection_ratio` | min 0.5 | (If defined in your pipeline) detection coverage |
| `pre_depth_map_valid_pixels` | 8k–35k | Below 8k: too sparse; above 35k: over-filled |
| `pre_theta_coverage_pct` | 98–102% | Outside: incomplete or wraparound unfolding |
| `pre_ready_for_detection` | must be true | Hard gate before considering tie-break |

Example structure (align names with your pipeline):

```python
GUARDRAIL_THRESHOLDS = {
    "pre_point_retention_pct": {"min": 0.4},
    "pre_real_detection_ratio": {"min": 0.5},   # if used
    "pre_depth_map_valid_pixels": {"min": 8000, "max": 35000},
    "pre_theta_coverage_pct": {"min": 98.0, "max": 102.0},
    "pre_ready_for_detection": True,
}
```

## 6. Generalizability

These numbers are **R4Tun/P4Tun-pipeline guardrails**, not universal constants:

- They are defined in this project’s preprocessing intrinsic extractor and used by the detection stage that consumes `depth_map_outlier.npy`.
- The “good range” depends on this data format, interpolation method, and resolution choices.
- Preprocessing has small effect on mIoU compared with detection/SAM, so the ranges are best interpreted as **readiness guardrails for this stack**.
- **General across current R4Tun tunnels**, but not guaranteed for other datasets or pipelines without recalibration.
