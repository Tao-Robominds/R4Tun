# SAM Segmentation Output Intrinsic Metrics

Critical metrics to determine if SAM segmentation output is of acceptable quality.
Each metric has a threshold for pass/fail decision.

## Critical Metrics (Currently Extracted)

| Metric | Good Range | What It Detects |
|--------|------------|-----------------|
| `sam_segment_type_completeness` | == True | All expected block types present |
| `sam_ring_completeness_avg` | >= 0.85 | Average fraction of expected types per ring |
| `sam_mask_coverage_pct` | [55%, 90%] | Segmented / mappable points |
| `sam_k_size_ratio` | [3%, 20%] | K-block proportion of segmented area |

### 1. sam_segment_type_completeness (Type Completeness)

- **Source:** `final.csv` pred column
- **Formula:** Check if all expected segment types (1-6 for 6-segment tunnels, 1-7 for 7-segment tunnels) appear in pred column
- **Threshold:** Must be True (all expected types present)
- **Failure mode:**
  - `False`: One or more expected block types (K, B1, A1, A2, A3, A4, B2) are entirely missing from segmentation. Indicates fundamental segmentation failure - SAM failed to produce masks for certain block types. Retune template mask parameters (`k_mask_width`, `k_mask_height_*`, `ab_mask_width`, `ab_mask_height`) or check detection quality.

### 2. sam_ring_completeness_avg (Ring Completeness)

- **Source:** `final.csv` pred and pred_ring columns
- **Formula:** For each ring with segments, compute fraction of expected types present. Return average across all rings.
- **Threshold:** >= 0.85 (at least 85% of expected types per ring on average)
- **Failure mode:**
  - <0.85: Many rings are missing certain segment types. Indicates inconsistent segmentation quality across rings. May be caused by poor K-position detection (some rings have bad anchor points), depth map quality issues (sparse regions), or template mask misalignment. Check detection intrinsic metrics first, then retune SAM template mask parameters.

### 3. sam_mask_coverage_pct (Mask Coverage)

- **Source:** `final.csv` pred column
- **Formula:** (segmented points / mappable points) × 100, where:
  - Mappable = all points where pred != 8 (unmapped)
  - Segmented = points where pred > 0 and pred < 8
- **Threshold:** [55%, 90%]
- **Failure mode:**
  - <55%: Under-segmentation. Masks are too small or many regions were not segmented. Indicates template masks too narrow/short, or SAM failed to generate masks for many regions. Retune `segment_width`, `k_mask_width`, `ab_mask_width`, `k_mask_height_*`, `ab_mask_height`, `padding`, `crop_margin`.
  - >90%: Over-segmentation. Masks are too large, eating into grooves or background. Indicates template masks too wide/tall, or `min_quality_threshold` too low (including low-quality masks). Retune template mask dimensions or increase `min_quality_threshold`.

### 4. sam_k_size_ratio (K-Block Size Ratio)

- **Source:** `final.csv` pred column
- **Formula:** (K-block points / segmented points) × 100, where K-blocks are pred == 1
- **Threshold:** [3%, 20%]
- **Failure mode:**
  - <3%: K-blocks are too small. Template mask for K-blocks (`k_mask_width`, `k_mask_height_*`) is too narrow/short. Retune K-block template mask parameters.
  - >20%: K-blocks are too large. Template mask for K-blocks is too wide/tall, or K-block masks are bleeding into adjacent segments. Retune K-block template mask parameters.

## Guardrail Summary

| Metric | Guardrail | Action if Failed |
|--------|-----------|-----------------|
| `sam_segment_type_completeness` | == True | Retune template mask parameters or check detection quality |
| `sam_ring_completeness_avg` | >= 0.85 | Check detection intrinsic metrics, retune SAM template mask parameters |
| `sam_mask_coverage_pct` | [55%, 90%] | Retune `segment_width`, template mask dimensions, `padding`, `crop_margin`, `min_quality_threshold` |
| `sam_k_size_ratio` | [3%, 20%] | Retune K-block template mask parameters (`k_mask_width`, `k_mask_height_*`) |

## Output Format

```json
{
  "sam_segment_type_completeness": true,
  "sam_ring_completeness_avg": 1.000,
  "sam_mask_coverage_pct": 71.7,
  "sam_k_size_ratio": 10.5,
  "sam_ready_for_evaluation": true,
  "sam_guardrail_violations": []
}
```

## Known-Good Values

| Tunnel | mIoU | segment_type_completeness | ring_completeness_avg | mask_coverage_pct | k_size_ratio | Status |
|--------|------|---------------------------|----------------------|-------------------|--------------|--------|
| 1-4 | 0.717 | True | 1.000 | 71.7% | 10.5% | ✅ All pass |
| 2-2 | 0.775 | True | 1.000 | 70.8% | 10.8% | ✅ All pass |
| 3-1 | 0.687 | True | 0.857 | 67.5% | 9.6% | ✅ All pass (ring_completeness borderline) |

## Correlation with mIoU

Based on `reports/INTRINSIC_METRICS_REPORT.md`:

- **`sam_mask_fill_rate`** (equivalent to `sam_mask_coverage_pct`): Strong negative correlation (r=-0.82) with mIoU for simple patterns. Lower coverage → higher mIoU (counterintuitive but validated).
- **`complex_sam_ring_completeness`** (equivalent to `sam_ring_completeness_avg`): Strong positive correlation (r≈0.80) with mIoU for complex patterns.

These metrics are validated predictors of segmentation quality and can be used for:
1. **Guardrails**: Hard thresholds to reject poor segmentation outputs
2. **BO objectives**: Proxy metrics for mIoU when GT is unavailable
3. **Early stopping**: Stop BO trials that fail guardrails
