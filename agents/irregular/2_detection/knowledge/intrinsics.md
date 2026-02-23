# Detection Output Intrinsic Metrics (Irregular / Complex Staggered)

Metrics to assess detection quality for segmentation readiness.
Split into K-position quality and non-K expansion quality.

## K-Position Metrics (adapted from regular)

| Metric | Threshold | What It Detects |
|--------|-----------|-----------------|
| `det_k_count_match` | == ring_count | Missing or phantom K positions |
| `det_k_x_spacing_cv` | <= 0.20 | Uneven horizontal K spacing |
| `det_k_confidence_avg` | >= 0.50 | Low-confidence detections (many fallbacks) |

### det_k_count_match
- **Source:** `detected.csv` row count vs `ring_count.txt`
- **Threshold:** Must match exactly
- **Failure mode:** Wrong number of rings → all downstream positions wrong

### det_k_x_spacing_cv
- **Source:** `detected.csv` X column
- **Formula:** std(X_gaps) / mean(X_gaps)
- **Threshold:** <= 0.20 (relaxed from regular 0.15; more rings, wider variation)
- **Note:** Geometric detection uses evenly-spaced bands so CV is typically ~0.00

### det_k_confidence_avg
- **Source:** `detected.csv` Confidence column
- **Formula:** mean(Confidence)
- **Threshold:** >= 0.50
- **Failure mode:** Many fallback detections (confidence 0.1-0.35) → inaccurate K Y positions
- **Known values:** 5-1: 0.95 (all geometric_midpoint), 4-1: 0.64 (mix of midpoint/neg_only/fallback)

## Expansion Quality Metrics (NEW for irregular)

| Metric | Threshold | What It Detects |
|--------|-----------|-----------------|
| `det_block_count_per_ring` | == True (all rings have 7) | Missing or extra blocks |
| `det_y_coverage_pct` | [85%, 115%] | Blocks tile circumference? |
| `det_min_y_gap_px` | >= 0 (informational) | Block overlap |
| `det_y_order_consistency` | >= 0.0 (informational) | Canonical cyclic order preserved |

### det_block_count_per_ring
- **Source:** `all_segments.csv` grouped by Ring
- **Threshold:** All rings must have exactly 7 blocks (K, B1, B2, A1-A4)
- **Failure mode:** Missing blocks → segmentation gaps

### det_y_coverage_pct
- **Source:** `all_segments.csv` Y positions per ring
- **Formula:** Sum of wrap-aware Y gaps / image_height, averaged across rings
- **Threshold:** [85%, 115%]
- **Note:** Should be ~100% for correct tiling

### det_min_y_gap_px
- **Source:** `all_segments.csv` pairwise Y distances within each ring
- **Threshold:** >= 0 (informational only)
- **Note:** BO-tuned offsets can place centroids at same Y because angular boundaries handle separation, not centroid spacing. For template expansion, expect gaps >= 80px.

### det_y_order_consistency
- **Source:** `all_segments.csv` block order sorted by Y per ring
- **Formula:** Fraction of rings where Y-sorted block sequence matches any rotation of K-B1-A1-A2-A3-A4-B2
- **Threshold:** >= 0.0 (informational only)
- **Note:** BO-tuned offsets produce non-canonical orders (5-1: 0.29, 4-1: 0.00). Template expansion should score ~1.0 since it preserves cyclic order by construction.

## Differences from Regular Pipeline

| Aspect | Regular | Irregular |
|--------|---------|-----------|
| Blocks/ring | 6 (K, B1, B2, A1-A3) | 7 (K, B1, B2, A1-A4) |
| midpoint_ratio metric | Yes (from Type column) | Replaced by k_confidence_avg |
| y_pattern_consistency | Even/odd alternation | Not applicable (complex stagger) |
| Expansion metrics | None needed | 4 new metrics for non-K blocks |
| X spacing | Varies with line detection | ~0.00 (geometric bands) |

## Known Good Values

| Metric | 5-1 (BO-tuned) | 4-1 (BO-tuned) |
|--------|----------------|----------------|
| k_count_match | True (7) | True (9) |
| k_x_spacing_cv | 0.0000 | 0.0000 |
| k_confidence_avg | 0.95 | 0.64 |
| block_count_per_ring | True | True |
| y_coverage_pct | 100.0% | 100.0% |
| min_y_gap_px | 75 | 0 |
| y_order_consistency | 0.29 | 0.00 |
