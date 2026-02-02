# No-GT BO Report Collection

This folder contains all documentation and evidence for the intrinsic-metrics-based
no-ground-truth Bayesian Optimization system.

## Main Reports

| File | Description |
|------|-------------|
| [NO_GT_BO_COMPREHENSIVE_REPORT.md](NO_GT_BO_COMPREHENSIVE_REPORT.md) | **Main report** - Complete journey, methodology, and reflection system |
| [NUMERICAL_EVIDENCE_SUMMARY.md](NUMERICAL_EVIDENCE_SUMMARY.md) | All numerical data, thresholds, and statistics |
| [MISSING_EXPERIMENTS_AND_RECOMMENDATIONS.md](MISSING_EXPERIMENTS_AND_RECOMMENDATIONS.md) | Gaps and future work |

## Analysis Reports

| File | Description |
|------|-------------|
| [2-2_INTRINSIC_QUALITY_ANALYSIS.md](2-2_INTRINSIC_QUALITY_ANALYSIS.md) | Why 2-2 failed and how it was fixed (det_x_spacing_cv) |
| [COMPLEX_STAGGERED_INTRINSIC_ANALYSIS.md](COMPLEX_STAGGERED_INTRINSIC_ANALYSIS.md) | Complex patterns (4-1, 5-1) analysis |

## Data Files

| File | Description |
|------|-------------|
| [training_data_simple.csv](training_data_simple.csv) | Training data (20 samples) with intrinsic metrics |
| [predictor_evaluation.json](predictor_evaluation.json) | Correlation analysis results |

## Quick Reference

### Key Predictive Features

**Simple Patterns (1-4, 2-2, 3-1):**
- det_midpoint_ratio (r = +0.87)
- sam_mask_fill_rate (r = -0.82)
- det_real_detection_ratio (r = +0.69)
- det_k_count_match (r = +0.52)

**Complex Patterns (4-1, 5-1):**
- segment_width (r = -0.789) ← DOMINANT
- ab_height (r = -0.341)
- k_height (r = -0.269)

### Reflection Thresholds

| Tier | Trigger | Action |
|------|---------|--------|
| 1 | det_k_count error > 3 | **MUST RERUN** |
| 1 | det_x_spacing_cv > 0.50 | **MUST RERUN** |
| 2 | predicted_mIoU < 0.40 | Rerun if budget |
| 3 | det_assume_default_ratio > 0.20 | Log warning |

### Model Performance

| Model | Spearman | MAE | Training Size |
|-------|----------|-----|---------------|
| Simple patterns | 0.84 | 0.09 | n=20 |
| Complex patterns | 0.87 | 0.0125 | n=70 |

---

*Generated: 2026-02-02*
