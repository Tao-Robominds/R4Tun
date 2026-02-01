# P4Tun Surrogate Model Pipeline

Fast parameter search using Gaussian Process surrogate models trained on existing Bayesian Optimization data.

## Overview

This module implements a surrogate model approach to efficiently search for pipeline parameters that achieve target metrics, without running the computationally expensive full pipeline.

### Pipeline Stages

1. **Data Extraction** - Extract parameter vectors and scores from BO JSON logs
2. **GP Training** - Train Gaussian Process surrogate on extracted data
3. **Inverse Search** - Search parameter space using GP (fast) to find candidates
4. **Validation** - Validate top candidates with full pipeline

## Quick Start

```python
from p4tun.surrogate import SurrogatePipeline

# Initialize pipeline for detection stage
pipeline = SurrogatePipeline(stage='detection')

# Fit (extract data and train GP)
pipeline.fit()

# Search for parameters targeting mIoU >= 0.75
search_result = pipeline.search(target_miou=0.75, n_candidates=10)

# Validate top candidates
validation_report = pipeline.validate(search_result, n_validate=5)
```

## Command Line Usage

```bash
# Run detection surrogate pipeline
python -m p4tun.surrogate.pipeline --stage detection --target 0.75

# Run SAM surrogate pipeline
python -m p4tun.surrogate.pipeline --stage sam --target 0.78 --n-validate 5

# Load existing model and search
python -m p4tun.surrogate.pipeline --stage detection --load-model --target 0.80

# Extract data only
python -m p4tun.surrogate.data_extractor --stage detection --output data.json

# Train GP model only
python -m p4tun.surrogate.gp_surrogate --stage detection --output model.pkl

# Inverse search only
python -m p4tun.surrogate.inverse_search --stage detection --target 0.75 --model model.pkl
```

## Module Components

### DataExtractor (`data_extractor.py`)

Extracts training data from BO JSON logs in `p4tun/bo/results/`.

```python
from p4tun.surrogate import DataExtractor

extractor = DataExtractor(stage='detection')
data = extractor.extract_all()
# data.X: parameter vectors (n_samples, n_features)
# data.y: scores (n_samples,)
# data.param_names: parameter names
```

### GPSurrogate (`gp_surrogate.py`)

Gaussian Process surrogate model for fast parameter evaluation.

```python
from p4tun.surrogate import GPSurrogate

surrogate = GPSurrogate(kernel_type='matern')
surrogate.fit(data)

# Predict score for parameters
prediction = surrogate.predict_single({'binary_threshold': 127, ...})
print(f"Predicted: {prediction.mean[0]:.3f} ± {prediction.std[0]:.3f}")

# Feature importance
importance = surrogate.get_feature_importance()
```

### InverseSearch (`inverse_search.py`)

Search for parameters that achieve target metrics.

```python
from p4tun.surrogate import InverseSearch

searcher = InverseSearch(surrogate, acquisition='ei')
result = searcher.search(target_metric=0.75, n_candidates=10)

for candidate in result.candidates:
    print(f"Predicted: {candidate.predicted_mean:.3f}")
```

### Validator (`validator.py`)

Validate candidates by running the full pipeline.

```python
from p4tun.surrogate import Validator

validator = Validator(stage='detection', tunnel_id='2-2')
report = validator.validate_search_result(search_result, n_candidates=5)

print(f"Success rate: {report.success_rate:.1%}")
print(f"Mean prediction error: {report.mean_prediction_error:.4f}")
```

## Key Parameters (Sensitivity Analysis)

### Detection Stage
- `binary_threshold` - Most sensitive
- `hough_oblique_threshold`
- `angle_positive_min`, `angle_positive_max`
- `hough_vertical_threshold`

### SAM Stage
- `segment_width`, `k_height`, `ab_height`
- `angle_deg`
- `k_mask_width`, `k_mask_height_pos`, `k_mask_height_neg`
- `ab_mask_width`, `ab_mask_height`
- `min_quality_threshold`

## Output Files

Results are saved to `p4tun/surrogate/outputs/`:
- `{stage}_search_{timestamp}.json` - Search results
- `{stage}_validation_{timestamp}.json` - Validation reports

Models are saved to `p4tun/surrogate/models/`:
- `{stage}_gp.pkl` - Trained GP model

## Acquisition Functions

- **ei** (Expected Improvement) - Default, balances exploitation and exploration
- **ucb** (Upper Confidence Bound) - Higher exploration
- **poi** (Probability of Improvement) - Focus on exceeding target
- **mean** - Pure exploitation (no exploration)

## Active Learning

When validation reveals high prediction errors, candidates are flagged for
inclusion in the next training iteration:

```python
updates = validator.get_training_updates(report, error_threshold=0.05)
# Returns list of (params, actual_score) for retraining
```
