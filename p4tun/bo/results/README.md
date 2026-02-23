# BO Results Directory

Organized Bayesian Optimization results for P4Tun pipeline.

## Directory Structure

```
results/
├── *.json                  # Latest vanilla BO results (best configs)
├── analysis/               # Analysis reports and evaluations
├── no_gt_bo/              # No-ground-truth BO experiment results
├── proxy_experiments/     # Proxy BO validation experiments
└── archive/               # Historical data
    ├── checkpoints/       # BO checkpoint files (.pkl)
    ├── convergence_plots/ # Convergence visualizations (.png)
    ├── logs/             # Execution logs
    └── vanilla_bo/       # Older BO history files
```

## Main Results (Latest)

| Tunnel | Stage | File | Best mIoU |
|--------|-------|------|-----------|
| 1-4 | combined | 1-4_combined_20260123_040930.json | 0.807 |
| 1-4 | detection | 1-4_detection_20260126_125324.json | - |
| 2-2 | detection | 2-2_detection_20260122_101404.json | - |
| 2-2 | sam | 2-2_sam_20260122_120958.json | 0.765 |
| 3-1 | combined | 3-1_combined_20260124_123138.json | 0.769 |
| 4-1 | sam_wraparound | 4-1_sam_wraparound_20260127_124127.json | 0.428 |
| 5-1 | complex_sam | 5-1_complex_sam_20260127_185432.json | 0.431 |

## Key Subfolders

### analysis/
Analysis reports including:
- `TUNING_GUIDELINE.md` - Parameter tuning guide
- `PREDICTOR_EVALUATION_REPORT.md` - mIoU predictor evaluation
- `*_INTRINSIC_QUALITY_ANALYSIS.md` - Per-tunnel intrinsic metric analysis
- `*_optimization_report.md` - Per-stage optimization reports

### no_gt_bo/
No-ground-truth BO experiments using intrinsic metrics to predict mIoU.

### proxy_experiments/
Validation experiments comparing proxy BO (predicted mIoU) vs true mIoU BO.

### archive/
Historical data for reference:
- `checkpoints/` - Resumable BO state (can be deleted to save space)
- `convergence_plots/` - Visualization of optimization progress
- `logs/` - Detailed execution logs
- `vanilla_bo/` - Older BO runs (superseded by latest)

## Usage

Load a result file:
```python
import json
with open('p4tun/bo/results/2-2_detection_20260122_101404.json') as f:
    result = json.load(f)
best_params = result['best_params']
```

---

*Last organized: 2026-02-02*
