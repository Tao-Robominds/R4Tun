# Missing Experiments and Recommendations

## Summary of Gaps

| Gap | Priority | Effort | Impact |
|-----|----------|--------|--------|
| Full ablation study | High | Medium | Better understand feature importance |
| Preprocessing intrinsic metrics | Low | Low | Complete picture but minimal mIoU impact |
| Cross-tunnel validation | Medium | Medium | Verify generalization |
| More training data | Medium | High | More robust predictor |
| Error analysis by failure mode | High | Medium | Better guardrails |
| **Per-stage impact across all tunnels** | **High** | **High** | **Accurate stage prioritization** |

---

## Gap 1: Full Ablation Study

### Current State
- Only compared: detection-only vs SAM-only vs combined
- Did NOT: Remove individual metrics to measure contribution

### Recommended Experiment

```python
# Leave-one-out feature importance
features = ['det_midpoint_ratio', 'det_real_detection_ratio', 
            'det_k_count_match', 'sam_mask_fill_rate']

results = {}
for remove_feature in features:
    subset = [f for f in features if f != remove_feature]
    model = Ridge().fit(X[subset], y)
    cv_mae = cross_val_score(model, X[subset], y, scoring='neg_mae').mean()
    results[remove_feature] = {
        'mae_without': -cv_mae,
        'delta_mae': (-cv_mae) - baseline_mae,
    }

# Report: ΔMAE when each feature is removed
```

### Expected Outcome
- Quantify: "Removing det_midpoint_ratio increases MAE by X"
- May discover: Some features are redundant

---

## Gap 2: Preprocessing Intrinsic Metrics

### Current State
- No preprocessing metrics in predictor
- Preprocessing has <0.1% mIoU impact per vanilla BO

### Proposed Metrics

| Stage | Metric | Data Source | Good Range |
|-------|--------|-------------|------------|
| Unfolding | theta_coverage | unfolded.csv | 98-102% |
| Unfolding | centerline_rmse | ellipse fitting | <1mm |
| Denoising | point_retention | before/after count | >90% |
| Enhancing | interpolation_coverage | depth_map | >95% |

### Recommended Implementation

```python
def compute_preprocessing_metrics(tunnel_id, data_dir):
    """Compute preprocessing stage intrinsic metrics."""
    metrics = {}
    
    # Unfolding metrics
    unfolded_path = Path(data_dir) / tunnel_id / "unfolded.csv"
    if unfolded_path.exists():
        df = pd.read_csv(unfolded_path)
        theta_min, theta_max = df['theta'].min(), df['theta'].max()
        metrics['theta_coverage'] = (theta_max - theta_min) / (2 * np.pi) * 100
    
    # Denoising metrics
    # (would need before/after point counts)
    
    # Enhancing metrics
    depth_map_path = Path(data_dir) / tunnel_id / "depth_map_outlier.npy"
    if depth_map_path.exists():
        depth_map = np.load(depth_map_path)
        valid_pixels = np.sum(~np.isnan(depth_map))
        total_pixels = depth_map.size
        metrics['interpolation_coverage'] = valid_pixels / total_pixels
    
    return metrics
```

### Decision
- **Add as guardrails only, not to predictor**
- Reason: <0.1% mIoU impact doesn't justify predictor complexity

---

## Gap 3: Cross-Tunnel Validation

### Current State
- Predictor trained on all tunnels together
- No leave-one-tunnel-out validation

### Recommended Experiment

```python
# Leave-one-tunnel-out cross-validation
tunnels = ['1-4', '2-2', '3-1', '4-1', '5-1']
results = {}

for test_tunnel in tunnels:
    train_data = df[df['tunnel_id'] != test_tunnel]
    test_data = df[df['tunnel_id'] == test_tunnel]
    
    model = Ridge().fit(train_data[features], train_data['mIoU'])
    predictions = model.predict(test_data[features])
    
    results[test_tunnel] = {
        'mae': mean_absolute_error(test_data['mIoU'], predictions),
        'spearman': spearmanr(test_data['mIoU'], predictions)[0],
        'n_test': len(test_data),
    }
```

### Expected Outcome
- Identify: Which tunnels generalize well
- May find: Some tunnels need tunnel-specific models (like 2-2)

---

## Gap 4: More Training Data

### Current State
- Simple patterns: 20 samples
- Complex patterns: 70 samples

### Data Collection Strategy

1. **Add failure cases**
   - Intentionally run with bad params (binary_threshold=50, etc.)
   - Capture low mIoU configurations
   - Important for guardrail calibration

2. **Expand parameter diversity**
   - Current data mostly from BO (optimized)
   - Add grid search configurations
   - Better coverage of param space

3. **Balance by tunnel**
   - Current: 5 samples per tunnel (simple)
   - Target: 15-20 samples per tunnel

### Implementation

```bash
# Run diverse configs for each tunnel
for tunnel in 1-4 2-2 3-1; do
    for binary_thresh in 100 125 150 175; do
        for hough_thresh in 30 50 70 90; do
            python run_config.py --tunnel $tunnel \
                --binary-threshold $binary_thresh \
                --hough-threshold $hough_thresh
        done
    done
done
```

---

## Gap 5: Error Analysis by Failure Mode

### Current State
- Know 2-2 failed due to x_spacing_cv
- Don't have systematic failure mode catalog

### Recommended Experiment

1. **Collect all no-GT BO failures** (predicted > true + 0.1)
2. **Categorize by violation type**:
   - K-count error
   - X-spacing irregularity
   - Low detection confidence
   - SAM over/under-segmentation
3. **For each category**:
   - What guardrail would have caught it?
   - What threshold is needed?

### Expected Output

| Failure Mode | Frequency | Root Cause | Guardrail Fix |
|--------------|-----------|------------|---------------|
| Irregular X-spacing | 30% | Wrong angle params | det_x_spacing_cv < 0.15 |
| K-count error | 20% | Threshold too low | det_k_count in expected±2 |
| Over-segmentation | 15% | Template too large | sam_mask_fill_rate < 0.95 |
| ... | ... | ... | ... |

---

## Gap 6: Per-Stage Impact Across All Tunnels

### Current State
- Per-stage impact percentages (Detection +6.3%, SAM +4-7%, etc.) are based **only on tunnel 2-2**
- These were measured in a specific optimization order: SAM → Detection → SAM → Preprocessing → Unfolding
- Other tunnels (1-4, 3-1, 4-1, 5-1) may have different per-stage impacts

### Problem
- The current numbers may not generalize
- Different tunnels have different characteristics (simple vs complex staggered patterns)
- Optimization order affects measured incremental impact

### Recommended Experiment

Run systematic per-stage BO optimization on **all tunnels** with consistent methodology:

```python
# For each tunnel, optimize each stage independently from a common baseline
tunnels = ['1-4', '2-2', '3-1', '4-1', '5-1']
stages = ['detection', 'sam', 'preprocessing', 'unfolding']

results = {}
for tunnel in tunnels:
    # Start from default parameters
    baseline_miou = evaluate_with_defaults(tunnel)
    
    results[tunnel] = {'baseline': baseline_miou}
    
    for stage in stages:
        # Reset to defaults, then optimize only this stage
        reset_to_defaults(tunnel)
        best_miou = run_bo(tunnel, stage=stage, n_iter=30)
        
        results[tunnel][stage] = {
            'best_miou': best_miou,
            'absolute_gain': best_miou - baseline_miou,
            'relative_gain': (best_miou - baseline_miou) / baseline_miou * 100,
        }
```

### Expected Output

| Tunnel | Detection Impact | SAM Impact | Preprocessing Impact | Unfolding Impact |
|--------|------------------|------------|----------------------|------------------|
| 1-4 | +X.X% | +X.X% | +X.X% | +X.X% |
| 2-2 | +6.3% | +7.4% | +0.1% | +0.0% |
| 3-1 | +X.X% | +X.X% | +X.X% | +X.X% |
| 4-1 | +X.X% | +X.X% | +X.X% | +X.X% |
| 5-1 | +X.X% | +X.X% | +X.X% | +X.X% |
| **Average** | **+X.X%** | **+X.X%** | **+X.X%** | **+X.X%** |

### Methodology Notes

1. **Independent optimization**: Each stage optimized independently from the same baseline (not sequentially)
2. **Same iteration budget**: 30 iterations per stage per tunnel
3. **Report both absolute and relative gains**
4. **Separate simple (1-4, 2-2, 3-1) from complex (4-1, 5-1) patterns**

### Why This Matters
- Provides accurate guidance for where to focus optimization effort
- Current numbers may over-estimate Detection impact or under-estimate SAM impact for other tunnels
- Critical for prioritizing engineering resources

---

## Recommendations Summary

### Immediate (Before Production)

1. ✅ **Implement reflection triggers** (documented in main report)
2. **Run error analysis** on existing no-GT BO failures
3. **Add preprocessing guardrails** (theta_coverage, interpolation_coverage)

### Short-term (Next Sprint)

4. **Run per-stage impact experiment** across all tunnels (see Gap 6)
5. **Run full ablation study** on current features
6. **Cross-tunnel validation** to verify generalization
7. **Collect 10 more samples per tunnel** with diverse params

### Long-term (Future Roadmap)

8. **Neural network predictor** if more data collected
9. **Active learning** to efficiently sample informative configs
10. **Multi-objective optimization** (mIoU + speed + robustness)

---

## Priority Matrix

| Experiment | Effort | Impact | Priority |
|------------|--------|--------|----------|
| Reflection triggers | Low | High | **P0** |
| Error analysis | Medium | High | **P1** |
| **Per-stage impact (all tunnels)** | **High** | **High** | **P1** |
| Preprocessing guardrails | Low | Medium | P2 |
| Full ablation | Medium | Medium | P2 |
| Cross-tunnel validation | Medium | Medium | P2 |
| More training data | High | High | P3 |

---

*Generated: 2026-02-02*  
*Updated: 2026-02-02 - Added Gap 6 (per-stage impact across all tunnels)*
