# Detection BO Recommendations

## Summary

Based on the BO convergence analysis, here are the tunnels that need more tuning:

## Tunnels Needing More BO

### 1. **4-1 (complex_staggered)** - HIGH PRIORITY
- **Current best wF1**: 0.5272
- **Issue**: Highly unstable (gap=0.2141), best found early at eval 53, then degraded
- **Reason**: 29D search space is challenging, many low-scoring evals (0.0-0.4 range)
- **Recommendation**: 
  - Run 200+ more calls (total 320+)
  - Consider narrowing bounds around best parameters from eval 53
  - May need to fix some complex-specific parameters based on best result

### 2. **5-1 (complex_staggered)** - MEDIUM PRIORITY
- **Current best wF1**: 0.6810
- **Issue**: Low absolute score, best found at eval 62, stable but not improving
- **Reason**: 29D space needs more exploration, many evals in 0.4-0.6 range
- **Recommendation**:
  - Run 100+ more calls (total 220+)
  - Consider narrowing bounds around best parameters

### 3. **3-1 (continuous)** - LOW PRIORITY
- **Current best wF1**: 0.9508 (excellent!)
- **Issue**: Best found at eval 48, then degraded to 0.7822 at eval 80
- **Reason**: Unstable convergence, may have over-explored after finding good solution
- **Recommendation**:
  - Run 50-80 more calls to stabilize
  - Consider narrowing bounds around eval 48 parameters

### 4. **2-2 (simple_staggered)** - LOW PRIORITY
- **Current best wF1**: 0.9311 (excellent!)
- **Issue**: Best found at eval 72, degraded to 0.8484 at eval 80
- **Reason**: Slight instability at end
- **Recommendation**:
  - Run 30-50 more calls to stabilize
  - Or accept current best (0.9311 is very good)

## Tunnels OK (No More BO Needed)

### **1-4 (simple_staggered)** - CONVERGED
- **Best wF1**: 0.9523
- **Status**: Perfect convergence, best found at final eval 80
- **Action**: No further tuning needed

## Execution Priority

1. **4-1**: Most critical - very low score and unstable
2. **5-1**: Moderate - needs improvement but stable
3. **3-1**: Optional - excellent score but unstable
4. **2-2**: Optional - excellent score, minor instability

## Notes

- Complex tunnels (4-1, 5-1) have 29D search space vs 14D for simple/continuous
- Complex tunnels consistently show lower wF1 scores, suggesting the algorithm or parameter ranges may need adjustment
- All simple/continuous tunnels achieved >0.93 wF1, indicating the detection logic works well for those patterns
