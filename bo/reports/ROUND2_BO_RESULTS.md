# Round 2 Detection BO Results

## Summary

All recommended additional BO runs have been completed. Results:

| Tunnel | Round 1 Best | Round 2 Best | Improvement | Total Calls | Status |
|--------|--------------|-------------|-------------|-------------|--------|
| **4-1** | 0.5272 | **0.5909** | +0.0637 | 320 | Improved |
| **5-1** | 0.6810 | 0.6810 | 0.0000 | 220 | No change |
| **3-1** | 0.9508 | 0.9508 | 0.0000 | 160 | Maintained |
| **2-2** | 0.9311 | **0.9432** | +0.0121 | 130 | Improved |

## Detailed Results

### 4-1 (complex_staggered) - HIGH PRIORITY
- **Round 1**: Best wF1=0.5272 at eval 53
- **Round 2**: Best wF1=0.5909 at eval 133 (new best found)
- **Improvement**: +0.0637 (+12.1%)
- **Status**: Significant improvement, but still moderate score
- **Note**: Still has room for improvement, but much better than initial 0.5272

### 5-1 (complex_staggered) - MEDIUM PRIORITY
- **Round 1**: Best wF1=0.6810 at eval 62
- **Round 2**: Found 0.6747 at eval 214, but previous best (0.6810) remains
- **Improvement**: 0.0000 (no improvement)
- **Status**: Stable, but no further improvement found
- **Note**: May have reached local optimum for current search space

### 3-1 (continuous) - LOW PRIORITY
- **Round 1**: Best wF1=0.9508 at eval 48
- **Round 2**: Found 0.9508 again at eval 136 (same score, different params)
- **Improvement**: 0.0000 (maintained)
- **Status**: Excellent score maintained, stabilized
- **Note**: Already at excellent performance level

### 2-2 (simple_staggered) - LOW PRIORITY
- **Round 1**: Best wF1=0.9311 at eval 72
- **Round 2**: Best wF1=0.9432 at eval 111 (new best found)
- **Improvement**: +0.0121 (+1.3%)
- **Status**: Small but meaningful improvement
- **Note**: Now at excellent performance level

## Recommendations

1. **4-1**: Still the lowest performer (0.5909). Consider:
   - Narrowing bounds around best parameters from eval 133
   - Investigating if complex detection algorithm needs adjustment
   - May need different approach for this tunnel type

2. **5-1**: No improvement found. Consider:
   - Narrowing bounds around best parameters
   - Or accept current 0.6810 as reasonable for complex tunnel

3. **3-1 & 2-2**: Both now at excellent levels (>0.94). No further tuning needed.

## Final Status

- **Excellent (>0.93)**: 1-4 (0.9523), 3-1 (0.9508), 2-2 (0.9432)
- **Good (0.65-0.93)**: 5-1 (0.6810)
- **Moderate (0.5-0.65)**: 4-1 (0.5909)

All best parameters have been saved to respective `parameters_detection.json` files.
