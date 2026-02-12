# Preprocessing Tuning Experience - 1-4

**Timestamp**: 2026-02-03 23:34:51
**Outcome**: IMPROVED

---

## Parameter Changes

### Old Parameters
```json
{
  "ring_spacing": 1.2,
  "tunnel_diameter": 5.04,
  "radius_min": 2.47,
  "radius_max": 2.57,
  "gradient_threshold": 0.2,
  "target_distances": [
    0.0580590233,
    0.0387060155,
    0.0193530078
  ],
  "curvature_neighbors": 20,
  "depth_map_resolution": 0.0019353008,
  "interpolation_window": 9
}
```

### New Parameters
```json
{
  "ring_spacing": 1.2,
  "tunnel_diameter": 5.54,
  "radius_min": 2.72,
  "radius_max": 2.82,
  "gradient_threshold": 0.2,
  "target_distances": [
    0.0580590233,
    0.0387060155,
    0.0193530078
  ],
  "curvature_neighbors": 20,
  "depth_map_resolution": 0.0019353008,
  "interpolation_window": 9
}
```

### Changed Parameters
- **tunnel_diameter**: 5.04 → 5.54
- **radius_min**: 2.47 → 2.72
- **radius_max**: 2.57 → 2.82

---

## Intrinsics Comparison

### Old Intrinsics
```json
{
  "pre_theta_coverage_pct": 99.9944625628869,
  "pre_point_retention_pct": 1.544966099902471,
  "pre_depth_map_valid_pixels": 3736,
  "pre_ready_for_detection": false,
  "pre_guardrail_violations": [
    "point_retention=1.5% < 70.0%",
    "depth_map_valid_pixels=3736 < 8000 (too sparse)"
  ]
}
```

### New Intrinsics
```json
{
  "pre_theta_coverage_pct": 99.9944625628869,
  "pre_point_retention_pct": 72.92060796916677,
  "pre_depth_map_valid_pixels": 15198,
  "pre_ready_for_detection": true,
  "pre_guardrail_violations": []
}
```

---

## Analysis

### Metrics Assessment

**Improved:**
- pre_point_retention_pct: 1.5450 → 72.9206
- pre_depth_map_valid_pixels: 3736.0000 → 15198.0000

**Unchanged:**
- pre_theta_coverage_pct: 99.9945 → 99.9945

### Overall Assessment

**IMPROVED**

---

## Lessons Learned

✅ **Improvement achieved.** Key insights:
- The parameter adjustment was effective
- Consider this configuration as a new baseline
