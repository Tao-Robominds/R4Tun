# Continuous-tunnel v3 summary (prompt-only detecting docs)

Model: **opus4.6**. Tunnels: `3-1-1`, `3-1-2`, `3-1-3`.
Walk-direction code **not** deployed (GT-free cue accuracy <80% gate).

## mIoU

| tunnel | run1 | v1 hint | v3 hint | Δv3 vs run1 |
|--------|------|---------|---------|-------------|
| 3-1-1 | 0.287 | 0.332 | 0.330 | +0.043 |
| 3-1-2 | 0.237 | 0.222 | 0.242 | +0.005 |
| 3-1-3 | 0.229 | — | 0.229 | +0.000 |

**Mean Δv3 vs run1:** +0.0160 (std 0.0192, n=3)

## Detecting state (GT-free signals, v3 run)

| tunnel | fallback rate | Y spread (px) | midpoint rate |
|--------|---------------|---------------|---------------|
| 3-1-1 | 50% | 1684 | 40% |
| 3-1-2 | 50% | 296 | 30% |
| 3-1-3 | 10% | 514 | 30% |

## Detecting parameter changes (v3 vs run1)

### 3-1-1
- `detecting.binary_threshold`: 120.0 → 127.0
- `detecting.dilation_iterations`: 2.0 → 1.0
- `detecting.hough_threshold_horizontal`: 30.0 → 50.0
- `detecting.hough_threshold_oblique`: 30.0 → 50.0
- `detecting.maxLineGap_horizontal`: 25.0 → 12.0
- `detecting.maxLineGap_oblique`: 50.0 → 40.0

### 3-1-2
- `detecting.binary_threshold`: 120.0 → 127.0
- `detecting.hough_threshold_horizontal`: 30.0 → 50.0
- `detecting.hough_threshold_oblique`: 35.0 → 50.0
- `detecting.hough_threshold_vertical`: 350.0 → 500.0
- `detecting.maxLineGap_horizontal`: 20.0 → 10.0
- `detecting.maxLineGap_oblique`: 55.0 → 40.0
- `detecting.minLineLength_horizontal`: 60.0 → 100.0
- `detecting.minLineLength_oblique`: 80.0 → 100.0

### 3-1-3
- `detecting.hough_threshold_horizontal`: 35.0 → 50.0
- `detecting.hough_threshold_oblique`: 40.0 → 50.0
- `detecting.hough_threshold_vertical`: 450.0 → 500.0
- `detecting.maxLineGap_horizontal`: 18.0 → 10.0
- `detecting.maxLineGap_oblique`: 50.0 → 40.0
- `detecting.minLineLength_horizontal`: 80.0 → 100.0
- `detecting.minLineLength_oblique`: 80.0 → 100.0

## Per-ring handedness (evaluation only — uses GT)

Mirrored = best transform has s=-1, k=0.

### 3-1-1
- k=0 rings: 7; mirrored among those: 2

| pred_ring | k | acc | mirrored |
|-----------|---|-----|----------|
| 0 | 0 | 0.69 | True |
| 1 | 0 | 0.78 | False |
| 2 | 0 | 0.64 | False |
| 3 | 2 | 0.52 | False |
| 4 | 5 | 0.57 | False |
| 5 | 5 | 0.56 | False |
| 6 | 0 | 0.61 | False |
| 7 | 0 | 0.63 | False |
| 8 | 0 | 0.64 | False |
| 9 | 0 | 0.43 | True |

### 3-1-2
- k=0 rings: 10; mirrored among those: 6

| pred_ring | k | acc | mirrored |
|-----------|---|-----|----------|
| 0 | 0 | 0.82 | False |
| 1 | 0 | 0.53 | True |
| 2 | 0 | 0.83 | True |
| 3 | 0 | 0.65 | True |
| 4 | 0 | 0.85 | True |
| 5 | 0 | 0.85 | True |
| 6 | 0 | 0.80 | True |
| 7 | 0 | 0.81 | False |
| 8 | 0 | 0.87 | False |
| 9 | 0 | 0.88 | False |

### 3-1-3
- k=0 rings: 6; mirrored among those: 4

| pred_ring | k | acc | mirrored |
|-----------|---|-----|----------|
| 0 | 0 | 0.77 | False |
| 1 | 1 | 0.53 | False |
| 2 | 1 | 0.64 | False |
| 3 | 1 | 0.52 | False |
| 4 | 0 | 0.75 | True |
| 5 | 1 | 0.55 | False |
| 6 | 0 | 0.79 | True |
| 7 | 0 | 0.84 | True |
| 8 | 0 | 0.62 | True |
| 9 | 0 | 0.84 | False |

## Walk-direction feasibility (design-time)

```
=== 3-1-1 ===
GT orders match ref or reverse: 6/10 rings
ref order: [5, 6, 4, 1, 2, 3]
  ring 0 type=horizontal       k=0 mir=True acc=0.80
  ring 1 type=assume           k=0 mir=False acc=0.52
  ring 2 type=assume           k=1 mir=False acc=0.55
  ring 3 type=assume           k=1 mir=False acc=0.54
  ring 4 type=assume           k=1 mir=False acc=0.56
  ring 5 type=midpoint         k=1 mir=False acc=0.54
  ring 6 type=midpoint         k=1 mir=False acc=0.51
  ring 7 type=midpoint         k=1 mir=False acc=0.50
  ring 8 type=midpoint         k=1 mir=False acc=0.55
  ring 9 type=midpoint         k=1 mir=False acc=0.32
slope rule: 0/0 = 0.00
majority mirrored=True on k=0: 0.50 (2 rings)

=== 3-1-2 ===
GT orders match ref or reverse: 2/10 rings
ref order: [5, 4, 6, 1, 2, 3]
  ring 0 type=default          k=0 mir=False acc=0.80
  ring 1 type=assume           k=0 mir=True acc=0.83
  ring 2 type=negative_slope   k=0 mir=True acc=0.81
  ring 3 type=positive_slope   k=0 mir=True acc=0.74
  ring 4 type=midpoint         k=0 mir=True acc=0.82
  ring 5 type=midpoint         k=0 mir=True acc=0.59
  ring 6 type=midpoint         k=0 mir=True acc=0.86
  ring 7 type=midpoint         k=0 mir=False acc=0.74
  ring 8 type=midpoint         k=0 mir=False acc=0.86
  ring 9 type=midpoint         k=0 mir=False acc=0.84
slope rule: 1/2 = 0.50
majority mirrored=True on k=0: 0.60 (10 rings)

=== 3-1-3 ===
GT orders match ref or reverse: 3/10 rings
ref order: [5, 6, 1, 4, 2, 3]
  ring 0 type=positive_slope   k=0 mir=False acc=0.79
  ring 1 type=positive_slope   k=1 mir=False acc=0.60
  ring 2 type=negative_slope   k=1 mir=False acc=0.56
  ring 3 type=negative_slope   k=1 mir=False acc=0.57
  ring 4 type=assume           k=1 mir=False acc=0.65
  ring 5 type=midpoint         k=1 mir=False acc=0.55
  ring 6 type=negative_slope   k=1 mir=False acc=0.53
  ring 7 type=negative_slope   k=0 mir=True acc=0.85
  ring 8 type=midpoint         k=0 mir=True acc=0.67
  ring 9 type=midpoint         k=0 mir=False acc=0.85
slope rule: 3/6 = 0.50
majority mirrored=True on k=0: 0.50 (4 rings)
```

**Verdict:** slope-sign rule 50%; majority vote 50–60% on k=0 rings.
Below 80% gate → no `walk_direction` code change.
