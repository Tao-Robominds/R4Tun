# Segment Coverage and Wraparound Analysis Report

**Date:** January 23, 2026  
**Analysis Type:** Ground Truth Segment Arrangement Verification  
**Tunnels Analyzed:** 1-4, 2-2, 3-1, 4-1, 5-1

---

## Executive Summary

This report analyzes the angular coverage of tunnel segments across five different tunnel configurations, with a focus on identifying wraparound issues where segments span the 0°/360° boundary in the unfolded depth map. The analysis reveals significant variation in wraparound severity across tunnels, with tunnels 4-1 and 5-1 exhibiting the most severe cases.

### Key Findings

- **Tunnels 1-4, 2-2, 3-1** (6 segments): Moderate wraparound with 2-3 segments spanning boundaries
- **Tunnels 4-1, 5-1** (7 segments): **Severe wraparound** with ALL segments spanning the 360° boundary
- **Critical Issue**: For 7-segment tunnels, ANY `theta_offset` will split multiple segments, making wraparound correction extremely challenging

---

## Segment Coverage by Tunnel

### Tunnel 1-4 (6 segments per ring)

**Segment Order:** K, B1, A1, A2, A3, B2

```
     0°       90°      180°      270°    360°
     |         |         |         |       |
K   |   ########     ########            |
B1  |##################   ###############| ← spans boundary
A1  |          ######################### |
A2  |          ######                    |
A3  |###############             ########| ← spans boundary
B2  |######    ##########################| ← spans boundary
```

**Analysis:**
- **Segments spanning boundary:** B1, A3, B2 (3 out of 6)
- **Coverage gaps:** Present between some segments
- **Wraparound severity:** **Moderate**
- **Impact:** `theta_offset` can help move some segments away from boundaries, but B1, A3, and B2 will always be problematic

---

### Tunnel 2-2 (6 segments per ring)

**Segment Order:** K, B1, A1, A2, A3, B2

```
     0°       90°      180°      270°    360°
     |         |         |         |       |
K   |   ########     ########            |
B1  |##################   ###############| ← spans boundary
A1  |           ######################## |
A2  |           #####                    |
A3  |######## #######             #######| ← spans boundary
B2  |#######   ##########################| ← spans boundary
```

**Analysis:**
- **Segments spanning boundary:** B1, A3, B2 (3 out of 6)
- **Coverage gaps:** Present between some segments
- **Wraparound severity:** **Moderate**
- **Impact:** Similar to Tunnel 1-4, with B1, A3, and B2 consistently crossing boundaries

---

### Tunnel 3-1 (6 segments per ring)

**Segment Order:** K, B1, A1, A2, A3, B2

```
     0°       90°      180°      270°    360°
     |         |         |         |       |
K   |          #######                   |
B1  |############                 #######| ← spans boundary
A1  |           ###################      |
A2  |        ####   #####                |
A3  |################                 ###| ← spans boundary
B2  |              ####################  |
```

**Analysis:**
- **Segments spanning boundary:** B1, A3 (2 out of 6)
- **Coverage gaps:** A2 has an internal gap (unusual pattern)
- **Wraparound severity:** **Moderate** (better than 1-4 and 2-2)
- **Impact:** Only 2 segments span boundaries, but A2's internal gap suggests potential detection issues

---

### Tunnel 4-1 (7 segments per ring)

**Segment Order:** K, B1, A1, A2, A3, A4, B2

#### Visual Explanation

The diagram below shows how segments are arranged around the tunnel's 360° circumference when "unfolded" into a 2D depth map. The left edge (0°) and right edge (360°) represent the **same physical location** - they're connected because the tunnel is circular.

```
     0°       90°      180°      270°    360° (same as 0°)
     |         |         |         |       |
     |<-- Tunnel circumference (360°) -->|
     |                                     |
K   |#######     ##################  ####| ← K-block wraps around
B1  |##############################  ####| ← B1 wraps around
A1  |############################       #| ← A1 wraps around
A2  |##############################  ####| ← A2 wraps around
A3  |##############################  ####| ← A3 wraps around
A4  |############################       #| ← A4 wraps around
B2  |##############################    ##| ← B2 wraps around
     |                                     |
     └─────────────────────────────────────┘
          (This is a continuous circle!)
```

**What "spans boundary" means:**
- When a segment's coverage extends from the **right side** (near 360°) and continues to the **left side** (near 0°), it "spans the boundary"
- In the diagram, `####` on the right edge and `####` on the left edge are **the same segment** split across the image boundary
- This happens because the tunnel is circular, but the depth map is a flat rectangle

#### Why This Is a Problem

**The Issue:**
1. **All 7 segments** cross the 0°/360° boundary in the unfolded image
2. This means **every segment** is split into two parts: one on the left edge and one on the right edge
3. When SAM tries to segment, it sees each segment as **two separate pieces** instead of one continuous segment

**Why It Happens:**
- With 7 segments in a 360° circle, each segment occupies ~51.4° of arc (360° ÷ 7 = 51.4°)
- The K-block (starting point) is positioned such that segments extend across the boundary
- **No matter where you "cut" the circle** to create the flat image, multiple segments will be split

**Analogy:**
Imagine cutting a circular pizza into 7 slices, then trying to lay it flat on a rectangular table. No matter where you make the cut, some slices will be split across the table's edges.

#### Impact on Processing

**Analysis:**
- **Segments spanning boundary:** **ALL 7 segments** (100%)
- **Coverage:** Nearly full 360° coverage with minimal gaps
- **Wraparound severity:** **SEVERE**
- **Impact:** **CRITICAL** - ANY `theta_offset` (rotation of where we "cut" the circle) will split multiple segments. Standard wraparound correction strategies are ineffective.

**Why `theta_offset` doesn't help:**
- `theta_offset` rotates where the 0°/360° cut is made
- But with 7 segments, **every possible cut location** will split multiple segments
- It's like rotating the pizza - you still have slices crossing the table edge

**Root Cause:** With 7 segments distributed around a 360° circumference, each segment occupies approximately 51.4° of arc. Given the segment arrangement and K-block positioning, all segments extend across the 0°/360° boundary in the unfolded image. This is a **geometric inevitability** for 7-segment tunnels with this arrangement.

#### Conceptual Example

Think of it like this:

**Physical Reality (3D tunnel ring):**
```
        K
       / \
      /   \
     /     \
    B1     B2
   /         \
  A1         A4
 /             \
A2─────────────A3
```
All segments form a complete circle (360°)

**Unfolded into 2D depth map:**
```
0°                   180°                  360° (wraps to 0°)
|                     |                     |
|  K  |  B1  |  A1  |  A2  |  A3  |  A4  |  B2  |
|     |      |      |      |      |      |      |
└────────────────────────────────────────────────┘
     ↑                                    ↑
   Same physical location (wraparound!)
```

**The Problem:**
- When we "cut" the circle to make it flat, we create an artificial boundary at 0°/360°
- With 7 segments, each segment is ~51.4° wide
- No matter where we cut, segments will span across this boundary
- In the depth map, a segment that spans the boundary appears as **two disconnected pieces** (left edge + right edge)

---

### Tunnel 5-1 (7 segments per ring)

**Segment Order:** K, B1, A1, A2, A3, A4, B2

```
     0°       90°      180°      270°    360°
     |         |         |         |       |
K   |#######    #################     ###| ← spans boundary
B1  |####################################| ← spans boundary
A1  |##############################   ###| ← spans boundary
A2  |####################################| ← spans boundary
A3  |####################################| ← spans boundary
A4  |####################################| ← spans boundary
B2  |####################################| ← spans boundary
```

**Analysis:**
- **Segments spanning boundary:** **ALL 7 segments** (100%)
- **Coverage:** Full 360° coverage with no gaps
- **Wraparound severity:** **SEVERE** (worst case)
- **Impact:** **CRITICAL** - Complete wraparound with full coverage. Standard `theta_offset` approach is completely ineffective.

**Root Cause:** Similar to Tunnel 4-1, but with even more uniform distribution. The K-block positioning and segment arrangement results in every segment crossing the boundary.

---

## Comparative Analysis

### Wraparound Severity Ranking

| Rank | Tunnel | Segments | Segments Spanning Boundary | Severity | Mitigation Difficulty |
|------|--------|----------|---------------------------|----------|----------------------|
| 1 | 5-1 | 7 | 7 (100%) | **SEVERE** | **Extremely High** |
| 2 | 4-1 | 7 | 7 (100%) | **SEVERE** | **Extremely High** |
| 3 | 1-4 | 6 | 3 (50%) | Moderate | Medium |
| 4 | 2-2 | 6 | 3 (50%) | Moderate | Medium |
| 5 | 3-1 | 6 | 2 (33%) | Moderate | Low-Medium |

### Key Observations

1. **7-segment tunnels (4-1, 5-1) are fundamentally problematic**
   - All segments span boundaries regardless of `theta_offset`
   - Standard wraparound correction cannot solve this issue
   - Requires specialized handling (e.g., per-segment processing, wraparound-aware SAM)

2. **6-segment tunnels have manageable wraparound**
   - Only 2-3 segments span boundaries
   - `theta_offset` can help reduce impact
   - Per-ring alignment may be effective

3. **Segment count is the primary factor**
   - More segments = higher probability of wraparound
   - 7 segments in 360° = ~51.4° per segment (too wide to avoid boundaries)
   - 6 segments in 360° = ~60° per segment (more manageable)

---

## Implications for Pipeline Processing

### Current Wraparound Handling

The codebase includes several wraparound mitigation strategies:

1. **`theta_offset` parameter** (`parameters_unfolding.json`)
   - Shifts the "cut" position of the cylindrical unwrapping
   - Range: 0-120° (from `bo/search_space.py`)
   - **Limitation**: Ineffective for 7-segment tunnels where ALL segments span boundaries

2. **Per-ring alignment** (`per_ring_alignment`)
   - Aligns K-blocks to a consistent angular position across rings
   - Target position: typically 180° (`k_target_position`)
   - **Limitation**: Still problematic for 7-segment tunnels

3. **Wraparound-aware SAM** (`4-2_sam_wraparound.py`)
   - Processes segments individually at their ground-truth positions
   - Requires `all_segments.csv` with GT segment positions
   - **Status**: Available but may need enhancement for 7-segment cases

### Recommendations

#### For 6-Segment Tunnels (1-4, 2-2, 3-1)

✅ **Recommended Approach:**
1. Use `theta_offset` optimization to minimize boundary crossings
2. Enable per-ring alignment for consistent K-block positioning
3. Standard SAM processing should work with minor adjustments

#### For 7-Segment Tunnels (4-1, 5-1)

⚠️ **Critical - Requires Special Handling:**

1. **DO NOT rely on `theta_offset` alone** - it cannot solve the fundamental problem
2. **Use wraparound-aware SAM** (`4-2_sam_wraparound.py`) with GT segment positions
3. **Consider per-segment processing** - process each segment individually at its known position
4. **Alternative approach**: Use pattern-based detection to identify segment boundaries before SAM processing
5. **Future enhancement**: Implement boundary-aware template masks that handle split segments

### Code References

- **Wraparound handling**: `p4tun/1_unfolding.py` (lines 1036-1047)
- **Wraparound-aware SAM**: `sam4tun/4-2_sam_wraparound.py`
- **Search space**: `p4tun/bo/search_space.py` (line 37: `unfold_theta_offset` range 0-120°)
- **Pattern discovery**: `agents/detecting/pattern_discovery.py` (for GT-free segment detection)

---

## Technical Details

### Angular Distribution

For a tunnel with circumference C and N segments:

- **Average segment arc length**: C / N
- **For 5.5m diameter tunnel** (C ≈ 17.28m): ~2.88m per segment (6 segments) or ~2.47m per segment (7 segments)
- **For 7.5m diameter tunnel** (C ≈ 23.56m): ~3.93m per segment (6 segments) or ~3.37m per segment (7 segments)

### Boundary Crossing Probability

The probability of a segment crossing the 0°/360° boundary depends on:
1. **Segment width** (angular extent)
2. **K-block position** (starting angle)
3. **Segment arrangement** (staggered vs. continuous)

For 7-segment tunnels:
- Each segment occupies ~51.4° of arc
- If K-block starts anywhere except the exact center of a segment, multiple segments will span boundaries
- **Conclusion**: Boundary crossing is nearly unavoidable

---

## Validation Checklist

- [x] Segment counts verified (6 for 1-4/2-2/3-1, 7 for 4-1/5-1)
- [x] Boundary-spanning segments identified
- [x] Coverage gaps noted where present
- [x] Wraparound severity classified
- [x] Code references identified
- [x] Recommendations provided

---

## Conclusion

The segment coverage analysis confirms that:

1. **7-segment tunnels (4-1, 5-1) have severe wraparound issues** that cannot be solved with standard `theta_offset` approaches
2. **6-segment tunnels (1-4, 2-2, 3-1) have manageable wraparound** that can be mitigated with existing techniques
3. **Specialized processing is required** for 7-segment tunnels, likely involving:
   - Individual segment processing at GT positions
   - Wraparound-aware template masks
   - Pattern-based boundary detection

The current codebase includes infrastructure for handling wraparound (`4-2_sam_wraparound.py`), but may need enhancement to fully address the severe cases in tunnels 4-1 and 5-1.

---

## Subsection Selection Strategy: Avoiding Wraparound Through Strategic Ring Selection

### The Problem

When creating experimental subsections from full tunnels, the selected ring range can significantly impact wraparound severity. If subsections are selected without considering K-block alignment and segment positioning, wraparound issues may be inadvertently introduced or exacerbated.

### Can Subsection Selection Avoid Wraparound?

**Short Answer:** 
- **For 6-segment tunnels**: **YES** - Strategic selection can significantly reduce wraparound
- **For 7-segment tunnels**: **PARTIALLY** - Can reduce severity but not eliminate it completely

### Selection Criteria for Minimizing Wraparound

#### 1. K-Block Position Consistency (Primary Criterion)

**Objective:** Select rings where K-blocks are in similar angular positions

**Method:**
1. Compute K-block center theta for each ring (using `compute_per_ring_k_centers()`)
2. Calculate variance/std deviation of K-block positions
3. Select contiguous ring ranges with **low variance** (< 15° standard deviation)

**Implementation:**
```python
# Pseudo-code for ring selection
k_centers = compute_per_ring_k_centers(df, cylindrical_coords, diameter)
k_positions = list(k_centers.values())

# Find contiguous ranges with low variance
for start_ring in range(len(k_positions)):
    for end_ring in range(start_ring + 5, len(k_positions)):  # Min 5 rings
        subset = k_positions[start_ring:end_ring]
        variance = np.var(subset)
        if variance < 225:  # < 15° std dev
            # Good candidate subsection
            candidate_ranges.append((start_ring, end_ring, variance))
```

**Expected Benefit:**
- **6-segment tunnels**: Can reduce boundary-spanning segments from 3 to 1-2
- **7-segment tunnels**: Can reduce severity but all segments may still span boundaries

#### 2. K-Block Position Away from Boundaries (Secondary Criterion)

**Objective:** Select rings where K-blocks are positioned away from 0°/360° boundary

**Method:**
1. Identify rings where K-block center is in "safe zone" (e.g., 45° to 315°)
2. Avoid rings where K-block is near 0° or 360° (±30°)

**Safe Zones:**
- **Optimal**: K-block between 90° and 270° (middle half of circle)
- **Acceptable**: K-block between 45° and 315° (avoids immediate boundary)
- **Avoid**: K-block between 330° and 30° (too close to boundary)

**Implementation:**
```python
safe_rings = []
for ring, k_center in k_centers.items():
    # Normalize to [0, 360)
    k_normalized = k_center % 360
    
    # Check if in safe zone (45° to 315°)
    if 45 <= k_normalized <= 315:
        safe_rings.append(ring)
    # Or check if NOT in danger zone (330° to 30°)
    elif not (330 <= k_normalized or k_normalized <= 30):
        safe_rings.append(ring)
```

#### 3. Contiguous Ring Selection (Practical Criterion)

**Objective:** Select contiguous ranges to maintain spatial coherence

**Why Important:**
- Non-contiguous rings may have different K-block alignments
- Contiguous ranges preserve ring-to-ring relationships
- Easier to process and validate

**Minimum Ring Count:**
- **For experiments**: At least 5-10 rings for statistical validity
- **For evaluation**: At least 20-30 rings for robust metrics
- **For training**: 50+ rings preferred

#### 4. Segment Count Consideration

**For 6-Segment Tunnels:**
- Subsection selection can be **highly effective**
- Target: Select rings where K-block variance < 10°
- Expected: Reduce boundary-spanning segments to 0-1

**For 7-Segment Tunnels:**
- Subsection selection has **limited effectiveness**
- Even with perfect K-block alignment, segments may still span boundaries
- Target: Minimize variance, but expect some wraparound
- Expected: May reduce from 7 to 5-6 boundary-spanning segments

### Practical Selection Algorithm

```python
def select_optimal_subsection(
    df: pd.DataFrame,
    cylindrical_coords: np.ndarray,
    diameter: float,
    min_rings: int = 10,
    max_variance: float = 225.0,  # 15° std dev
    k_safe_zone: Tuple[float, float] = (45.0, 315.0)
) -> Tuple[int, int]:
    """
    Select optimal ring range to minimize wraparound.
    
    Returns:
        (start_ring, end_ring) - optimal contiguous range
    """
    # Compute K-block positions
    k_centers = compute_per_ring_k_centers(df, cylindrical_coords, diameter)
    rings = sorted(k_centers.keys())
    k_positions = [k_centers[r] for r in rings]
    
    best_range = None
    best_score = float('inf')
    
    # Try all contiguous ranges
    for start_idx in range(len(rings) - min_rings + 1):
        for end_idx in range(start_idx + min_rings, len(rings) + 1):
            subset_rings = rings[start_idx:end_idx]
            subset_k = k_positions[start_idx:end_idx]
            
            # Criterion 1: Low variance
            variance = np.var(subset_k)
            if variance > max_variance:
                continue
            
            # Criterion 2: K-blocks in safe zone
            in_safe_zone = sum(
                1 for k in subset_k 
                if k_safe_zone[0] <= (k % 360) <= k_safe_zone[1]
            )
            safe_ratio = in_safe_zone / len(subset_k)
            
            # Criterion 3: Prefer longer ranges
            length_bonus = len(subset_rings) * 0.1
            
            # Score: lower is better (variance + unsafe penalty - length bonus)
            score = variance + (1 - safe_ratio) * 1000 - length_bonus
            
            if score < best_score:
                best_score = score
                best_range = (subset_rings[0], subset_rings[-1])
    
    return best_range
```

### Recommendations by Tunnel Type

#### For 6-Segment Tunnels (1-4, 2-2, 3-1)

✅ **Recommended Approach:**
1. **Analyze full tunnel first** - Compute K-block positions for all rings
2. **Identify low-variance ranges** - Look for 10+ contiguous rings with < 10° K-block variance
3. **Prefer middle positions** - Select ranges where K-blocks are between 90°-270°
4. **Validate selection** - Check that selected subsection has fewer boundary-spanning segments

**Expected Outcome:**
- Can reduce wraparound from "moderate" to "minimal"
- May eliminate boundary-spanning segments entirely
- Standard `theta_offset` becomes more effective

#### For 7-Segment Tunnels (4-1, 5-1)

⚠️ **Limited Effectiveness:**
1. **Still analyze full tunnel** - Compute K-block positions
2. **Select best available range** - Choose lowest-variance contiguous range
3. **Accept some wraparound** - Even optimal selection may have 5-6 boundary-spanning segments
4. **Use specialized processing** - Combine with wraparound-aware SAM

**Expected Outcome:**
- Can reduce wraparound severity from "severe" to "moderate"
- May reduce boundary-spanning segments from 7 to 5-6
- Still requires wraparound-aware processing

### Retrospective Analysis: Could Current Datasets Be Improved?

#### Tunnel 1-4, 2-2, 3-1
- **Potential improvement**: If selected based on K-block alignment, could reduce wraparound
- **Action**: Analyze full tunnel to identify better subsection ranges
- **Benefit**: Moderate wraparound → Minimal wraparound

#### Tunnel 4-1, 5-1
- **Limited improvement**: Even optimal selection won't eliminate wraparound
- **Action**: Still beneficial to select low-variance ranges
- **Benefit**: Severe wraparound → Moderate wraparound (still needs special handling)

### Implementation in Pipeline

To implement subsection selection in the current pipeline:

1. **Add ring filtering to `1_unfolding.py`**:
   ```python
   # After computing cylindrical coordinates
   if ring_range is not None:
       start_ring, end_ring = ring_range
       df = df[(df['ring'] >= start_ring) & (df['ring'] <= end_ring)]
   ```

2. **Create selection utility script**:
   ```python
   # scripts/select_optimal_subsection.py
   # Analyzes full tunnel and recommends optimal ring range
   ```

3. **Update parameter files**:
   ```json
   {
     "subsection": {
       "enabled": true,
       "start_ring": 10,
       "end_ring": 50,
       "selection_criteria": "low_variance_k_blocks"
     }
   }
   ```

### Trade-offs

**Benefits:**
- ✅ Reduced wraparound severity
- ✅ Better compatibility with standard processing
- ✅ Improved segmentation quality
- ✅ Easier parameter tuning

**Costs:**
- ❌ Reduced dataset size (fewer rings)
- ❌ May exclude interesting edge cases
- ❌ Requires full tunnel analysis first
- ❌ Additional preprocessing step

**Recommendation:**
- **For new experiments**: Always analyze full tunnel and select optimal subsections
- **For existing datasets**: Consider re-selecting if wraparound is problematic
- **For 7-segment tunnels**: Combine subsection selection with wraparound-aware processing

---

**Report Generated:** January 23, 2026  
**Next Steps:** 
- Test wraparound-aware SAM on tunnels 4-1 and 5-1
- Evaluate per-segment processing performance
- Consider boundary-aware template mask enhancements
- **NEW**: Implement subsection selection utility for future experiments
- **NEW**: Retrospectively analyze full tunnels to identify optimal subsections
