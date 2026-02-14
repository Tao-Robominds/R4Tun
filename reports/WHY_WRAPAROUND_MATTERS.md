# Why Wraparound Matters During Segmentation (Not Just Reprojection)

## Your Question

> "Why can't we just do normal segmentation first, and then handle the wraparound during reprojection?"

This is a great question! The issue is that **the segmentation itself fails** when segments span the boundary. It's not just about reprojection - SAM doesn't correctly segment split segments in the first place.

---

## The Problem: SAM Operates on 2D Image, Not 3D Space

### Key Insight

SAM (Segment Anything Model) works on the **2D depth map image**. It doesn't know the image represents a circular tunnel. To SAM, the left edge and right edge are just **two separate, disconnected edges** of a flat image.

---

## Concrete Problems During Segmentation

### Problem 1: Template Masks Fail

**What template masks do:**
- They define a rectangular/polygonal region where the segment should be
- Example: K-block template is ~625mm wide, centered at the detected K position

**What happens with wraparound:**

```
Normal case (no wraparound):
┌─────────────────────────────────────┐
│  [K-block template mask]            │
│  ┌─────────────┐                    │
│  │   K-block   │                    │
│  └─────────────┘                    │
│                                     │
└─────────────────────────────────────┘
Template mask covers the entire K-block ✓

Wraparound case (Tunnel 4-1):
┌─────────────────────────────────────┐
│┌───┐                        ┌──────┐│
││K  │                        │  K   ││
│└───┘                        └──────┘│
│  ↑                              ↑   │
│Left part                    Right part│
│                                     │
└─────────────────────────────────────┘
Template mask can only cover ONE part ✗
```

**Code example from `generate_template_mask()`:**
```python
# Template mask is centered at (x, y) with width w
vertices = [[x-w, y-h], [x-w, y+h], [x+w, y+h], [x+w, y-h]]
```

If the K-block center `x` is near the right edge, the template mask `[x-w, x+w]` will:
- Cover the right part of K-block ✓
- **Miss the left part entirely** ✗ (it's on the opposite edge!)

**Result:** SAM only segments half the K-block.

---

### Problem 2: Crop Regions Fail

**What crop regions do:**
- The code crops a region around each segment for SAM to process
- Example: Crop width = `segment_width + padding` (~1200 + 150 = 1350 pixels)

**What happens with wraparound:**

```python
# From crop_image_and_mask_logits()
# Crop centered at (cx, cy) with width/height
crop_x_start = cx - crop_width/2
crop_x_end = cx + crop_width/2
```

**Normal case:**
```
Segment center at x=1000
Crop: [1000-675, 1000+675] = [325, 1675]
┌─────────────────────────────────────┐
│     [──────────Crop──────────]     │
│          [Segment]                  │
└─────────────────────────────────────┘
Crop includes entire segment ✓
```

**Wraparound case (segment spans boundary):**
```
Segment center at x=3100 (near right edge, width=3256)
Crop: [3100-675, 3100+675] = [2425, 3775]

┌─────────────────────────────────────┐
│                    [Crop───]        │
│                         [Seg]       │
│[Seg]                                │
│↑                                    │
│Left part (at x=100) is OUTSIDE crop!│
└─────────────────────────────────────┘
```

**Result:** The crop only includes the right part. The left part is completely outside the crop region, so SAM never sees it!

---

### Problem 3: Prompt Points Fail

**What prompt points do:**
- They tell SAM "segment this region" by providing example points
- Points are placed around the detected K-block position

**What happens with wraparound:**

```python
# From generate_prompt_points_k()
# Points are placed relative to K-block center (x, y)
points = [
    [x-outer, y],  # Left side
    [x, y],        # Center
    [x+outer, y]   # Right side
]
```

**Normal case:**
```
K-block at x=1000
Points: [325, 1000, 1675]
┌─────────────────────────────────────┐
│  •     •     •                      │
│  [───K-block───]                    │
│  ↑     ↑     ↑                      │
│  All points inside K-block ✓        │
└─────────────────────────────────────┘
```

**Wraparound case:**
```
K-block center at x=3100 (near right edge)
Points: [2400, 3100, 3775] → but image width is only 3256!

┌─────────────────────────────────────┐
│•                                    •│
│  [K]                    [K]          │
│  ↑                      ↑            │
│Left part              Right part     │
│(no prompts!)         (has prompts)   │
└─────────────────────────────────────┘
```

**Result:** 
- Prompt points only cover the right part
- Left part has **no prompt points**
- SAM doesn't know to segment the left part!

---

### Problem 4: Visual Discontinuity

**What SAM sees:**

```
Normal segment (continuous):
┌─────────────────────────────────────┐
│  ████████████████████████████       │
│  ████████████████████████████       │
│  ████████████████████████████       │
│  (one continuous region)             │
└─────────────────────────────────────┘
SAM: "This is one object" ✓

Wraparound segment (split):
┌─────────────────────────────────────┐
│███                                  │
│███                                  │
│                                     │
│                                  ███│
│                                  ███│
│  (two disconnected regions)          │
└─────────────────────────────────────┘
SAM: "These are TWO separate objects" ✗
```

**Result:** SAM segments them as different classes or misses one entirely.

---

## Real Example from Code

Looking at `p4tun/4-2_sam.py`, the `process_row()` function:

```python
# For each segment, it:
1. Calculates crop region: [cx - delta_x, cx + delta_x]
2. Crops image to that region
3. Generates template mask in that crop
4. Generates prompt points in that crop
5. Runs SAM on the crop
```

**If segment spans boundary:**
- Step 1: Crop region `[cx - delta_x, cx + delta_x]` only includes one edge
- Step 2: Cropped image only shows one part of segment
- Step 3: Template mask only covers one part
- Step 4: Prompt points only on one part
- Step 5: SAM only segments one part ✗

**The other part is completely invisible to SAM!**

---

## Why Reprojection Can't Fix This

You might think: "Can't we just merge the left and right parts during reprojection?"

**The problem:**
1. **SAM never segmented the left part** - it's missing from the segmentation results
2. **Even if both parts exist**, they might be labeled as **different classes**:
   - Left part: labeled as "Background" or "A1"
   - Right part: labeled as "K-block"
   - How do you know they're the same segment?

3. **No way to identify matching parts:**
   - Which left-edge region corresponds to which right-edge region?
   - You'd need to know the segment arrangement beforehand (which defeats the purpose)

---

## The Solution: Wraparound-Aware Segmentation

This is why `4-2_sam_wraparound.py` exists! It:

1. **Uses ground truth segment positions** (`all_segments.csv`)
2. **Processes each segment individually** at its known position
3. **Handles boundary-crossing explicitly:**
   - If segment spans boundary, crop includes both edges
   - Template mask accounts for wraparound
   - Prompt points placed on both sides

**Key difference:**
```python
# Normal processing (fails with wraparound):
crop = image[cx-delta:cx+delta]  # Only one edge

# Wraparound-aware processing:
if segment_spans_boundary:
    # Crop includes both edges
    left_part = image[0:cx+delta]
    right_part = image[cx-delta:width]
    crop = np.concatenate([right_part, left_part], axis=1)
```

---

## Summary

| Issue | Normal Segmentation | Wraparound-Aware |
|-------|---------------------|------------------|
| **Template mask** | Only covers one edge | Covers both edges |
| **Crop region** | Misses other edge | Includes both edges |
| **Prompt points** | Only on one side | On both sides |
| **Visual continuity** | SAM sees two objects | SAM sees one object |
| **Result** | ✗ Incomplete/mislabeled | ✓ Complete segmentation |

**Bottom line:** The segmentation step itself fails because SAM operates on a 2D image and doesn't understand the circular nature. You can't fix it later during reprojection because the segmentation is already wrong.

---

## Visual Summary

```
┌─────────────────────────────────────────────────┐
│  NORMAL SEGMENTATION (fails with wraparound)   │
├─────────────────────────────────────────────────┤
│                                                 │
│  Segment K spans boundary:                      │
│  ┌───┐                              ┌──────┐  │
│  │K  │                              │  K   │  │
│  └───┘                              └──────┘  │
│   ↑                                    ↑      │
│  Left part (x=100)          Right part (x=3100)│
│                                                 │
│  SAM processing:                               │
│  1. Crop at x=3100: [2425, 3775]               │
│     → Only includes right part                 │
│  2. Template mask: centered at x=3100          │
│     → Only covers right part                   │
│  3. Prompt points: around x=3100               │
│     → Only on right part                       │
│  4. SAM result:                                │
│     → Only segments right part ✗               │
│     → Left part is MISSING                     │
│                                                 │
│  Reprojection:                                  │
│  → Can't fix it! Left part was never segmented!│
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  WRAPAROUND-AWARE SEGMENTATION (works!)        │
├─────────────────────────────────────────────────┤
│                                                 │
│  Segment K spans boundary:                     │
│  ┌───┐                              ┌──────┐  │
│  │K  │                              │  K   │  │
│  └───┘                              └──────┘  │
│                                                 │
│  SAM processing:                               │
│  1. Detect wraparound: segment crosses boundary│
│  2. Crop: [right_edge, left_edge] concatenated│
│     → Includes BOTH parts                      │
│  3. Template mask: accounts for wraparound     │
│     → Covers BOTH parts                        │
│  4. Prompt points: on both edges                │
│     → Guides SAM to both parts                  │
│  5. SAM result:                                 │
│     → Segments BOTH parts ✓                    │
│                                                 │
│  Reprojection:                                  │
│  → Works correctly! Both parts are segmented   │
└─────────────────────────────────────────────────┘
```
