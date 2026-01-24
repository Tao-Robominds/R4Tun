# Wraparound Visualization Guide for Tunnel 4-1

## Understanding the Wraparound Issue

When you look at the `depth_map.png` for Tunnel 4-1, here's what to look for to understand the wraparound issue:

### Key Visual Indicators

1. **The Left and Right Edges Are Connected**
   - The **leftmost edge** (0°) and **rightmost edge** (360°) represent the **same physical location** in the tunnel
   - In the real tunnel, these edges touch each other (it's a circle!)

2. **Continuous Patterns Across Edges**
   - Look at features (bright spots, patterns, depth variations) near the **left edge**
   - Now look at the **right edge** of the same horizontal band
   - You should see that patterns **continue** from right to left - they "wrap around"

3. **No Clean Break Points**
   - Unlike a tunnel where segments might be cleanly separated, here the data flows continuously
   - There's no vertical "empty zone" where you could cleanly cut without splitting a segment

## How to Manually Mark Wraparound

### Method 1: Using Image Viewer

1. Open `data/4-1/depth_map.png` in any image viewer
2. Draw these annotations mentally or with an image editor:

   ```
   ┌─────────────────────────────────────────────────┐
   │ ← 0° (LEFT EDGE)                   360° (RIGHT)│
   │                                                 │
   │ [RED LINE] ←──────────────────────→ [RED LINE]│
   │     ↑                                        ↑  │
   │  Same physical location!                     │
   │                                                 │
   │ [YELLOW ARROW] ←───────────────→ [YELLOW ARROW]│
   │     ↑ Features wrap around!                   ↑│
   │                                                 │
   └─────────────────────────────────────────────────┘
   ```

3. **Key observations:**
   - Red lines mark the 0°/360° boundary (left and right edges)
   - Yellow arrows show how features "wrap around" from right edge back to left edge
   - Notice how patterns on the right edge match patterns on the left edge

### Method 2: Using Python (Simple Version)

If you have Python with PIL/Pillow installed:

```python
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load image
img = Image.open('data/4-1/depth_map.png')
width, height = img.size

# Create a copy for annotation
annotated = img.copy()
draw = ImageDraw.Draw(annotated)

# Draw red boundary lines
# Left edge (0°)
draw.line([(0, 0), (0, height)], fill='red', width=5)
# Right edge (360°)
draw.line([(width-1, 0), (width-1, height)], fill='red', width=5)

# Draw yellow arrows showing wraparound (simplified as lines)
for y in [height*0.2, height*0.5, height*0.8]:
    # Arrow from right to left
    draw.line([(width-50, y), (50, y)], fill='yellow', width=3)
    # Arrow head (simple triangle)
    draw.polygon([(50, y), (70, y-10), (70, y+10)], fill='yellow')
    draw.polygon([(width-50, y), (width-70, y-10), (width-70, y+10)], fill='yellow')

# Add text labels
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
except:
    font = ImageFont.load_default()

draw.text((20, 20), "0° (LEFT EDGE)", fill='red', font=font)
draw.text((width-200, 20), "360° (RIGHT EDGE)\nSame as 0°!", fill='red', font=font)
draw.text((width//2 - 150, height//2), "WRAPAROUND:\nLeft and Right connect!", 
          fill='yellow', font=font)

# Save annotated image
annotated.save('data/4-1/depth_map_wraparound_marked.png')
print("Annotated image saved!")
```

### Method 3: Visual Comparison

1. **Extract edge strips:**
   - Take a 50-pixel wide strip from the **left edge** (columns 0-50)
   - Take a 50-pixel wide strip from the **right edge** (last 50 columns)
   - Place them side by side

2. **Compare the patterns:**
   - If you see similar patterns, colors, and features in both strips, that's wraparound!
   - The features on the right edge are the continuation of features on the left edge

## What the Wraparound Means

### For Tunnel 4-1 Specifically:

1. **ALL 7 segments** cross the 0°/360° boundary
2. Each segment appears as **two disconnected pieces** in the image:
   - One piece on the left edge
   - One piece on the right edge
3. **No theta_offset can fix this** because every possible cut location splits segments

### Visual Example:

```
Physical Tunnel (3D, circular):
        K
       / \
      /   \
     /     \
    B1     B2
   /         \
  A1         A4
 /             \
A2─────────────A3
    (complete circle)

Unfolded Depth Map (2D, flat):
┌─────────────────────────────────────┐
│ K  | B1 | A1 | A2 | A3 | A4 | B2  │
│    |    |    |    |    |    |     │
│ ←──┴────┴────┴────┴────┴────┴──────→│
│                                    │
│ Notice: K-block is split!         │
│ Left edge has part of K           │
│ Right edge has part of K           │
│ They're the SAME segment!          │
└─────────────────────────────────────┘
```

## Quick Check: Is This Wraparound?

Ask yourself:
1. ✅ Do features on the left edge look like they continue on the right edge?
2. ✅ Is there continuous data from left to right with no clear breaks?
3. ✅ Would "wrapping" the image into a cylinder make the edges match?

If **YES** to all three**, you have wraparound!

## Next Steps

Once you understand the wraparound visually:
- See `SEGMENT_COVERAGE_REPORT.md` for detailed analysis
- Use wraparound-aware processing (see `sam4tun/4-2_sam_wraparound.py`)
- Consider subsection selection to minimize wraparound (see report section on subsection selection)
