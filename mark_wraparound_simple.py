"""
Simple script to mark wraparound issues on depth map.
Shows that left and right edges are connected (same physical location).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os

tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "4-1"
depth_map_path = f"data/{tunnel_id}/depth_map.png"

if not os.path.exists(depth_map_path):
    print(f"Error: {depth_map_path} not found")
    sys.exit(1)

# Load image
img = mpimg.imread(depth_map_path)
if img is None:
    print(f"Error: Could not load {depth_map_path}")
    sys.exit(1)

height, width = img.shape[:2]

# Create figure
fig, ax = plt.subplots(figsize=(20, 12))
ax.imshow(img, cmap='gray')

# Mark left edge (0° boundary)
ax.axvline(x=0, color='red', linewidth=3, linestyle='--', alpha=0.7, label='Left Edge (0°)')
ax.text(10, height - 50, 'LEFT EDGE (0°)\nThis is the START', 
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
        color='white', fontsize=14, fontweight='bold')

# Mark right edge (360° boundary)  
ax.axvline(x=width-1, color='red', linewidth=3, linestyle='--', alpha=0.7, label='Right Edge (360°)')
ax.text(width - 200, height - 50, 'RIGHT EDGE (360°)\nSame as LEFT EDGE!', 
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
        color='white', fontsize=14, fontweight='bold')

# Draw arrows showing wraparound connection
arrow_props = dict(arrowstyle='<->', lw=4, color='yellow', alpha=0.8)
# Top arrow
ax.annotate('', xy=(width-1, 100), xytext=(0, 100),
            arrowprops=arrow_props)
ax.text(width/2, 80, 'WRAPAROUND:\nLeft and Right are CONNECTED!', 
        ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        fontsize=16, fontweight='bold')

# Middle arrow
ax.annotate('', xy=(width-1, height/2), xytext=(0, height/2),
            arrowprops=arrow_props)

# Bottom arrow
ax.annotate('', xy=(width-1, height-100), xytext=(0, height-100),
            arrowprops=arrow_props)

# Add explanation box
explanation = """
WRAPAROUND EXPLANATION:

1. The tunnel is CIRCULAR (360°)
2. When unfolded into a flat image:
   - LEFT edge (x=0) = 0° 
   - RIGHT edge (x=width) = 360°
   - These are the SAME physical location!

3. Segments that span the boundary appear as:
   - Part on LEFT edge
   - Part on RIGHT edge
   - But they're ONE continuous segment!

4. For Tunnel 4-1: ALL 7 segments span this boundary
"""

ax.text(50, 50, explanation, 
        bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.9),
        fontsize=12, fontweight='bold', verticalalignment='top')

ax.set_title(f'Tunnel {tunnel_id} Depth Map - Wraparound Visualization', 
             fontsize=18, fontweight='bold')
ax.axis('off')

# Save
output_path = f"data/{tunnel_id}/depth_map_wraparound_marked.png"
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved annotated image to: {output_path}")
plt.show()
