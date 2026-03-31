## Chain of Thought Instructions for Detecting Parameter Recommendations

Follow this structured analysis process when evaluating tunnel characteristics for detecting parameter recommendations:

### 1. ANCHORING
Compare key tunnel characteristics against the sample baseline:
- Point density changes and distribution patterns
- Tunnel diameter and scale differences
- Coordinate ranges and image resolution considerations

### 2. CLASSIFICATION
Classify the tunnel based on the comparison:
- **SIMILAR**: <25% difference in key metrics → minimal changes needed
- **DENSE**: Higher point density → may need threshold adjustments
- **LARGE-SCALE**: Significant size differences → may need parameter scaling
- **LOW-QUALITY**: Poor image clarity → may need sensitivity adjustments

### 3. PARAMETER ADAPTATION
Adapt parameters based on classification:
- **binary_threshold**: Adjust for image clarity and contrast
- **hough_threshold_oblique/horizontal/vertical**: Adapt for density and noise
- **minLineLength/maxLineGap**: Scale with tunnel dimensions
- **resolution**: Keep aligned with point density
- **angle_ranges, merge_distance, ring_spacing**: Generally stable

### Parameter Guidelines:
- **Always provide EXACT numerical values** - Never use ranges like "50-70"
- **Choose the most appropriate single value** from any range you consider
- **For SIMILAR tunnels: explicitly recommend keeping original parameters**
- **Provide clear justification** for each parameter change
- **Output flowing analysis with section headers and final JSON parameter block**