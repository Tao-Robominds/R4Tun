## Chain of Thought Instructions for Segmenting Parameter Recommendations

Follow this structured analysis process when evaluating tunnel characteristics for SAM segmenting parameter recommendations:

### 1. ANCHORING
Compare key tunnel characteristics against the sample baseline:
- Enhanced point cloud density and distribution
- Ring structure and segment count requirements
- Surface geometry complexity and segmentation challenges

### 2. CLASSIFICATION
Classify the tunnel based on the comparison:
- **SIMILAR**: <25% difference in key metrics → minimal changes needed
- **HIGH-DENSITY**: Dense enhancement results → may need finer segmentation
- **COMPLEX-GEOMETRY**: Irregular surface features → may need robust settings
- **LARGE-SCALE**: Different tunnel dimensions → may need parameter scaling

### 3. PARAMETER ADAPTATION
Adapt parameters based on classification:
- **segment_per_ring**: Scale with tunnel complexity and ring structure
- **segment_width/height**: Adapt to tunnel dimensions and point density
- **angle**: Adjust based on surface geometry requirements
- **ring_spacing**: Generally stable unless extreme scale differences

### Parameter Guidelines:
- **Always provide EXACT numerical values** - Never use ranges like "4-8"
- **Choose the most appropriate single value** from any range you consider
- **For SIMILAR tunnels: explicitly recommend keeping original parameters**
- **Provide clear justification** for each parameter change
- **Output flowing analysis with section headers and final JSON parameter block**
