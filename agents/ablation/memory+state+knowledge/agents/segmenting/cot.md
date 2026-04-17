## Chain of Thought Instructions for Segmenting Parameter Recommendations

Follow this structured analysis process when evaluating tunnel characteristics for SAM segmenting parameter recommendations:

### 0. CONSERVATIVE DEFAULT PRINCIPLE (read first, applies to every parameter)

When uncertain whether a parameter should deviate from the SAM4Tun default,
keep the default. Only change a parameter when you have clear evidence from
the tunnel characteristics that the default would cause a specific problem.

**Physical constants** — set from the tunnel type (always justified):
- segment_per_ring: 6 (1-\*, 2-\*, 3-\*) or 7 (4-\*, 5-\*)
- segment_order: match the tunnel's actual segment layout

**Template and prompt geometry** — keep SAM4Tun defaults unless the state
context (e.g., depth map dimensions, detection output) shows concrete evidence
that the default template geometry produces poor masks. Scaling template
dimensions proportionally with diameter is justified; arbitrary changes to
spacing factors or ring counts are not.

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
