#!/usr/bin/env python3
"""
Create Separate Ablation Study Visualizations

Generates individual HTML files for each ablation approach,
showing all 5 tunnels in a row with density-proportional sampling.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def load_approach_data(approach_path):
    """Load final segmentation data for all 5 tunnels for one approach"""
    tunnel_ids = ['1-4', '2-2', '3-1', '4-1', '5-1']
    
    tunnel_data = {}
    
    for tunnel_id in tunnel_ids:
        final_file = f"{approach_path}/{tunnel_id}/final.csv"
        if os.path.exists(final_file):
            df = pd.read_csv(final_file)
            tunnel_data[tunnel_id] = df
            print(f"  ✅ Loaded tunnel {tunnel_id}: {len(df):,} points")
        else:
            print(f"  ❌ File not found: {final_file}")
    
    return tunnel_data

def create_approach_visualization(tunnel_data, approach_name):
    """Create 3D visualization showing all 5 tunnels for one approach"""
    
    tunnel_ids = ['1-4', '2-2', '3-1', '4-1', '5-1']
    
    # Point density (pts/m³) - sample proportionally to reflect real density differences
    point_densities = {
        '1-4': 2076,
        '2-2': 1945,
        '3-1': 6894,  # Highest density
        '4-1': 838,   # Lowest density
        '5-1': 1556
    }
    
    # Calculate sampling size proportional to density
    base_sample = 30000  # Increased by 100% to show more segmentation detail
    max_density = max(point_densities.values())
    sample_sizes = {
        tid: int(base_sample * point_densities[tid] / max_density)
        for tid in tunnel_ids
    }
    
    # Segment colors - same as 1_original_visualization.html
    segment_colors = {
        0: 'lightblue',   # Background
        1: 'orange',      # K block
        2: 'lightgreen',  # B1 block
        3: 'pink',        # A1 block
        4: 'purple',      # A2 block
        5: 'brown',       # A3 block
        6: 'yellow',      # B2 block
        7: 'gray',
        8: 'cyan'
    }
    
    # Create 1x5 subplot grid (all tunnels in a row)
    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=[
            f'Tunnel {tid}' for tid in tunnel_ids
        ],
        specs=[[{'type': 'scatter3d'} for _ in range(5)]],
        horizontal_spacing=0.005
    )
    
    # Process each tunnel
    for idx, tunnel_id in enumerate(tunnel_ids, start=1):
        if tunnel_id not in tunnel_data:
            print(f"  ⚠️  Skipping tunnel {tunnel_id} - no data available")
            continue
        
        df = tunnel_data[tunnel_id]
        
        # Sample proportionally to point density to show real density differences
        sample_size = sample_sizes[tunnel_id]
        
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"  Tunnel {tunnel_id}: Sampled {sample_size:,} from {len(df):,} points (density: {point_densities[tunnel_id]} pts/m³)")
        else:
            df_sample = df
            print(f"  Tunnel {tunnel_id}: Using all {len(df):,} points (density: {point_densities[tunnel_id]} pts/m³)")
        
        # Determine colors based on segment - subtle error highlighting
        if 'pred' in df_sample.columns and 'segment' in df_sample.columns:
            # Color based on prediction, but mark errors with soft coral red
            is_error = df_sample['pred'] != df_sample['segment']
            colors = ['#FF6B6B' if err else segment_colors.get(pred, 'black') 
                     for err, pred in zip(is_error, df_sample['pred'])]
        elif 'pred' in df_sample.columns:
            colors = [segment_colors.get(seg, 'black') for seg in df_sample['pred']]
        elif 'segment' in df_sample.columns:
            colors = [segment_colors.get(seg, 'black') for seg in df_sample['segment']]
        else:
            colors = 'blue'  # Default if no segment info
        
        # Use consistent marker size to avoid fogginess
        marker_size = 1.5
        
        fig.add_trace(
            go.Scatter3d(
                x=df_sample['x'],
                y=df_sample['y'],
                z=df_sample['z'],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=colors,
                    opacity=0.8
                ),
                name=f'Tunnel {tunnel_id}',
                showlegend=False,
                text=[f'Tunnel {tunnel_id}'] * len(df_sample),
                hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ),
            row=1,
            col=idx
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{approach_name} - Segmentation Results (All 5 Tunnels)',
            x=0.5,
            font=dict(size=24)
        ),
        height=600,
        width=2400,
        showlegend=False
    )
    
    # Camera settings - user's perfect view angles
    camera_settings = {
        '1-4': dict(eye=dict(x=2.082, y=-1.172, z=0.343), center=dict(x=0, y=0, z=0)),
        '2-2': dict(eye=dict(x=-2.172, y=0.406, z=0.365), center=dict(x=0, y=0, z=0)),
        '3-1': dict(eye=dict(x=-1.93, y=0.32, z=0.754), center=dict(x=0, y=0, z=0)),
        '4-1': dict(eye=dict(x=-0.992, y=1.45, z=0.533), center=dict(x=0, y=0, z=0)),
        '5-1': dict(eye=dict(x=1.768, y=0.172, z=0.643), center=dict(x=0, y=0, z=0))
    }
    
    for idx, tunnel_id in enumerate(tunnel_ids, start=1):
        fig.update_scenes(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera_settings[tunnel_id],
            row=1, col=idx
        )
    
    # Adjust tunnel positions to balance spacing
    shift2 = 0.02
    shift3 = 0.04
    
    current_domain2 = fig.layout.scene2.domain
    if current_domain2:
        fig.update_layout(
            scene2=dict(
                domain=dict(
                    x=[current_domain2['x'][0] - shift2, current_domain2['x'][1] - shift2],
                    y=current_domain2['y']
                )
            )
        )
    
    current_domain3 = fig.layout.scene3.domain
    if current_domain3:
        fig.update_layout(
            scene3=dict(
                domain=dict(
                    x=[current_domain3['x'][0] - shift3, current_domain3['x'][1] - shift3],
                    y=current_domain3['y']
                )
            )
        )
    
    return fig

def main():
    """Main function to create separate visualizations for each approach"""
    
    base_path = "data/fyr"
    
    # Ablation approaches
    approaches = [
        ('2.baseline', 'Baseline'),
        ('3.point_context', 'Point Context'),
        ('4.point_context+CoT+Knowledge', 'Point Context + CoT + Knowledge'),
        ('5.image_CoT+Knowledge', 'Image CoT + Knowledge'),
        ('6.self-reflection', 'Self-Reflection')
    ]
    
    for approach_dir, approach_name in approaches:
        print(f"\n{'='*60}")
        print(f"Creating visualization for: {approach_name}")
        print(f"{'='*60}\n")
        
        # Load tunnel data
        approach_path = f"{base_path}/{approach_dir}"
        tunnel_data = load_approach_data(approach_path)
        
        if not tunnel_data:
            print(f"❌ No data loaded for {approach_name}. Skipping.")
            continue
        
        print(f"\n✅ Loaded {len(tunnel_data)} tunnels")
        
        # Create visualization
        fig = create_approach_visualization(tunnel_data, approach_name)
        
        # Save as HTML with camera tracking
        safe_name = approach_dir.replace('.', '_').replace('+', '_')
        output_file = f"report/3d/{safe_name}.html"
        
        # Add JavaScript to track camera position changes
        camera_tracking_script = """
    <div id="camera-info" style="position: fixed; bottom: 10px; right: 10px; background: white; padding: 15px; border: 2px solid #333; border-radius: 5px; font-family: monospace; font-size: 12px; z-index: 1000; max-width: 400px;">
        <div style="font-weight: bold; margin-bottom: 10px;">📷 Camera Settings (updates as you rotate):</div>
        <div id="camera-params" style="white-space: pre; background: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto;">
camera_settings = {
    '1-4': dict(eye=dict(x=2.082, y=-1.172, z=0.343)),
    '2-2': dict(eye=dict(x=-2.172, y=0.406, z=0.365)),
    '3-1': dict(eye=dict(x=-1.93, y=0.32, z=0.754)),
    '4-1': dict(eye=dict(x=-0.992, y=1.45, z=0.533)),
    '5-1': dict(eye=dict(x=1.768, y=0.172, z=0.643)),
}</div>
        <button onclick="copyCamera()" style="margin-top: 10px; padding: 5px 10px; cursor: pointer;">Copy to Clipboard</button>
    </div>
    <script>
        var tunnelIds = ['1-4', '2-2', '3-1', '4-1', '5-1'];
        var sceneNames = ['scene', 'scene2', 'scene3', 'scene4', 'scene5'];
        
        var cameraSettings = {
            '1-4': {x: 2.082, y: -1.172, z: 0.343},
            '2-2': {x: -2.172, y: 0.406, z: 0.365},
            '3-1': {x: -1.93, y: 0.32, z: 0.754},
            '4-1': {x: -0.992, y: 1.45, z: 0.533},
            '5-1': {x: 1.768, y: 0.172, z: 0.643}
        };
        
        // Wait for the plot to be ready
        setTimeout(function() {
            var plotDiv = document.querySelector('.plotly-graph-div');
            
            if (plotDiv) {
                plotDiv.on('plotly_relayout', function(eventData) {
                    // Check each scene for camera updates
                    sceneNames.forEach(function(sceneName, index) {
                        var cameraKey = sceneName + '.camera';
                        if (eventData && eventData[cameraKey]) {
                            var camera = eventData[cameraKey];
                            if (camera.eye) {
                                updateCameraDisplay(index, camera.eye);
                            }
                        }
                    });
                });
            }
        }, 1000);
        
        function updateCameraDisplay(index, eye) {
            var tunnelId = tunnelIds[index];
            cameraSettings[tunnelId] = {
                x: Math.round(eye.x * 1000) / 1000,
                y: Math.round(eye.y * 1000) / 1000,
                z: Math.round(eye.z * 1000) / 1000
            };
            
            var text = "camera_settings = {\\n";
            tunnelIds.forEach(function(tid) {
                var cam = cameraSettings[tid];
                text += "    '" + tid + "': dict(eye=dict(x=" + cam.x + ", y=" + cam.y + ", z=" + cam.z + ")),\\n";
            });
            text += "}";
            
            document.getElementById('camera-params').textContent = text;
        }
        
        function copyCamera() {
            var text = document.getElementById('camera-params').textContent;
            navigator.clipboard.writeText(text).then(function() {
                alert('Camera settings copied to clipboard!');
            });
        }
    </script>
    """
        
        fig.write_html(output_file, include_plotlyjs='cdn')
        
        # Add the camera tracking script
        with open(output_file, 'r') as f:
            html_content = f.read()
        
        html_content = html_content.replace('</body>', camera_tracking_script + '\n</body>')
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"\n✅ Created: {output_file}")
        
        # Print statistics
        print(f"\n📊 Tunnel Statistics for {approach_name}:")
        for tunnel_id, df in tunnel_data.items():
            print(f"\n  TUNNEL {tunnel_id}:")
            print(f"    Total points: {len(df):,}")
            
            if 'pred' in df.columns and 'segment' in df.columns:
                accuracy = (df['pred'] == df['segment']).mean()
                correct = (df['pred'] == df['segment']).sum()
                incorrect = len(df) - correct
                print(f"    Accuracy: {accuracy*100:.1f}%")
                print(f"    Correct predictions: {correct:,}")
                print(f"    Incorrect predictions: {incorrect:,}")

if __name__ == "__main__":
    main()



