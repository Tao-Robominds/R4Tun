#!/usr/bin/env python3
"""
5 Tunnels Original Form Visualization

Creates a 3D visualization showing all 5 tunnels in their original unwrapped form
displayed in a single row with pure blue color (no segmentation).

Tunnels: 1-4, 2-2, 3-1, 4-1, 5-1
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def load_all_tunnels_original():
    """Load unwrapped data for all 5 tunnels from the original dataset"""
    base_path = "data/fyr/1.original"
    tunnel_ids = ['1-4', '2-2', '3-1', '4-1', '5-1']
    
    tunnel_data = {}
    
    for tunnel_id in tunnel_ids:
        unwrapped_file = f"{base_path}/{tunnel_id}/unwrapped.csv"
        if os.path.exists(unwrapped_file):
            df = pd.read_csv(unwrapped_file)
            tunnel_data[tunnel_id] = df
            print(f"✅ Loaded tunnel {tunnel_id}: {len(df):,} points")
        else:
            print(f"❌ File not found: {unwrapped_file}")
    
    return tunnel_data

def create_tunnels_visualization(tunnel_data):
    """Create 3D visualization showing all 5 tunnels in a row"""
    
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
    base_sample = 10000
    max_density = max(point_densities.values())
    sample_sizes = {
        tid: int(base_sample * point_densities[tid] / max_density)
        for tid in tunnel_ids
    }
    
    # Camera settings - user's perfect view angles
    camera_settings = {
        '1-4': dict(eye=dict(x=2.082, y=-1.172, z=0.343), center=dict(x=0, y=0, z=0)),
        '2-2': dict(eye=dict(x=-2.172, y=0.406, z=0.365), center=dict(x=0, y=0, z=0)),
        '3-1': dict(eye=dict(x=-1.93, y=0.32, z=0.754), center=dict(x=0, y=0, z=0)),
        '4-1': dict(eye=dict(x=-0.992, y=1.45, z=0.533), center=dict(x=0, y=0, z=0)),
        '5-1': dict(eye=dict(x=1.768, y=0.172, z=0.643), center=dict(x=0, y=0, z=0))
    }
    
    # Create subplot titles with camera info
    subplot_titles = []
    for tid in tunnel_ids:
        cam = camera_settings[tid]
        eye = cam['eye']
        subplot_titles.append(
            f'Tunnel {tid}<br><sub>eye: ({eye["x"]}, {eye["y"]}, {eye["z"]})</sub>'
        )
    
    # Create 1x5 subplot grid (all tunnels in a row)
    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=subplot_titles,
        specs=[[{'type': 'scatter3d'} for _ in range(5)]],
        horizontal_spacing=0.005
    )
    
    # Process each tunnel
    for idx, tunnel_id in enumerate(tunnel_ids, start=1):
        if tunnel_id not in tunnel_data:
            print(f"⚠️  Skipping tunnel {tunnel_id} - no data available")
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
        
        # Use same marker size for all tunnels - let density differences show naturally
        marker_size = 1.5
        
        fig.add_trace(
            go.Scatter3d(
                x=df_sample['x'],
                y=df_sample['y'],
                z=df_sample['z'],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color='#4064c5',
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
            text='Original Forms of 5 Tunnels (Unwrapped Point Clouds)',
            x=0.5,
            font=dict(size=24)
        ),
        height=600,
        width=2400,
        showlegend=False
    )
    
    # Apply camera settings to each subplot
    for idx, tunnel_id in enumerate(tunnel_ids, start=1):
        fig.update_scenes(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera_settings[tunnel_id],
            row=1, col=idx
        )
    
    # Adjust tunnel positions to balance spacing
    # Scene2 (tunnel 2-2) - shift left by 2%
    # Scene3 (tunnel 3-1) - shift left by 4% (double)
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
    
    # Print camera settings for reference
    print("\n📷 Camera Settings (copy these values to reuse):")
    print("camera_settings = {")
    for tid in tunnel_ids:
        cam = camera_settings[tid]
        eye = cam['eye']
        print(f"    '{tid}': dict(eye=dict(x={eye['x']}, y={eye['y']}, z={eye['z']}), center=dict(x=0, y=0, z=0)),")
    print("}")
    
    return fig

def main():
    """Main function to create and save the visualization"""
    print("Creating 3D visualization for all 5 tunnels - Original Forms...")
    print()
    
    # Load tunnel data
    tunnel_data = load_all_tunnels_original()
    
    if not tunnel_data:
        print("❌ No data loaded. Exiting.")
        return
    
    print(f"\n✅ Loaded {len(tunnel_data)} tunnels")
    
    # Create visualization
    fig = create_tunnels_visualization(tunnel_data)
    
    # Save as HTML with camera tracking
    output_file = "report/3d/all_tunnels_original.html"
    
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
    
    # Add the camera tracking script to the HTML
    with open(output_file, 'r') as f:
        html_content = f.read()
    
    # Insert the script before </body>
    html_content = html_content.replace('</body>', camera_tracking_script + '</body>')
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Created: {output_file}")
    print("📷 Camera tracking enabled - rotate the tunnels and see real-time camera values!")
    
    # Print statistics
    print(f"\n📊 Tunnel Statistics:")
    for tunnel_id, df in tunnel_data.items():
        print(f"\n  TUNNEL {tunnel_id}:")
        print(f"    Total points: {len(df):,}")
        print(f"    X range: {df['x'].min():.2f} to {df['x'].max():.2f}")
        print(f"    Y range: {df['y'].min():.2f} to {df['y'].max():.2f}")
        print(f"    Z range: {df['z'].min():.2f} to {df['z'].max():.2f}")

if __name__ == "__main__":
    main()

