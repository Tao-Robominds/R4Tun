#!/usr/bin/env python3
"""
Regenerate skills/3d/best_categories_anthropic.html: 2×3 grid SAM4Tun (top) vs best
Anthropic runs (bottom). Point counts are proportional within each row; tunnel 5-5
gets an extra multiplier (larger extent). Bottom-row 2-2 gets an extra multiplier
because proportioning to the 5-5 cloud size otherwise undersamples it vs the other columns.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_HTML = REPO_ROOT / "skills" / "3d" / "best_categories_anthropic.html"

COLUMNS: list[dict] = [
    {
        "tunnel": "2-2",
        "anthropic_subdir": "memory+state+knowledge",
        "bottom_title": "Best (memory+state+knowledge): 2-2",
    },
    {
        "tunnel": "3-1-1",
        "anthropic_subdir": "memory+state",
        "bottom_title": "Best (memory+state): 3-1-1",
    },
    {
        "tunnel": "5-5",
        "anthropic_subdir": "memory+state+knowledge",
        "bottom_title": "Best (memory+state+knowledge): 5-5",
    },
]

SEGMENT_COLORS: dict[int, str] = {
    0: "lightblue",
    1: "orange",
    2: "lightgreen",
    3: "pink",
    4: "purple",
    5: "brown",
    6: "yellow",
    7: "gray",
    8: "cyan",
}

CAMERAS: dict[str, dict] = {
    "2-2": dict(eye=dict(x=-0.734, y=1.883, z=0.776), center=dict(x=0, y=0, z=0)),
    "3-1-1": dict(eye=dict(x=-0.548, y=2.005, z=0.607), center=dict(x=0, y=0, z=0)),
    "5-5": dict(eye=dict(x=2.062, y=0.427, z=0.795), center=dict(x=0, y=0, z=0)),
}

HTML_FOOTER = r"""
<div id="legend-panel" style="position: fixed; top: 10px; right: 10px; background: white; padding: 12px 14px; border: 2px solid #333; border-radius: 5px; font-family: Arial, sans-serif; font-size: 12px; z-index: 1000; width: 260px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
  <div style="font-weight: bold; margin-bottom: 8px;">Color Legend</div>
  <div style="display:flex; align-items:center; gap:8px; margin:2px 0;"><span style="display:inline-block; width:12px; height:12px; background:#add8e6;"></span>Background (0)</div>
  <div style="display:flex; align-items:center; gap:8px; margin:2px 0;"><span style="display:inline-block; width:12px; height:12px; background:orange;"></span>K-block (1)</div>
  <div style="display:flex; align-items:center; gap:8px; margin:2px 0;"><span style="display:inline-block; width:12px; height:12px; background:lightgreen;"></span>B1-block (2)</div>
  <div style="display:flex; align-items:center; gap:8px; margin:2px 0;"><span style="display:inline-block; width:12px; height:12px; background:pink;"></span>A1-block (3)</div>
  <div style="display:flex; align-items:center; gap:8px; margin:2px 0;"><span style="display:inline-block; width:12px; height:12px; background:purple;"></span>A2-block (4)</div>
  <div style="display:flex; align-items:center; gap:8px; margin:2px 0;"><span style="display:inline-block; width:12px; height:12px; background:brown;"></span>A3-block (5)</div>
  <div style="display:flex; align-items:center; gap:8px; margin:2px 0;"><span style="display:inline-block; width:12px; height:12px; background:yellow; border:1px solid #999;"></span>B2-block (6)</div>
  <div style="display:flex; align-items:center; gap:8px; margin-top:6px;"><span style="display:inline-block; width:12px; height:12px; background:#FF6B6B;"></span>Prediction error</div>
  <div style="margin-top:8px; padding-top:8px; border-top:1px solid #ddd; font-size:11px; color:#444;">
    Sampling: proportional within each row; tunnel 5-5 uses extra points (larger extent). Bottom-row 2-2 uses extra points so it matches the other Anthropic panels after proportioning to the longest cloud in that row.
  </div>
</div>

<div id="camera-info" style="position: fixed; bottom: 10px; right: 10px; background: white; padding: 15px; border: 2px solid #333; border-radius: 5px; font-family: monospace; font-size: 12px; z-index: 1000; max-width: 430px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
    <div style="font-weight: bold; margin-bottom: 10px;">📷 Camera Settings (updates as you rotate):</div>
    <div id="camera-params" style="white-space: pre; background: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto;">camera_settings = {
    '2-2': dict(eye=dict(x=-0.734, y=1.883, z=0.776)),
    '3-1-1': dict(eye=dict(x=-0.548, y=2.005, z=0.607)),
    '5-5': dict(eye=dict(x=2.062, y=0.427, z=0.795)),
}</div>
    <button onclick="copyCamera()" style="margin-top: 10px; padding: 5px 10px; cursor: pointer;">Copy to Clipboard</button>
</div>

<script>
    var tunnelIds = ['2-2', '3-1-1', '5-5', '2-2', '3-1-1', '5-5'];
    var sceneNames = ['scene', 'scene2', 'scene3', 'scene4', 'scene5', 'scene6'];
    var cameraSettings = {
        '2-2': {x: -0.734, y: 1.883, z: 0.776},
        '3-1-1': {x: -0.548, y: 2.005, z: 0.607},
        '5-5': {x: 2.062, y: 0.427, z: 0.795}
    };
    setTimeout(function() {
        var plotDiv = document.querySelector('.plotly-graph-div');
        if (plotDiv) {
            plotDiv.on('plotly_relayout', function(eventData) {
                sceneNames.forEach(function(sceneName, index) {
                    var cameraKey = sceneName + '.camera';
                    if (eventData && eventData[cameraKey] && eventData[cameraKey].eye) {
                        var eye = eventData[cameraKey].eye;
                        var tid = tunnelIds[index];
                        cameraSettings[tid] = {x: Math.round(eye.x*1000)/1000, y: Math.round(eye.y*1000)/1000, z: Math.round(eye.z*1000)/1000};
                        var text = "camera_settings = {\\n";
                        ['2-2','3-1-1','5-5'].forEach(function(t){ var c=cameraSettings[t]; text += "    '"+t+"': dict(eye=dict(x="+c.x+", y="+c.y+", z="+c.z+")),\\n";});
                        text += "}";
                        document.getElementById('camera-params').textContent = text;
                    }
                });
            });
        }
    }, 1000);
    function copyCamera() { navigator.clipboard.writeText(document.getElementById('camera-params').textContent).then(function(){ alert('Camera settings copied to clipboard!'); }); }
</script>

<script>
(function(){
  function padRange(minV, maxV, frac){
    var span = maxV - minV;
    if (!isFinite(span) || span <= 0) span = 1.0;
    var pad = span * frac;
    return [minV - pad, maxV + pad];
  }

  function fitScenes(gd){
    var scenes = ['scene','scene2','scene3','scene4','scene5','scene6'];
    var update = {};

    scenes.forEach(function(sceneName){
      var xs = [], ys = [], zs = [];
      gd.data.forEach(function(tr){
        var trScene = tr.scene || 'scene';
        if (trScene !== sceneName) return;
        if (!tr.x || !tr.y || !tr.z) return;
        for (var i=0; i<tr.x.length; i++) {
          var x = +tr.x[i], y = +tr.y[i], z = +tr.z[i];
          if (isFinite(x) && isFinite(y) && isFinite(z)) {
            xs.push(x); ys.push(y); zs.push(z);
          }
        }
      });

      if (xs.length === 0) return;

      var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
      var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
      var zmin = Math.min.apply(null, zs), zmax = Math.max.apply(null, zs);

      var xr = padRange(xmin, xmax, 0.02);
      var yr = padRange(ymin, ymax, 0.02);
      var zr = padRange(zmin, zmax, 0.02);

      update[sceneName + '.xaxis.range'] = xr;
      update[sceneName + '.yaxis.range'] = yr;
      update[sceneName + '.zaxis.range'] = zr;
      update[sceneName + '.aspectmode'] = 'data';
    });

    if (Object.keys(update).length > 0) {
      Plotly.relayout(gd, update);
    }
  }

  function init(){
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd || !window.Plotly) return;

    setTimeout(function(){ fitScenes(gd); }, 700);

    gd.on('plotly_doubleclick', function(){
      setTimeout(function(){ fitScenes(gd); }, 50);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>
"""


def _row_sample_targets(
    lengths: dict[str, int],
    *,
    base: int,
    tunnel_boosts: dict[str, float],
) -> dict[str, int]:
    max_len = max(lengths.values())
    out: dict[str, int] = {}
    for tid, ln in lengths.items():
        n = int(round(base * ln / max_len))
        n = min(n, ln)
        mult = tunnel_boosts.get(tid, 1.0)
        if mult > 1.0:
            n = min(ln, int(round(n * mult)))
        out[tid] = max(n, 1)
    return out


def _segment_colors(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    if "pred" in df.columns and "segment" in df.columns:
        pcol = df["pred"].fillna(-1).astype(np.int64)
        scol = df["segment"].fillna(-1).astype(np.int64)
        err = pcol != scol
        base = pcol.map(SEGMENT_COLORS).fillna("black").to_numpy(dtype=object)
        return np.where(err, "#FF6B6B", base)
    if "pred" in df.columns:
        pcol = df["pred"].fillna(-1).astype(np.int64)
        return pcol.map(SEGMENT_COLORS).fillna("black").to_numpy(dtype=object)
    if "segment" in df.columns:
        scol = df["segment"].fillna(-1).astype(np.int64)
        return scol.map(SEGMENT_COLORS).fillna("black").to_numpy(dtype=object)
    return np.full(n, "blue", dtype=object)


def _add_scatter(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    row: int,
    col: int,
    label: str,
) -> None:
    colors = _segment_colors(df)
    fig.add_trace(
        go.Scatter3d(
            x=df["x"].astype(np.float64),
            y=df["y"].astype(np.float64),
            z=df["z"].astype(np.float64),
            mode="markers",
            marker=dict(size=0.8, color=colors, opacity=0.65),
            name=label,
            showlegend=False,
            text=[label] * len(df),
            hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def build_figure(
    *,
    base_row: int,
    five_five_boost: float,
    anthropic_2_2_boost: float,
    random_state: int,
) -> go.Figure:
    sam_root = REPO_ROOT / "data" / "sam4tun"
    ant_root = REPO_ROOT / "data" / "ablation_anthropic"

    lengths_sam = {}
    lengths_ant = {}
    dfs_sam: dict[str, pd.DataFrame] = {}
    dfs_ant: dict[str, pd.DataFrame] = {}

    for col in COLUMNS:
        tid = col["tunnel"]
        p_sam = sam_root / tid / "final.csv"
        p_ant = ant_root / col["anthropic_subdir"] / tid / "final.csv"
        if not p_sam.is_file():
            raise FileNotFoundError(p_sam)
        if not p_ant.is_file():
            raise FileNotFoundError(p_ant)
        cols = ["x", "y", "z", "pred", "segment"]
        dfs_sam[tid] = pd.read_csv(p_sam, usecols=cols)
        dfs_ant[tid] = pd.read_csv(p_ant, usecols=cols)
        lengths_sam[tid] = len(dfs_sam[tid])
        lengths_ant[tid] = len(dfs_ant[tid])

    n_sam = _row_sample_targets(
        lengths_sam,
        base=base_row,
        tunnel_boosts={"5-5": five_five_boost},
    )
    n_ant = _row_sample_targets(
        lengths_ant,
        base=base_row,
        tunnel_boosts={"5-5": five_five_boost, "2-2": anthropic_2_2_boost},
    )

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}],
            [{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}],
        ],
        vertical_spacing=0.03,
        horizontal_spacing=0.003,
    )

    ann_x = [0.16566666666666666, 0.5, 0.8343333333333334]

    for idx, colspec in enumerate(COLUMNS, start=1):
        tid = colspec["tunnel"]
        df_s = dfs_sam[tid]
        df_a = dfs_ant[tid]
        if len(df_s) > n_sam[tid]:
            df_s = df_s.sample(n=n_sam[tid], random_state=random_state)
        if len(df_a) > n_ant[tid]:
            df_a = df_a.sample(n=n_ant[tid], random_state=random_state)

        _add_scatter(fig, df_s, row=1, col=idx, label=f"SAM4Tun {tid}")
        _add_scatter(fig, df_a, row=2, col=idx, label=f"Anthropic {tid}")

    annotations: list[dict] = []
    for idx, colspec in enumerate(COLUMNS):
        tid = colspec["tunnel"]
        x = ann_x[idx]
        annotations.append(
            dict(
                text=f"SAM4Tun: {tid}",
                xref="paper",
                yref="paper",
                x=x,
                y=1.0,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=16),
            )
        )
        annotations.append(
            dict(
                text=colspec["bottom_title"],
                xref="paper",
                yref="paper",
                x=x,
                y=0.485,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=16),
            )
        )

    fig.update_layout(
        title=dict(
            text="SAM4Tun (Top) vs Best Anthropic (Bottom) - Proportional Sampling per Tunnel",
            x=0.5,
            font=dict(size=22),
        ),
        height=1200,
        width=1750,
        showlegend=False,
        annotations=annotations,
        margin=dict(l=0, r=0, t=80, b=0),
    )

    for idx, colspec in enumerate(COLUMNS, start=1):
        tid = colspec["tunnel"]
        cam = CAMERAS[tid]
        fig.update_scenes(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            camera=cam,
            row=1,
            col=idx,
        )
        fig.update_scenes(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            camera=cam,
            row=2,
            col=idx,
        )

    print("Sample counts SAM4Tun row:", n_sam)
    print("Sample counts Anthropic row:", n_ant)
    return fig


def write_full_html(fig: go.Figure, path: Path) -> None:
    div_id = str(uuid.uuid4())
    inner = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"responsive": False, "displayModeBar": True},
    )
    doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Best categories - SAM4Tun vs Anthropic</title>
  <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
</head>
<body style="margin:0;">
{inner}
{HTML_FOOTER}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-row", type=int, default=70000, help="Target count for largest tunnel in a row")
    parser.add_argument(
        "--five-five-boost",
        type=float,
        default=2.25,
        help="Extra multiplier for tunnel 5-5 after proportional sizing (both rows)",
    )
    parser.add_argument(
        "--anthropic-2-2-boost",
        type=float,
        default=1.75,
        help="Extra multiplier for tunnel 2-2 on the Anthropic row only (proportion vs 5-5 undersamples)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=OUT_HTML)
    args = parser.parse_args()

    fig = build_figure(
        base_row=args.base_row,
        five_five_boost=args.five_five_boost,
        anthropic_2_2_boost=args.anthropic_2_2_boost,
        random_state=args.seed,
    )
    write_full_html(fig, args.output)


if __name__ == "__main__":
    main()
