#!/usr/bin/env python3
"""Extract Plotly figure from best_categories_anthropic.html and export static PNGs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio


def _skip_string(s: str, i: int) -> int:
    q = s[i]
    i += 1
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == q:
            return i + 1
        i += 1
    raise ValueError("unterminated string")


def _parse_balanced(s: str, start: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    depth = 0
    i = start
    if s[i] != open_ch:
        raise ValueError(f"expected {open_ch} at {start}")
    while i < len(s):
        c = s[i]
        if c in "\"'":
            i = _skip_string(s, i)
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[start : i + 1], i + 1
        i += 1
    raise ValueError("unbalanced brackets")


def extract_newplot_args(html: str) -> tuple[str, str]:
    m = re.search(r"Plotly\.newPlot\s*\(\s*\"[^\"]+\"\s*,\s*", html)
    if not m:
        raise RuntimeError("Plotly.newPlot not found")
    pos = m.end()
    data_json, pos = _parse_balanced(html, pos, "[", "]")
    while pos < len(html) and html[pos] in ", \t\n\r":
        pos += 1
    layout_json, pos = _parse_balanced(html, pos, "{", "}")
    return data_json, layout_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    html_path = root / "best_categories_anthropic.html"
    if not html_path.exists():
        print(f"Missing {html_path}", file=sys.stderr)
        sys.exit(1)

    print("Reading HTML…")
    html = html_path.read_text(encoding="utf-8", errors="replace")

    print("Parsing Plotly.newPlot…")
    data_json, layout_json = extract_newplot_args(html)
    data = json.loads(data_json)
    layout = json.loads(layout_json)
    fig = go.Figure(data=data, layout=layout)

    for name, scale in [("best_categories_anthropic_hd", 3), ("best_categories_anthropic_ultra_hd", 4)]:
        out = root / f"{name}.png"
        print(f"Writing {out} (scale={scale})…")
        pio.write_image(fig, str(out), format="png", scale=scale, engine="kaleido")

    print("Done.")


if __name__ == "__main__":
    main()
