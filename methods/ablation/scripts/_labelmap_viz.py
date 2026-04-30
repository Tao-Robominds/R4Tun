"""Shared labelmap visualization helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PALETTE_8 = np.array(
    [
        [0, 0, 0],        # 0 BG
        [220, 20, 60],    # 1 K
        [65, 105, 225],   # 2 B1
        [50, 205, 50],    # 3 A1
        [255, 165, 0],    # 4 A2
        [186, 85, 211],   # 5 A3
        [255, 215, 0],    # 6 A4
        [30, 144, 255],   # 7 B2
    ],
    dtype=np.uint8,
)


def render_labelmap_png(labelmap: np.ndarray, out_path: str) -> None:
    """Render int labelmap (0..7) to PNG with a fixed palette."""
    arr = np.asarray(labelmap)
    rgb = PALETTE_8[np.clip(arr, 0, 7).astype(np.int64)]
    bgr = rgb[..., ::-1]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), bgr)
