"""
crop.py — the SINGLE source of truth for crop geometry.

Both data preparation (training) and inference (prediction) cut a square window
around the ball at native resolution. If those two used different code or sizes,
the detector would see the ball at one scale in training and another at
inference, a silent, accuracy-killing mismatch. To make that impossible, both
paths go through `crop_origin` / `extract_crop` here, with ONE `crop_size`.

"""

from __future__ import annotations

import numpy as np


def crop_origin(cx: float, cy: float, size: int, w: int, h: int) -> tuple[int, int]:
    """Top-left (x0, y0) of a `size`x`size` window centred on (cx, cy),
    clamped so the window stays fully inside a `w`x`h` frame."""
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    x0 = max(0, min(x0, w - size))
    y0 = max(0, min(y0, h - size))
    return x0, y0


def extract_crop(frame: np.ndarray, cx: float, cy: float,
                 size: int) -> tuple[np.ndarray, int, int]:
    """Return (crop, x0, y0): the `size`x`size` native-resolution patch centred
    on (cx, cy). Add x0/y0 back to any in-crop coordinate to map to full frame."""
    h, w = frame.shape[:2]
    x0, y0 = crop_origin(cx, cy, size, w, h)
    return frame[y0:y0 + size, x0:x0 + size], x0, y0