from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from vector_engine.vector_trace import mask_to_paths
from vector_engine.eps_generator import paths_to_eps


@dataclass
class VectorPipelineResult:
    eps_bytes: bytes
    preview_png_bytes: bytes
    palette: list[str]


def _rgba_to_preview_png(rgba: Image.Image) -> bytes:
    bgr = cv2.cvtColor(np.array(rgba.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
    # preview için alpha'yı beyazla blend edip png kaydet
    a = bgr[:, :, 3:4].astype(np.float32) / 255.0
    rgb = bgr[:, :, :3].astype(np.float32)
    white = np.ones_like(rgb) * 255.0
    out = (rgb * a + white * (1.0 - a)).astype(np.uint8)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise ValueError("PNG encode failed")
    return buf.tobytes()


def rgba_to_eps(rgba: Image.Image, *, simplify_factor: float = 0.003, min_area: float = 20.0) -> VectorPipelineResult:
    rgba = rgba.convert("RGBA")
    np_rgba = np.array(rgba)
    h, w = np_rgba.shape[:2]

    # BGR quantized image
    rgb = np_rgba[:, :, :3]
    alpha = np_rgba[:, :, 3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # palette (unique colors from image, excluding transparent)
    fg = alpha > 10
    fg_pixels = bgr[fg]
    if fg_pixels.size == 0:
        # empty foreground -> blank eps
        eps = paths_to_eps([], width=w, height=h).eps_bytes
        return VectorPipelineResult(eps_bytes=eps, preview_png_bytes=_rgba_to_preview_png(rgba), palette=[])

    uniq = np.unique(fg_pixels.reshape(-1, 3), axis=0)
    # build layers
    layers = []
    palette = []
    for c in uniq:
        mask = cv2.inRange(bgr, c, c)
        mask = cv2.bitwise_and(mask, (fg.astype(np.uint8) * 255))
        # cleanup a bit
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        paths = mask_to_paths(mask, simplify_factor=simplify_factor, min_area=min_area)
        if not paths:
            continue
        rgb_hex = f"#{int(c[2]):02x}{int(c[1]):02x}{int(c[0]):02x}"
        palette.append(rgb_hex)
        layers.append(((int(c[2]), int(c[1]), int(c[0])), paths))

    eps_data = paths_to_eps(layers, width=w, height=h).eps_bytes
    return VectorPipelineResult(
        eps_bytes=eps_data,
        preview_png_bytes=_rgba_to_preview_png(rgba),
        palette=palette,
    )


