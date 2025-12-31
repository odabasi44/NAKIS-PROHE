from __future__ import annotations

from dataclasses import dataclass
import io

import cv2
import numpy as np
from PIL import Image

@dataclass
class VectorPipelineResult:
    eps_bytes: bytes
    preview_png_bytes: bytes
    palette: list[str]

def _rgba_to_eps_bytes(rgba: Image.Image) -> bytes:
    rgba = rgba.convert("RGBA")
    bgra = cv2.cvtColor(np.array(rgba), cv2.COLOR_RGBA2BGRA)
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    bgr = bgra[:, :, :3].astype(np.float32)
    white = np.ones_like(bgr) * 255.0
    out_bgr = (bgr * a + white * (1.0 - a)).astype(np.uint8)
    rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="EPS")
    return buf.getvalue()

def _rgba_to_preview_png(rgba: Image.Image) -> bytes:
    bgra = cv2.cvtColor(np.array(rgba.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    rgb = bgra[:, :, :3].astype(np.float32)
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

    rgb = np_rgba[:, :, :3]
    alpha = np_rgba[:, :, 3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    fg = alpha > 10
    fg_pixels = bgr[fg]
    if fg_pixels.size == 0:
        blank = Image.new("RGBA", (int(w), int(h)), (255, 255, 255, 255))
        eps = _rgba_to_eps_bytes(blank)
        return VectorPipelineResult(eps_bytes=eps, preview_png_bytes=_rgba_to_preview_png(blank), palette=[])

    uniq = np.unique(fg_pixels.reshape(-1, 3), axis=0)

    palette = []
    for c in uniq:
        palette.append(f"#{int(c[2]):02x}{int(c[1]):02x}{int(c[0]):02x}")

    eps_data = _rgba_to_eps_bytes(rgba)
    return VectorPipelineResult(eps_bytes=eps_data, preview_png_bytes=_rgba_to_preview_png(rgba), palette=palette)
