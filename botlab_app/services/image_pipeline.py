from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort


@dataclass
class ImagePipelineResult:
    rgba: Image.Image  # RGBA


class ImagePipeline:
    """
    Pure-python-ish pipeline:
    - enhance (denoise + unsharp + CLAHE)
    - background remove (U2Net ONNX varsa), yoksa border heuristics
    - color reduction (OpenCV kmeans)
    - resize (keep_ratio)
    """

    def __init__(self, u2net_path: str):
        self.u2net_session = None
        self.u2net_input_name = "input"
        if u2net_path and os.path.exists(u2net_path):
            try:
                self.u2net_session = ort.InferenceSession(u2net_path, providers=["CPUExecutionProvider"])
                self.u2net_input_name = self.u2net_session.get_inputs()[0].name
                print(f"[botlab_app] ✅ U2Net loaded: {u2net_path}")
            except Exception as e:
                # Model bozuk/eksikse servis ayağa kalksın; fallback background remove devreye girer.
                self.u2net_session = None
                print(f"[botlab_app] ⚠️ U2Net load failed ({u2net_path}): {e}")

    def enhance(self, rgb: Image.Image) -> Image.Image:
        bgr = cv2.cvtColor(np.array(rgb.convert("RGB")), cv2.COLOR_RGB2BGR)
        den = cv2.fastNlMeansDenoisingColored(bgr, None, 6, 6, 7, 21)
        blur = cv2.GaussianBlur(den, (0, 0), 1.0)
        sharp = cv2.addWeighted(den, 1.4, blur, -0.4, 0)
        lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), mode="RGB")

    def resize(self, img: Image.Image, width: int | None, height: int | None, keep_ratio: bool) -> Image.Image:
        w0, h0 = img.size
        tw = int(width) if width else 0
        th = int(height) if height else 0
        if tw <= 0 and th <= 0:
            return img
        if keep_ratio:
            if tw > 0 and th <= 0 and w0 > 0:
                th = int(round((tw * h0) / w0))
            elif th > 0 and tw <= 0 and h0 > 0:
                tw = int(round((th * w0) / h0))
            elif tw > 0 and th > 0:
                s = min(tw / w0, th / h0)
                tw = int(round(w0 * s))
                th = int(round(h0 * s))
        tw = max(64, min(tw if tw > 0 else w0, 3000))
        th = max(64, min(th if th > 0 else h0, 3000))
        if (tw, th) == (w0, h0):
            return img
        return img.resize((tw, th), Image.LANCZOS)

    def remove_background(self, rgb: Image.Image, threshold: float = 0.5) -> Image.Image:
        rgb = rgb.convert("RGB")
        np_rgb = np.array(rgb)
        h, w = np_rgb.shape[:2]
        alpha = np.full((h, w), 255, dtype=np.uint8)

        if self.u2net_session is not None:
            small = rgb.resize((320, 320))
            inp = np.transpose(np.array(small).astype(np.float32) / 255.0, (2, 0, 1))
            inp = np.expand_dims(inp, 0)
            out = self.u2net_session.run(None, {self.u2net_input_name: inp})[0].squeeze()
            mask = cv2.resize(out, (w, h))
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            alpha = ((mask >= float(threshold)) * 255).astype(np.uint8)
        else:
            bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
            border = np.concatenate([bgr[0, :, :], bgr[-1, :, :], bgr[:, 0, :], bgr[:, -1, :]], axis=0)
            bg = np.median(border.reshape(-1, 3), axis=0).astype(np.uint8)
            diff = np.linalg.norm(bgr.astype(np.int16) - bg.astype(np.int16), axis=2)
            alpha = (diff > 18).astype(np.uint8) * 255

        rgba = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = alpha
        return Image.fromarray(rgba, mode="RGBA")

    def reduce_colors(self, rgba: Image.Image, num_colors: int = 4) -> tuple[Image.Image, list[str]]:
        # alpha varsa beyaz ile blend (kmeans stabil) ama alpha'yı koruruz
        rgba = rgba.convert("RGBA")
        np_rgba = np.array(rgba)
        alpha = np_rgba[:, :, 3:4].astype(np.float32) / 255.0
        rgb = np_rgba[:, :, :3].astype(np.float32)
        white = np.ones_like(rgb) * 255.0
        blended = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
        bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        k = int(max(3, min(num_colors, 5)))
        data = np.float32(bgr).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quant_bgr = centers[labels.flatten()].reshape(bgr.shape)
        quant_rgb = cv2.cvtColor(quant_bgr, cv2.COLOR_BGR2RGB)

        out_rgba = np.zeros_like(np_rgba)
        out_rgba[:, :, :3] = quant_rgb
        out_rgba[:, :, 3] = np_rgba[:, :, 3]

        # palette (hex)
        uniq = np.unique(centers, axis=0)
        colors = []
        for c in uniq:
            colors.append(f"#{int(c[2]):02x}{int(c[1]):02x}{int(c[0]):02x}")
        return Image.fromarray(out_rgba, mode="RGBA"), colors

    def run(self, image: Image.Image, *, num_colors: int, width: int | None, height: int | None, keep_ratio: bool) -> ImagePipelineResult:
        rgb = self.resize(image.convert("RGB"), width, height, keep_ratio)
        enh = self.enhance(rgb)
        rgba = self.remove_background(enh)
        rgba2, _ = self.reduce_colors(rgba, num_colors=num_colors)
        return ImagePipelineResult(rgba=rgba2)


