from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

try:
    import pyembroidery
    HAS_EMBROIDERY = True
except ImportError:
    HAS_EMBROIDERY = False

def eps_layers_to_bot(rgba: np.ndarray, *, unit: str = "px") -> Dict[str, Any]:
    h, w = rgba.shape[:2]
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    fg = alpha > 10
    fg_pixels = bgr[fg]
    uniq = np.unique(fg_pixels.reshape(-1, 3), axis=0) if fg_pixels.size else np.zeros((0, 3), dtype=np.uint8)

    objects = []
    i = 1
    for c in uniq:
        mask = cv2.inRange(bgr, c, c)
        mask = cv2.bitwise_and(mask, (fg.astype(np.uint8) * 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            ap = cv2.approxPolyDP(cnt, 0.003 * cv2.arcLength(cnt, True), True)
            if ap is None or len(ap) < 3:
                continue
            pts = [[int(p[0][0]), int(p[0][1])] for p in ap]
            objects.append({
                "id": f"obj_{i:03d}",
                "type": "tatami",
                "points": pts,
                "angle": 45,
                "density": 4.0,
                "pull_comp": 0.3,
                "underlay": {"type": "zigzag", "density": 1.2},
                "thread": {"brand": "Generic", "code": None, "rgb": [int(c[2]), int(c[1]), int(c[0])]},
            })
            i += 1

    return {"version": "1.0", "metadata": {"width": w, "height": h, "unit": unit}, "objects": objects}

def _ranges_from_sorted(xs: np.ndarray):
    if xs.size == 0:
        return []
    out = []
    s = int(xs[0]); p = int(xs[0])
    for v in xs[1:]:
        v = int(v)
        if v == p + 1:
            p = v; continue
        out.append((s, p)); s = v; p = v
    out.append((s, p))
    return out

def bot_to_pattern(bot_json: Dict[str, Any]) -> "pyembroidery.EmbPattern":
    if not HAS_EMBROIDERY:
        raise ImportError("pyembroidery yüklü değil.")

    meta = bot_json.get("metadata") or {}
    w = int(meta.get("width") or 0)
    h = int(meta.get("height") or 0)
    if w <= 0 or h <= 0:
        raise ValueError("BOT metadata width/height geçersiz")

    pattern = pyembroidery.EmbPattern()
    color_change_cmd = getattr(pyembroidery, "COLOR_CHANGE", None)
    first_color = True

    for obj in bot_json.get("objects") or []:
        t = (obj.get("type") or "").lower()
        if t not in ("tatami", "satin"):
            continue
        pts = obj.get("points") or []
        if len(pts) < 3:
            continue

        thr = obj.get("thread") or {}
        rgb = thr.get("rgb") or [0, 0, 0]
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])

        if (not first_color) and (color_change_cmd is not None):
            pattern.add_command(color_change_cmd)
        pattern.add_thread(pyembroidery.EmbThread(r, g, b))
        first_color = False

        poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        density = float(obj.get("density") or 4.0)
        row_step = max(2, int(round(10.0 / max(1.0, density))))
        col_step = max(2, int(round(row_step / 1.5)))

        direction = 1
        for y in range(0, h, row_step):
            xs = np.where(mask[y] > 0)[0]
            segs = _ranges_from_sorted(xs)
            if not segs:
                continue
            if direction < 0:
                segs = list(reversed(segs))
            for (x0, x1) in segs:
                xr = range(int(x0), int(x1) + 1, col_step) if direction > 0 else range(int(x1), int(x0) - 1, -col_step)
                started = False
                for x in xr:
                    if not started:
                        pattern.add_stitch_absolute(pyembroidery.JUMP, float(x), float(y))
                        started = True
                    pattern.add_stitch_absolute(pyembroidery.STITCH, float(x), float(y))
            direction *= -1

        pattern.add_command(pyembroidery.JUMP)

    return pattern

def export_pattern(pattern: "pyembroidery.EmbPattern", fmt: str) -> bytes:
    import io
    fmt = (fmt or "dst").lower()
    out = io.BytesIO()
    if fmt == "dst":
        pyembroidery.write_dst(pattern, out)
    elif fmt == "pes":
        pyembroidery.write_pes(pattern, out)
    elif fmt == "exp":
        pyembroidery.write_exp(pattern, out)
    elif fmt == "jef":
        pyembroidery.write_jef(pattern, out)
    elif fmt == "xxx":
        pyembroidery.write_xxx(pattern, out)
    elif fmt == "vp3":
        pyembroidery.write_vp3(pattern, out)
    else:
        pyembroidery.write_dst(pattern, out)
    return out.getvalue()
