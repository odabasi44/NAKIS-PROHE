from __future__ import annotations

from typing import Any, Dict

import os
import cv2
import numpy as np

try:
    import pyembroidery
    HAS_EMBROIDERY = True
except ImportError:
    HAS_EMBROIDERY = False

def eps_layers_to_bot(rgba: np.ndarray, *, unit: str = "px", prefer_text_satin: bool = False) -> Dict[str, Any]:
    h, w = rgba.shape[:2]
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    fg = alpha > 10
    fg_pixels = bgr[fg]

    try:
        px_per_mm = float(os.getenv("BOTLAB_PX_PER_MM", "10"))
    except Exception:
        px_per_mm = 10.0
    px_per_mm = max(0.1, px_per_mm)

    try:
        satin_threshold_mm = float(os.getenv("BOTLAB_SATIN_THRESHOLD_MM", "8"))
    except Exception:
        satin_threshold_mm = 8.0
    satin_threshold_mm = max(0.1, satin_threshold_mm)

    try:
        text_satin_height_cm = float(os.getenv("BOTLAB_TEXT_SATIN_HEIGHT_CM", "8"))
    except Exception:
        text_satin_height_cm = 8.0
    text_satin_height_mm = max(0.1, text_satin_height_cm) * 10.0

    if fg_pixels.size:
        uniq, counts = np.unique(fg_pixels.reshape(-1, 3), axis=0, return_counts=True)
        order = np.lexsort((uniq[:, 2], uniq[:, 1], uniq[:, 0]))
        uniq = uniq[order]
        counts = counts[order]
        uniq = uniq[np.argsort(-counts, kind="stable")]
    else:
        uniq = np.zeros((0, 3), dtype=np.uint8)

    objects = []
    i = 1
    for c in uniq:
        mask = cv2.inRange(bgr, c, c)
        mask = cv2.bitwise_and(mask, (fg.astype(np.uint8) * 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue

            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            mm_min = (min(float(rw), float(rh)) / px_per_mm) if rw and rh else 0.0
            x, y, bw, bh = cv2.boundingRect(cnt)
            height_mm = float(bh) / px_per_mm if bh else 0.0

            if prefer_text_satin and height_mm > 0.0:
                stype = "satin" if height_mm <= text_satin_height_mm else "tatami"
            else:
                stype = "satin" if (mm_min > 0.0 and mm_min < satin_threshold_mm) else "tatami"

            ap = cv2.approxPolyDP(cnt, 0.003 * cv2.arcLength(cnt, True), True)
            if ap is None or len(ap) < 3:
                continue
            pts = [[int(p[0][0]), int(p[0][1])] for p in ap]
            objects.append({
                "id": f"obj_{i:03d}",
                "type": stype,
                "points": pts,
                "angle": float(rect[2]) if stype == "satin" else 45,
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

    try:
        px_per_mm = float(os.getenv("BOTLAB_PX_PER_MM", "10"))
    except Exception:
        px_per_mm = 10.0
    px_per_mm = max(0.1, px_per_mm)

    try:
        units_per_mm = float(os.getenv("BOTLAB_UNITS_PER_MM", "10"))
    except Exception:
        units_per_mm = 10.0
    units_per_mm = max(0.1, units_per_mm)

    def to_units(x_px: float, y_px: float) -> tuple[float, float]:
        return (float(x_px) / px_per_mm * units_per_mm, float(y_px) / px_per_mm * units_per_mm)

    objs = bot_json.get("objects") or []
    groups = {}
    order_keys = []
    for obj in objs:
        thr = obj.get("thread") or {}
        rgb = thr.get("rgb") or [0, 0, 0]
        key = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        if key not in groups:
            groups[key] = []
            order_keys.append(key)
        groups[key].append(obj)

    first_color = True

    for (r, g, b) in order_keys:
        if (not first_color) and (color_change_cmd is not None):
            pattern.add_command(color_change_cmd)
        pattern.add_thread(pyembroidery.EmbThread(int(r), int(g), int(b)))
        first_color = False

        for obj in groups.get((r, g, b)) or []:
            t = (obj.get("type") or "").lower()
            if t not in ("tatami", "satin"):
                continue
            pts = obj.get("points") or []
            if len(pts) < 3:
                continue

            poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)

            density = float(obj.get("density") or 4.0)
            row_step = max(2, int(round(10.0 / max(1.0, density))))
            col_step = max(2, int(round(row_step / 1.5)))

            if t == "tatami":
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
                            ux, uy = to_units(float(x), float(y))
                            if not started:
                                pattern.add_stitch_absolute(pyembroidery.JUMP, ux, uy)
                                started = True
                            pattern.add_stitch_absolute(pyembroidery.STITCH, ux, uy)
                    direction *= -1
                pattern.add_command(pyembroidery.JUMP)
                continue

            angle = float(obj.get("angle") or 0.0)
            m = cv2.moments(mask)
            if m["m00"]:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
            else:
                cx = float(w) / 2.0
                cy = float(h) / 2.0

            rot = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
            rot_mask = cv2.warpAffine(mask, rot, (w, h), flags=cv2.INTER_NEAREST)

            inv = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

            step = max(2, int(round(col_step / 1.2)))
            direction = 1
            for x in range(0, w, step):
                ys = np.where(rot_mask[:, x] > 0)[0]
                segs = _ranges_from_sorted(ys)
                if not segs:
                    continue
                if direction < 0:
                    segs = list(reversed(segs))
                for (y0, y1) in segs:
                    yr = range(int(y0), int(y1) + 1, row_step) if direction > 0 else range(int(y1), int(y0) - 1, -row_step)
                    started = False
                    for y in yr:
                        rx = float(x)
                        ry = float(y)
                        ox = inv[0, 0] * rx + inv[0, 1] * ry + inv[0, 2]
                        oy = inv[1, 0] * rx + inv[1, 1] * ry + inv[1, 2]
                        ux, uy = to_units(ox, oy)
                        if not started:
                            pattern.add_stitch_absolute(pyembroidery.JUMP, ux, uy)
                            started = True
                        pattern.add_stitch_absolute(pyembroidery.STITCH, ux, uy)
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
