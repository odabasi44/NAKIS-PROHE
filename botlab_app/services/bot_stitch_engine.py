import json
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    import pyembroidery
    HAS_EMBROIDERY = True
except ImportError:
    HAS_EMBROIDERY = False


def _ranges_from_sorted_indices(indices: np.ndarray) -> List[Tuple[int, int]]:
    if indices.size == 0:
        return []
    ranges = []
    start = int(indices[0])
    prev = int(indices[0])
    for v in indices[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
            continue
        ranges.append((start, prev))
        start = v
        prev = v
    ranges.append((start, prev))
    return ranges


def bot_json_to_pattern(bot_json: Dict[str, Any]) -> "pyembroidery.EmbPattern":
    """
    BOT -> pyembroidery pattern (basit tatami fill).
    Bu profesyonel digitizing değildir; temel "çalışır" çıktıdır.
    """
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
        if (obj.get("type") or "").lower() != "tatami":
            continue

        pts = obj.get("points") or []
        if len(pts) < 3:
            continue

        thr = obj.get("thread") or {}
        rgb = thr.get("rgb") or [0, 0, 0]
        try:
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        except Exception:
            r, g, b = 0, 0, 0

        if (not first_color) and (color_change_cmd is not None):
            pattern.add_command(color_change_cmd)
        pattern.add_thread(pyembroidery.EmbThread(r, g, b))
        first_color = False

        poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        density = float(obj.get("density") or 4.0)
        # px spacing: density yükseldikçe sıklaşsın (basit)
        row_step = max(2, int(round(10.0 / max(1.0, density))))
        col_step = max(2, int(round(row_step / 1.5)))

        # basit zig-zag satır tarama
        direction = 1
        for y in range(0, h, row_step):
            xs = np.where(mask[y] > 0)[0]
            segs = _ranges_from_sorted_indices(xs)
            if not segs:
                continue
            if direction < 0:
                segs = list(reversed(segs))

            for (x0, x1) in segs:
                if direction > 0:
                    xr = range(int(x0), int(x1) + 1, col_step)
                else:
                    xr = range(int(x1), int(x0) - 1, -col_step)

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
    fmt = (fmt or "dst").lower()
    import io

    out = io.BytesIO()
    if fmt == "dst":
        pyembroidery.write_dst(pattern, out)
    elif fmt == "pes":
        pyembroidery.write_pes(pattern, out)
    elif fmt == "exp":
        pyembroidery.write_exp(pattern, out)
    elif fmt == "jef":
        pyembroidery.write_jef(pattern, out)
    elif fmt == "vp3":
        pyembroidery.write_vp3(pattern, out)
    elif fmt == "xxx":
        pyembroidery.write_xxx(pattern, out)
    else:
        pyembroidery.write_dst(pattern, out)
    return out.getvalue()


