from __future__ import annotations

from typing import List, Optional
import cv2
import numpy as np
from .path_tools import Path

def _approx_contour(cnt: np.ndarray, simplify_factor: float) -> Optional[list[tuple[float, float]]]:
    if cnt is None or len(cnt) < 3:
        return None
    eps = float(simplify_factor) * cv2.arcLength(cnt, True)
    ap = cv2.approxPolyDP(cnt, eps, True)
    if ap is None or len(ap) < 3:
        return None
    return [(float(p[0][0]), float(p[0][1])) for p in ap]

def mask_to_paths(mask: np.ndarray, *, simplify_factor: float = 0.003, min_area: float = 20.0) -> List[Path]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    if hierarchy is None or len(contours) == 0:
        return []
    h = hierarchy[0]
    out: List[Path] = []
    for i, cnt in enumerate(contours):
        if int(h[i][3]) != -1:
            continue
        if cv2.contourArea(cnt) < float(min_area):
            continue
        outer_pts = _approx_contour(cnt, simplify_factor)
        if not outer_pts:
            continue

        holes: list[Path] = []
        child = int(h[i][2])
        while child != -1:
            hc = contours[child]
            if cv2.contourArea(hc) >= float(min_area):
                hp = _approx_contour(hc, simplify_factor)
                if hp:
                    holes.append(Path(points=hp, closed=True, holes=None))
            child = int(h[child][0])

        out.append(Path(points=outer_pts, closed=True, holes=holes or None))
    return out
