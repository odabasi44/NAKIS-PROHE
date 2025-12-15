from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from .path_tools import Path

@dataclass
class EPSData:
    eps_bytes: bytes
    width: int
    height: int

def _eps_poly(path: Path, *, canvas_h: int) -> bytes:
    parts: list[bytes] = []
    parts.append(b"newpath\n")

    def emit(points: list[tuple[float, float]]):
        x0, y0 = points[0]
        y0 = canvas_h - y0
        parts.append(f"{int(round(x0))} {int(round(y0))} moveto\n".encode("ascii"))
        for x, y in points[1:]:
            y = canvas_h - y
            parts.append(f"{int(round(x))} {int(round(y))} lineto\n".encode("ascii"))
        parts.append(b"closepath\n")

    emit(path.points)
    if path.holes:
        for h in path.holes:
            emit(h.points)
    parts.append(b"eofill\n")
    return b"".join(parts)

def paths_to_eps(layers: List[Tuple[Tuple[int, int, int], List[Path]]], *, width: int, height: int) -> EPSData:
    out: list[bytes] = []
    out.append(b"%!PS-Adobe-3.0 EPSF-3.0\n")
    out.append(f"%%BoundingBox: 0 0 {int(width)} {int(height)}\n".encode("ascii"))
    out.append(b"%%LanguageLevel: 2\n%%EndComments\n")
    out.append(b"gsave\n")

    for (r, g, b), paths in layers:
        if not paths:
            continue
        out.append(f"gsave {r/255.0:.4f} {g/255.0:.4f} {b/255.0:.4f} setrgbcolor\n".encode("ascii"))
        for p in paths:
            out.append(_eps_poly(p, canvas_h=int(height)))
        out.append(b"grestore\n")

    out.append(b"grestore\nshowpage\n%%EOF\n")
    return EPSData(eps_bytes=b"".join(out), width=int(width), height=int(height))
