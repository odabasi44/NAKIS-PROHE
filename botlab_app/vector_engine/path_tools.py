from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[float, float]

@dataclass
class Path:
    points: List[Point]
    closed: bool = True
    holes: List["Path"] | None = None
