from __future__ import annotations
import os
import uuid

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def new_id() -> str:
    return uuid.uuid4().hex
