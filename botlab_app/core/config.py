from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    output_dir: str = os.getenv("BOTLAB_OUTPUT_DIR", "static/output")
    models_dir: str = os.getenv("BOTLAB_MODELS_DIR", "models")
    u2net_path: str = os.getenv("BOTLAB_U2NET_PATH", os.path.join("models", "u2net.onnx"))

settings = Settings()
