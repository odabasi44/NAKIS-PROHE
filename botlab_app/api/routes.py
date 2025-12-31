from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

from api.schemas import (
    UploadResponse,
    VectorProcessRequest,
    VectorProcessResponse,
    EmbroideryProcessRequest,
    EmbroideryProcessResponse,
)
from core.config import settings
from core.utils import ensure_dir, new_id
from services.image_pipeline import ImagePipeline
from services.vector_pipeline import rgba_to_eps
from services.embroidery_pipeline import eps_layers_to_bot, bot_to_pattern, export_pattern

router = APIRouter()

ensure_dir(settings.output_dir)
_pipeline = ImagePipeline(settings.u2net_path)

@router.get("/health")
async def health():
    return {"ok": True}

def _path_for(id_: str, ext: str) -> str:
    return str(Path(settings.output_dir) / f"{id_}.{ext}")

@router.post("/upload", response_model=UploadResponse)
async def upload(image: UploadFile = File(...)):
    if not image:
        raise HTTPException(status_code=400, detail="image required")
    id_ = new_id()
    raw_path = _path_for(id_, "source")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    with open(raw_path, "wb") as f:
        f.write(data)
    return UploadResponse(id=id_)

@router.post("/process/vector", response_model=VectorProcessResponse)
async def process_vector(req: VectorProcessRequest):
    raw_path = _path_for(req.id, "source")
    if not os.path.exists(raw_path):
        raise HTTPException(status_code=404, detail="id not found")

    with Image.open(raw_path) as im:
        res = _pipeline.run(im, num_colors=req.num_colors, width=req.width, height=req.height, keep_ratio=req.keep_ratio)

    eps_res = rgba_to_eps(res.rgba, simplify_factor=0.003, min_area=20.0)
    eps_path = _path_for(req.id, "eps")
    png_path = _path_for(req.id, "png")
    with open(eps_path, "wb") as f:
        f.write(eps_res.eps_bytes)
    with open(png_path, "wb") as f:
        f.write(eps_res.preview_png_bytes)

    return VectorProcessResponse(
        id=req.id,
        eps_url=f"/api/static/{req.id}.eps",
        png_url=f"/api/static/{req.id}.png",
        colors=eps_res.palette,
    )

@router.post("/process/embroidery", response_model=EmbroideryProcessResponse)
async def process_embroidery(req: EmbroideryProcessRequest):
    raw_path = _path_for(req.id, "source")
    if not os.path.exists(raw_path):
        raise HTTPException(status_code=404, detail="id not found")

    fmt = (req.format or "dst").lower()
    if fmt not in ("bot", "dst", "pes", "exp", "jef", "xxx", "vp3"):
        raise HTTPException(status_code=400, detail="unsupported format")

    with Image.open(raw_path) as im:
        res = _pipeline.run(im, num_colors=req.num_colors, width=req.width, height=req.height, keep_ratio=req.keep_ratio)

    rgba_np = __import__("numpy").array(res.rgba.convert("RGBA"))
    bot = eps_layers_to_bot(rgba_np, unit="px")
    bot_path = _path_for(req.id, "bot")
    with open(bot_path, "w", encoding="utf-8") as f:
        import json
        json.dump(bot, f, ensure_ascii=False, indent=2)

    if fmt == "bot":
        return EmbroideryProcessResponse(id=req.id, bot_url=f"/api/static/{req.id}.bot", file_url=f"/api/static/{req.id}.bot")

    pattern = bot_to_pattern(bot)
    out_bytes = export_pattern(pattern, fmt)
    out_path = _path_for(req.id, fmt)
    with open(out_path, "wb") as f:
        f.write(out_bytes)

    return EmbroideryProcessResponse(id=req.id, bot_url=f"/api/static/{req.id}.bot", file_url=f"/api/static/{req.id}.{fmt}")

@router.get("/static/{filename}")
async def get_static(filename: str):
    if not filename:
        raise HTTPException(status_code=400, detail="filename required")

    safe = filename.replace("/", "").replace("\\", "")
    if safe in (".", "..") or ".." in safe:
        raise HTTPException(status_code=400, detail="invalid filename")

    base_dir = Path(settings.output_dir).resolve()
    target = (base_dir / safe).resolve()
    if base_dir not in target.parents and target != base_dir:
        raise HTTPException(status_code=400, detail="invalid filename")

    path = str(target)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path)
