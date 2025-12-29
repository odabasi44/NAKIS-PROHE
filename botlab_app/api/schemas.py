from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    id: str


class VectorProcessRequest(BaseModel):
    id: str = Field(..., description="Upload ID")
    # Fotoğraf modunda daha doğal tonlar için 8 gibi değerler istenebilir.
    num_colors: int = Field(3, ge=2, le=12)
    width: int | None = Field(None, ge=64, le=3000)
    height: int | None = Field(None, ge=64, le=3000)
    keep_ratio: bool = True
    mode: str = Field("photo", description="photo|logo")
    outline: bool = True
    outline_thickness: int = Field(1, ge=1, le=6)


class VectorProcessResponse(BaseModel):
    id: str
    eps_url: str
    png_url: str
    colors: list[str]


class EmbroideryProcessRequest(BaseModel):
    id: str
    format: str = Field("dst", description="bot|dst|pes|exp|jef|xxx|vp3")
    num_colors: int = Field(3, ge=2, le=12)
    width: int | None = Field(None, ge=64, le=3000)
    height: int | None = Field(None, ge=64, le=3000)
    keep_ratio: bool = True
    mode: str = Field("photo", description="photo|logo")
    outline: bool = True
    outline_thickness: int = Field(1, ge=1, le=6)


class EmbroideryProcessResponse(BaseModel):
    id: str
    bot_url: str
    file_url: str
