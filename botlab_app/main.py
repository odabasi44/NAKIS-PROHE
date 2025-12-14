from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from botlab_app.api.routes import router as api_router


app = FastAPI(title="Botlab Web Embroidery Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


