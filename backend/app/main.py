from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_data import router as data_router
from app.api.routes_examples import router as examples_router
from app.api.routes_model import router as model_router
from app.api.routes_predict import router as predict_router
from app.api.routes_session import router as session_router

app = FastAPI(title="LightGBM Visualizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_router)
app.include_router(data_router)
app.include_router(examples_router)
app.include_router(predict_router)
app.include_router(session_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
