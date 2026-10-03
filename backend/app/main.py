from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.services.classifier import FruitClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = FruitClassifier(
        settings.model_path, settings.labels_path
    )
    yield


app = FastAPI(
    title="Fruit Classification API",
    description="Classify fruit images with a pretrained Swin model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
