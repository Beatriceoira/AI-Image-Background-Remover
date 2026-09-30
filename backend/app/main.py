"""
MatteForge - AI Background Remover FastAPI Application.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .api.routes_health import router as health_router
from .api.routes_remove import router as remove_router
from .core.logging_config import LoggingConfig
from .core.config import Config
from .services.inference import get_inference_service

# Initialize logging
LoggingConfig()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - eager model loading and warmup on startup."""
    logger.info("Starting MatteForge application...")

    # Eagerly load the model on startup
    inference = get_inference_service()
    try:
        inference.load_model()
        logger.info("Model loaded successfully on startup.")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Don't crash - allow the app to start so /health can report the error
        # The endpoint will return model_loaded: false

    yield

    # Shutdown cleanup
    logger.info("Shutting down MatteForge application...")
    inference.unload()


app = FastAPI(
    title="MatteForge",
    description="AI-powered background remover using U²-Net segmentation",
    version="0.1.0",
    lifespan=lifespan,
)

# Register API routers
app.include_router(health_router, tags=["health"])
app.include_router(remove_router, prefix="/api", tags=["background removal"])

# Serve frontend static files
import os
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def root():
    """Serve the frontend index.html"""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "MatteForge API is running. Visit /docs for API documentation."}


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon if exists"""
    favicon_path = os.path.join(frontend_dir, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return {"message": "No favicon"}