"""
Health check API endpoint.
"""
import logging
from fastapi import APIRouter
from ..core.config import Config
from ..services.inference import get_inference_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health():
    """Health check endpoint with model and device status."""
    inference = get_inference_service()
    return {
        "status": "ok",
        "model": Config.MODEL_NAME,
        "device": Config.get_device(),
        "model_loaded": inference.is_ready(),
    }