"""
Inference service — singleton model manager with eager loading & warmup.
Provides a single loaded instance for the entire application lifetime.
"""
import os
import logging
import threading
import torch
from ..models.base import SegmentationModel
from ..models.u2net import U2NetModel
from ..core.config import Config

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Singleton manager for segmentation model.
    Loads model once at startup, provides inference method.
    """

    _instance: "InferenceService | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "InferenceService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._model: SegmentationModel | None = None
        self._device: torch.device = torch.device("cpu")
        self._model_input_size = (320, 320)
        self._initialized = True

    @property
    def model(self) -> SegmentationModel:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model_input_size(self) -> tuple[int, int]:
        return self._model_input_size

    def load_model(self, weights_path: str | None = None, device: torch.device | None = None) -> None:
        """Load the U²-Net model onto the specified device with pretrained weights."""
        if self._model is not None:
            logger.info("Model already loaded.")
            return

        if device is None:
            if Config.ENABLE_GPU and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self._device = device

        if weights_path is None:
            weights_path = Config.MODEL_PATH

        logger.info(f"Loading U²-Net model onto {device} from {weights_path}...")
        try:
            self._model = U2NetModel()
            self._model.load(weights_path, device)
        except FileNotFoundError:
            logger.error(f"Model weights not found at {weights_path}. Cannot proceed.")
            raise RuntimeError(f"Model weights not found at {weights_path}. Run scripts/download_model.py to download.")
        except RuntimeError as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Warmup inference
        self._warmup()
        logger.info("Model loaded and warmed up successfully.")

    def _warmup(self) -> None:
        """Run a dummy forward pass to warm up GPU kernels and caches."""
        if self._model is None:
            return
        logger.info("Warming up model...")
        dummy = torch.randn(1, 3, *self._model_input_size, device=self._device)
        try:
            with torch.inference_mode():
                _ = self._model.predict(dummy)
            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a preprocessed image tensor.

        Args:
            image_tensor: Tensor of shape (1, 3, H, W) normalized to [0, 1]

        Returns:
            Tensor of shape (1, 1, H, W) with values in [0, 1] (sigmoid activated)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model.predict(image_tensor)

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            self._model.unload()
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded.")

    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._model is not None and self._model.is_loaded


def get_inference_service() -> InferenceService:
    """Dependency-injection style getter for the inference singleton."""
    return InferenceService()