"""
Abstract base class for segmentation models.
Supports hot-swapping models (U²-Net, MODNet, SAM, SAM2) behind a unified interface.
"""
from abc import ABC, abstractmethod
import numpy as np
import torch


class SegmentationModel(ABC):
    """Abstract interface for foreground/salient object segmentation models."""

    @abstractmethod
    def load(self, weights_path: str, device: torch.device) -> None:
        """Load model weights onto the specified device."""
        pass

    @abstractmethod
    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a preprocessed image tensor (1, C, H, W).
        Returns a soft probability mask tensor (1, 1, H, W) in range [0, 1].
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model from GPU/CPU memory."""
        pass
