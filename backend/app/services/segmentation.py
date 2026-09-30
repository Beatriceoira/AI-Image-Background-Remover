"""
Segmentation service — runs model inference and resizes soft mask to original image dimensions.
"""
import numpy as np
import torch
from PIL import Image
from .inference import get_inference_service


class SegmentationService:
    """
    Generates a soft foreground probability mask from an image.
    """

    def __init__(self):
        self._inference = get_inference_service()

    def segment(self, original_image: np.ndarray, preprocessed_tensor: torch.Tensor) -> np.ndarray:
        """
        Run segmentation inference and resize mask to original image size.

        Args:
            original_image: Original RGB image array (H, W, 3) uint8
            preprocessed_tensor: Preprocessed tensor (1, 3, 320, 320)

        Returns:
            Foreground mask as uint8 array (H_orig, W_orig) in range [0, 255]
        """
        original_h, original_w = original_image.shape[:2]

        # Run inference (returns sigmoid-activated mask in [0, 1])
        with torch.inference_mode():
            mask_tensor = self._inference.predict(preprocessed_tensor)  # (1, 1, 320, 320)

        # Convert to numpy and squeeze
        mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()  # (320, 320)

        # Resize back to original image size
        mask_pil = Image.fromarray((mask_np * 255).astype("uint8"), mode="L")
        mask_pil = mask_pil.resize((original_w, original_h), Image.BILINEAR)
        mask_final = np.array(mask_pil).astype(np.uint8)  # (H, W) in [0, 255]

        return mask_final