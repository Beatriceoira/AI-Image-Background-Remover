"""
Image preprocessing pipeline for segmentation model input.
Handles EXIF orientation, RGB conversion, resizing, normalization, tensor conversion.
"""
import numpy as np
import torch
from PIL import Image, ImageOps
from .inference import get_inference_service


class PreprocessingService:
    """
    Preprocessing pipeline:
    1. Ensure RGB format (drop alpha, convert grayscale)
    2. Apply EXIF orientation correction
    3. Validate dimensions against configured limits
    4. Resize to model input size (bilinear)
    5. Normalize to [0, 1] float32
    6. Transpose to (C, H, W) and add batch dimension
    """

    def __init__(self):
        self._inference = get_inference_service()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a NumPy RGB image (H, W, 3) uint8 for model inference.

        Returns:
            Tensor of shape (1, 3, H_model, W_model) float32 in [0, 1]
        """
        # 1. Ensure RGB
        if image.ndim == 2:
            # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            # RGBA -> RGB (drop alpha)
            image = image[:, :, :3]

        # 2. EXIF orientation correction (via PIL)
        pil_img = Image.fromarray(image.astype("uint8"), mode="RGB")
        pil_img = ImageOps.exif_transpose(pil_img)
        image = np.array(pil_img)

        # 3. Dimension validation (using config limits)
        from ..core.config import Config
        h, w = image.shape[:2]
        if w > Config.MAX_IMAGE_WIDTH or h > Config.MAX_IMAGE_HEIGHT:
            raise ValueError(
                f"Image dimensions ({w}x{h}) exceed maximum allowed "
                f"({Config.MAX_IMAGE_WIDTH}x{Config.MAX_IMAGE_HEIGHT})"
            )

        # 4. Resize to model input size
        model_h, model_w = self._inference.model_input_size
        pil_img = pil_img.resize((model_w, model_h), Image.BILINEAR)
        image_resized = np.array(pil_img)

        # 5. Normalize to [0, 1] and transpose to (C, H, W)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_transposed = np.transpose(image_normalized, (2, 0, 1))

        # 6. Add batch dimension
        image_tensor = torch.from_numpy(image_transposed).unsqueeze(0)

        return image_tensor