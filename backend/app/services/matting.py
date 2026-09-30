"""
Matting service — applies mask to create RGBA foreground with optional background replacement.
Supports: transparent, solid color, gradient, and custom background image.
"""
import numpy as np
from PIL import Image
from ..utils.image_utils import parse_hex_color


class MattingService:
    """
    Alpha matting and background compositing.
    """

    def __init__(self):
        pass

    def matte(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background_mode: str = "transparent",
        background_color: tuple[int, int, int] | None = None,
        background_image: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply alpha mask to create RGBA foreground composited onto optional background.

        Args:
            image: RGB image (H, W, 3) uint8
            mask: Foreground mask (H, W) uint8 [0, 255] or float [0, 1]
            background_mode: 'transparent' | 'solid' | 'gradient' | 'image'
            background_color: RGB tuple for solid background
            background_image: RGB array for image background

        Returns:
            RGBA image (H, W, 4) uint8
        """
        # Ensure image is RGB
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        # Normalize mask to float [0, 1]
        mask_float = self._normalize_mask(mask)

        # Get background according to mode
        background = self._get_background(
            image.shape[:2], background_mode, background_color, background_image
        )

        # Alpha composite
        rgba = self._composite(image, mask_float, background)

        return rgba

    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Normalize mask to float [0, 1]."""
        if mask.ndim == 3:
            mask = mask.squeeze(axis=-1)
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        return np.clip(mask, 0.0, 1.0)

    def _get_background(
        self,
        shape: tuple[int, int],
        mode: str,
        color: tuple[int, int, int] | None,
        image: np.ndarray | None,
    ) -> np.ndarray | None:
        """Create background array or return None for transparency."""
        h, w = shape[:2]

        if mode == "transparent":
            return None
        elif mode == "solid":
            if color is None:
                color = (0, 0, 0)
            return np.full((h, w, 3), color, dtype=np.uint8)
        elif mode == "gradient":
            return self._create_gradient(h, w)
        elif mode == "image":
            if image is not None:
                bg_h, bg_w = image.shape[:2]
                if bg_h != h or bg_w != w:
                    pil_img = Image.fromarray(image.astype("uint8"))
                    pil_img = pil_img.resize((w, h), Image.BILINEAR)
                    return np.array(pil_img)
                return image
            return np.zeros((h, w, 3), dtype=np.uint8)
        else:
            return np.zeros((h, w, 3), dtype=np.uint8)

    def _create_gradient(self, height: int, width: int) -> np.ndarray:
        """Create a linear gradient from white (top) to gray (bottom)."""
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            intensity = int(255 * (1 - i / height))
            gradient[i, :, :] = (intensity, intensity, intensity)
        return gradient

    def _composite(
        self,
        foreground: np.ndarray,
        mask: np.ndarray,
        background: np.ndarray | None,
    ) -> np.ndarray:
        """
        Alpha composite foreground onto background using the mask as alpha.
        """
        # Ensure mask is float [0, 1]
        mask = self._normalize_mask(mask)
        alpha = (mask * 255).astype(np.uint8)

        if background is None:
            # Transparent background - just attach alpha channel
            rgba = np.dstack([foreground, alpha])
        else:
            # Composite onto background
            fg_f = foreground.astype(np.float32) / 255.0
            bg_f = background.astype(np.float32) / 255.0
            mask_f = mask[:, :, np.newaxis]

            # Alpha blending: result = fg * alpha + bg * (1 - alpha)
            composite = fg_f * mask_f + bg_f * (1 - mask_f)
            composite = (composite * 255).astype(np.uint8)

            rgba = np.dstack([composite, alpha])

        return rgba