"""
Post-processing service for mask refinement and color decontamination.
Configurable options: Gaussian blur, median filter, hole filling, edge smoothing, color decontamination.
"""
import numpy as np
from scipy import ndimage

try:
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class PostprocessingService:
    """
    Configurable post-processing for masks and foreground images.
    """

    def __init__(self):
        self.default_config = {
            "blur_radius": 0,              # Gaussian blur radius (0 = no blur)
            "median_filter": 0,            # Median filter radius (0 = no filter)
            "hole_fill": True,             # Fill small holes in mask
            "edge_smooth": True,           # Smooth edges with morphological ops
            "color_decontamination": False,  # Remove background color bleed at edges
        }

    def refine_mask(self, mask: np.ndarray, config: dict | None = None) -> np.ndarray:
        """
        Apply refinement operations to the mask.

        Args:
            mask: Input mask (H, W) - float [0, 1] or uint8 [0, 255]
            config: Processing configuration dict

        Returns:
            Refined mask as float array in [0, 1]
        """
        if config is None:
            config = {}
        process_config = {**self.default_config, **config}

        # Convert to float [0, 1] if needed
        if mask.dtype == np.uint8:
            mask_float = mask.astype(np.float32) / 255.0
        else:
            mask_float = mask.astype(np.float32)
            mask_float = np.clip(mask_float, 0.0, 1.0)

        # Apply Gaussian blur for edge smoothing
        blur_radius = process_config.get("blur_radius", 0)
        if blur_radius > 0:
            mask_float = self._gaussian_blur(mask_float, blur_radius)

        # Apply median filter for noise removal
        median_radius = process_config.get("median_filter", 0)
        if median_radius > 0 and HAS_SCIPY:
            mask_float = self._median_filter(mask_float, median_radius)

        # Fill holes in the mask
        if process_config.get("hole_fill", True) and HAS_SCIPY:
            mask_float = self._fill_holes(mask_float)

        # Apply edge smoothing
        if process_config.get("edge_smooth", True) and HAS_SCIPY:
            mask_float = self._smooth_edges(mask_float)

        return mask_float

    def decontaminate_colors(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bg_samples: int = 100,
    ) -> np.ndarray:
        """
        Remove background color contamination at foreground edges.

        Args:
            image: RGB image (H, W, 3) uint8
            mask: Foreground mask (H, W) float [0, 1]
            bg_samples: Number of background pixels to sample for color estimation

        Returns:
            Cleaned RGB image (H, W, 3) uint8
        """
        if not self.default_config.get("color_decontamination", False):
            return image

        # Get background pixels (where mask < 0.5)
        bg_mask = mask < 0.5
        if not np.any(bg_mask):
            return image

        # Sample background colors
        bg_pixels = image[bg_mask]
        if len(bg_pixels) < bg_samples:
            bg_samples = len(bg_pixels)
        idx = np.random.choice(len(bg_pixels), bg_samples, replace=False)
        bg_color = bg_pixels[idx].mean(axis=0).astype(np.float32)

        # Estimate edge pixels (transition zone)
        edge_mask = (mask > 0.1) & (mask < 0.9)
        if not np.any(edge_mask):
            return image

        # Replace edge pixel colors by blending with background color
        edge_pixels = image[edge_mask].astype(np.float32)
        edge_alpha = mask[edge_mask][:, np.newaxis]
        cleaned_pixels = edge_pixels * edge_alpha + bg_color * (1 - edge_alpha)

        result = image.copy()
        result[edge_mask] = cleaned_pixels.astype(np.uint8)
        return result

    def postprocess(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        config: dict | None = None,
    ) -> np.ndarray:
        """
        Full post-processing: refine mask, optional color decontamination, create RGBA output.

        Args:
            image: RGB image (H, W, 3) uint8
            mask: Foreground mask (H, W) uint8 or float
            config: Processing configuration

        Returns:
            RGBA image (H, W, 4) uint8
        """
        if config is None:
            config = {}
        process_config = {**self.default_config, **config}

        # Ensure image is RGB
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        # Refine mask
        refined_mask = self.refine_mask(mask, process_config)

        # Optional color decontamination
        if process_config.get("color_decontamination", False):
            image = self.decontaminate_colors(image, refined_mask)

        # Create RGBA output
        alpha = (np.clip(refined_mask, 0.0, 1.0) * 255).astype(np.uint8)
        rgba = np.dstack([image, alpha])

        return rgba

    # --- Internal refinement methods ---

    def _gaussian_blur(self, mask: np.ndarray, radius: int) -> np.ndarray:
        """Apply Gaussian blur to mask."""
        if radius <= 0:
            return mask
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(mask, sigma=radius)
        except ImportError:
            return self._box_blur(mask, radius * 2 + 1)

    def _box_blur(self, mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """Simple box blur fallback."""
        if kernel_size < 1:
            return mask
        h, w = mask.shape
        result = np.zeros_like(mask)
        half = kernel_size // 2
        # Vectorized box blur using integral image
        integral = np.cumsum(np.cumsum(mask, axis=0), axis=1)
        integral = np.pad(integral, ((half + 1, 0), (half + 1, 0)), mode="constant")
        for i in range(h):
            i0, i1 = i, i + kernel_size
            for j in range(w):
                j0, j1 = j, j + kernel_size
                result[i, j] = (
                    integral[i1, j1] - integral[i0, j1] - integral[i1, j0] + integral[i0, j0]
                ) / (kernel_size * kernel_size)
        return result

    def _median_filter(self, mask: np.ndarray, radius: int) -> np.ndarray:
        """Apply median filter to remove noise."""
        if radius <= 0:
            return mask
        try:
            from scipy.ndimage import median_filter
            return median_filter(mask, size=radius * 2 + 1)
        except ImportError:
            return mask

    def _fill_holes(self, mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Fill small holes in the foreground mask."""
        if not HAS_SCIPY:
            return mask

        try:
            # Binarize
            binary = mask > threshold
            # Label connected components
            labeled, num_features = ndimage.label(binary)
            if num_features == 0:
                return mask
            # Count pixels in each component
            counts = ndimage.sum(binary, labeled, range(1, num_features + 1))
            # Find small components (holes) - less than 1% of image
            small_threshold = mask.size * 0.01
            small_components = counts < small_threshold
            # Fill small components (set to foreground)
            for i, is_small in enumerate(small_components):
                if is_small:
                    mask[labeled == i + 1] = 1.0
        except Exception:
            pass
        return mask

    def _smooth_edges(self, mask: np.ndarray) -> np.ndarray:
        """Smooth edges using morphological opening and closing."""
        if not HAS_SCIPY:
            return mask

        try:
            # Binary mask at threshold
            binary = mask > 0.5
            # Opening to remove small protrusions
            opened = ndimage.binary_opening(binary)
            # Closing to fill small gaps
            closed = ndimage.binary_closing(opened)
            # Blend with original soft mask
            return np.where(closed, np.maximum(mask, 0.5), mask * 0.5)
        except Exception:
            return mask