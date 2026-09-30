import numpy as np
from PIL import Image

class ImageService:
    def __init__(self):
        pass

    def read_image(self, path):
        """
        Read an image from a file path.
        Returns a numpy array in RGB format.
        """
        try:
            pil_image = Image.open(path)
            # Ensure RGB mode (handles RGBA, grayscale, etc.)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return np.array(pil_image)
        except Exception as e:
            raise ValueError(f"Failed to read image from {path}: {e}")

    def save_image(self, image_array, path, format='PNG'):
        """
        Save a numpy array as an image.
        image_array: numpy array (H, W, C) dtype uint8
        """
        try:
            pil_image = Image.fromarray(image_array)
            pil_image.save(path, format=format)
            return True
        except Exception as e:
            raise ValueError(f"Failed to save image to {path}: {e}")