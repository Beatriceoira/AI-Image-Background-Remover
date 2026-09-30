"""
Image utility helpers for encoding, decoding, EXIF handling, and color conversion.
"""
import io
import uuid
import numpy as np
from PIL import Image, ImageOps


def bytes_to_numpy(image_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Decode image bytes into an RGB NumPy array (H, W, 3) uint8.
    Automatically applies EXIF orientation correction.
    Returns: (numpy_array, original_size_hw)
    """
    img = Image.open(io.BytesIO(image_bytes))
    # Correct EXIF rotation if present
    img = ImageOps.exif_transpose(img)
    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    np_arr = np.array(img, dtype=np.uint8)
    return np_arr, np_arr.shape[:2]


def numpy_to_bytes(image_array: np.ndarray, format: str = "PNG", quality: int = 95) -> tuple[bytes, str]:
    """
    Convert a NumPy image array (H, W, 3 or 4) to encoded image bytes.
    Returns: (bytes, media_type)
    """
    if image_array.ndim == 2:
        # Grayscale mask
        pil_img = Image.fromarray(image_array, mode="L")
    elif image_array.shape[2] == 4:
        pil_img = Image.fromarray(image_array, mode="RGBA")
    elif image_array.shape[2] == 3:
        pil_img = Image.fromarray(image_array, mode="RGB")
    else:
        raise ValueError(f"Unsupported image array shape: {image_array.shape}")

    fmt_upper = format.upper()
    buf = io.BytesIO()

    if fmt_upper == "JPEG" or fmt_upper == "JPG":
        if pil_img.mode == "RGBA":
            # Flatten RGBA against white for JPEG
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])
            pil_img = bg
        pil_img.save(buf, format="JPEG", quality=quality)
        media_type = "image/jpeg"
    elif fmt_upper == "WEBP":
        pil_img.save(buf, format="WEBP", quality=quality)
        media_type = "image/webp"
    else:
        # Default PNG
        pil_img.save(buf, format="PNG")
        media_type = "image/png"

    buf.seek(0)
    return buf.getvalue(), media_type


def parse_hex_color(hex_color: str) -> tuple[int, int, int]:
    """
    Parse a hex color string (e.g. '#FFFFFF', '#00ff00', '123456') to an (R, G, B) tuple.
    """
    cleaned = hex_color.lstrip("#").strip()
    if len(cleaned) == 3:
        cleaned = "".join([c * 2 for c in cleaned])
    if len(cleaned) != 6:
        raise ValueError(f"Invalid hex color format: '{hex_color}'. Expected format like '#FFFFFF' or 'FFF'.")
    try:
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
        return (r, g, b)
    except ValueError:
        raise ValueError(f"Invalid hex color format: '{hex_color}'.")


def generate_safe_filename(prefix: str = "matteforge", ext: str = "png") -> str:
    """Generate a safe, unique server-side filename."""
    unique_id = uuid.uuid4().hex
    return f"{prefix}_{unique_id}.{ext}"
