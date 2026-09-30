"""
Input validation utilities for image uploads.
Enforces security, MIME checking, magic-byte signatures, size & dimension limits,
and decompression bomb protection.
"""
import io
from fastapi import HTTPException, status, UploadFile
from PIL import Image
from ..core.config import Config

# Magic bytes for common image formats
MAGIC_BYTES = {
    "JPEG": [b"\xFF\xD8\xFF"],
    "PNG": [b"\x89PNG\r\n\x1a\n"],
    "WEBP": [b"RIFF"],  # standard RIFF header; RIFF...WEBP
    "BMP": [b"BM"],
    "TIFF": [b"II*\x00", b"MM\x00*"],
}

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


async def validate_image_upload(file: UploadFile) -> bytes:
    """
    Comprehensively validate an uploaded image file:
    1. Checks MIME type header.
    2. Enforces maximum file size limit.
    3. Verifies magic bytes signature against image format.
    4. Protects against decompression bombs.
    5. Validates image dimensions against configured maximums.

    Returns raw byte contents if valid.
    Raises HTTPException on validation failure.
    """
    # 1. Validate MIME type
    if not file.content_type or file.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported media type: '{file.content_type}'. Allowed types: JPEG, PNG, WEBP, BMP, TIFF."
        )

    # 2. Read contents and check file size limit
    contents = await file.read()
    max_bytes = Config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({len(contents) / (1024*1024):.2f} MB) exceeds maximum allowed limit of {Config.MAX_UPLOAD_SIZE_MB} MB."
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # 3. Verify magic bytes signature
    matched_magic = False
    for signatures in MAGIC_BYTES.values():
        for sig in signatures:
            if contents.startswith(sig):
                matched_magic = True
                break
        if matched_magic:
            break

    if not matched_magic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content does not match a valid image magic-byte signature."
        )

    # 4. Decompression bomb protection & Image decode validation
    Image.MAX_IMAGE_PIXELS = Config.MAX_IMAGE_WIDTH * Config.MAX_IMAGE_HEIGHT
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Image.DecompressionBombError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image pixel count exceeds maximum limit (potential decompression bomb)."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or unreadable image file: {str(e)}"
        )

    # Re-open after verify()
    try:
        img = Image.open(io.BytesIO(contents))
        width, height = img.size
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to inspect image dimensions: {str(e)}"
        )

    # 5. Dimension validation
    if width > Config.MAX_IMAGE_WIDTH or height > Config.MAX_IMAGE_HEIGHT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image dimensions ({width}x{height}) exceed maximum allowed dimensions of {Config.MAX_IMAGE_WIDTH}x{Config.MAX_IMAGE_HEIGHT}."
        )

    return contents
