"""
Background removal API endpoint.
Full pipeline: upload → validation → preprocessing → segmentation → mask refinement → matting → output.
"""
import io
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Query, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np

from ..core.config import Config
from ..services.inference import get_inference_service
from ..services.preprocessing import PreprocessingService
from ..services.segmentation import SegmentationService
from ..services.matting import MattingService
from ..services.postprocessing import PostprocessingService
from ..utils.validation import validate_image_upload
from ..utils.image_utils import bytes_to_numpy, numpy_to_bytes, parse_hex_color, generate_safe_filename

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
preprocess_service = PreprocessingService()
segmentation_service = SegmentationService()
matting_service = MattingService()
postprocess_service = PostprocessingService()


@router.post("/remove-background")
async def remove_background(
    image: UploadFile = File(...),
    background_mode: str = Query("transparent", pattern="^(transparent|solid|gradient|image)$"),
    background_color: str | None = Query(None, description="Hex color for solid background (e.g., #FFFFFF)"),
    refine_edges: bool = Query(False, description="Apply edge refinement"),
    blur_radius: int = Query(0, ge=0, le=10, description="Gaussian blur radius for mask smoothing (0-10)"),
    median_filter: int = Query(0, ge=0, le=10, description="Median filter radius for noise removal (0-10)"),
    hole_fill: bool = Query(True, description="Fill small holes in mask"),
    edge_smooth: bool = Query(True, description="Smooth mask edges with morphological operations"),
    color_decontamination: bool = Query(False, description="Remove background color bleed at edges"),
    output_format: str = Query("PNG", pattern="^(PNG|JPEG|WEBP)$"),
    background_image: UploadFile | None = File(None, description="Optional background image for 'image' mode"),
):
    """
    Remove background from uploaded image.

    Parameters:
        image: Foreground image file (required)
        background_mode: 'transparent', 'solid', 'gradient', or 'image'
        background_color: Hex color string (e.g., '#FFFFFF') for solid background mode
        refine_edges: Enable full edge refinement pipeline
        blur_radius: Gaussian blur radius (0-10)
        median_filter: Median filter radius (0-10)
        hole_fill: Fill small holes in mask
        edge_smooth: Apply morphological edge smoothing
        color_decontamination: Reduce background color contamination at edges
        output_format: Output format - 'PNG', 'JPEG', or 'WEBP'
        background_image: Optional background image for 'image' mode

    Returns:
        Processed image as binary response with appropriate Content-Type.
    """
    # 1. Validate main image
    contents = await validate_image_upload(image)

    # 2. Validate background image if provided
    bg_contents = None
    if background_mode == "image":
        if background_image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="background_image is required when background_mode='image'"
            )
        bg_contents = await validate_image_upload(background_image)

    # 3. Parse background color
    bg_color = None
    if background_color:
        try:
            bg_color = parse_hex_color(background_color)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

    # 4. Decode image to NumPy RGB array
    try:
        np_image, original_size = bytes_to_numpy(contents)
    except Exception as e:
        logger.error(f"Image decode failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image"
        )

    # 5. Decode background image if provided
    bg_np = None
    if bg_contents:
        try:
            bg_np, _ = bytes_to_numpy(bg_contents)
        except Exception as e:
            logger.error(f"Background image decode failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to decode background image"
            )

    # 6. Preprocessing
    try:
        preprocessed = preprocess_service.preprocess(np_image)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Preprocessing failed"
        )

    # 7. Segmentation
    try:
        mask = segmentation_service.segment(np_image, preprocessed)
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Segmentation failed"
        )

    # 8. Post-processing / mask refinement
    if refine_edges or blur_radius > 0 or median_filter > 0 or not hole_fill or not edge_smooth or color_decontamination:
        try:
            post_config = {
                "blur_radius": blur_radius,
                "median_filter": median_filter,
                "hole_fill": hole_fill,
                "edge_smooth": edge_smooth,
                "color_decontamination": color_decontamination,
            }
            # Refine mask
            mask = postprocess_service.refine_mask(mask, post_config)
        except Exception as e:
            logger.error(f"Mask refinement failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Mask refinement failed"
            )

    # 9. Matting with optional background
    try:
        matted = matting_service.matte(
            np_image,
            mask,
            background_mode=background_mode,
            background_color=bg_color,
            background_image=bg_np,
        )
    except Exception as e:
        logger.error(f"Matting failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Matting failed"
        )

    # 10. Final post-processing (optional color decontamination applied in postprocess)
    # Note: Matting already handled basic compositing; postprocess applies decontamination if enabled
    try:
        final = postprocess_service.postprocess(np_image, mask, {
            "color_decontamination": color_decontamination,
        })
        # If background_mode was not transparent, we need to re-composite
        if background_mode != "transparent":
            final = matting_service.matte(
                np_image,
                mask,
                background_mode=background_mode,
                background_color=bg_color,
                background_image=bg_np,
            )
    except Exception as e:
        logger.error(f"Final postprocessing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Postprocessing failed"
        )

    # 11. Encode and return
    try:
        img_bytes, media_type = numpy_to_bytes(final, format=output_format)
    except Exception as e:
        logger.error(f"Image encoding failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image encoding failed"
        )

    # Return as streaming response
    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type=media_type,
        headers={
            "X-Processing-Info": f"size={original_size[1]}x{original_size[0]};format={output_format};device={Config.get_device()}"
        }
    )


@router.get("/remove-background/info")
async def remove_background_info():
    """
    Get information about the background removal endpoint capabilities.
    """
    return {
        "endpoint": "/api/remove-background",
        "method": "POST",
        "parameters": {
            "image": "Required: Foreground image file",
            "background_mode": "Optional: 'transparent', 'solid', 'gradient', 'image' (default: 'transparent')",
            "background_color": "Optional: Hex color for solid background (e.g., '#FFFFFF')",
            "background_image": "Optional: Background image file for 'image' mode",
            "refine_edges": "Optional: Enable edge refinement pipeline (default: false)",
            "blur_radius": "Optional: Gaussian blur radius 0-10 (default: 0)",
            "median_filter": "Optional: Median filter radius 0-10 (default: 0)",
            "hole_fill": "Optional: Fill small holes in mask (default: true)",
            "edge_smooth": "Optional: Morphological edge smoothing (default: true)",
            "color_decontamination": "Optional: Remove background color bleed (default: false)",
            "output_format": "Optional: 'PNG', 'JPEG', 'WEBP' (default: 'PNG')",
        },
        "supported_formats": ["JPEG", "PNG", "WEBP", "BMP", "TIFF"],
        "max_file_size_mb": Config.MAX_UPLOAD_SIZE_MB,
        "max_dimensions": f"{Config.MAX_IMAGE_WIDTH}x{Config.MAX_IMAGE_HEIGHT}",
    }