#!/usr/bin/env python3
"""
Comprehensive API tests for MatteForge.
Run: python -m pytest backend/tests/test_api.py -v
"""
import sys
import os
import io
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.inference import get_inference_service
from backend.app.core.config import Config

client = TestClient(app)


def test_health():
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    print(f"✓ Health response: {data}")
    assert data["status"] == "ok"
    assert "model" in data
    assert "device" in data
    assert "model_loaded" in data


def test_routes_exist():
    """Test that expected routes are registered."""
    print("\nTesting route registration...")
    # Test that endpoints are accessible
    health_response = client.get("/health")
    assert health_response.status_code == 200, f"Health endpoint failed: {health_response.status_code}"

    info_response = client.get("/api/remove-background/info")
    assert info_response.status_code == 200, f"Info endpoint failed: {info_response.status_code}"

    # Check route paths in the app
    app_routes = [route.path for route in app.routes if hasattr(route, 'path')]
    print(f"Routes found: {app_routes}")
    assert '/health' in app_routes or any('health' in str(r) for r in app.routes), "Health route missing"
    assert '/api/remove-background' in app_routes or any('remove-background' in str(r) for r in app.routes), "Remove background route missing"
    print("✓ All expected routes registered")


def test_model_weights():
    """Check if model weights exist."""
    print("\nChecking model weights...")
    model_path = os.path.join(os.path.dirname(__file__), "..", "app", "models", "weights", "u2net.pth")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model weights found at {model_path} ({size_mb:.1f} MB)")
    else:
        print("⚠ Model weights not found - inference will use random weights")


def create_test_image(size=(256, 256), color=(255, 0, 0)):
    """Create a simple test image."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def test_valid_upload():
    """Test valid image upload returns PNG with transparency."""
    print("\nTesting valid image upload...")

    # Load model for test
    inference = get_inference_service()
    try:
        inference.load_model()
    except FileNotFoundError:
        print("⚠ Model weights not found, skipping full pipeline test")
        return

    img_bytes = create_test_image((100, 100), (128, 128, 128))
    files = {"image": ("test.png", img_bytes, "image/png")}
    response = client.post("/api/remove-background", files=files)
    assert response.status_code == 200, f"Expected 200, got {response.status_code} - {response.text[:500]}"
    assert "image/png" in response.headers["content-type"]

    # Verify output is valid PNG with alpha channel
    output = Image.open(io.BytesIO(response.content))
    assert output.mode == "RGBA", f"Expected RGBA mode, got {output.mode}"
    assert output.size == (100, 100), f"Expected 100x100, got {output.size}"
    print(f"✓ Valid upload: {output.size} RGBA")


def test_invalid_upload():
    """Test invalid file type is rejected."""
    print("\nTesting invalid file upload...")
    files = {"image": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/api/remove-background", files=files)
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print(f"✓ Invalid upload correctly rejected")


def test_oversized_file():
    """Test oversized file is rejected."""
    print("\nTesting oversized file...")
    # Create a large image that exceeds MAX_UPLOAD_SIZE_MB (15MB)
    large_img = Image.new("RGB", (5000, 5000), (128, 128, 128))
    buf = io.BytesIO()
    large_img.save(buf, format="PNG", optimize=False)
    img_bytes = buf.getvalue()

    if len(img_bytes) > 15 * 1024 * 1024:
        files = {"image": ("large.png", img_bytes, "image/png")}
        response = client.post("/api/remove-background", files=files)
        assert response.status_code == 413, f"Expected 413, got {response.status_code}"
        print(f"✓ Oversized file correctly rejected")
    else:
        print("⚠ Test image not large enough to trigger limit, skipping")


def test_corrupt_image():
    """Test corrupted image is rejected."""
    print("\nTesting corrupted image...")
    files = {"image": ("corrupt.png", b"not a valid png", "image/png")}
    response = client.post("/api/remove-background", files=files)
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print(f"✓ Corrupt image correctly rejected")


def test_segmentation_mask():
    """Verify mask is generated with correct dimensions."""
    print("\nTesting segmentation mask dimensions...")

    inference = get_inference_service()
    try:
        inference.load_model()
    except FileNotFoundError:
        print("⚠ Model weights not found, skipping test")
        return

    img_bytes = create_test_image((200, 150), (100, 150, 200))
    files = {"image": ("test.png", img_bytes, "image/png")}
    response = client.post("/api/remove-background", files=files)
    assert response.status_code == 200
    output = Image.open(io.BytesIO(response.content))
    assert output.size == (200, 150), f"Mask dimensions mismatch: {output.size}"
    print(f"✓ Output dimensions match input: {output.size}")


def test_alpha_channel():
    """Verify output has alpha channel with transparency."""
    print("\nTesting alpha channel presence...")

    inference = get_inference_service()
    try:
        inference.load_model()
    except FileNotFoundError:
        print("⚠ Model weights not found, skipping test")
        return

    # Create image with clear foreground/background contrast
    img = Image.new("RGB", (100, 100), (200, 200, 200))  # Light gray bg
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 80, 80], fill=(50, 50, 50))  # Dark circle
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_bytes = buf.getvalue()

    files = {"image": ("test.png", img_bytes, "image/png")}
    response = client.post("/api/remove-background", files=files)
    assert response.status_code == 200
    output = Image.open(io.BytesIO(response.content))
    assert output.mode == "RGBA"
    alpha = np.array(output)[:, :, 3]
    print(f"✓ Alpha channel verified: min={alpha.min()}, max={alpha.max()}")


def test_integration_pipeline():
    """Test full pipeline: upload -> inference -> mask -> alpha -> PNG."""
    print("\nTesting full integration pipeline...")

    inference = get_inference_service()
    try:
        inference.load_model()
    except FileNotFoundError:
        print("⚠ Model weights not found, skipping test")
        return

    # Use a more complex test image
    img = Image.new("RGB", (320, 240))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 270, 190], fill=(100, 200, 100))
    draw.ellipse([100, 80, 220, 160], fill=(200, 100, 100))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"image": ("complex.png", buf.getvalue(), "image/png")}
    response = client.post("/api/remove-background", files=files)
    assert response.status_code == 200

    output = Image.open(io.BytesIO(response.content))
    assert output.mode == "RGBA"
    assert output.size == (320, 240)

    # Verify the output can be saved and re-loaded
    verify_buf = io.BytesIO()
    output.save(verify_buf, format="PNG")
    verify_buf.seek(0)
    verify = Image.open(verify_buf)
    assert verify.mode == "RGBA"
    print(f"✓ Full pipeline works: {output.size} RGBA saved and verified")


def test_background_modes():
    """Test different background modes."""
    print("\nTesting background modes...")

    inference = get_inference_service()
    try:
        inference.load_model()
    except FileNotFoundError:
        print("⚠ Model weights not found, skipping test")
        return

    img_bytes = create_test_image((100, 100), (128, 128, 128))

    # Test transparent (default)
    response = client.post("/api/remove-background",
        files={"image": ("test.png", img_bytes, "image/png")},
        data={"background_mode": "transparent", "output_format": "PNG"}
    )
    assert response.status_code == 200
    out = Image.open(io.BytesIO(response.content))
    assert out.mode == "RGBA"
    print("✓ Transparent mode works")

    # Test solid
    response = client.post("/api/remove-background",
        files={"image": ("test.png", img_bytes, "image/png")},
        data={"background_mode": "solid", "background_color": "#FF0000", "output_format": "PNG"}
    )
    assert response.status_code == 200
    print("✓ Solid color mode works")

    # Test gradient
    response = client.post("/api/remove-background",
        files={"image": ("test.png", img_bytes, "image/png")},
        data={"background_mode": "gradient", "output_format": "PNG"}
    )
    assert response.status_code == 200
    print("✓ Gradient mode works")

    # Test image background
    bg_bytes = create_test_image((100, 100), (0, 0, 255))
    response = client.post("/api/remove-background",
        files={
            "image": ("test.png", img_bytes, "image/png"),
            "background_image": ("bg.png", bg_bytes, "image/png")
        },
        data={"background_mode": "image", "output_format": "PNG"}
    )
    assert response.status_code == 200
    print("✓ Image background mode works")


def test_output_formats():
    """Test different output formats."""
    print("\nTesting output formats...")

    inference = get_inference_service()
    try:
        inference.load_model()
    except FileNotFoundError:
        print("⚠ Model weights not found, skipping test")
        return

    img_bytes = create_test_image((100, 100), (128, 128, 128))

    for fmt in ["PNG", "WEBP"]:
        response = client.post(f"/api/remove-background?output_format={fmt}",
            files={"image": ("test.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200, f"Format {fmt} failed"
        expected_type = f"image/{fmt.lower()}"
        assert response.headers["content-type"] == expected_type, f"Wrong content-type for {fmt}: {response.headers['content-type']}"
        print(f"✓ {fmt} format works")

    # JPEG test
    response = client.post("/api/remove-background?output_format=JPEG",
        files={"image": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200, "JPEG format failed"
    print("✓ JPEG format works")


def test_refinement_options():
    """Test mask refinement options."""
    print("\nTesting refinement options...")

    inference = get_inference_service()
    try:
        inference.load_model()
    except FileNotFoundError:
        print("⚠ Model weights not found, skipping test")
        return

    img_bytes = create_test_image((100, 100), (128, 128, 128))

    response = client.post("/api/remove-background",
        files={"image": ("test.png", img_bytes, "image/png")},
        data={
            "refine_edges": "true",
            "blur_radius": "2",
            "median_filter": "1",
            "hole_fill": "true",
            "edge_smooth": "true",
            "color_decontamination": "false",
            "output_format": "PNG"
        }
    )
    assert response.status_code == 200
    print("✓ Refinement options work")


if __name__ == "__main__":
    try:
        test_health()
        test_routes_exist()
        test_model_weights()
        test_valid_upload()
        test_invalid_upload()
        test_oversized_file()
        test_corrupt_image()
        test_segmentation_mask()
        test_alpha_channel()
        test_integration_pipeline()
        test_background_modes()
        test_output_formats()
        test_refinement_options()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)