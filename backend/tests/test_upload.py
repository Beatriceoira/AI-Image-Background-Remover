#!/usr/bin/env python3
"""
Tests for image upload validation.
Run: python -m pytest backend/tests/test_upload.py -v
"""
import sys
import os
import io
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import UploadFile
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.utils.validation import validate_image_upload, MAGIC_BYTES, ALLOWED_MIME_TYPES
from backend.app.core.config import Config
from PIL import Image


client = TestClient(app)


def create_test_image(size=(100, 100), color=(128, 128, 128), fmt="PNG"):
    """Create a simple test image."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


class TestMagicBytes:
    """Test magic byte validation."""

    def test_png_magic(self):
        """PNG magic bytes should match."""
        contents = create_test_image(fmt="PNG")
        assert contents.startswith(b"\x89PNG\r\n\x1a\n")

    def test_jpeg_magic(self):
        """JPEG magic bytes should match."""
        contents = create_test_image(fmt="JPEG")
        assert contents.startswith(b"\xFF\xD8\xFF")

    def test_bmp_magic(self):
        """BMP magic bytes should match."""
        img = Image.new("RGB", (10, 10), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        buf.seek(0)
        contents = buf.getvalue()
        assert contents.startswith(b"BM")


class TestMIMEValidation:
    """Test MIME type validation."""

    def test_valid_png_mime(self):
        """PNG should be accepted."""
        assert "image/png" in ALLOWED_MIME_TYPES

    def test_valid_jpeg_mime(self):
        """JPEG should be accepted."""
        assert "image/jpeg" in ALLOWED_MIME_TYPES

    def test_invalid_mime(self):
        """Text file should be rejected."""
        assert "text/plain" not in ALLOWED_MIME_TYPES


class TestValidateImageUpload:
    """Test the main validation function."""

    def test_valid_png_upload(self):
        """Valid PNG should pass validation."""
        contents = create_test_image()
        file = UploadFile(
            filename="test.png",
            file=io.BytesIO(contents),
            headers={"content-type": "image/png"},
        )
        # Synchronous version using run_in_threadpool
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(validate_image_upload(file))
            assert isinstance(result, bytes)
            assert len(result) > 0
        finally:
            loop.close()

    def test_valid_jpeg_upload(self):
        """Valid JPEG should pass validation."""
        contents = create_test_image(fmt="JPEG")
        file = UploadFile(
            filename="test.jpg",
            file=io.BytesIO(contents),
            headers={"content-type": "image/jpeg"},
        )
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(validate_image_upload(file))
            assert isinstance(result, bytes)
            assert len(result) > 0
        finally:
            loop.close()

    def test_missing_content_type(self):
        """Missing content type should be rejected."""
        file = UploadFile(
            filename="test.png",
            file=io.BytesIO(create_test_image()),
            headers={},
        )
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with pytest.raises(Exception) as exc_info:
                loop.run_until_complete(validate_image_upload(file))
            assert exc_info.value.status_code == 400
        finally:
            loop.close()

    def test_invalid_content_type(self):
        """Invalid content type should be rejected."""
        file = UploadFile(
            filename="test.txt",
            file=io.BytesIO(b"not an image"),
            headers={"content-type": "text/plain"},
        )
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with pytest.raises(Exception) as exc_info:
                loop.run_until_complete(validate_image_upload(file))
            assert exc_info.value.status_code == 400
        finally:
            loop.close()

    def test_empty_file(self):
        """Empty file should be rejected."""
        file = UploadFile(
            filename="empty.png",
            file=io.BytesIO(b""),
            headers={"content-type": "image/png"},
        )
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with pytest.raises(Exception) as exc_info:
                loop.run_until_complete(validate_image_upload(file))
            assert exc_info.value.status_code == 400
        finally:
            loop.close()

    def test_corrupt_image(self):
        """Corrupt image should be rejected."""
        file = UploadFile(
            filename="corrupt.png",
            file=io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00not_valid"),
            headers={"content-type": "image/png"},
        )
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with pytest.raises(Exception) as exc_info:
                loop.run_until_complete(validate_image_upload(file))
            assert exc_info.value.status_code == 400
        finally:
            loop.close()

    def test_oversized_file(self):
        """Oversized file should be rejected."""
        large_size = (Config.MAX_UPLOAD_SIZE_MB + 1) * 1024 * 1024
        file = UploadFile(
            filename="large.png",
            file=io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * large_size),
            headers={"content-type": "image/png"},
        )
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with pytest.raises(Exception) as exc_info:
                loop.run_until_complete(validate_image_upload(file))
            assert exc_info.value.status_code == 413
        finally:
            loop.close()

    def test_dimension_validation(self):
        """Small valid image should pass dimension validation."""
        contents = create_test_image((50, 50))
        file = UploadFile(
            filename="small.png",
            file=io.BytesIO(contents),
            headers={"content-type": "image/png"},
        )
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(validate_image_upload(file))
            assert isinstance(result, bytes)
        finally:
            loop.close()


class TestIntegration:
    """Integration tests with FastAPI."""

    def test_api_rejects_invalid_mime(self):
        """POST should reject non-image files."""
        files = {"image": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/api/remove-background", files=files)
        assert response.status_code == 400

    def test_api_rejects_corrupt_image(self):
        """POST should reject corrupted images."""
        files = {"image": ("corrupt.png", b"not a png", "image/png")}
        response = client.post("/api/remove-background", files=files)
        assert response.status_code == 400

    def test_api_rejects_oversized_file(self):
        """POST should reject oversized files."""
        large_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * ((Config.MAX_UPLOAD_SIZE_MB + 1) * 1024 * 1024)
        files = {"image": ("large.png", large_data, "image/png")}
        response = client.post("/api/remove-background", files=files)
        assert response.status_code == 413

    def test_api_accepts_valid_image(self):
        """POST should accept valid images."""
        contents = create_test_image((100, 100))
        files = {"image": ("test.png", contents, "image/png")}
        response = client.post("/api/remove-background", files=files)
        # May fail due to model loading, but should not be 400/413
        assert response.status_code != 400
        assert response.status_code != 413
