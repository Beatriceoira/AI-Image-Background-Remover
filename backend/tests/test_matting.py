#!/usr/bin/env python3
"""
Tests for matting service.
Run: python -m pytest backend/tests/test_matting.py -v
"""
import sys
import os
import io
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.services.matting import MattingService
from backend.app.utils.image_utils import parse_hex_color


class TestMattingService:
    """Test matting service."""

    @pytest.fixture
    def service(self):
        """Create matting service instance."""
        return MattingService()

    @pytest.fixture
    def sample_image(self):
        """Create sample RGB image."""
        img = Image.new("RGB", (100, 100), (200, 100, 50))
        return np.array(img)

    @pytest.fixture
    def sample_mask(self):
        """Create sample mask (center circle)."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Create circular mask
        center = 50
        radius = 30
        for y in range(100):
            for x in range(100):
                if (x - center) ** 2 + (y - center) ** 2 <= radius ** 2:
                    mask[y, x] = 255
        return mask

    def test_transparent_mode(self, service, sample_image, sample_mask):
        """Transparent mode should produce RGBA output."""
        result = service.matte(sample_image, sample_mask, background_mode="transparent")
        assert result.shape == (100, 100, 4)
        assert result.dtype == np.uint8

    def test_solid_background(self, service, sample_image, sample_mask):
        """Solid background mode should composite correctly."""
        bg_color = (255, 0, 0)  # Red
        result = service.matte(
            sample_image,
            sample_mask,
            background_mode="solid",
            background_color=bg_color,
        )
        assert result.shape == (100, 100, 4)

        # Check that background pixels are red where mask is 0
        # Bottom-left corner should be background (mask=0)
        corner_value = result[95, 5, :3]
        assert corner_value[0] == 255  # R
        assert corner_value[1] == 0    # G
        assert corner_value[2] == 0    # B

    def test_gradient_background(self, service, sample_image, sample_mask):
        """Gradient background mode should work."""
        result = service.matte(
            sample_image,
            sample_mask,
            background_mode="gradient",
        )
        assert result.shape == (100, 100, 4)

    def test_image_background(self, service, sample_image, sample_mask):
        """Image background mode should work."""
        bg_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = service.matte(
            sample_image,
            sample_mask,
            background_mode="image",
            background_image=bg_img,
        )
        assert result.shape == (100, 100, 4)

    def test_alpha_channel_values(self, service, sample_image, sample_mask):
        """Alpha channel should reflect mask values."""
        result = service.matte(sample_image, sample_mask, background_mode="transparent")

        # Extract alpha channel
        alpha = result[:, :, 3]

        # Check center (should be opaque)
        assert alpha[50, 50] == 255

        # Check corner (should be transparent)
        assert alpha[0, 0] == 0

    def test_rgba_composite(self, service, sample_image, sample_mask):
        """Composite should preserve foreground colors."""
        result = service.matte(sample_image, sample_mask, background_mode="transparent")

        # Check foreground pixel (should match original)
        fg_pixel = result[50, 50, :3]
        assert fg_pixel[0] == 200  # R
        assert fg_pixel[1] == 100  # G
        assert fg_pixel[2] == 50   # B

    def test_hard_edge_mask(self, service):
        """Hard edge mask should produce sharp alpha transition."""
        # Create hard edge mask (left half foreground)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:, 50:] = 255

        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = service.matte(img, mask, background_mode="transparent")

        # Check transition at x=50
        assert result[50, 49, 3] == 0    # Left side transparent
        assert result[50, 50, 3] == 255  # Right side opaque

    def test_soft_edge_mask(self, service):
        """Soft edge mask should produce smooth alpha transition."""
        # Create gradient mask
        mask = np.linspace(0, 255, 100, dtype=np.uint8).reshape(1, 100)
        mask = np.repeat(mask, 100, axis=0)

        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = service.matte(img, mask, background_mode="transparent")

        # Check smooth transition
        alpha_row = result[:, :, 3][50, :]
        assert alpha_row[0] < alpha_row[25] < alpha_row[50] < alpha_row[75] < alpha_row[99]

    def test_rgba_input(self, service):
        """RGBA input should be handled correctly."""
        # Create RGBA image
        img_rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        np_img = np.array(img_rgba)

        mask = np.ones((100, 100), dtype=np.uint8) * 128

        result = service.matte(np_img, mask, background_mode="transparent")
        assert result.shape == (100, 100, 4)

    def test_mask_normalization(self, service):
        """Mask in [0, 1] float range should work."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask_float = np.ones((100, 100), dtype=np.float32) * 0.5

        result = service.matte(img, mask_float, background_mode="transparent")
        assert result.shape == (100, 100, 4)
        # 0.5 * 255 = 127.5 -> 127 after conversion
        assert np.all(result[:, :, 3] == 127)


class TestParseHexColor:
    """Test hex color parsing."""

    def test_full_hex(self):
        """Full hex format should parse."""
        assert parse_hex_color("#FF0000") == (255, 0, 0)

    def test_short_hex(self):
        """Short hex format should parse."""
        assert parse_hex_color("#F00") == (255, 0, 0)

    def test_without_hash(self):
        """Hex without hash should parse."""
        assert parse_hex_color("FF0000") == (255, 0, 0)

    def test_invalid_format(self):
        """Invalid hex should raise ValueError."""
        with pytest.raises(ValueError):
            parse_hex_color("GGGGGG")

    def test_invalid_length(self):
        """Wrong length should raise ValueError."""
        with pytest.raises(ValueError):
            parse_hex_color("#FF00")  # 4 chars
        with pytest.raises(ValueError):
            parse_hex_color("#FF")    # 2 chars


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test full pipeline from image to RGBA output."""
        # Create test image
        img = Image.new("RGB", (100, 100), (200, 100, 50))
        np_img = np.array(img)

        # Create mask (circle in center)
        mask = np.zeros((100, 100), dtype=np.uint8)
        for y in range(100):
            for x in range(100):
                if (x - 50) ** 2 + (y - 50) ** 2 <= 2500:
                    mask[y, x] = 255

        # Run matting
        service = MattingService()
        result = service.matte(np_img, mask, background_mode="transparent")

        # Verify
        assert result.shape == (100, 100, 4)
        assert result.dtype == np.uint8

        # Verify colors preserved
        assert result[50, 50, 0] == 200  # R
        assert result[50, 50, 1] == 100  # G
        assert result[50, 50, 2] == 50   # B
        assert result[50, 50, 3] == 255  # A

        # Verify background
        assert result[0, 0, 3] == 0  # Transparent
