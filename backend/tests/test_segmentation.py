#!/usr/bin/env python3
"""
Tests for segmentation service.
Run: python -m pytest backend/tests/test_segmentation.py -v
"""
import sys
import os
import io
import numpy as np
import pytest
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.services.segmentation import SegmentationService
from backend.app.services.preprocessing import PreprocessingService
from backend.app.services.inference import InferenceService
from backend.app.models.u2net import U2Net
from backend.app.core.config import Config


class TestPreprocessing:
    """Test image preprocessing."""

    def test_rgb_conversion(self):
        """Image should be converted to RGB."""
        service = PreprocessingService()
        # Create RGBA image
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        np_img = np.array(img)
        # Should drop alpha channel
        assert np_img.shape[2] == 4
        # After preprocessing, tensor should have 3 channels
        tensor = service.preprocess(np_img)
        assert tensor.shape == (1, 3, 320, 320)

    def test_grayscale_conversion(self):
        """Grayscale image should be converted to RGB."""
        service = PreprocessingService()
        img = Image.new("L", (100, 100), 128)
        np_img = np.array(img)
        assert np_img.ndim == 2
        tensor = service.preprocess(np_img)
        assert tensor.shape == (1, 3, 320, 320)

    def test_dimensions_match_input(self):
        """Output tensor dimensions should match model input."""
        service = PreprocessingService()
        img = Image.new("RGB", (200, 150), (128, 128, 128))
        np_img = np.array(img)
        tensor = service.preprocess(np_img)
        assert tensor.shape == (1, 3, 320, 320)

    def test_normalized_values(self):
        """Tensor values should be in [0, 1] range."""
        service = PreprocessingService()
        # White image
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        np_img = np.array(img)
        tensor = service.preprocess(np_img)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_batch_dimension(self):
        """Tensor should have batch dimension."""
        service = PreprocessingService()
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        np_img = np.array(img)
        tensor = service.preprocess(np_img)
        assert tensor.dim() == 4
        assert tensor.shape[0] == 1


class TestSegmentationService:
    """Test segmentation service."""

    @pytest.fixture
    def inference_service(self):
        """Create and initialize inference service."""
        service = InferenceService()
        try:
            service.load_model()
        except FileNotFoundError:
            pytest.skip("Model weights not found")
        return service

    @pytest.fixture
    def segmentation_service(self, inference_service):
        """Create segmentation service."""
        return SegmentationService()

    @pytest.fixture
    def preprocessing_service(self):
        """Create preprocessing service."""
        return PreprocessingService()

    def test_mask_shape_matches_original(self, inference_service, segmentation_service, preprocessing_service):
        """Output mask should match original image dimensions."""
        original_h, original_w = 200, 150
        img = Image.new("RGB", (original_w, original_h), (128, 128, 128))
        np_img = np.array(img)

        tensor = preprocessing_service.preprocess(np_img)
        mask = segmentation_service.segment(np_img, tensor)

        assert mask.shape == (original_h, original_w)

    def test_mask_value_range(self, inference_service, segmentation_service, preprocessing_service):
        """Mask values should be in valid range."""
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        np_img = np.array(img)

        tensor = preprocessing_service.preprocess(np_img)
        mask = segmentation_service.segment(np_img, tensor)

        assert mask.min() >= 0
        assert mask.max() <= 255

    def test_mask_has_variation(self, inference_service, segmentation_service, preprocessing_service):
        """Mask should have some variation (not all uniform)."""
        # Create a simple shape on contrasting background
        img = Image.new("RGB", (100, 100), (200, 200, 200))  # Light gray background
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.ellipse([20, 20, 80, 80], fill=(50, 50, 50))  # Dark circle
        np_img = np.array(img)

        tensor = preprocessing_service.preprocess(np_img)
        mask = segmentation_service.segment(np_img, tensor)

        # Mask should have both foreground and background regions
        assert mask.min() < 255, "Should have some background pixels"
        assert mask.max() > 0, "Should have some foreground pixels"

    def test_consistent_output(self, inference_service, segmentation_service, preprocessing_service):
        """Same input should produce same output."""
        img = Image.new("RGB", (100, 100), (100, 150, 200))
        np_img = np.array(img)

        tensor1 = preprocessing_service.preprocess(np_img)
        mask1 = segmentation_service.segment(np_img, tensor1)

        tensor2 = preprocessing_service.preprocess(np_img)
        mask2 = segmentation_service.segment(np_img, tensor2)

        np.testing.assert_array_equal(mask1, mask2)


class TestModelLoad:
    """Test model loading."""

    def test_model_loads(self):
        """Model should load successfully if weights exist."""
        service = InferenceService()
        try:
            service.load_model()
            assert service.is_ready()
        except FileNotFoundError:
            pytest.skip("Model weights not found")

    def test_model_device(self):
        """Model should load on correct device."""
        service = InferenceService()
        try:
            service.load_model()
            device = service.device
            assert device.type in ["cpu", "cuda"]
        except FileNotFoundError:
            pytest.skip("Model weights not found")

    def test_model_unload(self):
        """Model should unload successfully."""
        service = InferenceService()
        try:
            service.load_model()
            service.unload()
            assert not service.is_ready()
        except FileNotFoundError:
            pytest.skip("Model weights not found")


class TestEdgeCases:
    """Test edge cases."""

    def test_single_pixel_image(self):
        """Very small image should work."""
        service = PreprocessingService()
        img = Image.new("RGB", (1, 1), (128, 128, 128))
        np_img = np.array(img)
        tensor = service.preprocess(np_img)
        assert tensor.shape == (1, 3, 320, 320)

    def test_large_image(self):
        """Large image should be handled."""
        service = PreprocessingService()
        # Use a moderate size to avoid memory issues
        img = Image.new("RGB", (600, 400), (128, 128, 128))
        np_img = np.array(img)
        tensor = service.preprocess(np_img)
        assert tensor.shape == (1, 3, 320, 320)

    def test_all_white(self):
        """All-white image should produce valid mask."""
        inference = InferenceService()
        seg = SegmentationService()
        prep = PreprocessingService()
        try:
            inference.load_model()
        except FileNotFoundError:
            pytest.skip("Model weights not found")

        img = Image.new("RGB", (50, 50), (255, 255, 255))
        np_img = np.array(img)
        tensor = prep.preprocess(np_img)
        mask = seg.segment(np_img, tensor)
        assert mask.shape == (50, 50)

    def test_all_black(self):
        """All-black image should produce valid mask."""
        inference = InferenceService()
        seg = SegmentationService()
        prep = PreprocessingService()
        try:
            inference.load_model()
        except FileNotFoundError:
            pytest.skip("Model weights not found")

        img = Image.new("RGB", (50, 50), (0, 0, 0))
        np_img = np.array(img)
        tensor = prep.preprocess(np_img)
        mask = seg.segment(np_img, tensor)
        assert mask.shape == (50, 50)
