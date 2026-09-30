#!/usr/bin/env python3
"""
Tests for health endpoint.
Run: python -m pytest backend/tests/test_health.py -v
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import Config
from backend.app.services.inference import get_inference_service


client = TestClient(app)


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_status(self):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        """Health response should have expected fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model" in data
        assert "device" in data
        assert "model_loaded" in data

    def test_health_status_ok(self):
        """Status should be 'ok'."""
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_model_name(self):
        """Model name should match config."""
        response = client.get("/health")
        data = response.json()
        assert data["model"] == Config.MODEL_NAME

    def test_health_device(self):
        """Device should be valid."""
        response = client.get("/health")
        data = response.json()
        assert data["device"] in ["cpu", "cuda"]

    def test_health_model_loaded(self):
        """Model loaded status should be boolean."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["model_loaded"], bool)


class TestHealthEndpointWithModel:
    """Test health when model is loaded."""

    def test_health_after_model_load(self):
        """Health should show model_loaded=true after load."""
        inference = get_inference_service()
        try:
            inference.load_model()
            response = client.get("/health")
            data = response.json()
            assert data["model_loaded"] == True
        except FileNotFoundError:
            pytest.skip("Model weights not found")

    def test_health_after_model_unload(self):
        """Health should show model_loaded=false after unload."""
        inference = get_inference_service()
        try:
            inference.load_model()
            inference.unload()
            response = client.get("/health")
            data = response.json()
            assert data["model_loaded"] == False
        except FileNotFoundError:
            pytest.skip("Model weights not found")


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_exists(self):
        """Root endpoint should exist."""
        response = client.get("/")
        # Should not return 404
        assert response.status_code != 404


class TestAPIDocs:
    """Test API documentation endpoints."""

    def test_docs_endpoint(self):
        """Swagger docs should be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint(self):
        """ReDoc should be accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_json(self):
        """OpenAPI schema should be accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200


class TestModelInfo:
    """Test model information endpoints."""

    def test_remove_background_info(self):
        """Info endpoint should return parameters."""
        response = client.get("/api/remove-background/info")
        assert response.status_code == 200
        data = response.json()
        assert "endpoint" in data
        assert "parameters" in data
        assert "supported_formats" in data
