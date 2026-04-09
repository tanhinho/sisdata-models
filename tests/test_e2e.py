import pytest
import os
import requests

API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8080')


def test_service_health():
    """Test that FastAPI service is healthy and model is loaded."""
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True


def test_prediction():
    """Test that the /predict endpoint returns a valid prediction."""
    # TODO: Implement this test
    pass


def test_prediction_invalid_input():
    """Test that the /predict endpoint handles invalid input gracefully."""
    # TODO: Implement this test
    pass
