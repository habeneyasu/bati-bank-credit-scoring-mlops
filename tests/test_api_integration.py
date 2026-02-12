"""
Integration tests for the Credit Scoring API.

These tests verify the complete API functionality including endpoints,
middleware, error handling, and model integration.

Run with: pytest tests/test_api_integration.py -v -m integration
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi import status
from fastapi.testclient import TestClient

from src.api.main import app, model, load_model_from_mlflow
from tests.conftest import api_client, sample_prediction_request, mock_model


@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for API endpoints."""
    
    def test_root_endpoint(self, api_client: TestClient):
        """Test the root endpoint returns correct response."""
        response = api_client.get("/")
        assert response.status_code == status.HTTP_200_OK
        assert "Credit Scoring API" in response.json()["message"]
    
    def test_health_check_without_model(self, api_client: TestClient):
        """Test health check endpoint when model is not loaded."""
        # Ensure model is None
        with patch('src.api.main.model', None):
            response = api_client.get("/health")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "degraded"  # API is degraded when model not loaded
            assert data["model_loaded"] is False
    
    def test_health_check_with_model(self, api_client: TestClient, mock_model):
        """Test health check endpoint when model is loaded."""
        with patch('src.api.main.model', mock_model):
            with patch('src.api.main.model_name', "test_model"):
                with patch('src.api.main.model_version', "1"):
                    response = api_client.get("/health")
                    assert response.status_code == status.HTTP_200_OK
                    data = response.json()
                    assert data["status"] == "healthy"
                    assert data["model_loaded"] is True
                    assert data["model_name"] == "test_model"
                    assert data["model_version"] == "1"
    
    def test_metrics_endpoint(self, api_client: TestClient):
        """Test metrics endpoint returns Prometheus-style metrics."""
        response = api_client.get("/metrics")
        assert response.status_code == status.HTTP_200_OK
        assert "predictions_total" in response.text
        assert "predictions_success" in response.text
        assert "predictions_errors" in response.text
    
    def test_predict_endpoint_success(self, api_client: TestClient, 
                                      sample_prediction_request: dict, 
                                      mock_model):
        """Test successful prediction request."""
        with patch('src.api.main.model', mock_model):
            response = api_client.post("/predict", json=sample_prediction_request)
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "risk_level" in data
            assert data["prediction"] in [0, 1]
            assert 0.0 <= data["probability"] <= 1.0
            assert data["risk_level"] in ["low", "medium", "high"]
    
    def test_predict_endpoint_missing_model(self, api_client: TestClient,
                                           sample_prediction_request: dict):
        """Test prediction request when model is not loaded."""
        with patch('src.api.main.model', None):
            response = api_client.post("/predict", json=sample_prediction_request)
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            assert "model not loaded" in response.json()["detail"].lower()
    
    def test_predict_endpoint_invalid_features_count(self, api_client: TestClient,
                                                     mock_model):
        """Test prediction request with incorrect number of features."""
        with patch('src.api.main.model', mock_model):
            invalid_request = {
                "features": [0.0] * 25  # Should be 26
            }
            response = api_client.post("/predict", json=invalid_request)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_endpoint_invalid_features_type(self, api_client: TestClient,
                                                    mock_model):
        """Test prediction request with invalid feature types."""
        with patch('src.api.main.model', mock_model):
            invalid_request = {
                "features": ["invalid"] * 26  # Should be floats
            }
            response = api_client.post("/predict", json=invalid_request)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_endpoint_missing_features(self, api_client: TestClient,
                                              mock_model):
        """Test prediction request with missing features field."""
        with patch('src.api.main.model', mock_model):
            response = api_client.post("/predict", json={})
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_endpoint_model_error(self, api_client: TestClient,
                                         sample_prediction_request: dict):
        """Test prediction request when model raises an error."""
        error_model = MagicMock()
        error_model.predict_proba.side_effect = Exception("Model error")
        
        with patch('src.api.main.model', error_model):
            response = api_client.post("/predict", json=sample_prediction_request)
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.integration
class TestAPIMiddleware:
    """Integration tests for API middleware."""
    
    def test_rate_limiting_middleware(self, api_client: TestClient):
        """Test rate limiting middleware."""
        # Make many requests quickly
        for _ in range(10):
            response = api_client.get("/health")
            # Health endpoint should not be rate limited
            assert response.status_code == status.HTTP_200_OK
    
    def test_request_logging_middleware(self, api_client: TestClient):
        """Test that requests are logged."""
        response = api_client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        # Check for process time header
        assert "X-Process-Time" in response.headers
    
    def test_error_handling_middleware(self, api_client: TestClient):
        """Test error handling middleware."""
        # Make a request that will fail
        response = api_client.post("/predict", json={"invalid": "data"})
        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]


@pytest.mark.integration
class TestAPIPydanticModels:
    """Integration tests for Pydantic models."""
    
    def test_prediction_request_validation(self):
        """Test PredictionRequest model validation."""
        from src.api.pydantic_models import PredictionRequest
        
        # Valid request
        valid_request = PredictionRequest(
            features=[0.0] * 26
        )
        assert len(valid_request.features) == 26
        
        # Invalid: too few features
        with pytest.raises(Exception):
            PredictionRequest(features=[0.0] * 25)
        
        # Invalid: too many features
        with pytest.raises(Exception):
            PredictionRequest(features=[0.0] * 27)
    
    def test_prediction_response_validation(self):
        """Test PredictionResponse model validation."""
        from src.api.pydantic_models import PredictionResponse
        
        # Valid response
        valid_response = PredictionResponse(
            prediction=0,
            probability=0.15,
            risk_level="low"
        )
        assert valid_response.prediction in [0, 1]
        assert 0.0 <= valid_response.probability <= 1.0
        assert valid_response.risk_level in ["low", "medium", "high"]
        
        # Invalid: probability out of range
        with pytest.raises(Exception):
            PredictionResponse(
                prediction=0,
                probability=1.5,  # Invalid
                risk_level="low"
            )
    
    def test_health_response_validation(self):
        """Test HealthResponse model validation."""
        from src.api.pydantic_models import HealthResponse
        
        # Valid response
        valid_response = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name="test_model",
            model_version="1"
        )
        assert valid_response.status == "healthy"
        assert valid_response.model_loaded is True


@pytest.mark.integration
class TestAPIModelLoading:
    """Integration tests for model loading."""
    
    @patch('mlflow.sklearn.load_model')
    def test_model_loading_success(self, mock_load_model, mock_model):
        """Test successful model loading."""
        mock_load_model.return_value = mock_model
        
        loaded_model = load_model_from_mlflow(
            model_name="test_model",
            stage="Production"
        )
        
        assert loaded_model is not None
        mock_load_model.assert_called_once()
    
    @patch('mlflow.sklearn.load_model')
    def test_model_loading_failure(self, mock_load_model):
        """Test model loading failure."""
        mock_load_model.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception):
            load_model_from_mlflow(
                model_name="nonexistent_model",
                stage="Production"
            )


@pytest.mark.integration
class TestAPIRiskLevels:
    """Integration tests for risk level classification."""
    
    def test_low_risk_classification(self, api_client: TestClient,
                                    sample_prediction_request: dict,
                                    mock_model):
        """Test low risk classification."""
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])  # 10% risk
        
        with patch('src.api.main.model', mock_model):
            response = api_client.post("/predict", json=sample_prediction_request)
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["risk_level"] == "low"
            assert data["probability"] == 0.1
    
    def test_medium_risk_classification(self, api_client: TestClient,
                                       sample_prediction_request: dict,
                                       mock_model):
        """Test medium risk classification."""
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])  # 50% risk
        
        with patch('src.api.main.model', mock_model):
            response = api_client.post("/predict", json=sample_prediction_request)
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["risk_level"] == "medium"
            assert data["probability"] == 0.5
    
    def test_high_risk_classification(self, api_client: TestClient,
                                     sample_prediction_request: dict,
                                     mock_model):
        """Test high risk classification."""
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 80% risk
        
        with patch('src.api.main.model', mock_model):
            response = api_client.post("/predict", json=sample_prediction_request)
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["risk_level"] == "high"
            assert data["probability"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
