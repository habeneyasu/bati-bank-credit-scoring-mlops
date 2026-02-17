"""
Tests for Customer Prediction History API endpoint.
"""

import pytest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app
from src.database.models import Prediction
from decimal import Decimal


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_prediction():
    """Create a mock prediction object."""
    pred = MagicMock(spec=Prediction)
    pred.prediction_id = "pred_123"
    pred.customer_id = "cust_123"
    pred.prediction = 0
    pred.probability = Decimal("0.25")
    pred.risk_level = "low"
    pred.customer_score = 75
    pred.latency_ms = Decimal("150.50")
    pred.model_name = "credit_scoring_model"
    pred.model_version = "v1.0.0"
    pred.model_stage = "Production"
    pred.created_at = datetime.now(timezone.utc)
    pred.created_at_date = date.today()
    pred.request_metadata = None
    return pred


class TestCustomerPredictionHistoryEndpoint:
    """Test customer prediction history endpoint."""
    
    def test_get_customer_predictions_basic(self, client, mock_prediction):
        """Test basic customer prediction history retrieval."""
        with patch('src.api.main.get_db_session') as mock_db:
            with patch('src.database.services.PredictionService') as mock_service:
                mock_service_instance = MagicMock()
                mock_service.return_value.__enter__.return_value = mock_service_instance
                mock_service_instance.get_customer_predictions.return_value = [mock_prediction]
                mock_service_instance.count_customer_predictions.return_value = 1
                
                response = client.get("/api/customers/cust_123/predictions")
                
                assert response.status_code == 200
                data = response.json()
                assert data["customer_id"] == "cust_123"
                assert len(data["predictions"]) == 1
                assert data["pagination"]["total"] == 1
                assert data["pagination"]["returned"] == 1
    
    def test_get_customer_predictions_with_limit(self, client, mock_prediction):
        """Test customer prediction history with limit."""
        with patch('src.api.main.get_db_session') as mock_db:
            with patch('src.database.services.PredictionService') as mock_service:
                mock_service_instance = MagicMock()
                mock_service.return_value.__enter__.return_value = mock_service_instance
                mock_service_instance.get_customer_predictions.return_value = [mock_prediction] * 10
                mock_service_instance.count_customer_predictions.return_value = 50
                
                response = client.get("/api/customers/cust_123/predictions?limit=10")
                
                assert response.status_code == 200
                data = response.json()
                assert len(data["predictions"]) == 10
                assert data["pagination"]["limit"] == 10
                assert data["pagination"]["total"] == 50
                assert data["pagination"]["has_more"] is True
    
    def test_get_customer_predictions_with_offset(self, client, mock_prediction):
        """Test customer prediction history with offset."""
        with patch('src.api.main.get_db_session') as mock_db:
            with patch('src.database.services.PredictionService') as mock_service:
                mock_service_instance = MagicMock()
                mock_service.return_value.__enter__.return_value = mock_service_instance
                mock_service_instance.get_customer_predictions.return_value = [mock_prediction] * 5
                mock_service_instance.count_customer_predictions.return_value = 20
                
                response = client.get("/api/customers/cust_123/predictions?limit=5&offset=10")
                
                assert response.status_code == 200
                data = response.json()
                assert len(data["predictions"]) == 5
                assert data["pagination"]["offset"] == 10
                assert data["pagination"]["has_more"] is True
    
    def test_get_customer_predictions_with_date_range(self, client, mock_prediction):
        """Test customer prediction history with date range filtering."""
        with patch('src.api.main.get_db_session') as mock_db:
            with patch('src.database.services.PredictionService') as mock_service:
                mock_service_instance = MagicMock()
                mock_service.return_value.__enter__.return_value = mock_service_instance
                mock_service_instance.get_customer_predictions.return_value = [mock_prediction]
                mock_service_instance.count_customer_predictions.return_value = 1
                
                start_date = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
                end_date = date.today().strftime("%Y-%m-%d")
                
                response = client.get(
                    f"/api/customers/cust_123/predictions?start_date={start_date}&end_date={end_date}"
                )
                
                assert response.status_code == 200
                # Verify date filters were passed
                call_args = mock_service_instance.get_customer_predictions.call_args
                assert call_args[1]["start_date"] is not None
                assert call_args[1]["end_date"] is not None
    
    def test_get_customer_predictions_with_risk_level(self, client, mock_prediction):
        """Test customer prediction history with risk level filtering."""
        with patch('src.api.main.get_db_session') as mock_db:
            with patch('src.database.services.PredictionService') as mock_service:
                mock_service_instance = MagicMock()
                mock_service.return_value.__enter__.return_value = mock_service_instance
                mock_service_instance.get_customer_predictions.return_value = [mock_prediction]
                mock_service_instance.count_customer_predictions.return_value = 1
                
                response = client.get("/api/customers/cust_123/predictions?risk_level=low")
                
                assert response.status_code == 200
                call_args = mock_service_instance.get_customer_predictions.call_args
                assert call_args[1]["risk_level"] == "low"
    
    def test_get_customer_predictions_with_model_version(self, client, mock_prediction):
        """Test customer prediction history with model version filtering."""
        with patch('src.api.main.get_db_session') as mock_db:
            with patch('src.database.services.PredictionService') as mock_service:
                mock_service_instance = MagicMock()
                mock_service.return_value.__enter__.return_value = mock_service_instance
                mock_service_instance.get_customer_predictions.return_value = [mock_prediction]
                mock_service_instance.count_customer_predictions.return_value = 1
                
                response = client.get("/api/customers/cust_123/predictions?model_version=v1.0.0")
                
                assert response.status_code == 200
                call_args = mock_service_instance.get_customer_predictions.call_args
                assert call_args[1]["model_version"] == "v1.0.0"
    
    def test_get_customer_predictions_with_analytics(self, client, mock_prediction):
        """Test customer prediction history with analytics."""
        with patch('src.api.main.get_db_session') as mock_db:
            with patch('src.database.services.PredictionService') as mock_service:
                mock_service_instance = MagicMock()
                mock_service.return_value.__enter__.return_value = mock_service_instance
                mock_service_instance.get_customer_predictions.return_value = [mock_prediction]
                mock_service_instance.count_customer_predictions.return_value = 1
                mock_service_instance.get_customer_prediction_analytics.return_value = {
                    "total_count": 1,
                    "average_probability": 0.25,
                    "average_score": 75.0,
                    "risk_level_distribution": {"low": 1, "medium": 0, "high": 0}
                }
                
                response = client.get("/api/customers/cust_123/predictions?include_analytics=true")
                
                assert response.status_code == 200
                data = response.json()
                assert "analytics" in data
                assert data["analytics"]["total_count"] == 1
                assert data["analytics"]["average_probability"] == 0.25
    
    def test_get_customer_predictions_invalid_date_format(self, client):
        """Test customer prediction history with invalid date format."""
        response = client.get("/api/customers/cust_123/predictions?start_date=invalid-date")
        
        assert response.status_code == 400
        assert "Invalid start_date format" in response.json()["detail"]
    
    def test_get_customer_predictions_invalid_date_range(self, client):
        """Test customer prediction history with invalid date range."""
        start_date = date.today().strftime("%Y-%m-%d")
        end_date = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        response = client.get(
            f"/api/customers/cust_123/predictions?start_date={start_date}&end_date={end_date}"
        )
        
        assert response.status_code == 400
        assert "start_date must be before or equal to end_date" in response.json()["detail"]
    
    def test_get_customer_predictions_invalid_risk_level(self, client):
        """Test customer prediction history with invalid risk level."""
        response = client.get("/api/customers/cust_123/predictions?risk_level=invalid")
        
        assert response.status_code == 422  # Validation error
    
    def test_get_customer_predictions_limit_validation(self, client):
        """Test customer prediction history with invalid limit."""
        response = client.get("/api/customers/cust_123/predictions?limit=0")
        
        assert response.status_code == 422  # Validation error
    
    def test_get_customer_predictions_offset_validation(self, client):
        """Test customer prediction history with invalid offset."""
        response = client.get("/api/customers/cust_123/predictions?offset=-1")
        
        assert response.status_code == 422  # Validation error
    
    def test_legacy_endpoint_redirect(self, client, mock_prediction):
        """Test that legacy endpoint redirects to new endpoint."""
        with patch('src.api.main.get_db_session') as mock_db:
            with patch('src.database.services.PredictionService') as mock_service:
                mock_service_instance = MagicMock()
                mock_service.return_value.__enter__.return_value = mock_service_instance
                mock_service_instance.get_customer_predictions.return_value = [mock_prediction]
                mock_service_instance.count_customer_predictions.return_value = 1
                
                response = client.get("/api/predictions/customer/cust_123?limit=50")
                
                assert response.status_code == 200
                data = response.json()
                assert data["customer_id"] == "cust_123"


class TestCustomerPredictionHistoryResponseFormat:
    """Test response format of customer prediction history."""
    
    def test_prediction_data_format(self, client, mock_prediction):
        """Test that prediction data is properly formatted."""
        with patch('src.api.main.get_db_session') as mock_db:
            with patch('src.database.services.PredictionService') as mock_service:
                mock_service_instance = MagicMock()
                mock_service.return_value.__enter__.return_value = mock_service_instance
                mock_service_instance.get_customer_predictions.return_value = [mock_prediction]
                mock_service_instance.count_customer_predictions.return_value = 1
                
                response = client.get("/api/customers/cust_123/predictions")
                
                assert response.status_code == 200
                data = response.json()
                pred = data["predictions"][0]
                
                assert "prediction_id" in pred
                assert "customer_id" in pred
                assert "prediction" in pred
                assert "probability" in pred
                assert "risk_level" in pred
                assert "customer_score" in pred
                assert "model_name" in pred
                assert "model_version" in pred
                assert "created_at" in pred
    
    def test_pagination_format(self, client, mock_prediction):
        """Test that pagination metadata is properly formatted."""
        with patch('src.api.main.get_db_session') as mock_db:
            with patch('src.database.services.PredictionService') as mock_service:
                mock_service_instance = MagicMock()
                mock_service.return_value.__enter__.return_value = mock_service_instance
                mock_service_instance.get_customer_predictions.return_value = [mock_prediction] * 10
                mock_service_instance.count_customer_predictions.return_value = 50
                
                response = client.get("/api/customers/cust_123/predictions?limit=10&offset=20")
                
                assert response.status_code == 200
                data = response.json()
                
                assert "pagination" in data
                pagination = data["pagination"]
                assert pagination["limit"] == 10
                assert pagination["offset"] == 20
                assert pagination["total"] == 50
                assert pagination["returned"] == 10
                assert pagination["has_more"] is True
    
    def test_filters_format(self, client, mock_prediction):
        """Test that filters are included in response."""
        with patch('src.api.main.get_db_session') as mock_db:
            with patch('src.database.services.PredictionService') as mock_service:
                mock_service_instance = MagicMock()
                mock_service.return_value.__enter__.return_value = mock_service_instance
                mock_service_instance.get_customer_predictions.return_value = [mock_prediction]
                mock_service_instance.count_customer_predictions.return_value = 1
                
                response = client.get(
                    "/api/customers/cust_123/predictions?risk_level=high&model_version=v2.0.0"
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert "filters" in data
                filters = data["filters"]
                assert filters["risk_level"] == "high"
                assert filters["model_version"] == "v2.0.0"
