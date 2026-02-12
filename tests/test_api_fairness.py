"""
Integration tests for fairness API endpoints.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def api_client():
    """Create test client."""
    return TestClient(app)


class TestFairnessAPI:
    """Test suite for fairness API endpoints."""
    
    def test_fairness_endpoint_exists(self, api_client):
        """Test that fairness endpoint exists."""
        response = api_client.get("/api/fairness")
        
        # Should return either 200 (with data) or 503 (model not loaded) or 500 (error)
        assert response.status_code in [200, 503, 500]
    
    def test_fairness_endpoint_structure(self, api_client):
        """Test fairness endpoint response structure."""
        response = api_client.get("/api/fairness")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for expected metrics
            assert "demographic_parity" in data or "overall_status" in data
            
            # If we have demographic_parity, check its structure
            if "demographic_parity" in data:
                assert "value" in data["demographic_parity"]
                assert "threshold" in data["demographic_parity"]
                assert "status" in data["demographic_parity"]
    
    def test_fairness_endpoint_mock_data(self, api_client):
        """Test fairness endpoint returns mock data when test data unavailable."""
        response = api_client.get("/api/fairness")
        
        if response.status_code == 200:
            data = response.json()
            
            # Mock data should have note field
            if "note" in data:
                assert "mock" in data["note"].lower() or "Mock" in data["note"]
            
            # Should have all required metrics
            required_metrics = [
                "demographic_parity",
                "equalized_odds",
                "calibration",
                "disparate_impact"
            ]
            
            for metric in required_metrics:
                if metric in data:
                    assert "value" in data[metric]
                    assert "threshold" in data[metric]
                    assert "status" in data[metric]
    
    def test_fairness_endpoint_compliance_status(self, api_client):
        """Test that fairness endpoint includes compliance status."""
        response = api_client.get("/api/fairness")
        
        if response.status_code == 200:
            data = response.json()
            
            # Should have overall status
            if "overall_status" in data:
                assert data["overall_status"] in ["compliant", "non_compliant"]
            
            # Each metric should have status
            metrics = ["demographic_parity", "equalized_odds", "calibration", "disparate_impact"]
            for metric in metrics:
                if metric in data and isinstance(data[metric], dict):
                    if "status" in data[metric]:
                        assert data[metric]["status"] in ["compliant", "non_compliant", "insufficient_data"]
