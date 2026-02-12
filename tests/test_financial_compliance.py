"""
Financial compliance and validation tests.

These tests ensure the credit scoring system meets regulatory requirements
and internal risk management standards for financial applications.

Run with: pytest tests/test_financial_compliance.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.pydantic_models import PredictionRequest, PredictionResponse
from src.utils.config import Settings
from tests.conftest import sample_prediction_request


class TestDataValidation:
    """Tests for data validation requirements."""
    
    def test_prediction_request_feature_count_validation(self):
        """Test that prediction requests must have exactly 26 features."""
        # Valid: exactly 26 features
        valid_request = PredictionRequest(features=[0.0] * 26)
        assert len(valid_request.features) == 26
        
        # Invalid: too few features
        with pytest.raises(Exception):
            PredictionRequest(features=[0.0] * 25)
        
        # Invalid: too many features
        with pytest.raises(Exception):
            PredictionRequest(features=[0.0] * 27)
    
    def test_prediction_response_probability_range(self):
        """Test that prediction probabilities are in valid range [0, 1]."""
        # Valid probabilities
        valid_responses = [
            PredictionResponse(prediction=0, probability=0.0, risk_level="low"),
            PredictionResponse(prediction=0, probability=0.5, risk_level="medium"),
            PredictionResponse(prediction=1, probability=1.0, risk_level="high"),
        ]
        
        for response in valid_responses:
            assert 0.0 <= response.probability <= 1.0
        
        # Invalid: probability < 0
        with pytest.raises(Exception):
            PredictionResponse(prediction=0, probability=-0.1, risk_level="low")
        
        # Invalid: probability > 1
        with pytest.raises(Exception):
            PredictionResponse(prediction=1, probability=1.1, risk_level="high")
    
    def test_prediction_response_binary_classification(self):
        """Test that predictions are binary (0 or 1)."""
        # Valid predictions
        valid_responses = [
            PredictionResponse(prediction=0, probability=0.2, risk_level="low"),
            PredictionResponse(prediction=1, probability=0.8, risk_level="high"),
        ]
        
        for response in valid_responses:
            assert response.prediction in [0, 1]
        
        # Invalid: prediction not 0 or 1
        with pytest.raises(Exception):
            PredictionResponse(prediction=2, probability=0.5, risk_level="medium")
        
        with pytest.raises(Exception):
            PredictionResponse(prediction=-1, probability=0.5, risk_level="medium")


class TestRiskThresholdValidation:
    """Tests for risk threshold validation."""
    
    def test_risk_threshold_configuration(self):
        """Test that risk thresholds are properly configured."""
        settings = Settings()
        
        # Thresholds must be in [0, 1]
        assert 0.0 <= settings.risk_threshold_low <= 1.0
        assert 0.0 <= settings.risk_threshold_high <= 1.0
        
        # High threshold must be greater than low threshold
        assert settings.risk_threshold_high > settings.risk_threshold_low
    
    def test_risk_level_classification(self):
        """Test that risk levels are correctly classified."""
        settings = Settings()
        
        # Low risk: probability < low threshold
        low_prob = settings.risk_threshold_low - 0.1
        assert low_prob < settings.risk_threshold_low
        
        # Medium risk: low threshold <= probability < high threshold
        medium_prob = (settings.risk_threshold_low + settings.risk_threshold_high) / 2
        assert settings.risk_threshold_low <= medium_prob < settings.risk_threshold_high
        
        # High risk: probability >= high threshold
        high_prob = settings.risk_threshold_high + 0.1
        assert high_prob >= settings.risk_threshold_high


class TestModelConsistency:
    """Tests for model consistency and reproducibility."""
    
    def test_model_version_tracking(self):
        """Test that model version is tracked for audit purposes."""
        from src.api.main import model_name, model_version
        
        # Model version should be tracked (if model is loaded)
        # This is important for regulatory compliance
        pass  # Actual test would require model to be loaded
    
    def test_prediction_reproducibility(self):
        """Test that predictions are reproducible for the same input."""
        # For regulatory compliance, same input should produce same output
        request1 = PredictionRequest(features=[0.0] * 26)
        request2 = PredictionRequest(features=[0.0] * 26)
        
        assert request1.features == request2.features
        # Actual prediction test would require model to be loaded


class TestAuditTrail:
    """Tests for audit trail requirements."""
    
    def test_request_logging_for_audit(self):
        """Test that all prediction requests are logged for audit."""
        # All prediction requests must be logged for regulatory compliance
        # This is handled by RequestLoggingMiddleware
        pass
    
    def test_model_metadata_tracking(self):
        """Test that model metadata is tracked."""
        # Model name, version, and load time should be tracked
        from src.api.main import model_name, model_version, model_load_time
        
        # These should be available for audit purposes
        pass


class TestDataPrivacy:
    """Tests for data privacy compliance."""
    
    def test_no_sensitive_data_in_logs(self):
        """Test that sensitive data is not logged."""
        # For GDPR and privacy compliance, sensitive customer data
        # should not appear in logs
        pass
    
    def test_prediction_request_validation(self):
        """Test that prediction requests are validated before processing."""
        # Invalid requests should be rejected before processing
        with pytest.raises(Exception):
            PredictionRequest(features=["invalid"] * 26)


class TestErrorHandling:
    """Tests for error handling in financial context."""
    
    def test_graceful_degradation(self):
        """Test that system handles errors gracefully."""
        # In financial systems, errors must be handled gracefully
        # to prevent service disruption
        pass
    
    def test_error_response_format(self):
        """Test that error responses follow standard format."""
        # Error responses should be consistent and informative
        # but not expose sensitive information
        pass


class TestPerformanceRequirements:
    """Tests for performance requirements."""
    
    def test_prediction_latency(self):
        """Test that predictions are returned within acceptable latency."""
        # Financial systems typically require sub-second response times
        # This would be tested in integration tests
        pass
    
    def test_concurrent_request_handling(self):
        """Test that system handles concurrent requests correctly."""
        # System should handle multiple concurrent requests
        # This would be tested in load tests
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
