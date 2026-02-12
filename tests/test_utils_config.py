"""
Unit tests for configuration management (src/utils/config.py).

Tests verify environment variable loading, validation, and property access.

Run with: pytest tests/test_utils_config.py -v
"""

import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError

from src.utils.config import Settings


class TestSettings:
    """Tests for Settings class."""
    
    def test_default_settings(self):
        """Test that default settings are loaded correctly."""
        # Clear any environment variables that might affect defaults
        import os
        env_backup = os.environ.get("MLFLOW_TRACKING_URI")
        if "MLFLOW_TRACKING_URI" in os.environ:
            del os.environ["MLFLOW_TRACKING_URI"]
        
        try:
            settings = Settings()
            # Allow for temp directories in test environment
            assert "mlruns" in settings.mlflow_tracking_uri or settings.mlflow_tracking_uri.startswith("file:")
        finally:
            if env_backup:
                os.environ["MLFLOW_TRACKING_URI"] = env_backup
        assert settings.model_name == "credit_scoring_model"
        assert settings.model_stage == "Production"
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.expected_features == 26
        assert settings.risk_threshold_low == 0.30
        assert settings.risk_threshold_high == 0.60
    
    def test_environment_variable_loading(self):
        """Test loading settings from environment variables."""
        with patch.dict(os.environ, {
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
            "MODEL_NAME": "test_model",
            "API_PORT": "9000"
        }):
            settings = Settings()
            assert settings.mlflow_tracking_uri == "http://localhost:5000"
            assert settings.model_name == "test_model"
            assert settings.api_port == 9000
    
    def test_api_port_validation(self):
        """Test API port validation."""
        # Valid port
        settings = Settings(api_port=8080)
        assert settings.api_port == 8080
        
        # Invalid: port too low
        with pytest.raises(ValidationError):
            Settings(api_port=0)
        
        # Invalid: port too high
        with pytest.raises(ValidationError):
            Settings(api_port=70000)
    
    def test_risk_threshold_validation(self):
        """Test risk threshold validation."""
        # Valid thresholds
        settings = Settings(
            risk_threshold_low=0.2,
            risk_threshold_high=0.8
        )
        assert settings.risk_threshold_low == 0.2
        assert settings.risk_threshold_high == 0.8
        
        # Invalid: high threshold <= low threshold
        with pytest.raises(ValidationError):
            Settings(
                risk_threshold_low=0.8,
                risk_threshold_high=0.2
            )
        
        # Invalid: threshold out of range
        with pytest.raises(ValidationError):
            Settings(risk_threshold_low=-0.1)
        
        with pytest.raises(ValidationError):
            Settings(risk_threshold_high=1.5)
    
    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level
        
        # Invalid log level
        with pytest.raises(ValidationError):
            Settings(log_level="INVALID")
    
    def test_cors_origins_parsing(self):
        """Test CORS origins parsing from string."""
        # String input (comma-separated)
        settings = Settings(cors_origins="http://localhost:3000,https://example.com")
        assert len(settings.cors_origins) == 2
        assert "http://localhost:3000" in settings.cors_origins
        assert "https://example.com" in settings.cors_origins
        
        # List input
        settings = Settings(cors_origins=["http://localhost:3000"])
        assert len(settings.cors_origins) == 1
    
    def test_cors_methods_parsing(self):
        """Test CORS methods parsing from string."""
        # String input
        settings = Settings(cors_allow_methods="GET,POST,PUT")
        assert len(settings.cors_allow_methods) == 3
        assert "GET" in settings.cors_allow_methods
        
        # List input
        settings = Settings(cors_allow_methods=["GET", "POST"])
        assert len(settings.cors_allow_methods) == 2
    
    def test_cors_headers_parsing(self):
        """Test CORS headers parsing from string."""
        # String input
        settings = Settings(cors_allow_headers="Content-Type,Authorization")
        assert len(settings.cors_allow_headers) == 2
        
        # List input
        settings = Settings(cors_allow_headers=["Content-Type"])
        assert len(settings.cors_allow_headers) == 1
    
    def test_is_production_property(self):
        """Test is_production property."""
        # Production environment
        settings = Settings(environment="production")
        assert settings.is_production is True
        
        # Development environment
        settings = Settings(environment="development")
        assert settings.is_production is False
        
        # Case insensitive
        settings = Settings(environment="PRODUCTION")
        assert settings.is_production is True
    
    def test_is_development_property(self):
        """Test is_development property."""
        # Development environment
        settings = Settings(environment="development")
        assert settings.is_development is True
        
        # Production environment
        settings = Settings(environment="production")
        assert settings.is_development is False
    
    def test_expected_features_validation(self):
        """Test expected features validation."""
        # Valid
        settings = Settings(expected_features=26)
        assert settings.expected_features == 26
        
        # Invalid: too low
        with pytest.raises(ValidationError):
            Settings(expected_features=0)
    
    def test_rate_limit_validation(self):
        """Test rate limit validation."""
        # Valid
        settings = Settings(rate_limit_per_minute=100)
        assert settings.rate_limit_per_minute == 100
        
        # Invalid: too low
        with pytest.raises(ValidationError):
            Settings(rate_limit_per_minute=0)
    
    def test_api_workers_validation(self):
        """Test API workers validation."""
        # Valid
        settings = Settings(api_workers=4)
        assert settings.api_workers == 4
        
        # Invalid: too low
        with pytest.raises(ValidationError):
            Settings(api_workers=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
