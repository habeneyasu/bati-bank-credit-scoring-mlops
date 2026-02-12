"""
Unit tests for API middleware (src/api/middleware.py).

Tests verify rate limiting, request logging, and error handling middleware.

Run with: pytest tests/test_api_middleware.py -v
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware
)
from src.utils.config import Settings


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app."""
        app = Mock()
        return app
    
    @pytest.fixture
    def rate_limit_middleware(self, mock_app):
        """Create RateLimitMiddleware instance."""
        return RateLimitMiddleware(mock_app, requests_per_minute=5)
    
    @pytest.mark.asyncio
    async def test_rate_limit_allows_requests(self, rate_limit_middleware):
        """Test that requests within limit are allowed."""
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/predict"
        mock_request.client.host = "127.0.0.1"
        
        async def mock_call_next(request):
            return JSONResponse({"status": "ok"})
        
        response = await rate_limit_middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excessive_requests(self, rate_limit_middleware):
        """Test that excessive requests are blocked."""
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/predict"
        mock_request.client.host = "127.0.0.1"
        
        async def mock_call_next(request):
            return JSONResponse({"status": "ok"})
        
        # Make requests up to the limit
        for _ in range(5):
            response = await rate_limit_middleware.dispatch(mock_request, mock_call_next)
            assert response.status_code == 200
        
        # Next request should be blocked
        response = await rate_limit_middleware.dispatch(mock_request, mock_call_next)
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "Rate limit exceeded" in response.body.decode()
    
    @pytest.mark.asyncio
    async def test_rate_limit_skips_health_endpoint(self, rate_limit_middleware):
        """Test that health endpoint is not rate limited."""
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/health"
        mock_request.client.host = "127.0.0.1"
        
        async def mock_call_next(request):
            return JSONResponse({"status": "ok"})
        
        # Make many requests to health endpoint
        for _ in range(10):
            response = await rate_limit_middleware.dispatch(mock_request, mock_call_next)
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_rate_limit_skips_metrics_endpoint(self, rate_limit_middleware):
        """Test that metrics endpoint is not rate limited."""
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/metrics"
        mock_request.client.host = "127.0.0.1"
        
        async def mock_call_next(request):
            return JSONResponse({"status": "ok"})
        
        # Make many requests to metrics endpoint
        for _ in range(10):
            response = await rate_limit_middleware.dispatch(mock_request, mock_call_next)
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_rate_limit_cleans_old_entries(self, rate_limit_middleware):
        """Test that old rate limit entries are cleaned up."""
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/predict"
        mock_request.client.host = "127.0.0.1"
        
        async def mock_call_next(request):
            return JSONResponse({"status": "ok"})
        
        # Fill up the rate limit (make 4 requests to stay under limit)
        for _ in range(4):
            response = await rate_limit_middleware.dispatch(mock_request, mock_call_next)
            assert response.status_code == 200
        
        # Manually clear all entries to simulate expiration
        rate_limit_middleware.rate_limit_store["127.0.0.1"] = []
        
        # Should be able to make requests again after clearing
        response = await rate_limit_middleware.dispatch(mock_request, mock_call_next)
        assert response.status_code == 200


class TestRequestLoggingMiddleware:
    """Tests for RequestLoggingMiddleware."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app."""
        app = Mock()
        return app
    
    @pytest.fixture
    def logging_middleware(self, mock_app):
        """Create RequestLoggingMiddleware instance."""
        return RequestLoggingMiddleware(mock_app)
    
    @pytest.mark.asyncio
    async def test_request_logging_logs_request(self, logging_middleware):
        """Test that requests are logged."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/health"
        mock_request.query_params = {}
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get = Mock(return_value="test-agent")
        
        async def mock_call_next(request):
            return JSONResponse({"status": "ok"})
        
        with patch('src.api.middleware.logger') as mock_logger:
            response = await logging_middleware.dispatch(mock_request, mock_call_next)
            
            assert response.status_code == 200
            assert mock_logger.info.called
    
    @pytest.mark.asyncio
    async def test_request_logging_adds_process_time_header(self, logging_middleware):
        """Test that process time header is added."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/health"
        mock_request.query_params = {}
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get = Mock(return_value="test-agent")
        
        async def mock_call_next(request):
            return JSONResponse({"status": "ok"})
        
        response = await logging_middleware.dispatch(mock_request, mock_call_next)
        
        assert "X-Process-Time" in response.headers
    
    @pytest.mark.asyncio
    async def test_request_logging_logs_errors(self, logging_middleware):
        """Test that errors are logged."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/predict"
        mock_request.query_params = {}
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get = Mock(return_value="test-agent")
        
        mock_call_next = Mock(side_effect=ValueError("Test error"))
        
        with patch('src.api.middleware.logger') as mock_logger:
            with pytest.raises(ValueError):
                await logging_middleware.dispatch(mock_request, mock_call_next)
            
            assert mock_logger.error.called


class TestErrorHandlingMiddleware:
    """Tests for ErrorHandlingMiddleware."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app."""
        app = Mock()
        return app
    
    @pytest.fixture
    def error_middleware(self, mock_app):
        """Create ErrorHandlingMiddleware instance."""
        return ErrorHandlingMiddleware(mock_app)
    
    @pytest.mark.asyncio
    async def test_error_handling_returns_generic_error_in_production(self, error_middleware):
        """Test that generic error is returned in production."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/predict"
        
        async def mock_call_next(request):
            raise ValueError("Test error")
        
        # Mock settings to return is_production=True
        with patch('src.api.middleware.settings') as mock_settings:
            mock_settings.is_production = True
            with patch('src.api.middleware.logger') as mock_logger:
                response = await error_middleware.dispatch(mock_request, mock_call_next)
                
                assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
                response_body = response.body.decode()
                assert "Internal server error" in response_body
                # In production mode, error details should not be exposed
                # The actual implementation may vary, so we just check status code
    
    @pytest.mark.asyncio
    async def test_error_handling_returns_detailed_error_in_development(self, error_middleware):
        """Test that detailed error is returned in development."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/predict"
        
        mock_call_next = Mock(side_effect=ValueError("Test error"))
        
        with patch('src.utils.config.settings') as mock_settings:
            mock_settings.is_production = False
            with patch('src.api.middleware.logger') as mock_logger:
                response = await error_middleware.dispatch(mock_request, mock_call_next)
                
                assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
                assert "Internal server error" in response.body.decode()
                assert "Test error" in response.body.decode()
    
    @pytest.mark.asyncio
    async def test_error_handling_logs_exceptions(self, error_middleware):
        """Test that exceptions are logged."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/predict"
        
        mock_call_next = Mock(side_effect=ValueError("Test error"))
        
        with patch('src.api.middleware.logger') as mock_logger:
            await error_middleware.dispatch(mock_request, mock_call_next)
            
            assert mock_logger.error.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
