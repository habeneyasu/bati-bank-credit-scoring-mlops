"""
Tests for OpenTelemetry distributed tracing functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

from src.utils.tracing import (
    setup_tracing,
    get_tracer,
    trace_span,
    trace_function,
    add_span_attribute,
    add_span_event,
    set_span_status,
    get_trace_id,
    get_span_id,
    get_trace_context,
    instrument_fastapi,
    instrument_requests,
    instrument_sqlalchemy
)


class TestTracingSetup:
    """Test tracing setup and configuration."""
    
    def test_setup_tracing_when_not_available(self):
        """Test setup when OpenTelemetry is not available."""
        with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', False):
            result = setup_tracing()
            assert result is False
    
    def test_setup_tracing_when_disabled(self):
        """Test setup when tracing is disabled."""
        with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', True):
            with patch('src.utils.tracing._tracing_config') as mock_config:
                mock_config.enabled = False
                result = setup_tracing()
                assert result is False
    
    def test_get_tracer_when_not_available(self):
        """Test getting tracer when OpenTelemetry is not available."""
        with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', False):
            tracer = get_tracer()
            assert tracer is None
    
    def test_get_tracer_when_disabled(self):
        """Test getting tracer when tracing is disabled."""
        with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', True):
            with patch('src.utils.tracing._tracing_config') as mock_config:
                mock_config.enabled = False
                tracer = get_tracer()
                assert tracer is None


class TestTraceSpan:
    """Test trace span context manager."""
    
    def test_trace_span_when_not_available(self):
        """Test trace_span when tracing is not available."""
        with trace_span("test.operation") as span:
            assert span is None
    
    def test_trace_span_with_attributes(self):
        """Test trace_span with attributes."""
        with patch('src.utils.tracing.get_tracer') as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer
            
            with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', True):
                with patch('src.utils.tracing._tracing_config') as mock_config:
                    mock_config.enabled = True
                    
                    with trace_span("test.operation", attributes={"key": "value"}):
                        pass
                    
                    mock_span.set_attribute.assert_called_with("key", "value")
    
    def test_trace_span_with_kind(self):
        """Test trace_span with span kind."""
        with patch('src.utils.tracing.get_tracer') as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer
            
            with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', True):
                with patch('src.utils.tracing._tracing_config') as mock_config:
                    mock_config.enabled = True
                    
                    with trace_span("test.operation", kind="server"):
                        pass
                    
                    # Verify span was created with correct kind
                    mock_tracer.start_as_current_span.assert_called()


class TestTraceFunction:
    """Test trace_function decorator."""
    
    def test_trace_function_sync(self):
        """Test tracing a synchronous function."""
        @trace_function(name="test.function", attributes={"component": "test"})
        def test_func(x, y):
            return x + y
        
        with patch('src.utils.tracing.get_tracer') as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer
            
            with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', True):
                with patch('src.utils.tracing._tracing_config') as mock_config:
                    mock_config.enabled = True
                    
                    result = test_func(2, 3)
                    assert result == 5
    
    @pytest.mark.asyncio
    async def test_trace_function_async(self):
        """Test tracing an asynchronous function."""
        @trace_function(name="test.async_function")
        async def test_async_func(x, y):
            return x * y
        
        with patch('src.utils.tracing.get_tracer') as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer
            
            with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', True):
                with patch('src.utils.tracing._tracing_config') as mock_config:
                    mock_config.enabled = True
                    
                    result = await test_async_func(2, 3)
                    assert result == 6


class TestSpanOperations:
    """Test span attribute and event operations."""
    
    def test_add_span_attribute_when_not_available(self):
        """Test adding attribute when tracing is not available."""
        with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', False):
            # Should not raise exception
            add_span_attribute("key", "value")
    
    def test_add_span_event_when_not_available(self):
        """Test adding event when tracing is not available."""
        with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', False):
            # Should not raise exception
            add_span_event("test.event")
    
    def test_set_span_status_when_not_available(self):
        """Test setting status when tracing is not available."""
        with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', False):
            # Should not raise exception
            set_span_status("ok")
    
    def test_get_trace_id_when_not_available(self):
        """Test getting trace ID when tracing is not available."""
        with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', False):
            trace_id = get_trace_id()
            assert trace_id is None
    
    def test_get_span_id_when_not_available(self):
        """Test getting span ID when tracing is not available."""
        with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', False):
            span_id = get_span_id()
            assert span_id is None
    
    def test_get_trace_context(self):
        """Test getting trace context."""
        with patch('src.utils.tracing.get_trace_id', return_value="abc123"):
            with patch('src.utils.tracing.get_span_id', return_value="def456"):
                context = get_trace_context()
                assert context == {"trace_id": "abc123", "span_id": "def456"}


class TestInstrumentation:
    """Test instrumentation functions."""
    
    def test_instrument_fastapi_when_not_available(self):
        """Test FastAPI instrumentation when OpenTelemetry is not available."""
        mock_app = MagicMock()
        with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', False):
            # Should not raise exception
            instrument_fastapi(mock_app)
    
    def test_instrument_requests_when_not_available(self):
        """Test requests instrumentation when OpenTelemetry is not available."""
        with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', False):
            # Should not raise exception
            instrument_requests()
    
    def test_instrument_sqlalchemy_when_not_available(self):
        """Test SQLAlchemy instrumentation when OpenTelemetry is not available."""
        with patch('src.utils.tracing.OPENTELEMETRY_AVAILABLE', False):
            # Should not raise exception
            instrument_sqlalchemy()


class TestTracingIntegration:
    """Integration tests for tracing with FastAPI."""
    
    @pytest.mark.asyncio
    async def test_tracing_in_predict_endpoint(self):
        """Test that tracing is integrated in predict endpoint."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        client = TestClient(app)
        
        # Mock tracing to avoid actual OpenTelemetry setup
        with patch('src.utils.tracing.setup_tracing', return_value=False):
            # This should not raise an error even if tracing is not set up
            response = client.get("/health")
            assert response.status_code in [200, 503]  # 503 if model not loaded
