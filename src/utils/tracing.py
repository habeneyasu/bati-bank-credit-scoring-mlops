"""
OpenTelemetry Distributed Tracing Configuration

Provides distributed tracing capabilities for request correlation,
performance profiling, and debugging across microservices.

This module integrates OpenTelemetry with FastAPI to provide:
- Automatic request tracing
- Span creation for key operations
- Trace context propagation
- Performance profiling
- Error tracking with traces
"""

import functools
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from datetime import datetime

from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# OpenTelemetry imports (with graceful fallback if not installed)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.zipkin.json import ZipkinExporter
    
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning(
        "OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk "
        "opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-requests "
        "opentelemetry-instrumentation-sqlalchemy opentelemetry-exporter-otlp"
    )


class TracingConfig:
    """Configuration for distributed tracing."""
    
    def __init__(self):
        self.enabled = getattr(settings, 'enable_distributed_tracing', False)
        self.service_name = getattr(settings, 'tracing_service_name', 'credit-scoring-api')
        self.service_version = getattr(settings, 'tracing_service_version', '1.0.0')
        self.exporter_type = getattr(settings, 'tracing_exporter', 'console')  # console, otlp, jaeger, zipkin
        self.otlp_endpoint = getattr(settings, 'tracing_otlp_endpoint', 'http://localhost:4317')
        self.jaeger_endpoint = getattr(settings, 'tracing_jaeger_endpoint', 'http://localhost:14268/api/traces')
        self.zipkin_endpoint = getattr(settings, 'tracing_zipkin_endpoint', 'http://localhost:9411/api/v2/spans')
        self.sample_rate = getattr(settings, 'tracing_sample_rate', 1.0)  # 0.0 to 1.0


# Global tracing config
_tracing_config = TracingConfig()
_tracer_provider: Optional[Any] = None
_tracer: Optional[Any] = None


def get_tracer(name: Optional[str] = None) -> Optional[Any]:
    """
    Get OpenTelemetry tracer instance.
    
    Args:
        name: Tracer name (default: service name)
        
    Returns:
        Tracer instance or None if not available
    """
    if not OPENTELEMETRY_AVAILABLE or not _tracing_config.enabled:
        return None
    
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(name or _tracing_config.service_name)
    return _tracer


def setup_tracing() -> bool:
    """
    Set up OpenTelemetry tracing.
    
    Returns:
        True if setup successful, False otherwise
    """
    if not OPENTELEMETRY_AVAILABLE:
        logger.warning("OpenTelemetry not available, tracing disabled")
        return False
    
    if not _tracing_config.enabled:
        logger.info("Distributed tracing is disabled")
        return False
    
    try:
        global _tracer_provider
        
        # Create resource with service information
        resource = Resource.create({
            "service.name": _tracing_config.service_name,
            "service.version": _tracing_config.service_version,
            "service.namespace": "mlops",
            "deployment.environment": settings.environment
        })
        
        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)
        
        # Configure sampler
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        sampler = TraceIdRatioBased(_tracing_config.sample_rate)
        _tracer_provider = TracerProvider(resource=resource, sampler=sampler)
        
        # Create span exporter based on configuration
        exporter = None
        if _tracing_config.exporter_type == "console":
            exporter = ConsoleSpanExporter()
        elif _tracing_config.exporter_type == "otlp":
            exporter = OTLPSpanExporter(
                endpoint=_tracing_config.otlp_endpoint,
                insecure=True  # Set to False for TLS
            )
        elif _tracing_config.exporter_type == "jaeger":
            exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
                collector_endpoint=_tracing_config.jaeger_endpoint
            )
        elif _tracing_config.exporter_type == "zipkin":
            exporter = ZipkinExporter(
                endpoint=_tracing_config.zipkin_endpoint
            )
        else:
            logger.warning(f"Unknown exporter type: {_tracing_config.exporter_type}, using console")
            exporter = ConsoleSpanExporter()
        
        # Add span processor
        span_processor = BatchSpanProcessor(exporter)
        _tracer_provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(_tracer_provider)
        
        logger.info(
            f"OpenTelemetry tracing initialized: service={_tracing_config.service_name}, "
            f"exporter={_tracing_config.exporter_type}, sample_rate={_tracing_config.sample_rate}"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry tracing: {e}", exc_info=True)
        return False


def instrument_fastapi(app):
    """
    Instrument FastAPI application with OpenTelemetry.
    
    Args:
        app: FastAPI application instance
    """
    if not OPENTELEMETRY_AVAILABLE or not _tracing_config.enabled:
        return
    
    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented with OpenTelemetry")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}", exc_info=True)


def instrument_requests():
    """Instrument requests library with OpenTelemetry."""
    if not OPENTELEMETRY_AVAILABLE or not _tracing_config.enabled:
        return
    
    try:
        RequestsInstrumentor().instrument()
        logger.info("Requests library instrumented with OpenTelemetry")
    except Exception as e:
        logger.error(f"Failed to instrument requests: {e}", exc_info=True)


def instrument_sqlalchemy():
    """Instrument SQLAlchemy with OpenTelemetry."""
    if not OPENTELEMETRY_AVAILABLE or not _tracing_config.enabled:
        return
    
    try:
        SQLAlchemyInstrumentor().instrument()
        logger.info("SQLAlchemy instrumented with OpenTelemetry")
    except Exception as e:
        logger.error(f"Failed to instrument SQLAlchemy: {e}", exc_info=True)


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[str] = None
):
    """
    Context manager for creating a trace span.
    
    Args:
        name: Span name
        attributes: Span attributes (key-value pairs)
        kind: Span kind ('server', 'client', 'internal', 'producer', 'consumer')
        
    Yields:
        Span instance
    """
    tracer = get_tracer()
    if not tracer:
        yield None
        return
    
    try:
        # Map string kind to OpenTelemetry SpanKind
        span_kind = None
        if kind:
            from opentelemetry.trace import SpanKind
            kind_map = {
                'server': SpanKind.SERVER,
                'client': SpanKind.CLIENT,
                'internal': SpanKind.INTERNAL,
                'producer': SpanKind.PRODUCER,
                'consumer': SpanKind.CONSUMER
            }
            span_kind = kind_map.get(kind.lower(), SpanKind.INTERNAL)
        
        with tracer.start_as_current_span(name, kind=span_kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            
            yield span
    except Exception as e:
        logger.error(f"Error in trace span: {e}", exc_info=True)
        yield None


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_args: bool = False
):
    """
    Decorator for tracing function execution.
    
    Args:
        name: Span name (default: function name)
        attributes: Additional span attributes
        capture_args: Whether to capture function arguments as attributes
        
    Example:
        @trace_function(attributes={"component": "model"})
        def predict(features):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            span_attrs = attributes or {}
            
            if capture_args:
                # Add function arguments as attributes (be careful with PII)
                for i, arg in enumerate(args):
                    span_attrs[f"arg.{i}"] = str(arg)[:100]  # Truncate long values
                for key, value in kwargs.items():
                    span_attrs[f"kwarg.{key}"] = str(value)[:100]
            
            with trace_span(span_name, attributes=span_attrs):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            span_attrs = attributes or {}
            
            if capture_args:
                for i, arg in enumerate(args):
                    span_attrs[f"arg.{i}"] = str(arg)[:100]
                for key, value in kwargs.items():
                    span_attrs[f"kwarg.{key}"] = str(value)[:100]
            
            tracer = get_tracer()
            if tracer:
                with tracer.start_as_current_span(span_name) as span:
                    if span_attrs:
                        for key, value in span_attrs.items():
                            span.set_attribute(key, str(value))
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        if hasattr(func, '__code__') and 'async' in str(func.__code__.co_flags):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def add_span_attribute(key: str, value: Any):
    """
    Add attribute to current span.
    
    Args:
        key: Attribute key
        value: Attribute value
    """
    tracer = get_tracer()
    if not tracer:
        return
    
    try:
        span = trace.get_current_span()
        if span:
            span.set_attribute(key, str(value))
    except Exception as e:
        logger.debug(f"Failed to add span attribute: {e}")


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Add event to current span.
    
    Args:
        name: Event name
        attributes: Event attributes
    """
    tracer = get_tracer()
    if not tracer:
        return
    
    try:
        span = trace.get_current_span()
        if span:
            span.add_event(name, attributes=attributes or {})
    except Exception as e:
        logger.debug(f"Failed to add span event: {e}")


def set_span_status(status_code: str, description: Optional[str] = None):
    """
    Set status on current span.
    
    Args:
        status_code: 'ok', 'error', or 'unset'
        description: Optional status description
    """
    tracer = get_tracer()
    if not tracer:
        return
    
    try:
        span = trace.get_current_span()
        if span:
            status_map = {
                'ok': Status(StatusCode.OK),
                'error': Status(StatusCode.ERROR, description),
                'unset': Status(StatusCode.UNSET)
            }
            span.set_status(status_map.get(status_code.lower(), Status(StatusCode.UNSET)))
    except Exception as e:
        logger.debug(f"Failed to set span status: {e}")


def get_trace_id() -> Optional[str]:
    """
    Get current trace ID.
    
    Returns:
        Trace ID as hex string or None
    """
    tracer = get_tracer()
    if not tracer:
        return None
    
    try:
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().trace_id, '032x')
    except Exception:
        pass
    
    return None


def get_span_id() -> Optional[str]:
    """
    Get current span ID.
    
    Returns:
        Span ID as hex string or None
    """
    tracer = get_tracer()
    if not tracer:
        return None
    
    try:
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().span_id, '016x')
    except Exception:
        pass
    
    return None


def get_trace_context() -> Dict[str, Optional[str]]:
    """
    Get current trace context (trace_id, span_id).
    
    Returns:
        Dictionary with trace_id and span_id
    """
    return {
        "trace_id": get_trace_id(),
        "span_id": get_span_id()
    }
