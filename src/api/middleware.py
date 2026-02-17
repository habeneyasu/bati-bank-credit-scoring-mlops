"""
Custom middleware for the credit scoring API.

Includes rate limiting, request logging, error handling, and PII redaction.
"""

import time
import json
from typing import Callable, Dict, Any
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.config import settings
from src.utils.logging import get_logger
from src.utils.pii_redaction import redact_pii, get_redactor

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.rate_limit_store: dict = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limit before processing request."""
        # Skip rate limiting for health and metrics endpoints
        if request.url.path in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        now = datetime.now(timezone.utc)
        
        # Clean old entries (older than 1 minute)
        self.rate_limit_store[client_ip] = [
            timestamp for timestamp in self.rate_limit_store[client_ip]
            if now - timestamp < timedelta(minutes=1)
        ]
        
        # Check rate limit
        if len(self.rate_limit_store[client_ip]) >= self.requests_per_minute:
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_ip": client_ip,
                    "path": request.url.path,
                    "requests_in_window": len(self.rate_limit_store[client_ip])
                }
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Add current request timestamp
        self.rate_limit_store[client_ip].append(now)
        
        response = await call_next(request)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all requests and responses with PII redaction."""
    
    def __init__(self, app):
        super().__init__(app)
        self.redactor = get_redactor() if settings.enable_pii_redaction and settings.redact_in_logs else None
    
    def _redact_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PII from log data if enabled."""
        if self.redactor and settings.redact_in_logs:
            return self.redactor.redact_dict(data, recursive=True)
        return data
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details with PII redaction."""
        start_time = time.time()
        
        # Prepare request log data
        request_log_data = {
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown")
        }
        
        # Note: Request body is not read here to avoid consuming the stream
        # Body will be read by the endpoint handler
        # PII redaction in logs happens via the logging extra fields
        
        # Redact PII from log data
        redacted_request_data = self._redact_log_data(request_log_data)
        
        # Log request
        logger.info(
            "Incoming request",
            extra=redacted_request_data
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Prepare response log data
            response_log_data = {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time_seconds": round(process_time, 4)
            }
            
            # Redact PII from response log data
            redacted_response_data = self._redact_log_data(response_log_data)
            
            # Log successful response
            logger.info(
                "Request completed",
                extra=redacted_response_data
            )
            
            # Add process time header
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Prepare error log data
            error_log_data = {
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
                "error_type": type(e).__name__,
                "process_time_seconds": round(process_time, 4)
            }
            
            # Redact PII from error log data
            redacted_error_data = self._redact_log_data(error_log_data)
            
            logger.error(
                "Request failed",
                extra=redacted_error_data,
                exc_info=True
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling and formatting errors with PII redaction."""
    
    def __init__(self, app):
        super().__init__(app)
        self.redactor = get_redactor() if settings.enable_pii_redaction and settings.redact_in_logs else None
    
    def _redact_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PII from log data if enabled."""
        if self.redactor and settings.redact_in_logs:
            return self.redactor.redact_dict(data, recursive=True)
        return data
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors and format responses."""
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Prepare error log data
            error_log_data = {
                "path": request.url.path,
                "method": request.method,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            # Redact PII from error log data
            redacted_error_data = self._redact_log_data(error_log_data)
            
            logger.error(
                "Unhandled exception",
                extra=redacted_error_data,
                exc_info=True
            )
            
            # Return generic error in production, detailed in development
            if settings.is_production:
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": "Internal server error"}
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "detail": "Internal server error",
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )


class PIIRedactionMiddleware(BaseHTTPMiddleware):
    """Middleware for redacting PII from API responses (optional).
    
    WARNING: This is disabled by default (redact_in_responses=False).
    Redacting responses may break API contracts. Typically, PII redaction
    should only be applied to logs, not responses.
    
    If you need response redaction, implement it at the endpoint level
    rather than via middleware to maintain API contract integrity.
    """
    
    def __init__(self, app):
        super().__init__(app)
        # Response redaction is typically not recommended
        # This middleware is a placeholder for future implementation if needed
        self.enabled = settings.enable_pii_redaction and settings.redact_in_responses
        if self.enabled:
            logger.warning(
                "PII response redaction is enabled. This may break API contracts. "
                "Consider redacting only in logs instead."
            )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Pass through response (redaction handled at endpoint level if needed)."""
        # Response redaction is complex and may break API contracts
        # For now, we pass through. If needed, implement at endpoint level.
        response = await call_next(request)
        return response
