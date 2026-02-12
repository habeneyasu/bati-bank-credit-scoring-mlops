"""
Unit tests for retry utilities (src/utils/retry.py).

Tests verify retry logic, exponential backoff, and error handling.

Run with: pytest tests/test_utils_retry.py -v
"""

import pytest
import time
from unittest.mock import Mock, patch
import asyncio

from src.utils.retry import retry, async_retry


class TestRetryDecorator:
    """Tests for retry decorator."""
    
    def test_retry_success_on_first_attempt(self):
        """Test that function succeeds on first attempt."""
        @retry(max_attempts=3)
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_retry_success_after_failures(self):
        """Test that function succeeds after retries."""
        call_count = [0]
        
        @retry(max_attempts=3, delay=0.1)
        def flaky_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count[0] == 3
    
    def test_retry_exhausted_raises_exception(self):
        """Test that exception is raised after all retries are exhausted."""
        @retry(max_attempts=3, delay=0.1)
        def always_failing_function():
            raise ValueError("Permanent error")
        
        with pytest.raises(ValueError, match="Permanent error"):
            always_failing_function()
    
    def test_retry_only_catches_specified_exceptions(self):
        """Test that retry only catches specified exceptions."""
        @retry(max_attempts=2, delay=0.1, exceptions=(ValueError,))
        def function_with_different_exception():
            raise TypeError("Different error")
        
        with pytest.raises(TypeError):
            function_with_different_exception()
    
    def test_retry_exponential_backoff(self):
        """Test that retry uses exponential backoff."""
        call_times = []
        
        @retry(max_attempts=3, delay=0.1, backoff=2.0)
        def function_with_backoff():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Error")
            return "success"
        
        start_time = time.time()
        result = function_with_backoff()
        end_time = time.time()
        
        assert result == "success"
        assert len(call_times) == 3
        
        # Check that delays increase exponentially
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        
        # Allow some tolerance for timing
        assert delay2 > delay1 * 1.5  # Should be approximately 2x
    
    def test_retry_with_callback(self):
        """Test retry with callback function."""
        callback_calls = []
        
        def on_retry(attempt, exception):
            callback_calls.append((attempt, str(exception)))
        
        call_count = [0]
        
        @retry(max_attempts=3, delay=0.1, on_retry=on_retry)
        def function_with_callback():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Error")
            return "success"
        
        result = function_with_callback()
        
        assert result == "success"
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 1  # First retry attempt


class TestAsyncRetryDecorator:
    """Tests for async retry decorator."""
    
    @pytest.mark.asyncio
    async def test_async_retry_success_on_first_attempt(self):
        """Test that async function succeeds on first attempt."""
        async def successful_function():
            return "success"
        
        # Apply decorator manually to avoid pytest-asyncio issues
        decorated_func = async_retry(max_attempts=3)(successful_function)
        result = await decorated_func()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_async_retry_success_after_failures(self):
        """Test that async function succeeds after retries."""
        call_count = [0]
        
        async def flaky_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary error")
            return "success"
        
        # Apply decorator manually
        decorated_func = async_retry(max_attempts=3, delay=0.1)(flaky_function)
        result = await decorated_func()
        assert result == "success"
        assert call_count[0] == 3
    
    @pytest.mark.asyncio
    async def test_async_retry_exhausted_raises_exception(self):
        """Test that exception is raised after all async retries are exhausted."""
        async def always_failing_function():
            raise ValueError("Permanent error")
        
        # Apply decorator manually
        decorated_func = async_retry(max_attempts=3, delay=0.1)(always_failing_function)
        with pytest.raises(ValueError, match="Permanent error"):
            await decorated_func()
    
    @pytest.mark.asyncio
    async def test_async_retry_exponential_backoff(self):
        """Test that async retry uses exponential backoff."""
        call_times = []
        
        async def function_with_backoff():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Error")
            return "success"
        
        # Apply decorator manually
        decorated_func = async_retry(max_attempts=3, delay=0.1, backoff=2.0)(function_with_backoff)
        start_time = time.time()
        result = await decorated_func()
        end_time = time.time()
        
        assert result == "success"
        assert len(call_times) == 3
        
        # Check that delays increase exponentially
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        
        # Allow some tolerance for timing
        assert delay2 > delay1 * 1.5
    
    @pytest.mark.asyncio
    async def test_async_retry_with_callback(self):
        """Test async retry with callback function."""
        callback_calls = []
        
        def on_retry(attempt, exception):
            callback_calls.append((attempt, str(exception)))
        
        call_count = [0]
        
        async def function_with_callback():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Error")
            return "success"
        
        # Apply decorator manually
        decorated_func = async_retry(max_attempts=3, delay=0.1, on_retry=on_retry)(function_with_callback)
        result = await decorated_func()
        
        assert result == "success"
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 1  # First retry attempt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
