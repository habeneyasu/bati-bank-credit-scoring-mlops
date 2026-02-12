"""
Retry utilities for resilient API calls and model loading.
"""

import asyncio
import time
import logging
from typing import Callable, TypeVar, Optional, List
from functools import wraps

T = TypeVar('T')
logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch and retry on
        on_retry: Optional callback function called on each retry
    
    Returns:
        Decorated function
    
    Example:
        @retry(max_attempts=3, delay=1.0, exceptions=(ConnectionError,))
        def load_model():
            return mlflow.sklearn.load_model("models:/model/Production")
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts",
                            extra={"error": str(e), "attempts": max_attempts},
                            exc_info=True
                        )
                        raise
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt}/{max_attempts}), retrying...",
                        extra={
                            "error": str(e),
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "delay": current_delay
                        }
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


async def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Async retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch and retry on
        on_retry: Optional callback function called on each retry
    
    Returns:
        Decorated async function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"Async function {func.__name__} failed after {max_attempts} attempts",
                            extra={"error": str(e), "attempts": max_attempts},
                            exc_info=True
                        )
                        raise
                    
                    logger.warning(
                        f"Async function {func.__name__} failed (attempt {attempt}/{max_attempts}), retrying...",
                        extra={
                            "error": str(e),
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "delay": current_delay
                        }
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator
