"""
Caching utilities for API performance optimization.

Provides in-memory and Redis-based caching for predictions and explanations
to meet sub-200ms latency requirements for real-time lending decisions.
"""

import hashlib
import json
import time
from typing import Optional, Any, Dict
from functools import wraps
from collections import OrderedDict
import threading

from src.utils.logging import get_logger

logger = get_logger(__name__)


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation.
    
    Used for in-memory caching when Redis is not available.
    """
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl: Time-to-live in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.RLock()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamps.get(key, 0) > self.ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            if self._is_expired(key):
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)


class CacheManager:
    """
    Unified cache manager supporting both in-memory and Redis caching.
    """
    
    def __init__(self, use_redis: bool = False, redis_url: Optional[str] = None):
        """
        Initialize cache manager.
        
        Args:
            use_redis: Whether to use Redis (requires redis package)
            redis_url: Redis connection URL
        """
        self.use_redis = use_redis
        self.redis_client = None
        
        if use_redis:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url or "redis://localhost:6379/0")
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except ImportError:
                logger.warning("Redis not available, falling back to in-memory cache")
                self.use_redis = False
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, falling back to in-memory cache")
                self.use_redis = False
        
        # Fallback to in-memory cache
        if not self.use_redis:
            self.memory_cache = LRUCache(max_size=1000, ttl=300)  # 5 min TTL
            logger.info("Using in-memory LRU cache")
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """
        Generate cache key from data.
        
        Args:
            prefix: Key prefix
            data: Data to hash
        
        Returns:
            Cache key string
        """
        # Create deterministic hash from data
        if isinstance(data, (list, tuple)):
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        hash_obj = hashlib.sha256(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        if self.use_redis and self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                return None
        else:
            return self.memory_cache.get(key)
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if self.use_redis and self.redis_client:
            try:
                value_str = json.dumps(value)
                if ttl:
                    self.redis_client.setex(key, ttl, value_str)
                else:
                    self.redis_client.set(key, value_str)
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
        else:
            self.memory_cache.set(key, value)
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
        else:
            # In-memory cache doesn't support direct delete, but we can clear
            pass
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
        else:
            self.memory_cache.clear()


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        from src.utils.config import settings
        use_redis = getattr(settings, 'use_redis_cache', False)
        redis_url = getattr(settings, 'redis_url', None)
        _cache_manager = CacheManager(use_redis=use_redis, redis_url=redis_url)
    return _cache_manager


def cached_prediction(cache_key_prefix: str = "prediction", ttl: int = 300):
    """
    Decorator to cache prediction results.
    
    Args:
        cache_key_prefix: Prefix for cache keys
        ttl: Time-to-live in seconds
    
    Usage:
        @cached_prediction(ttl=300)
        async def predict(features):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache key from function arguments
            # For prediction, key is based on features
            if 'request' in kwargs:
                features = kwargs['request'].features
            elif args and hasattr(args[0], 'features'):
                features = args[0].features
            else:
                features = kwargs.get('features', args[0] if args else None)
            
            if features:
                cache_key = cache._generate_key(cache_key_prefix, features)
                
                # Try to get from cache
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {cache_key_prefix}")
                    return cached_result
                
                # Cache miss - compute result
                result = await func(*args, **kwargs)
                
                # Store in cache
                cache.set(cache_key, result, ttl=ttl)
                logger.debug(f"Cache miss for {cache_key_prefix}, stored result")
                
                return result
            else:
                # No features to cache - just call function
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator
