"""
Performance monitoring and benchmarking utilities.

Tracks response times, calculates percentiles, and provides performance metrics
to ensure 95th percentile latency under 200ms for real-time lending decisions.
"""

import time
import statistics
from typing import List, Dict, Optional, Callable
from collections import deque
from threading import Lock
import numpy as np

# Use numpy for faster percentile calculations
try:
    from numpy import percentile, mean, median, std, min as np_min, max as np_max
except ImportError:
    # Fallback if numpy not available
    percentile = statistics.quantiles
    mean = statistics.mean
    median = statistics.median
    std = statistics.stdev
    np_min = min
    np_max = max

from src.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """
    Performance monitor for tracking API response times and metrics.
    """
    
    def __init__(self, max_samples: int = 10000):
        """
        Initialize performance monitor.
        
        Args:
            max_samples: Maximum number of samples to keep in memory
        """
        self.max_samples = max_samples
        self.latencies: deque = deque(maxlen=max_samples)
        self.lock = Lock()
        
        # Track by endpoint
        self.endpoint_latencies: Dict[str, deque] = {}
        
        # Track errors
        self.error_count = 0
        self.total_requests = 0
    
    def record_latency(self, latency: float, endpoint: Optional[str] = None):
        """
        Record a latency measurement.
        
        Args:
            latency: Latency in seconds
            endpoint: Optional endpoint name
        """
        with self.lock:
            self.latencies.append(latency)
            self.total_requests += 1
            
            if endpoint:
                if endpoint not in self.endpoint_latencies:
                    self.endpoint_latencies[endpoint] = deque(maxlen=self.max_samples)
                self.endpoint_latencies[endpoint].append(latency)
    
    def record_error(self):
        """Record an error."""
        with self.lock:
            self.error_count += 1
    
    def get_stats(self, endpoint: Optional[str] = None) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Optimized for fast response - uses efficient numpy operations.
        
        Args:
            endpoint: Optional endpoint name to filter
        
        Returns:
            Dictionary with performance statistics
        """
        with self.lock:
            # Get latencies (copy to avoid holding lock during computation)
            if endpoint and endpoint in self.endpoint_latencies:
                latencies = list(self.endpoint_latencies[endpoint])
            else:
                latencies = list(self.latencies)
            
            if not latencies:
                return {
                    "count": 0,
                    "mean": 0.0,
                    "median": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "std": 0.0
                }
            
            # Convert to numpy array for efficient computation
            latencies_array = np.array(latencies) * 1000  # Convert to milliseconds
            
            # Use numpy for fast percentile calculation
            return {
                "count": len(latencies),
                "mean": float(np.mean(latencies_array)),
                "median": float(np.median(latencies_array)),
                "p50": float(np.percentile(latencies_array, 50)),
                "p95": float(np.percentile(latencies_array, 95)),
                "p99": float(np.percentile(latencies_array, 99)),
                "min": float(np.min(latencies_array)),
                "max": float(np.max(latencies_array)),
                "std": float(np.std(latencies_array)) if len(latencies_array) > 1 else 0.0
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all endpoints.
        
        Optimized to minimize lock time and computation.
        
        Returns:
            Dictionary mapping endpoint names to statistics
        """
        # Get all data while holding lock (minimal time)
        with self.lock:
            endpoint_names = list(self.endpoint_latencies.keys())
            error_count = self.error_count
            total_requests = self.total_requests
        
        # Compute stats outside lock (faster)
        stats = {
            "all": self.get_stats()
        }
        
        # Get stats for each endpoint (computed outside lock)
        for endpoint in endpoint_names:
            stats[endpoint] = self.get_stats(endpoint)
        
        stats["error_rate"] = {
            "errors": error_count,
            "total": total_requests,
            "rate": error_count / total_requests if total_requests > 0 else 0.0
        }
        
        return stats
    
    def check_sla(self, percentile: float = 95, threshold_ms: float = 200) -> Dict[str, bool]:
        """
        Check if SLA requirements are met.
        
        Args:
            percentile: Percentile to check (default: 95)
            threshold_ms: Threshold in milliseconds (default: 200)
        
        Returns:
            Dictionary with SLA compliance status
        """
        stats = self.get_stats()
        p95 = stats.get("p95", 0.0)
        
        return {
            "compliant": p95 <= threshold_ms,
            "p95_ms": p95,
            "threshold_ms": threshold_ms,
            "margin_ms": threshold_ms - p95
        }
    
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.latencies.clear()
            self.endpoint_latencies.clear()
            self.error_count = 0
            self.total_requests = 0


class PerformanceTimer:
    """
    Context manager for timing code blocks.
    """
    
    def __init__(self, monitor: Optional[PerformanceMonitor] = None, endpoint: Optional[str] = None):
        """
        Initialize timer.
        
        Args:
            monitor: Optional performance monitor to record to
            endpoint: Optional endpoint name
        """
        self.monitor = monitor
        self.endpoint = endpoint
        self.start_time = None
        self.latency = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.latency = time.perf_counter() - self.start_time
        if self.monitor:
            self.monitor.record_latency(self.latency, self.endpoint)
        return False
    
    def get_latency(self) -> Optional[float]:
        """Get measured latency in seconds."""
        return self.latency


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def benchmark_function(func: Callable, n_iterations: int = 100, *args, **kwargs) -> Dict[str, float]:
    """
    Benchmark a function.
    
    Args:
        func: Function to benchmark
        n_iterations: Number of iterations
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        Dictionary with benchmark results
    """
    latencies = []
    
    for _ in range(n_iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        latency = time.perf_counter() - start
        latencies.append(latency)
    
    latencies_ms = [l * 1000 for l in latencies]
    
    return {
        "iterations": n_iterations,
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": np.percentile(latencies_ms, 95),
        "p99_ms": np.percentile(latencies_ms, 99),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    }
