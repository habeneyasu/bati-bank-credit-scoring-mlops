"""
API Performance Benchmarking Script

Benchmarks API endpoints to ensure 95th percentile latency under 200ms
for real-time lending decisions.

Usage:
    python examples/benchmark_api.py
"""

import sys
import time
import statistics
import asyncio
from pathlib import Path
from typing import List, Dict
import numpy as np

try:
    import requests
    import concurrent.futures
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_sample_features(n_features: int = 26) -> List[float]:
    """Generate sample feature values for testing."""
    import random
    return [random.uniform(-2, 2) for _ in range(n_features)]


def benchmark_endpoint(
    url: str,
    method: str = "POST",
    data: Dict = None,
    n_requests: int = 100,
    concurrent: int = 10
) -> Dict:
    """
    Benchmark an API endpoint.
    
    Args:
        url: Endpoint URL
        method: HTTP method
        data: Request data
        n_requests: Number of requests
        concurrent: Number of concurrent requests
    
    Returns:
        Dictionary with benchmark results
    """
    latencies = []
    errors = 0
    
    def make_request():
        start = time.perf_counter()
        try:
            if method == "POST":
                response = requests.post(url, json=data, timeout=5)
            else:
                response = requests.get(url, timeout=5)
            
            latency = time.perf_counter() - start
            
            if response.status_code == 200:
                latencies.append(latency)
                return True
            else:
                errors += 1
                return False
        except Exception as e:
            errors += 1
            return False
    
    # Run concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [executor.submit(make_request) for _ in range(n_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    if not latencies:
        return {
            "endpoint": url,
            "n_requests": n_requests,
            "successful": 0,
            "errors": errors,
            "error": "All requests failed"
        }
    
    latencies_ms = [l * 1000 for l in latencies]
    
    return {
        "endpoint": url,
        "n_requests": n_requests,
        "successful": len(latencies),
        "errors": errors,
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p50_ms": np.percentile(latencies_ms, 50),
        "p95_ms": np.percentile(latencies_ms, 95),
        "p99_ms": np.percentile(latencies_ms, 99),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    }


def main():
    """Run API benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark API performance")
    parser.add_argument("--url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--target-p95", type=float, default=200.0, help="Target P95 latency in ms")
    
    args = parser.parse_args()
    
    base_url = args.url
    n_requests = args.requests
    concurrent = args.concurrent
    target_p95 = args.target_p95
    
    print("=" * 80)
    print("API Performance Benchmark")
    print("=" * 80)
    print(f"Base URL: {base_url}")
    print(f"Requests: {n_requests}")
    print(f"Concurrent: {concurrent}")
    print(f"Target P95: {target_p95}ms")
    print()
    
    # Test health endpoint
    print("Testing /health endpoint...")
    health_result = benchmark_endpoint(
        f"{base_url}/health",
        method="GET",
        n_requests=n_requests,
        concurrent=concurrent
    )
    print_benchmark_result(health_result, target_p95)
    print()
    
    # Test predict endpoint
    print("Testing /predict endpoint...")
    sample_features = generate_sample_features(26)
    predict_result = benchmark_endpoint(
        f"{base_url}/predict",
        method="POST",
        data={"features": sample_features, "include_explanation": False},
        n_requests=n_requests,
        concurrent=concurrent
    )
    print_benchmark_result(predict_result, target_p95)
    print()
    
    # Test predict with explanation (slower)
    print("Testing /predict with explanation...")
    predict_explain_result = benchmark_endpoint(
        f"{base_url}/predict",
        method="POST",
        data={"features": sample_features, "include_explanation": True},
        n_requests=min(n_requests, 50),  # Fewer requests for explanation
        concurrent=min(concurrent, 5)
    )
    print_benchmark_result(predict_explain_result, target_p95 * 2)  # Explanation is slower
    print()
    
    # Test explain endpoint
    print("Testing /explain endpoint...")
    explain_result = benchmark_endpoint(
        f"{base_url}/explain",
        method="POST",
        data={"features": sample_features},
        n_requests=min(n_requests, 50),
        concurrent=min(concurrent, 5)
    )
    print_benchmark_result(explain_result, target_p95 * 2)
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    endpoints = [
        ("/health", health_result),
        ("/predict", predict_result),
        ("/predict (with explanation)", predict_explain_result),
        ("/explain", explain_result)
    ]
    
    all_compliant = True
    for name, result in endpoints:
        if "p95_ms" in result:
            compliant = result["p95_ms"] <= target_p95
            status = "✓ PASS" if compliant else "✗ FAIL"
            print(f"{name:30} P95: {result['p95_ms']:7.2f}ms {status}")
            if not compliant:
                all_compliant = False
        else:
            print(f"{name:30} ERROR: {result.get('error', 'Unknown error')}")
            all_compliant = False
    
    print()
    if all_compliant:
        print("✓ All endpoints meet SLA requirements!")
    else:
        print("✗ Some endpoints do not meet SLA requirements")
        print("  Consider: caching, query optimization, or model optimization")
    
    print()
    print("=" * 80)


def print_benchmark_result(result: Dict, target_p95: float):
    """Print benchmark result in readable format."""
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    
    print(f"  Successful: {result['successful']}/{result['n_requests']}")
    if result['errors'] > 0:
        print(f"  Errors: {result['errors']}")
    
    if "p95_ms" in result:
        print(f"  Mean:    {result['mean_ms']:7.2f}ms")
        print(f"  Median:  {result['median_ms']:7.2f}ms")
        print(f"  P50:     {result['p50_ms']:7.2f}ms")
        print(f"  P95:     {result['p95_ms']:7.2f}ms {'✓' if result['p95_ms'] <= target_p95 else '✗'}")
        print(f"  P99:     {result['p99_ms']:7.2f}ms")
        print(f"  Min:     {result['min_ms']:7.2f}ms")
        print(f"  Max:     {result['max_ms']:7.2f}ms")
        print(f"  Std Dev: {result['std_ms']:7.2f}ms")


if __name__ == "__main__":
    main()
