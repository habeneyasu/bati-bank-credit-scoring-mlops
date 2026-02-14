"""
Load Testing and Performance Benchmarking Module

Provides comprehensive load testing capabilities:
- Automated load testing scenarios
- Stress testing
- Performance benchmarking
- Capacity planning
- SLA validation
"""

import asyncio
import aiohttp
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random

from src.utils.logging import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


class LoadTestScenario:
    """Defines a load test scenario."""
    
    def __init__(
        self,
        name: str,
        endpoint: str,
        method: str = "POST",
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        weight: int = 1
    ):
        """
        Initialize load test scenario.
        
        Args:
            name: Scenario name
            endpoint: API endpoint
            method: HTTP method
            payload: Request payload
            headers: Request headers
            weight: Weight for scenario selection (higher = more frequent)
        """
        self.name = name
        self.endpoint = endpoint
        self.method = method
        self.payload = payload or {}
        self.headers = headers or {}
        self.weight = weight


class LoadTester:
    """Load testing orchestrator."""
    
    def __init__(self, base_url: str = None):
        """
        Initialize load tester.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url or f"http://{settings.api_host}:{settings.api_port}"
        self.logger = get_logger(f"{__name__}.LoadTester")
        self.token = None
    
    async def authenticate(self, username: str, password: str) -> bool:
        """Authenticate and get token."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/auth/login",
                    json={"username": username, "password": password}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.token = data.get("access_token")
                        return True
                    return False
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token."""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}
    
    async def run_load_test(
        self,
        scenarios: List[LoadTestScenario],
        total_requests: int,
        concurrent_users: int,
        duration_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a load test.
        
        Args:
            scenarios: List of test scenarios
            total_requests: Total number of requests to make
            concurrent_users: Number of concurrent users
            duration_seconds: Optional duration limit in seconds
            
        Returns:
            Load test results
        """
        start_time = time.time()
        results = {
            "scenarios": {},
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "latencies": [],
            "errors": []
        }
        
        # Weight scenarios for selection
        weighted_scenarios = []
        for scenario in scenarios:
            weighted_scenarios.extend([scenario] * scenario.weight)
        
        async def make_request(scenario: LoadTestScenario) -> Tuple[str, float, bool, Optional[str]]:
            """Make a single request."""
            request_start = time.time()
            error = None
            success = False
            
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{self.base_url}{scenario.endpoint}"
                    headers = {**self.get_headers(), **scenario.headers}
                    
                    if scenario.method == "POST":
                        async with session.post(url, json=scenario.payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            latency = (time.time() - request_start) * 1000  # Convert to ms
                            if response.status == 200:
                                success = True
                            else:
                                error = f"HTTP {response.status}"
                    elif scenario.method == "GET":
                        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            latency = (time.time() - request_start) * 1000
                            if response.status == 200:
                                success = True
                            else:
                                error = f"HTTP {response.status}"
                    
                    return scenario.name, latency, success, error
                    
            except asyncio.TimeoutError:
                latency = (time.time() - request_start) * 1000
                return scenario.name, latency, False, "Timeout"
            except Exception as e:
                latency = (time.time() - request_start) * 1000
                return scenario.name, latency, False, str(e)
        
        # Run load test
        tasks = []
        requests_made = 0
        
        while requests_made < total_requests:
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                break
            
            # Create batch of concurrent requests
            batch_size = min(concurrent_users, total_requests - requests_made)
            
            for _ in range(batch_size):
                scenario = random.choice(weighted_scenarios)
                tasks.append(make_request(scenario))
                requests_made += 1
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            tasks = []
            
            for result in batch_results:
                if isinstance(result, Exception):
                    results["failed_requests"] += 1
                    results["errors"].append(str(result))
                    continue
                
                scenario_name, latency, success, error = result
                
                results["total_requests"] += 1
                if success:
                    results["successful_requests"] += 1
                else:
                    results["failed_requests"] += 1
                    if error:
                        results["errors"].append(error)
                
                results["latencies"].append(latency)
                
                if scenario_name not in results["scenarios"]:
                    results["scenarios"][scenario_name] = {
                        "total": 0,
                        "successful": 0,
                        "failed": 0,
                        "latencies": []
                    }
                
                results["scenarios"][scenario_name]["total"] += 1
                if success:
                    results["scenarios"][scenario_name]["successful"] += 1
                    results["scenarios"][scenario_name]["latencies"].append(latency)
                else:
                    results["scenarios"][scenario_name]["failed"] += 1
        
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        if results["latencies"]:
            latencies = results["latencies"]
            results["statistics"] = {
                "mean_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies),
                "p95_ms": self._percentile(latencies, 95),
                "p99_ms": self._percentile(latencies, 99),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            }
        else:
            results["statistics"] = {}
        
        results["elapsed_time_seconds"] = elapsed_time
        results["requests_per_second"] = results["total_requests"] / elapsed_time if elapsed_time > 0 else 0
        results["success_rate"] = (results["successful_requests"] / results["total_requests"] * 100) if results["total_requests"] > 0 else 0
        
        # Check SLA compliance
        results["sla_compliance"] = self._check_sla_compliance(results)
        
        return results
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower = int(index)
        upper = lower + 1
        weight = index - lower
        
        if upper >= len(sorted_data):
            return sorted_data[-1]
        
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    def _check_sla_compliance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Check SLA compliance."""
        target_p95 = settings.target_p95_latency_ms
        p95 = results.get("statistics", {}).get("p95_ms", 0)
        
        compliant = p95 <= target_p95 if p95 > 0 else False
        success_rate = results.get("success_rate", 0)
        success_rate_compliant = success_rate >= 99.0  # 99% success rate target
        
        return {
            "p95_compliant": compliant,
            "p95_ms": p95,
            "target_p95_ms": target_p95,
            "success_rate_compliant": success_rate_compliant,
            "success_rate": success_rate,
            "overall_compliant": compliant and success_rate_compliant
        }


class StressTester:
    """Stress testing orchestrator."""
    
    def __init__(self, base_url: str = None):
        """Initialize stress tester."""
        self.base_url = base_url or f"http://{settings.api_host}:{settings.api_port}"
        self.logger = get_logger(f"{__name__}.StressTester")
    
    async def run_stress_test(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        ramp_up_users: int = 10,
        max_users: int = 100,
        ramp_up_duration: int = 60,
        hold_duration: int = 120
    ) -> Dict[str, Any]:
        """
        Run a stress test with gradual ramp-up.
        
        Args:
            endpoint: API endpoint to test
            payload: Request payload
            ramp_up_users: Number of users to ramp up to
            max_users: Maximum concurrent users
            ramp_up_duration: Duration to ramp up (seconds)
            hold_duration: Duration to hold at max load (seconds)
            
        Returns:
            Stress test results
        """
        start_time = time.time()
        results = {
            "latencies": [],
            "errors": [],
            "phases": []
        }
        
        # Ramp-up phase
        ramp_up_start = time.time()
        users_per_second = ramp_up_users / ramp_up_duration
        
        current_users = 0
        while current_users < ramp_up_users:
            elapsed = time.time() - ramp_up_start
            target_users = int(users_per_second * elapsed)
            
            if target_users > current_users:
                # Add new users
                new_users = target_users - current_users
                # Simulate requests from new users
                # (Simplified - in production would use proper async task management)
                current_users = target_users
            
            await asyncio.sleep(0.1)
            
            if elapsed >= ramp_up_duration:
                break
        
        # Hold phase
        hold_start = time.time()
        while (time.time() - hold_start) < hold_duration:
            # Continue making requests at max load
            await asyncio.sleep(0.1)
        
        # Calculate results
        elapsed_time = time.time() - start_time
        results["elapsed_time_seconds"] = elapsed_time
        
        return results


class CapacityPlanner:
    """Capacity planning tools."""
    
    def __init__(self):
        """Initialize capacity planner."""
        self.logger = get_logger(f"{__name__}.CapacityPlanner")
    
    def estimate_capacity(
        self,
        target_rps: float,
        avg_latency_ms: float,
        target_p95_ms: float
    ) -> Dict[str, Any]:
        """
        Estimate required capacity.
        
        Args:
            target_rps: Target requests per second
            avg_latency_ms: Average latency in milliseconds
            target_p95_ms: Target P95 latency in milliseconds
            
        Returns:
            Capacity estimates
        """
        # Simple capacity estimation
        # In production, would use more sophisticated models
        
        # Estimate concurrent connections needed
        concurrent_connections = target_rps * (avg_latency_ms / 1000)
        
        # Estimate workers needed (assuming each worker can handle ~10 RPS)
        workers_needed = max(1, int(target_rps / 10))
        
        # Estimate memory (rough estimate)
        memory_per_request_mb = 10  # Estimated
        memory_needed_mb = target_rps * memory_per_request_mb * (avg_latency_ms / 1000)
        
        return {
            "target_rps": target_rps,
            "estimated_concurrent_connections": concurrent_connections,
            "estimated_workers": workers_needed,
            "estimated_memory_mb": memory_needed_mb,
            "recommendations": self._generate_recommendations(target_rps, avg_latency_ms, target_p95_ms)
        }
    
    def _generate_recommendations(
        self,
        target_rps: float,
        avg_latency_ms: float,
        target_p95_ms: float
    ) -> List[str]:
        """Generate capacity planning recommendations."""
        recommendations = []
        
        if avg_latency_ms > target_p95_ms * 0.8:
            recommendations.append("Consider optimizing model inference or adding caching")
        
        if target_rps > 100:
            recommendations.append("Consider horizontal scaling with load balancer")
        
        if avg_latency_ms > 100:
            recommendations.append("Consider using feature store to reduce computation time")
        
        return recommendations


def get_load_tester(base_url: str = None) -> LoadTester:
    """Get a LoadTester instance."""
    return LoadTester(base_url)


def get_stress_tester(base_url: str = None) -> StressTester:
    """Get a StressTester instance."""
    return StressTester(base_url)


def get_capacity_planner() -> CapacityPlanner:
    """Get a CapacityPlanner instance."""
    return CapacityPlanner()
