"""
Locust Load Testing Configuration

This file defines load testing scenarios for the Credit Scoring API.
Run with: locust -f tests/load_testing/locustfile.py --host=http://localhost:8001
"""

import random
import time
from locust import HttpUser, task, between, events
from typing import Dict, Any
import json


class CreditScoringUser(HttpUser):
    """Simulates a user making credit scoring requests."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Login to get token
        try:
            response = self.client.post(
                "/api/auth/login",
                json={
                    "username": "test_user",
                    "password": "test_password"
                },
                catch_response=True
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                response.success()
            else:
                self.token = None
                response.failure(f"Login failed: {response.status_code}")
        except Exception as e:
            self.token = None
            print(f"Login error: {e}")
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token."""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}
    
    @task(3)
    def predict_credit_risk(self):
        """Test prediction endpoint (most common operation)."""
        features = self.generate_sample_features()
        
        response = self.client.post(
            "/predict",
            json={
                "features": features,
                "include_explanation": False
            },
            headers=self.get_headers(),
            catch_response=True,
            name="predict"
        )
        
        if response.status_code == 200:
            data = response.json()
            if "prediction" in data and "probability" in data:
                response.success()
            else:
                response.failure("Invalid response format")
        else:
            response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def predict_with_explanation(self):
        """Test prediction with explanation (slower operation)."""
        features = self.generate_sample_features()
        
        response = self.client.post(
            "/predict",
            json={
                "features": features,
                "include_explanation": True
            },
            headers=self.get_headers(),
            catch_response=True,
            name="predict_with_explanation"
        )
        
        if response.status_code == 200:
            response.success()
        else:
            response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def score_customer(self):
        """Test customer scoring endpoint."""
        if not self.token:
            return
        
        customer_id = f"customer_{random.randint(1000, 9999)}"
        
        response = self.client.post(
            "/api/customers/score",
            json={
                "customer_id": customer_id
            },
            headers=self.get_headers(),
            catch_response=True,
            name="score_customer"
        )
        
        if response.status_code in [200, 404]:  # 404 is acceptable if customer doesn't exist
            response.success()
        else:
            response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_health(self):
        """Test health check endpoint."""
        response = self.client.get(
            "/health",
            catch_response=True,
            name="health"
        )
        
        if response.status_code == 200:
            response.success()
        else:
            response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_performance_metrics(self):
        """Test performance metrics endpoint."""
        if not self.token:
            return
        
        response = self.client.get(
            "/api/performance",
            headers=self.get_headers(),
            catch_response=True,
            name="performance_metrics"
        )
        
        if response.status_code == 200:
            response.success()
        else:
            response.failure(f"Status code: {response.status_code}")
    
    def generate_sample_features(self, n_features: int = 26) -> list:
        """Generate random sample features for testing."""
        return [random.uniform(-2, 2) for _ in range(n_features)]


class StressTestUser(HttpUser):
    """Stress test user with aggressive load."""
    
    wait_time = between(0.1, 0.5)  # Very short wait time
    
    def on_start(self):
        """Login to get token."""
        try:
            response = self.client.post(
                "/api/auth/login",
                json={
                    "username": "test_user",
                    "password": "test_password"
                }
            )
            if response.status_code == 200:
                self.token = response.json().get("access_token")
            else:
                self.token = None
        except:
            self.token = None
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token."""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}
    
    @task(10)
    def rapid_predictions(self):
        """Rapid fire predictions for stress testing."""
        features = [random.uniform(-2, 2) for _ in range(26)]
        
        self.client.post(
            "/predict",
            json={"features": features, "include_explanation": False},
            headers=self.get_headers(),
            name="stress_predict"
        )


# Custom event handlers for reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 80)
    print("Load Test Started")
    print("=" * 80)
    print(f"Target host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("=" * 80)
    print("Load Test Completed")
    print("=" * 80)
    
    # Print summary statistics
    stats = environment.stats
    if stats:
        print("\nRequest Statistics:")
        print("-" * 80)
        for name, stat in stats.entries.items():
            if stat.num_requests > 0:
                print(f"{name}:")
                print(f"  Total Requests: {stat.num_requests}")
                print(f"  Failures: {stat.num_failures}")
                print(f"  Median Response Time: {stat.median_response_time}ms")
                print(f"  P95 Response Time: {stat.get_response_time_percentile(0.95)}ms")
                print(f"  P99 Response Time: {stat.get_response_time_percentile(0.99)}ms")
                print(f"  RPS: {stat.total_rps:.2f}")
                print()
