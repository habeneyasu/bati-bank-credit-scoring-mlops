"""
Test script for Credit Scoring API

This script tests the FastAPI endpoints to ensure they work correctly.
"""

import requests
import json
from pathlib import Path

# API base URL
API_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Health check passed: {data}")
    return data


def test_predict():
    """Test the /predict endpoint."""
    print("\nTesting /predict endpoint...")
    
    # Sample feature vector (26 features matching processed data)
    # This is a sample - in production, you'd use actual processed features
    sample_features = [
        0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994,
        -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    
    payload = {
        "features": sample_features
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Prediction successful: {data}")
    
    # Validate response structure
    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1
    assert data["risk_level"] in ["low", "high"]
    
    return data


def test_invalid_request():
    """Test error handling for invalid requests."""
    print("\nTesting error handling...")
    
    # Test with wrong number of features
    payload = {
        "features": [1.0, 2.0]  # Wrong number of features
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=payload
    )
    
    assert response.status_code == 400
    print(f"✓ Error handling works: {response.json()}")


def main():
    """Run all API tests."""
    print("=" * 100)
    print("Testing Credit Scoring API")
    print("=" * 100)
    print()
    
    try:
        # Test health check
        health_data = test_health_check()
        
        if not health_data.get("model_loaded"):
            print("\n⚠ Warning: Model not loaded. Predictions will fail.")
            print("Ensure MLflow model registry is accessible.")
            return
        
        # Test prediction
        prediction_data = test_predict()
        
        # Test error handling
        test_invalid_request()
        
        print("\n" + "=" * 100)
        print("All API tests passed!")
        print("=" * 100)
        
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Error: Could not connect to API at {API_URL}")
        print("Make sure the API is running:")
        print("  docker-compose up")
        print("  or")
        print("  uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()

