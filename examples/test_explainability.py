"""
Example script to test model explainability functionality.

This script demonstrates how to:
1. Test the explainability module directly
2. Test explainability via the API endpoints
3. Visualize SHAP explanations

Run with: python examples/test_explainability.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("ERROR: SHAP library not installed. Install with: pip install shap>=0.42")
    sys.exit(1)

from src.models.explainability import ModelExplainer, create_explainer
from sklearn.ensemble import RandomForestClassifier


def test_explainability_module():
    """Test the explainability module directly."""
    print("=" * 80)
    print("Testing Model Explainability Module")
    print("=" * 80)
    print()
    
    # Create a simple model for testing
    print("1. Creating a test model...")
    np.random.seed(42)
    n_samples = 200
    n_features = 26  # Match expected features
    
    # Generate synthetic training data
    X_train = np.random.randn(n_samples, n_features)
    y_train = ((X_train[:, 0] + X_train[:, 1] - X_train[:, 2]) > 0).astype(int)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    print(f"   ✓ Model trained: {type(model).__name__}")
    print(f"   ✓ Training samples: {n_samples}")
    print(f"   ✓ Features: {n_features}")
    print()
    
    # Create feature names
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    
    # Initialize explainer
    print("2. Initializing SHAP explainer...")
    try:
        explainer = ModelExplainer(
            model=model,
            background_data=X_train[:50],  # Use subset for background
            feature_names=feature_names,
            explainer_type="auto"  # Auto-detect (should use TreeExplainer)
        )
        print("   ✓ Explainer initialized successfully")
        print(f"   ✓ Explainer type: {type(explainer.explainer).__name__}")
        print()
    except Exception as e:
        print(f"   ✗ Failed to initialize explainer: {e}")
        return False
    
    # Test explaining a single instance
    print("3. Testing explanation for a single instance...")
    test_instance = np.random.randn(1, n_features)
    
    try:
        explanation = explainer.explain_instance(test_instance)
        
        print("   ✓ Explanation generated successfully")
        print(f"   ✓ Prediction: {explanation['prediction']} ({'High Risk' if explanation['prediction'] == 1 else 'Low Risk'})")
        print(f"   ✓ Probability: {explanation['probability']:.4f}")
        print(f"   ✓ Base value: {explanation['base_value']:.4f}")
        print()
        
        # Show top 5 features
        print("   Top 5 most important features:")
        for i, feat in enumerate(explanation['feature_importance'][:5], 1):
            direction = "increases" if feat['shap_value'] > 0 else "decreases"
            print(f"   {i}. {feat['feature']}: {direction} risk by {abs(feat['shap_value']):.4f}")
        print()
        
        # Show explanation summary
        print("   Explanation Summary:")
        print(f"   {explanation['explanation_summary']}")
        print()
        
    except Exception as e:
        print(f"   ✗ Failed to generate explanation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test global feature importance
    print("4. Testing global feature importance...")
    try:
        global_importance = explainer.get_feature_importance_global(
            sample_data=X_train[:30],
            n_samples=30
        )
        
        print("   ✓ Global feature importance calculated")
        print("   Top 5 globally important features:")
        for i, feat in enumerate(global_importance['feature_importance'][:5], 1):
            print(f"   {i}. {feat['feature']}: {feat['importance']:.4f}")
        print()
        
    except Exception as e:
        print(f"   ✗ Failed to calculate global importance: {e}")
        return False
    
    # Test batch explanation
    print("5. Testing batch explanation...")
    try:
        batch_instances = np.random.randn(3, n_features)
        batch_explanations = explainer.explain_batch(batch_instances)
        
        print(f"   ✓ Batch explanation successful for {len(batch_explanations)} instances")
        print()
        
    except Exception as e:
        print(f"   ✗ Failed batch explanation: {e}")
        return False
    
    print("=" * 80)
    print("✓ All explainability module tests passed!")
    print("=" * 80)
    print()
    
    return True


def test_api_explainability():
    """Test explainability via API endpoints."""
    print("=" * 80)
    print("Testing API Explainability Endpoints")
    print("=" * 80)
    print()
    
    try:
        import requests
    except ImportError:
        print("ERROR: requests library not installed. Install with: pip install requests")
        return False
    
    # API URL (adjust if needed)
    api_url = "http://localhost:8000"
    
    # Sample feature vector (26 features)
    sample_features = [
        0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994,
        -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    
    # Test 1: Prediction with explanation
    print("1. Testing /predict endpoint with explanation...")
    try:
        response = requests.post(
            f"{api_url}/predict",
            json={
                "features": sample_features,
                "include_explanation": True
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("   ✓ Prediction with explanation successful")
            print(f"   ✓ Prediction: {data['prediction']}")
            print(f"   ✓ Probability: {data['probability']:.4f}")
            print(f"   ✓ Risk level: {data['risk_level']}")
            
            if data.get('explanation'):
                print("   ✓ Explanation included in response")
                if 'feature_importance' in data['explanation']:
                    print(f"   ✓ Feature importance: {len(data['explanation']['feature_importance'])} features")
            else:
                print("   ⚠ Explanation not included (explainer may not be initialized)")
            print()
        else:
            print(f"   ✗ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            print()
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ⚠ API server not running. Start it with: uvicorn src.api.main:app")
        print()
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print()
        return False
    
    # Test 2: Dedicated explain endpoint
    print("2. Testing /explain endpoint...")
    try:
        response = requests.post(
            f"{api_url}/explain",
            json={"features": sample_features},
            params={"include_plot": False},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("   ✓ Explanation endpoint successful")
            print(f"   ✓ Prediction: {data['prediction']}")
            print(f"   ✓ Probability: {data['probability']:.4f}")
            print(f"   ✓ Base value: {data['base_value']:.4f}")
            print(f"   ✓ Explanation summary: {data['explanation_summary'][:100]}...")
            print(f"   ✓ Feature importance: {len(data['feature_importance'])} features")
            print()
            
            # Show top 3 features
            print("   Top 3 most important features:")
            for i, feat in enumerate(data['feature_importance'][:3], 1):
                direction = "increases" if feat['shap_value'] > 0 else "decreases"
                print(f"   {i}. {feat['feature']}: {direction} risk by {abs(feat['shap_value']):.4f}")
            print()
            
        else:
            print(f"   ✗ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            print()
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ⚠ API server not running. Start it with: uvicorn src.api.main:app")
        print()
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print()
        return False
    
    # Test 3: Explain endpoint with plot
    print("3. Testing /explain endpoint with waterfall plot...")
    try:
        response = requests.post(
            f"{api_url}/explain",
            json={"features": sample_features},
            params={"include_plot": True},
            timeout=60  # Longer timeout for plot generation
        )
        
        if response.status_code == 200:
            data = response.json()
            print("   ✓ Explanation with plot successful")
            
            if data.get('waterfall_plot'):
                print("   ✓ Waterfall plot generated (base64 encoded)")
                print(f"   ✓ Plot size: {len(data['waterfall_plot'])} characters")
            else:
                print("   ⚠ Waterfall plot not generated (matplotlib may not be available)")
            print()
        else:
            print(f"   ✗ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            print()
            
    except requests.exceptions.ConnectionError:
        print("   ⚠ API server not running")
        print()
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print()
    
    print("=" * 80)
    print("✓ API explainability tests completed!")
    print("=" * 80)
    print()
    
    return True


def main():
    """Main function to run all explainability tests."""
    print()
    print("Model Explainability Testing Script")
    print("=" * 80)
    print()
    
    # Test 1: Module tests
    module_success = test_explainability_module()
    
    # Test 2: API tests
    print()
    api_success = test_api_explainability()
    
    # Summary
    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Module tests: {'✓ PASSED' if module_success else '✗ FAILED'}")
    print(f"API tests: {'✓ PASSED' if api_success else '✗ FAILED or SKIPPED (API not running)'}")
    print()
    
    if module_success:
        print("✓ Explainability module is working correctly!")
    else:
        print("✗ Some tests failed. Check the output above for details.")
    
    print()
    print("To test the API endpoints manually:")
    print("1. Start the API server: uvicorn src.api.main:app --reload")
    print("2. Visit http://localhost:8000/docs for interactive API documentation")
    print("3. Or use curl:")
    print('   curl -X POST "http://localhost:8000/explain" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"features": [0.0, -0.046, -0.072, ...]}\'')
    print()


if __name__ == "__main__":
    main()
