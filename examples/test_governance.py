"""
Example script to test governance and fairness analysis.

This script demonstrates how to:
1. Load a trained model
2. Perform fairness analysis
3. Generate fairness reports
4. Test API endpoints

Usage:
    python examples/test_governance.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.models.fairness import FairnessAnalyzer, analyze_model_fairness
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_sample_data(n_samples=1000, n_features=5, random_state=42):
    """Create sample data for testing."""
    np.random.seed(random_state)
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create groups (simulate customer segments based on feature values)
    # Group 0: Low spending, Group 1: Medium spending, Group 2: High spending
    spending_level = X[:, 0] + X[:, 1]
    groups = np.zeros(n_samples, dtype=int)
    groups[spending_level < -0.5] = 0  # Low
    groups[(spending_level >= -0.5) & (spending_level < 0.5)] = 1  # Medium
    groups[spending_level >= 0.5] = 2  # High
    
    # Create labels with some correlation to features and groups
    y = ((X[:, 0] + X[:, 1] + 0.1 * groups) > 0).astype(int)
    
    return X, y, groups


def test_fairness_analyzer():
    """Test FairnessAnalyzer class."""
    print("\n" + "="*80)
    print("Testing FairnessAnalyzer")
    print("="*80)
    
    # Create sample data
    X, y, groups = create_sample_data()
    
    # Split data
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.3, random_state=42
    )
    
    # Train a simple model
    print("\nTraining model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Initialize analyzer
    analyzer = FairnessAnalyzer(
        threshold_demographic_parity=0.80,
        threshold_equalized_odds=0.75,
        threshold_calibration=0.85,
        threshold_disparate_impact=0.80
    )
    
    # Test individual metrics
    print("\n1. Testing Demographic Parity...")
    dp_result = analyzer.demographic_parity(y_pred, groups_test)
    print(f"   Value: {dp_result['value']:.4f}")
    print(f"   Threshold: {dp_result['threshold']:.4f}")
    print(f"   Status: {dp_result['status']}")
    print(f"   Group Rates: {dp_result['group_rates']}")
    
    print("\n2. Testing Equalized Odds...")
    eo_result = analyzer.equalized_odds(y_test, y_pred, groups_test)
    print(f"   Value: {eo_result['value']:.4f}")
    print(f"   Threshold: {eo_result['threshold']:.4f}")
    print(f"   Status: {eo_result['status']}")
    
    print("\n3. Testing Calibration...")
    cal_result = analyzer.calibration(y_test, y_pred_proba, groups_test)
    print(f"   Value: {cal_result['value']:.4f}")
    print(f"   Threshold: {cal_result['threshold']:.4f}")
    print(f"   Status: {cal_result['status']}")
    
    print("\n4. Testing Disparate Impact Ratio...")
    di_result = analyzer.disparate_impact_ratio(y_pred, groups_test)
    print(f"   Value: {di_result['value']:.4f}")
    print(f"   Threshold: {di_result['threshold']:.4f}")
    print(f"   Status: {di_result['status']}")
    print(f"   Protected Group: {di_result['protected_group']}")
    
    # Comprehensive analysis
    print("\n5. Comprehensive Fairness Analysis...")
    comprehensive = analyzer.comprehensive_analysis(
        y_test, y_pred, y_pred_proba, groups_test
    )
    
    print(f"\n   Overall Status: {comprehensive['overall_status']}")
    print(f"   Summary:")
    print(f"     - Total Metrics: {comprehensive['summary']['total_metrics']}")
    print(f"     - Compliant Metrics: {comprehensive['summary']['compliant_metrics']}")
    print(f"     - Non-Compliant Metrics: {comprehensive['summary']['non_compliant_metrics']}")
    
    return comprehensive


def test_analyze_model_fairness():
    """Test analyze_model_fairness convenience function."""
    print("\n" + "="*80)
    print("Testing analyze_model_fairness convenience function")
    print("="*80)
    
    # Create sample data
    X, y, groups = create_sample_data()
    
    # Split data
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.3, random_state=42
    )
    
    # Train model
    print("\nTraining model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Analyze fairness
    print("\nAnalyzing model fairness...")
    results = analyze_model_fairness(
        model,
        X_test,
        y_test,
        groups_test
    )
    
    print(f"\nOverall Status: {results['overall_status']}")
    print("\nMetric Details:")
    for metric_name, metric_data in results.items():
        if isinstance(metric_data, dict) and "value" in metric_data:
            print(f"  {metric_name}:")
            print(f"    Value: {metric_data['value']:.4f}")
            print(f"    Threshold: {metric_data['threshold']:.4f}")
            print(f"    Status: {metric_data['status']}")
    
    return results


def test_api_endpoint():
    """Test fairness API endpoint."""
    print("\n" + "="*80)
    print("Testing Fairness API Endpoint")
    print("="*80)
    
    try:
        import requests
        
        # Try to connect to API
        base_url = "http://localhost:8001"  # Adjust port if needed
        
        print(f"\nTesting API at {base_url}/api/fairness...")
        
        try:
            response = requests.get(f"{base_url}/api/fairness", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print("\n✅ API Response Received:")
                print(f"   Overall Status: {data.get('overall_status', 'N/A')}")
                
                metrics = ["demographic_parity", "equalized_odds", "calibration", "disparate_impact"]
                for metric in metrics:
                    if metric in data:
                        print(f"\n   {metric.replace('_', ' ').title()}:")
                        print(f"     Value: {data[metric].get('value', 'N/A')}")
                        print(f"     Status: {data[metric].get('status', 'N/A')}")
                
            elif response.status_code == 503:
                print("\n⚠️  Model not loaded. Start the API server first:")
                print("   uvicorn src.api.main:app --reload --port 8001")
            else:
                print(f"\n❌ API returned status code: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("\n❌ Could not connect to API. Is the server running?")
            print("   Start with: uvicorn src.api.main:app --reload --port 8001")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            
    except ImportError:
        print("\n⚠️  requests library not installed. Install with: pip install requests")


def main():
    """Run all governance tests."""
    print("\n" + "="*80)
    print("Governance and Fairness Analysis Testing")
    print("="*80)
    
    try:
        # Test 1: FairnessAnalyzer
        comprehensive_results = test_fairness_analyzer()
        
        # Test 2: Convenience function
        model_results = test_analyze_model_fairness()
        
        # Test 3: API endpoint
        test_api_endpoint()
        
        print("\n" + "="*80)
        print("✅ All tests completed!")
        print("="*80)
        print("\nNext steps:")
        print("1. Review the fairness metrics above")
        print("2. Start the API server: uvicorn src.api.main:app --reload --port 8001")
        print("3. Open the dashboard: http://localhost:5173")
        print("4. Click 'Governance' button to view interactive fairness analysis")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
