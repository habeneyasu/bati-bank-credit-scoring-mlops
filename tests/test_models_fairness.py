"""
Tests for fairness analysis module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.models.fairness import FairnessAnalyzer, analyze_model_fairness


@pytest.fixture
def sample_data():
    """Create sample data for fairness testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    X = np.random.randn(n_samples, 5)
    
    # Create groups (simulate customer segments)
    groups = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])
    
    # Create labels with some bias
    y = ((X[:, 0] + X[:, 1] + 0.1 * groups) > 0).astype(int)
    
    return X, y, groups


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for testing."""
    X, y, _ = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def analyzer():
    """Create a fairness analyzer instance."""
    return FairnessAnalyzer(
        threshold_demographic_parity=0.80,
        threshold_equalized_odds=0.75,
        threshold_calibration=0.85,
        threshold_disparate_impact=0.80
    )


class TestFairnessAnalyzer:
    """Test suite for FairnessAnalyzer class."""
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.threshold_demographic_parity == 0.80
        assert analyzer.threshold_equalized_odds == 0.75
        assert analyzer.threshold_calibration == 0.85
        assert analyzer.threshold_disparate_impact == 0.80
    
    def test_demographic_parity(self, analyzer, sample_data):
        """Test demographic parity calculation."""
        _, y, groups = sample_data
        y_pred = np.random.choice([0, 1], size=len(y), p=[0.7, 0.3])
        
        result = analyzer.demographic_parity(y_pred, groups)
        
        assert "metric" in result
        assert result["metric"] == "demographic_parity"
        assert "value" in result
        assert "threshold" in result
        assert "status" in result
        assert "group_rates" in result
        assert 0.0 <= result["value"] <= 1.0
        assert result["threshold"] == 0.80
    
    def test_demographic_parity_insufficient_groups(self, analyzer):
        """Test demographic parity with insufficient groups."""
        y_pred = np.array([0, 1, 0, 1])
        groups = np.array([0, 0, 0, 0])  # Only one group
        
        result = analyzer.demographic_parity(y_pred, groups)
        
        assert result["status"] == "insufficient_data"
        assert result["value"] == 1.0
    
    def test_equalized_odds(self, analyzer, sample_data):
        """Test equalized odds calculation."""
        _, y_true, groups = sample_data
        y_pred = np.random.choice([0, 1], size=len(y_true), p=[0.7, 0.3])
        
        result = analyzer.equalized_odds(y_true, y_pred, groups)
        
        assert "metric" in result
        assert result["metric"] == "equalized_odds"
        assert "value" in result
        assert "threshold" in result
        assert "status" in result
        assert "group_metrics" in result
        assert 0.0 <= result["value"] <= 1.0
    
    def test_calibration(self, analyzer, sample_data):
        """Test calibration calculation."""
        _, y_true, groups = sample_data
        y_pred_proba = np.random.rand(len(y_true))
        
        result = analyzer.calibration(y_true, y_pred_proba, groups)
        
        assert "metric" in result
        assert result["metric"] == "calibration"
        assert "value" in result
        assert "threshold" in result
        assert "status" in result
        assert "group_calibration" in result
        assert 0.0 <= result["value"] <= 1.0
    
    def test_disparate_impact_ratio(self, analyzer, sample_data):
        """Test disparate impact ratio calculation."""
        _, _, groups = sample_data
        y_pred = np.random.choice([0, 1], size=len(groups), p=[0.7, 0.3])
        
        result = analyzer.disparate_impact_ratio(y_pred, groups)
        
        assert "metric" in result
        assert result["metric"] == "disparate_impact"
        assert "value" in result
        assert "threshold" in result
        assert "status" in result
        assert "protected_group" in result
        assert "group_rates" in result
        assert 0.0 <= result["value"] <= 1.0
    
    def test_comprehensive_analysis(self, analyzer, sample_data, trained_model):
        """Test comprehensive fairness analysis."""
        X, y_true, groups = sample_data
        
        y_pred = trained_model.predict(X)
        y_pred_proba = trained_model.predict_proba(X)[:, 1]
        
        result = analyzer.comprehensive_analysis(y_true, y_pred, y_pred_proba, groups)
        
        assert "demographic_parity" in result
        assert "equalized_odds" in result
        assert "calibration" in result
        assert "disparate_impact" in result
        assert "overall_status" in result
        assert "summary" in result
        
        assert result["overall_status"] in ["compliant", "non_compliant"]
        assert "total_metrics" in result["summary"]
        assert "compliant_metrics" in result["summary"]
        assert "non_compliant_metrics" in result["summary"]
    
    def test_comprehensive_analysis_compliant(self, analyzer):
        """Test comprehensive analysis with compliant metrics."""
        np.random.seed(42)
        n = 200
        
        # Create fair predictions
        y_true = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
        y_pred = y_true.copy()  # Perfect predictions
        y_pred_proba = y_true.astype(float) + np.random.normal(0, 0.1, n)
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
        groups = np.random.choice([0, 1], size=n)
        
        result = analyzer.comprehensive_analysis(y_true, y_pred, y_pred_proba, groups)
        
        # With perfect predictions, should be compliant
        assert result["overall_status"] in ["compliant", "non_compliant"]


class TestAnalyzeModelFairness:
    """Test suite for analyze_model_fairness convenience function."""
    
    def test_analyze_model_fairness(self, sample_data, trained_model):
        """Test analyze_model_fairness function."""
        X, y, groups = sample_data
        
        # Split for testing
        X_test = X[:200]
        y_test = y[:200]
        groups_test = groups[:200]
        
        result = analyze_model_fairness(
            trained_model,
            X_test,
            y_test,
            groups_test
        )
        
        assert "demographic_parity" in result
        assert "equalized_odds" in result
        assert "calibration" in result
        assert "disparate_impact" in result
        assert "overall_status" in result
    
    def test_analyze_model_fairness_custom_thresholds(self, sample_data, trained_model):
        """Test analyze_model_fairness with custom thresholds."""
        X, y, groups = sample_data
        
        X_test = X[:200]
        y_test = y[:200]
        groups_test = groups[:200]
        
        custom_thresholds = {
            "threshold_demographic_parity": 0.90,
            "threshold_equalized_odds": 0.80,
        }
        
        result = analyze_model_fairness(
            trained_model,
            X_test,
            y_test,
            groups_test,
            thresholds=custom_thresholds
        )
        
        assert result["demographic_parity"]["threshold"] == 0.90
        assert result["equalized_odds"]["threshold"] == 0.80


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_groups(self, analyzer):
        """Test with empty groups array."""
        y_pred = np.array([0, 1, 0])
        groups = np.array([])
        
        # Should handle gracefully (returns insufficient_data status)
        result = analyzer.demographic_parity(y_pred, groups)
        assert result["status"] == "insufficient_data"
    
    def test_single_group(self, analyzer):
        """Test with single group."""
        y_pred = np.array([0, 1, 0, 1])
        groups = np.array([0, 0, 0, 0])
        
        result = analyzer.demographic_parity(y_pred, groups)
        assert result["status"] == "insufficient_data"
    
    def test_all_zeros_predictions(self, analyzer, sample_data):
        """Test with all zero predictions."""
        _, _, groups = sample_data
        y_pred = np.zeros(len(groups))
        
        result = analyzer.demographic_parity(y_pred, groups)
        assert "value" in result
        assert result["value"] >= 0.0
    
    def test_all_ones_predictions(self, analyzer, sample_data):
        """Test with all one predictions."""
        _, _, groups = sample_data
        y_pred = np.ones(len(groups))
        
        result = analyzer.demographic_parity(y_pred, groups)
        assert "value" in result
        assert result["value"] >= 0.0
