"""
Unit tests for WoE (Weight of Evidence) calculator.

Tests verify WoE and IV calculations for feature engineering.

Run with: pytest tests/test_features_woe.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.woe import (
    calculate_woe_iv,
    calculate_iv_for_features,
    apply_woe_transformation
)


class TestCalculateWoEIV:
    """Tests for calculate_woe_iv function."""
    
    def test_woe_calculation_categorical(self):
        """Test WoE calculation for categorical features."""
        # Create feature with clear relationship to target
        feature = pd.Series(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'])
        target = pd.Series([1, 1, 0, 1, 0, 0, 0, 0, 0])
        
        woe_df, iv = calculate_woe_iv(feature, target)
        
        assert iv >= 0
        assert len(woe_df) > 0
        assert 'woe' in woe_df.columns
        assert 'iv' in woe_df.columns
    
    def test_woe_calculation_numerical(self):
        """Test WoE calculation for numerical features."""
        np.random.seed(42)
        feature = pd.Series(np.random.normal(100, 20, 100))
        target = pd.Series(np.random.choice([0, 1], 100))
        
        woe_df, iv = calculate_woe_iv(feature, target, bins=5)
        
        assert iv >= 0
        assert len(woe_df) > 0
        assert 'woe' in woe_df.columns
    
    def test_woe_with_perfect_separation(self):
        """Test WoE calculation with perfect feature separation."""
        # Feature that perfectly separates target
        feature = pd.Series(['A'] * 50 + ['B'] * 50)
        target = pd.Series([1] * 50 + [0] * 50)
        
        woe_df, iv = calculate_woe_iv(feature, target)
        
        # Should have high IV value
        assert iv > 0
        assert len(woe_df) == 2  # Two categories
    
    def test_woe_with_no_separation(self):
        """Test WoE calculation with no feature separation."""
        # Feature with no relationship to target
        np.random.seed(42)
        feature = pd.Series(np.random.choice(['A', 'B', 'C'], 100))
        target = pd.Series(np.random.choice([0, 1], 100))
        
        woe_df, iv = calculate_woe_iv(feature, target)
        
        # Should have low IV value
        assert iv >= 0
        assert iv < 0.5  # Low information value
    
    def test_woe_with_missing_values(self):
        """Test WoE calculation with missing values."""
        feature = pd.Series(['A', 'A', 'B', np.nan, 'B', 'C'])
        target = pd.Series([1, 0, 1, 0, 0, 1])
        
        # Missing values are dropped by default
        woe_df, iv = calculate_woe_iv(feature, target)
        
        assert iv >= 0
        assert len(woe_df) > 0
    
    def test_woe_custom_bins(self):
        """Test WoE calculation with custom bins."""
        np.random.seed(42)
        feature = pd.Series(np.random.normal(100, 20, 100))
        target = pd.Series(np.random.choice([0, 1], 100))
        
        woe_df, iv = calculate_woe_iv(feature, target, bins=10)
        
        assert iv >= 0
        assert len(woe_df) <= 10  # Should have at most 10 bins


class TestCalculateIVForFeatures:
    """Tests for calculate_iv_for_features function."""
    
    def test_iv_calculation_multiple_features(self):
        """Test IV calculation for multiple features."""
        np.random.seed(42)
        n_samples = 200
        
        df = pd.DataFrame({
            'feature1': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature2': np.random.normal(100, 20, n_samples),
            'feature3': np.random.choice([0, 1, 2], n_samples),
        })
        
        target = pd.Series(np.random.choice([0, 1], n_samples))
        
        iv_results = calculate_iv_for_features(df, target)
        
        assert len(iv_results) > 0
        assert 'feature' in iv_results.columns
        assert 'iv' in iv_results.columns
    
    def test_iv_calculation_sorted_by_iv(self):
        """Test that IV results are sorted by IV value."""
        np.random.seed(42)
        n_samples = 200
        
        df = pd.DataFrame({
            'feature1': np.random.choice(['A', 'B'], n_samples),
            'feature2': np.random.choice(['X', 'Y'], n_samples),
        })
        
        # Create target with relationship to feature1
        target = pd.Series(
            (df['feature1'] == 'A').astype(int)
        )
        
        iv_results = calculate_iv_for_features(df, target)
        
        # Results should be sorted by IV (descending)
        iv_values = iv_results['iv'].values
        assert all(iv_values[i] >= iv_values[i+1] for i in range(len(iv_values)-1))


class TestApplyWoETransformation:
    """Tests for apply_woe_transformation function."""
    
    def test_apply_woe_transformation(self):
        """Test applying WoE transformation with mapping."""
        feature = pd.Series(['A', 'B', 'C', 'A', 'B'])
        woe_mapping = {'A': 0.5, 'B': -0.3, 'C': 0.8}
        
        transformed = apply_woe_transformation(feature, woe_mapping)
        
        assert len(transformed) == len(feature)
        assert transformed.dtype in [np.float64, np.float32]
        assert transformed.iloc[0] == 0.5  # A -> 0.5
        assert transformed.iloc[1] == -0.3  # B -> -0.3
    
    def test_apply_woe_with_missing_values(self):
        """Test WoE transformation handles missing values."""
        feature = pd.Series(['A', 'B', 'C', 'D'])  # 'D' not in mapping
        woe_mapping = {'A': 0.5, 'B': -0.3, 'C': 0.8}
        
        transformed = apply_woe_transformation(feature, woe_mapping)
        
        # Missing values should be filled with 0
        assert transformed.iloc[3] == 0.0  # 'D' -> 0.0 (default)
    
    def test_apply_woe_with_numerical_features(self):
        """Test WoE transformation with numerical features."""
        feature = pd.Series([1, 2, 3, 1, 2])
        woe_mapping = {1: 0.5, 2: -0.3, 3: 0.8}
        
        transformed = apply_woe_transformation(feature, woe_mapping)
        
        assert len(transformed) == len(feature)
        assert transformed.iloc[0] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
