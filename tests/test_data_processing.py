"""
Unit tests for data processing module.

Run with: pytest tests/test_data_processing.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import (
    DataProcessor,
    TemporalFeatureExtractor,
    CustomerAggregator,
    WoETransformer
)
from src.woe_calculator import calculate_woe_iv, calculate_iv_for_features


class TestTemporalFeatureExtractor:
    """Tests for TemporalFeatureExtractor."""
    
    def test_temporal_features_created(self):
        """Test that temporal features are extracted correctly."""
        dates = pd.date_range('2018-11-15', periods=5, freq='2H')
        df = pd.DataFrame({
            'TransactionStartTime': dates,
            'Amount': [100, 200, 300, 400, 500]
        })
        
        extractor = TemporalFeatureExtractor(datetime_col='TransactionStartTime')
        result = extractor.transform(df)
        
        assert 'transaction_hour' in result.columns
        assert 'transaction_day' in result.columns
        assert 'transaction_month' in result.columns
        assert 'transaction_year' in result.columns
        assert result['transaction_year'].iloc[0] == 2018
    
    def test_missing_datetime_column(self):
        """Test behavior when datetime column is missing."""
        df = pd.DataFrame({'Amount': [100, 200, 300]})
        extractor = TemporalFeatureExtractor(datetime_col='TransactionStartTime')
        result = extractor.transform(df)
        
        # Should return original dataframe unchanged
        assert 'transaction_hour' not in result.columns


class TestCustomerAggregator:
    """Tests for CustomerAggregator."""
    
    def test_aggregate_features_created(self):
        """Test that aggregate features are created correctly."""
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C1', 'C2', 'C2'],
            'Amount': [100, 200, 300, 150, 250]
        })
        
        aggregator = CustomerAggregator(customer_col='CustomerId', amount_col='Amount')
        result = aggregator.transform(df)
        
        assert 'total_transaction_amount' in result.columns
        assert 'avg_transaction_amount' in result.columns
        assert 'transaction_count' in result.columns
        assert 'std_transaction_amount' in result.columns
    
    def test_aggregation_correctness(self):
        """Test that aggregations are calculated correctly."""
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C1'],
            'Amount': [100, 200, 300]
        })
        
        aggregator = CustomerAggregator(customer_col='CustomerId', amount_col='Amount')
        result = aggregator.transform(df)
        
        c1_data = result[result['CustomerId'] == 'C1'].iloc[0]
        assert c1_data['total_transaction_amount'] == 600
        assert abs(c1_data['avg_transaction_amount'] - 200.0) < 0.01
        assert c1_data['transaction_count'] == 3


class TestDataProcessor:
    """Tests for DataProcessor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2018-11-15', periods=100, freq='1H')
        return pd.DataFrame({
            'CustomerId': np.random.choice(['C1', 'C2', 'C3'], 100),
            'TransactionStartTime': dates,
            'Amount': np.random.normal(1000, 500, 100),
            'Category': np.random.choice(['A', 'B', 'C'], 100),
            'Value': np.random.normal(1200, 600, 100)
        })
    
    def test_basic_processing(self, sample_data):
        """Test basic data processing."""
        processor = DataProcessor(
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount'
        )
        
        result = processor.process_step_by_step(sample_data)
        
        assert result.shape[0] > 0
        assert result.shape[1] > sample_data.shape[1]
        assert 'transaction_hour' in result.columns
        assert 'total_transaction_amount' in result.columns
    
    def test_onehot_encoding(self, sample_data):
        """Test one-hot encoding."""
        processor = DataProcessor(
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount',
            encoding_method='onehot'
        )
        
        result = processor.process_step_by_step(sample_data)
        
        # Check for one-hot encoded columns
        category_cols = [c for c in result.columns if 'Category' in c]
        assert len(category_cols) > 0
    
    def test_label_encoding(self, sample_data):
        """Test label encoding."""
        processor = DataProcessor(
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount',
            encoding_method='label'
        )
        
        result = processor.process_step_by_step(sample_data)
        
        # Category should be encoded as integer
        if 'Category' in result.columns:
            assert pd.api.types.is_numeric_dtype(result['Category'])
    
    def test_missing_value_imputation(self, sample_data):
        """Test missing value imputation."""
        # Add some missing values
        sample_data.loc[0:5, 'Amount'] = np.nan
        
        processor = DataProcessor(
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount',
            imputation_method='median'
        )
        
        result = processor.process_step_by_step(sample_data)
        
        # Should not have missing values in Amount
        assert result['Amount'].isna().sum() == 0
    
    def test_scaling_methods(self, sample_data):
        """Test different scaling methods."""
        for method in ['standardize', 'normalize', 'robust']:
            processor = DataProcessor(
                customer_col='CustomerId',
                datetime_col='TransactionStartTime',
                amount_col='Amount',
                scaling_method=method
            )
            
            result = processor.process_step_by_step(sample_data)
            assert result.shape[0] > 0


class TestWoECalculator:
    """Tests for WoE calculator."""
    
    def test_woe_calculation(self):
        """Test WoE and IV calculation."""
        # Create feature with clear relationship to target
        feature = pd.Series(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'])
        target = pd.Series([1, 1, 0, 1, 0, 0, 0, 0, 0])
        
        woe_df, iv = calculate_woe_iv(feature, target)
        
        assert iv >= 0
        assert len(woe_df) > 0
        assert 'woe' in woe_df.columns
        assert 'iv' in woe_df.columns
    
    def test_woe_with_numerical_feature(self):
        """Test WoE calculation with numerical feature."""
        np.random.seed(42)
        feature = pd.Series(np.random.normal(100, 20, 100))
        target = pd.Series(np.random.choice([0, 1], 100))
        
        woe_df, iv = calculate_woe_iv(feature, target, bins=5)
        
        assert iv >= 0
        assert len(woe_df) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
