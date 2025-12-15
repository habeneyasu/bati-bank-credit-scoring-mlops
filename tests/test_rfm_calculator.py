"""
Unit tests for RFM Calculator (Step 1 of Proxy Target Variable Engineering).

Run with: pytest tests/test_rfm_calculator.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rfm_calculator import RFMCalculator, calculate_rfm_metrics


class TestRFMCalculator:
    """Tests for RFMCalculator class."""
    
    def test_basic_rfm_calculation(self):
        """Test basic RFM calculation with simple data."""
        base_date = datetime(2019, 2, 13)
        
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2'],
            'TransactionStartTime': [
                base_date - timedelta(days=10),
                base_date - timedelta(days=5),   # C1's last transaction: 5 days ago
                base_date - timedelta(days=20),   # C2's last transaction: 20 days ago
            ],
            'Amount': [100, 200, 150]
        })
        
        calculator = RFMCalculator(
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount',
            snapshot_date=base_date
        )
        
        rfm_df = calculator.calculate_rfm(df)
        
        # Check structure
        assert 'CustomerId' in rfm_df.columns
        assert 'recency' in rfm_df.columns
        assert 'frequency' in rfm_df.columns
        assert 'monetary' in rfm_df.columns
        assert len(rfm_df) == 2  # Two customers
        
        # Check Customer 1 values
        c1_data = rfm_df[rfm_df['CustomerId'] == 'C1'].iloc[0]
        assert c1_data['recency'] == 5  # 5 days since last transaction
        assert c1_data['frequency'] == 2  # 2 transactions
        assert c1_data['monetary'] == 300  # 100 + 200
        
        # Check Customer 2 values
        c2_data = rfm_df[rfm_df['CustomerId'] == 'C2'].iloc[0]
        assert c2_data['recency'] == 20  # 20 days since last transaction
        assert c2_data['frequency'] == 1  # 1 transaction
        assert c2_data['monetary'] == 150  # 150
    
    def test_rfm_with_negative_amounts(self):
        """Test RFM calculation handles negative amounts (refunds)."""
        base_date = datetime(2019, 2, 13)
        
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1'],
            'TransactionStartTime': [
                base_date - timedelta(days=5),
                base_date - timedelta(days=3),
            ],
            'Amount': [100, -50]  # One refund
        })
        
        calculator = RFMCalculator(
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount',
            snapshot_date=base_date
        )
        
        rfm_df = calculator.calculate_rfm(df)
        
        c1_data = rfm_df[rfm_df['CustomerId'] == 'C1'].iloc[0]
        # Monetary should use absolute value
        assert c1_data['monetary'] == 150  # abs(100) + abs(-50)
        assert c1_data['frequency'] == 2
        assert c1_data['recency'] == 3  # 3 days since last transaction
    
    def test_rfm_with_auto_snapshot_date(self):
        """Test RFM calculation when snapshot_date is not provided (uses max date)."""
        base_date = datetime(2019, 2, 13)
        
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2'],
            'TransactionStartTime': [
                base_date - timedelta(days=10),
                base_date - timedelta(days=5),   # This is the max date
                base_date - timedelta(days=20),
            ],
            'Amount': [100, 200, 150]
        })
        
        calculator = RFMCalculator(
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount'
            # snapshot_date not provided - should use max date
        )
        
        rfm_df = calculator.calculate_rfm(df)
        
        # When snapshot_date is max date, recency for C1 should be 0
        c1_data = rfm_df[rfm_df['CustomerId'] == 'C1'].iloc[0]
        assert c1_data['recency'] == 0  # Last transaction is on snapshot date
        
        # Verify snapshot date was set to max date
        assert calculator.get_snapshot_date() == base_date - timedelta(days=5)
    
    def test_rfm_multiple_customers(self):
        """Test RFM calculation with multiple customers."""
        base_date = datetime(2019, 2, 13)
        n_customers = 10
        n_transactions_per_customer = 5
        
        data = []
        for i in range(n_customers):
            for j in range(n_transactions_per_customer):
                data.append({
                    'CustomerId': f'C{i}',
                    'TransactionStartTime': base_date - timedelta(days=j*2),
                    'Amount': 100 * (j + 1)
                })
        
        df = pd.DataFrame(data)
        
        calculator = RFMCalculator(
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount',
            snapshot_date=base_date
        )
        
        rfm_df = calculator.calculate_rfm(df)
        
        assert len(rfm_df) == n_customers
        assert all(rfm_df['frequency'] == n_transactions_per_customer)
        # All customers should have recency = 0 (last transaction on snapshot date)
        assert all(rfm_df['recency'] == 0)
        
        # Check monetary values (sum of 100, 200, 300, 400, 500 = 1500)
        assert all(rfm_df['monetary'] == 1500)
    
    def test_rfm_missing_columns(self):
        """Test that missing columns raise appropriate errors."""
        df = pd.DataFrame({
            'CustomerId': ['C1'],
            'Amount': [100]
            # Missing TransactionStartTime
        })
        
        calculator = RFMCalculator(
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount'
        )
        
        with pytest.raises(ValueError, match="Missing required columns"):
            calculator.calculate_rfm(df)
    
    def test_rfm_string_datetime(self):
        """Test RFM calculation with string datetime (should be converted)."""
        base_date = datetime(2019, 2, 13)
        
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1'],
            'TransactionStartTime': [
                (base_date - timedelta(days=5)).strftime('%Y-%m-%d'),
                (base_date - timedelta(days=3)).strftime('%Y-%m-%d'),
            ],
            'Amount': [100, 200]
        })
        
        calculator = RFMCalculator(
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount',
            snapshot_date=base_date
        )
        
        rfm_df = calculator.calculate_rfm(df)
        
        # Should work with string dates
        assert len(rfm_df) == 1
        c1_data = rfm_df.iloc[0]
        assert c1_data['recency'] == 3
    
    def test_get_summary_statistics(self):
        """Test summary statistics generation."""
        base_date = datetime(2019, 2, 13)
        
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C3'],
            'TransactionStartTime': [
                base_date - timedelta(days=5),
                base_date - timedelta(days=3),
                base_date - timedelta(days=10),
                base_date - timedelta(days=15),
            ],
            'Amount': [100, 200, 300, 400]
        })
        
        calculator = RFMCalculator(
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount',
            snapshot_date=base_date
        )
        
        calculator.calculate_rfm(df)
        summary = calculator.get_summary_statistics()
        
        # Check structure
        assert 'recency' in summary.columns
        assert 'frequency' in summary.columns
        assert 'monetary' in summary.columns
        assert 'mean' in summary.index
        assert 'std' in summary.index
    
    def test_get_summary_statistics_before_calculation(self):
        """Test that summary statistics raise error if not calculated yet."""
        calculator = RFMCalculator()
        
        with pytest.raises(ValueError, match="RFM metrics have not been calculated"):
            calculator.get_summary_statistics()
    
    def test_custom_column_names(self):
        """Test RFM calculation with custom column names."""
        base_date = datetime(2019, 2, 13)
        
        df = pd.DataFrame({
            'cust_id': ['C1', 'C1'],
            'txn_date': [
                base_date - timedelta(days=5),
                base_date - timedelta(days=3),
            ],
            'txn_amount': [100, 200]
        })
        
        calculator = RFMCalculator(
            customer_col='cust_id',
            datetime_col='txn_date',
            amount_col='txn_amount',
            snapshot_date=base_date
        )
        
        rfm_df = calculator.calculate_rfm(df)
        
        assert 'cust_id' in rfm_df.columns
        assert 'recency' in rfm_df.columns
        assert len(rfm_df) == 1


class TestCalculateRFMMetricsFunction:
    """Tests for convenience function calculate_rfm_metrics."""
    
    def test_convenience_function(self):
        """Test the convenience function works correctly."""
        base_date = datetime(2019, 2, 13)
        
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1'],
            'TransactionStartTime': [
                base_date - timedelta(days=5),
                base_date - timedelta(days=3),
            ],
            'Amount': [100, 200]
        })
        
        rfm_df = calculate_rfm_metrics(
            df,
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount',
            snapshot_date=base_date
        )
        
        assert 'recency' in rfm_df.columns
        assert 'frequency' in rfm_df.columns
        assert 'monetary' in rfm_df.columns
        assert len(rfm_df) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

