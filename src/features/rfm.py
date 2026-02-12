"""
Step 1: RFM Metrics Calculator for Proxy Target Variable Engineering

This module calculates RFM (Recency, Frequency, Monetary) metrics for each customer
from transaction history data.

RFM Metrics:
- Recency: Days since last transaction (calculated from snapshot date)
- Frequency: Number of transactions per customer
- Monetary: Total transaction amount per customer

The snapshot date is defined to ensure consistent Recency calculation across all customers.
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class RFMCalculator:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics per customer.
    
    This class implements Step 1 of the Proxy Target Variable Engineering process.
    It calculates three key behavioral metrics for each customer:
    
    1. Recency: Number of days between the snapshot date and the customer's last transaction
    2. Frequency: Total number of transactions per customer
    3. Monetary: Total transaction amount per customer (sum of all transaction amounts)
    
    The snapshot date is used as a reference point to ensure consistent Recency
    calculation across all customers.
    """
    
    def __init__(
        self,
        customer_col: str = 'CustomerId',
        datetime_col: str = 'TransactionStartTime',
        amount_col: str = 'Amount',
        snapshot_date: Optional[datetime] = None
    ):
        """
        Initialize RFM Calculator.
        
        Args:
            customer_col: Name of the customer ID column (default: 'CustomerId')
            datetime_col: Name of the datetime column (default: 'TransactionStartTime')
            amount_col: Name of the transaction amount column (default: 'Amount')
            snapshot_date: Reference date for Recency calculation. If None, uses the
                          maximum date in the transaction data (default: None)
        
        Example:
            >>> calculator = RFMCalculator(
            ...     customer_col='CustomerId',
            ...     datetime_col='TransactionStartTime',
            ...     amount_col='Amount',
            ...     snapshot_date=datetime(2019, 2, 13)
            ... )
        """
        self.customer_col = customer_col
        self.datetime_col = datetime_col
        self.amount_col = amount_col
        self.snapshot_date = snapshot_date
        self.rfm_data_ = None
        self.actual_snapshot_date_ = None
    
    def _validate_input(self, df: pd.DataFrame):
        """
        Validate that required columns exist in the input DataFrame.
        
        Args:
            df: Input DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = [self.customer_col, self.datetime_col, self.amount_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )
    
    def _determine_snapshot_date(self, df: pd.DataFrame) -> datetime:
        """
        Determine the snapshot date for Recency calculation.
        
        If snapshot_date was provided during initialization, use it.
        Otherwise, use the maximum date in the transaction data.
        
        Args:
            df: Input DataFrame with transaction data
            
        Returns:
            Snapshot date as datetime object
        """
        if self.snapshot_date is not None:
            return pd.to_datetime(self.snapshot_date)
        else:
            # Use maximum date in the data as snapshot date
            max_date = pd.to_datetime(df[self.datetime_col]).max()
            return max_date
    
    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer.
        
        This is the main method that performs the RFM calculation:
        
        1. Validates input data
        2. Determines snapshot date
        3. Converts datetime column if needed
        4. Groups transactions by customer
        5. Calculates Recency, Frequency, and Monetary for each customer
        
        Args:
            df: DataFrame with transaction data. Must contain columns:
                - customer_col: Customer identifier
                - datetime_col: Transaction datetime
                - amount_col: Transaction amount
        
        Returns:
            DataFrame with RFM metrics per customer. Columns:
                - customer_col: Customer identifier
                - recency: Days since last transaction (from snapshot date)
                - frequency: Number of transactions
                - monetary: Total transaction amount
        
        Example:
            >>> df = pd.DataFrame({
            ...     'CustomerId': ['C1', 'C1', 'C2'],
            ...     'TransactionStartTime': [
            ...         datetime(2019, 2, 1),
            ...         datetime(2019, 2, 5),
            ...         datetime(2019, 2, 3)
            ...     ],
            ...     'Amount': [100, 200, 150]
            ... })
            >>> calculator = RFMCalculator(snapshot_date=datetime(2019, 2, 13))
            >>> rfm_df = calculator.calculate_rfm(df)
            >>> print(rfm_df)
               CustomerId  recency  frequency  monetary
            0         C1        8          2       300
            1         C2       10          1       150
        """
        # Validate input
        self._validate_input(df)
        
        # Make a copy to avoid modifying original data
        df = df.copy()
        
        # Convert datetime column if needed
        if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_col]):
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        
        # Determine snapshot date
        snapshot_date = self._determine_snapshot_date(df)
        self.actual_snapshot_date_ = snapshot_date
        
        print(f"Calculating RFM metrics using snapshot date: {snapshot_date.date()}")
        
        # Calculate RFM metrics per customer
        rfm_list = []
        
        for customer_id in df[self.customer_col].unique():
            # Get all transactions for this customer
            customer_transactions = df[df[self.customer_col] == customer_id]
            
            # Calculate Recency: Days since last transaction
            last_transaction_date = customer_transactions[self.datetime_col].max()
            recency = (snapshot_date - last_transaction_date).days
            
            # Calculate Frequency: Number of transactions
            frequency = len(customer_transactions)
            
            # Calculate Monetary: Total transaction amount
            # Use absolute value to handle refunds (negative amounts)
            monetary = customer_transactions[self.amount_col].abs().sum()
            
            rfm_list.append({
                self.customer_col: customer_id,
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary
            })
        
        # Create DataFrame with RFM metrics
        rfm_df = pd.DataFrame(rfm_list)
        
        # Store for later use
        self.rfm_data_ = rfm_df
        
        print(f"RFM metrics calculated for {len(rfm_df)} customers")
        print(f"Recency range: {rfm_df['recency'].min()} to {rfm_df['recency'].max()} days")
        print(f"Frequency range: {rfm_df['frequency'].min()} to {rfm_df['frequency'].max()} transactions")
        print(f"Monetary range: {rfm_df['monetary'].min():.2f} to {rfm_df['monetary'].max():.2f}")
        
        return rfm_df
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for RFM metrics.
        
        Returns:
            DataFrame with summary statistics (mean, median, std, min, max) for each RFM metric
            
        Raises:
            ValueError: If RFM metrics have not been calculated yet
        """
        if self.rfm_data_ is None:
            raise ValueError(
                "RFM metrics have not been calculated yet. "
                "Call calculate_rfm() first."
            )
        
        summary = self.rfm_data_[['recency', 'frequency', 'monetary']].describe()
        return summary
    
    def get_snapshot_date(self) -> Optional[datetime]:
        """
        Get the snapshot date used for Recency calculation.
        
        Returns:
            Snapshot date as datetime object, or None if not calculated yet
        """
        return self.actual_snapshot_date_


def calculate_rfm_metrics(
    df: pd.DataFrame,
    customer_col: str = 'CustomerId',
    datetime_col: str = 'TransactionStartTime',
    amount_col: str = 'Amount',
    snapshot_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Convenience function to calculate RFM metrics.
    
    This is a simple wrapper around RFMCalculator for quick usage.
    
    Args:
        df: DataFrame with transaction data
        customer_col: Name of customer ID column
        datetime_col: Name of datetime column
        amount_col: Name of amount column
        snapshot_date: Reference date for Recency calculation (optional)
    
    Returns:
        DataFrame with RFM metrics per customer
    
    Example:
        >>> df = pd.read_csv('data/raw/data.csv')
        >>> rfm_df = calculate_rfm_metrics(df, snapshot_date=datetime(2019, 2, 13))
        >>> print(rfm_df.head())
    """
    calculator = RFMCalculator(
        customer_col=customer_col,
        datetime_col=datetime_col,
        amount_col=amount_col,
        snapshot_date=snapshot_date
    )
    
    return calculator.calculate_rfm(df)


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 100)
    print("RFM Metrics Calculator - Step 1 of Proxy Target Variable Engineering")
    print("=" * 100)
    print()
    
    # Create sample data for demonstration
    from datetime import datetime, timedelta
    
    base_date = datetime(2019, 2, 13)
    sample_data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C1', 'C2', 'C2', 'C3'],
        'TransactionStartTime': [
            base_date - timedelta(days=30),
            base_date - timedelta(days=20),
            base_date - timedelta(days=5),   # C1's last transaction: 5 days ago
            base_date - timedelta(days=15),
            base_date - timedelta(days=10),  # C2's last transaction: 10 days ago
            base_date - timedelta(days=25),  # C3's last transaction: 25 days ago
        ],
        'Amount': [100, 200, 150, 300, 250, 50]
    })
    
    print("Sample Transaction Data:")
    print(sample_data)
    print()
    
    # Calculate RFM metrics
    calculator = RFMCalculator(
        customer_col='CustomerId',
        datetime_col='TransactionStartTime',
        amount_col='Amount',
        snapshot_date=base_date
    )
    
    rfm_df = calculator.calculate_rfm(sample_data)
    
    print("\nRFM Metrics per Customer:")
    print(rfm_df)
    print()
    
    # Show summary statistics
    print("Summary Statistics:")
    print(calculator.get_summary_statistics())
    print()
    
    print(f"Snapshot Date Used: {calculator.get_snapshot_date().date()}")
    print()
    print("=" * 100)
    print("Step 1 Complete: RFM Metrics Calculated Successfully!")
    print("=" * 100)

