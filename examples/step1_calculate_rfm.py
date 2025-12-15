"""
Example: Step 1 - Calculate RFM Metrics

This script demonstrates how to calculate RFM (Recency, Frequency, Monetary) metrics
for each customer from transaction history data.

This is Step 1 of the Proxy Target Variable Engineering process.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rfm_calculator import RFMCalculator, calculate_rfm_metrics


def main():
    """Main function to demonstrate RFM calculation."""
    
    print("=" * 80)
    print("Step 1: Calculate RFM Metrics")
    print("=" * 80)
    print()
    
    # Load transaction data
    data_path = project_root / "data" / "raw" / "data.csv"
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please ensure the data file exists at data/raw/data.csv")
        return
    
    print(f"Loading transaction data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} transactions")
    print(f"Date range: {df['TransactionStartTime'].min()} to {df['TransactionStartTime'].max()}")
    print()
    
    # Define snapshot date
    # Using the maximum date in the data as the snapshot date
    # This represents the "current" date for Recency calculation
    max_date = pd.to_datetime(df['TransactionStartTime']).max()
    snapshot_date = max_date.to_pydatetime()
    
    print(f"Snapshot Date: {snapshot_date.date()}")
    print("(This is the reference date for calculating Recency)")
    print()
    
    # Calculate RFM metrics
    print("Calculating RFM metrics...")
    print("-" * 100)
    
    calculator = RFMCalculator(
        customer_col='CustomerId',
        datetime_col='TransactionStartTime',
        amount_col='Amount',
        snapshot_date=snapshot_date
    )
    
    rfm_df = calculator.calculate_rfm(df)
    
    print()
    print("=" * 100)
    print("RFM Metrics Summary")
    print("=" * 100)
    print()
    
    # Display first few rows
    print("First 10 customers:")
    print(rfm_df.head(10))
    print()
    
    # Display summary statistics
    print("Summary Statistics:")
    print(calculator.get_summary_statistics())
    print()
    
    # Display distribution information
    print("RFM Metrics Distribution:")
    print(f"  Recency:   {rfm_df['recency'].min():.0f} to {rfm_df['recency'].max():.0f} days")
    print(f"  Frequency: {rfm_df['frequency'].min():.0f} to {rfm_df['frequency'].max():.0f} transactions")
    print(f"  Monetary:  ${rfm_df['monetary'].min():,.2f} to ${rfm_df['monetary'].max():,.2f}")
    print()
    
    # Save RFM metrics
    output_path = project_root / "data" / "processed" / "rfm_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rfm_df.to_csv(output_path, index=False)
    print(f"RFM metrics saved to: {output_path}")
    print()
    
    print("=" * 100)
    print("Step 1 Complete: RFM Metrics Calculated Successfully!")
    print("=" * 100)
    print()
    print("Next Steps:")
    print("  1. Review the RFM metrics to understand customer behavior patterns")
    print("  2. Proceed to Step 2: Clustering customers based on RFM metrics")
    print()


if __name__ == "__main__":
    main()

