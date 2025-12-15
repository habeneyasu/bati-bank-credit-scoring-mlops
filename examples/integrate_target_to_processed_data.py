"""
Integrate Target Variable into Processed Dataset

This script merges the is_high_risk target variable into the main processed dataset
that contains all the engineered features from Task 3.

This creates the final dataset ready for model training.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main function to integrate target variable into processed data."""
    
    print("=" * 100)
    print("Integrate Target Variable into Processed Dataset")
    print("=" * 100)
    print()
    
    # Load processed data (feature-engineered from Task 3)
    processed_path = project_root / "data" / "processed" / "processed_data.csv"
    
    if not processed_path.exists():
        print(f"Processed data not found: {processed_path}")
        print("Please run feature engineering first (Task 3).")
        return
    
    print(f"Loading processed data from: {processed_path}")
    processed_df = pd.read_csv(processed_path)
    print(f"Loaded {len(processed_df):,} rows with {len(processed_df.columns)} features")
    print()
    
    # Load transactions with target (from Step 3)
    transactions_with_target_path = project_root / "data" / "processed" / "transactions_with_target.csv"
    
    if not transactions_with_target_path.exists():
        print(f"Transactions with target not found: {transactions_with_target_path}")
        print("Please run Step 3 first to create the target variable.")
        print("Run: python examples/step3_create_high_risk_target.py")
        return
    
    print(f"Loading transactions with target from: {transactions_with_target_path}")
    transactions_with_target = pd.read_csv(transactions_with_target_path)
    print(f"Loaded {len(transactions_with_target):,} transactions")
    print()
    
    # Check if target already exists in processed data
    if 'is_high_risk' in processed_df.columns:
        print("Warning: 'is_high_risk' column already exists in processed data.")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Aborting. Target variable not updated.")
            return
        processed_df = processed_df.drop(columns=['is_high_risk'])
    
    # Extract target mapping from transactions
    # We need to match on TransactionId or use CustomerId
    # Since processed_data.csv doesn't have TransactionId, we'll need to check what columns it has
    
    print("Checking available columns for merging...")
    print(f"Processed data columns: {list(processed_df.columns)[:10]}...")
    print(f"Transactions columns: {list(transactions_with_target.columns)[:10]}...")
    print()
    
    # Since processed_data.csv doesn't have TransactionId or CustomerId,
    # we need to match by row index (assuming same order as raw data)
    # Check if row counts match
    if len(processed_df) == len(transactions_with_target):
        print("Matching by row index (same number of rows, assuming same order)...")
        # Match by index position - processed data should be in same order as raw data
        processed_df['is_high_risk'] = transactions_with_target['is_high_risk'].values
        
        print("✓ Target variable added by row index matching")
    else:
        print(f"Warning: Row count mismatch!")
        print(f"  Processed data: {len(processed_df):,} rows")
        print(f"  Transactions with target: {len(transactions_with_target):,} rows")
        print()
        print("This suggests the processed data may have been filtered or transformed.")
        print("Options:")
        print("  1. Reprocess the data with target variable included")
        print("  2. Use transactions_with_target.csv directly for training")
        print()
        
        # Try to match by TransactionId if it exists in processed data
        if 'TransactionId' in processed_df.columns:
            print("Attempting merge by TransactionId...")
            target_mapping = transactions_with_target[['TransactionId', 'is_high_risk']].copy()
            processed_df = processed_df.merge(
                target_mapping,
                on='TransactionId',
                how='left'
            )
        else:
            print("Error: Cannot reliably match processed data with target variable.")
            print("Recommendation: Reprocess data with target variable included.")
            return
    
    # Check for missing values
    if processed_df['is_high_risk'].isna().any():
        missing_count = processed_df['is_high_risk'].isna().sum()
        print(f"Warning: {missing_count} rows have missing target values.")
        print("Filling missing values with 0 (low-risk)...")
        processed_df['is_high_risk'] = processed_df['is_high_risk'].fillna(0).astype(int)
    
    # Display target distribution
    print()
    print("Target Variable Distribution in Processed Data:")
    print("-" * 100)
    target_dist = processed_df['is_high_risk'].value_counts().sort_index()
    target_pct = processed_df['is_high_risk'].value_counts(normalize=True).sort_index() * 100
    
    for target_value in sorted(target_dist.index):
        count = target_dist[target_value]
        pct = target_pct[target_value]
        label = "High-Risk" if target_value == 1 else "Low-Risk"
        print(f"  {label} ({target_value}): {count:,} rows ({pct:.1f}%)")
    print()
    
    # Save integrated dataset
    output_path = project_root / "data" / "processed" / "processed_data_with_target.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data with target saved to: {output_path}")
    print(f"Final dataset shape: {processed_df.shape}")
    print(f"Features: {len(processed_df.columns)} (including is_high_risk)")
    print()
    
    print("=" * 100)
    print("Integration Complete: Target Variable Added to Processed Dataset!")
    print("=" * 100)
    print()
    print("The dataset is now ready for model training.")
    print(f"Use: {output_path}")
    print()


if __name__ == "__main__":
    main()

