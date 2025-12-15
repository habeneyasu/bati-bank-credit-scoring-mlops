"""
Example: Prepare Data Splits for Model Training

This script demonstrates how to split the processed data into training and testing sets
for model training and evaluation.

This is part of Task 5: Model Training and Tracking.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_splitting import (
    split_data_from_file,
    get_split_summary,
    load_splits
)


def main():
    """Main function to prepare data splits."""
    
    print("=" * 100)
    print("Data Preparation: Split Data into Training and Testing Sets")
    print("=" * 100)
    print()
    
    # Load processed data with target
    data_path = project_root / "data" / "processed" / "processed_data_with_target.csv"
    
    if not data_path.exists():
        print(f"Processed data with target not found: {data_path}")
        print("Please run the integration script first:")
        print("  python examples/integrate_target_to_processed_data.py")
        return
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    print()
    
    # Display target distribution
    print("Target Variable Distribution (Full Dataset):")
    print("-" * 100)
    target_dist = df['is_high_risk'].value_counts().sort_index()
    target_pct = df['is_high_risk'].value_counts(normalize=True).sort_index() * 100
    
    for target_value in sorted(target_dist.index):
        count = target_dist[target_value]
        pct = target_pct[target_value]
        label = "High-Risk" if target_value == 1 else "Low-Risk"
        print(f"  {label} ({target_value}): {count:,} samples ({pct:.1f}%)")
    print()
    
    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    print("-" * 100)
    print()
    
    splits = split_data_from_file(
        str(data_path),
        target_col='is_high_risk',
        test_size=0.2,  # 20% for testing
        random_state=42,  # For reproducibility
        stratify=True,  # Maintain class distribution
        save_splits=True,
        output_dir=str(project_root / "data" / "processed" / "splits")
    )
    
    X_train, X_test, y_train, y_test = splits
    
    print()
    print("=" * 100)
    print("Split Summary")
    print("=" * 100)
    print()
    
    # Get detailed summary
    summary = get_split_summary(y_train, y_test)
    print(summary.round(2))
    print()
    
    # Display split details
    print("Training Set Details:")
    print(f"  Samples: {len(X_train):,}")
    print(f"  Features: {len(X_train.columns)}")
    print(f"  Low-Risk (0): {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"  High-Risk (1): {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
    print()
    
    print("Test Set Details:")
    print(f"  Samples: {len(X_test):,}")
    print(f"  Features: {len(X_test.columns)}")
    print(f"  Low-Risk (0): {sum(y_test == 0):,} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
    print(f"  High-Risk (1): {sum(y_test == 1):,} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")
    print()
    
    # Verify class distribution is maintained
    print("Class Distribution Verification:")
    print("-" * 100)
    original_pct = (df['is_high_risk'].sum() / len(df)) * 100
    train_pct = (y_train.sum() / len(y_train)) * 100
    test_pct = (y_test.sum() / len(y_test)) * 100
    
    print(f"  Original dataset: {original_pct:.2f}% high-risk")
    print(f"  Training set:    {train_pct:.2f}% high-risk (difference: {abs(original_pct - train_pct):.2f}%)")
    print(f"  Test set:        {test_pct:.2f}% high-risk (difference: {abs(original_pct - test_pct):.2f}%)")
    print()
    
    if abs(original_pct - train_pct) < 1.0 and abs(original_pct - test_pct) < 1.0:
        print("✓ Class distribution maintained across splits (stratified splitting working correctly)")
    else:
        print("⚠ Warning: Class distribution differs significantly between splits")
    print()
    
    print("=" * 100)
    print("Data Preparation Complete!")
    print("=" * 100)
    print()
    print("Splits saved to: data/processed/splits/")
    print("  - X_train.csv, y_train.csv")
    print("  - X_test.csv, y_test.csv")
    print()
    print("Next Steps:")
    print("  1. Use these splits for model training")
    print("  2. Train models on training set")
    print("  3. Evaluate on test set")
    print()


if __name__ == "__main__":
    main()

