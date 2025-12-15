"""
Data Splitting Module for Model Training

This module provides functions to split data into training and testing sets
for model training and evaluation.

Features:
- Train/Test split with reproducibility (random_state)
- Stratified splitting for imbalanced datasets
- Optional validation set creation
- Data saving and loading utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import joblib
import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 42,
    stratify: bool = True,
    shuffle: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Split data into training and testing sets (and optionally validation set).
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of dataset to include in test split (default: 0.2)
        val_size: Proportion of dataset to include in validation split (default: None)
                 If provided, creates train/val/test split
        random_state: Random state for reproducibility (default: 42)
        stratify: Whether to use stratified splitting (default: True)
                 Helps maintain class distribution in splits
        shuffle: Whether to shuffle data before splitting (default: True)
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) if val_size is None
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test) if val_size is provided
    
    Example:
        >>> X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        >>> X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        ...     X, y, test_size=0.2, val_size=0.1, random_state=42
        ... )
    """
    # Validate inputs
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("y must be a pandas Series or numpy array")
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length. Got {len(X)} and {len(y)}")
    
    # Convert y to Series if needed
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    # Determine stratification
    stratify_param = y if stratify else None
    
    if val_size is None:
        # Simple train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param,
            shuffle=shuffle
        )
        
        print(f"Data split completed:")
        print(f"  Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Test set:     {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"  Random state: {random_state}")
        print(f"  Stratified:   {stratify}")
        
        return X_train, X_test, y_train, y_test
    
    else:
        # Train/Val/Test split
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param,
            shuffle=shuffle
        )
        
        # Second split: separate train and validation from remaining data
        # Adjust val_size relative to the remaining data
        val_size_adjusted = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=(y_temp if stratify else None),
            shuffle=shuffle
        )
        
        print(f"Data split completed:")
        print(f"  Training set:    {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set:  {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"  Random state:   {random_state}")
        print(f"  Stratified:     {stratify}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def split_data_from_file(
    data_path: str,
    target_col: str = 'is_high_risk',
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 42,
    stratify: bool = True,
    save_splits: bool = True,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Load data from file and split into training and testing sets.
    
    Args:
        data_path: Path to CSV file with features and target
        target_col: Name of target column (default: 'is_high_risk')
        test_size: Proportion for test set (default: 0.2)
        val_size: Proportion for validation set (default: None)
        random_state: Random state for reproducibility (default: 42)
        stratify: Whether to use stratified splitting (default: True)
        save_splits: Whether to save splits to disk (default: True)
        output_dir: Directory to save splits (default: data/processed/splits)
    
    Returns:
        Tuple of data splits (same as split_data function)
    
    Example:
        >>> splits = split_data_from_file(
        ...     'data/processed/processed_data_with_target.csv',
        ...     test_size=0.2,
        ...     random_state=42
        ... )
    """
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    
    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {list(df.columns)}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Features: {len(X.columns)}")
    print(f"Target distribution:")
    print(y.value_counts().sort_index())
    print()
    
    # Split data
    splits = split_data(
        X, y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=stratify
    )
    
    # Save splits if requested
    if save_splits:
        if output_dir is None:
            output_dir = Path(data_path).parent / "splits"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if val_size is None:
            X_train, X_test, y_train, y_test = splits
            
            # Save training set
            X_train.to_csv(output_dir / "X_train.csv", index=False)
            y_train.to_csv(output_dir / "y_train.csv", index=False)
            
            # Save test set
            X_test.to_csv(output_dir / "X_test.csv", index=False)
            y_test.to_csv(output_dir / "y_test.csv", index=False)
            
            print(f"\nSplits saved to: {output_dir}")
            print(f"  - X_train.csv, y_train.csv")
            print(f"  - X_test.csv, y_test.csv")
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = splits
            
            # Save training set
            X_train.to_csv(output_dir / "X_train.csv", index=False)
            y_train.to_csv(output_dir / "y_train.csv", index=False)
            
            # Save validation set
            X_val.to_csv(output_dir / "X_val.csv", index=False)
            y_val.to_csv(output_dir / "y_val.csv", index=False)
            
            # Save test set
            X_test.to_csv(output_dir / "X_test.csv", index=False)
            y_test.to_csv(output_dir / "y_test.csv", index=False)
            
            print(f"\nSplits saved to: {output_dir}")
            print(f"  - X_train.csv, y_train.csv")
            print(f"  - X_val.csv, y_val.csv")
            print(f"  - X_test.csv, y_test.csv")
    
    return splits


def load_splits(
    splits_dir: str,
    include_val: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Load previously saved data splits from disk.
    
    Args:
        splits_dir: Directory containing saved splits
        include_val: Whether to load validation set (default: False)
    
    Returns:
        Tuple of data splits (same as split_data function)
    
    Example:
        >>> X_train, X_test, y_train, y_test = load_splits('data/processed/splits')
    """
    splits_dir = Path(splits_dir)
    
    if include_val:
        # Load train/val/test splits
        X_train = pd.read_csv(splits_dir / "X_train.csv")
        y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze()
        X_val = pd.read_csv(splits_dir / "X_val.csv")
        y_val = pd.read_csv(splits_dir / "y_val.csv").squeeze()
        X_test = pd.read_csv(splits_dir / "X_test.csv")
        y_test = pd.read_csv(splits_dir / "y_test.csv").squeeze()
        
        print(f"Loaded splits from: {splits_dir}")
        print(f"  Training set:   {len(X_train):,} samples")
        print(f"  Validation set: {len(X_val):,} samples")
        print(f"  Test set:       {len(X_test):,} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        # Load train/test splits
        X_train = pd.read_csv(splits_dir / "X_train.csv")
        y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze()
        X_test = pd.read_csv(splits_dir / "X_test.csv")
        y_test = pd.read_csv(splits_dir / "y_test.csv").squeeze()
        
        print(f"Loaded splits from: {splits_dir}")
        print(f"  Training set: {len(X_train):,} samples")
        print(f"  Test set:     {len(X_test):,} samples")
        
        return X_train, X_test, y_train, y_test


def get_split_summary(
    y_train: pd.Series,
    y_test: pd.Series,
    y_val: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Get summary statistics of data splits.
    
    Args:
        y_train: Training target
        y_test: Test target
        y_val: Validation target (optional)
    
    Returns:
        DataFrame with split summary statistics
    """
    summary_data = []
    
    # Training set
    train_dist = y_train.value_counts().sort_index()
    summary_data.append({
        'split': 'train',
        'n_samples': len(y_train),
        'class_0': train_dist.get(0, 0),
        'class_1': train_dist.get(1, 0),
        'class_0_pct': (train_dist.get(0, 0) / len(y_train)) * 100,
        'class_1_pct': (train_dist.get(1, 0) / len(y_train)) * 100
    })
    
    # Validation set (if provided)
    if y_val is not None:
        val_dist = y_val.value_counts().sort_index()
        summary_data.append({
            'split': 'validation',
            'n_samples': len(y_val),
            'class_0': val_dist.get(0, 0),
            'class_1': val_dist.get(1, 0),
            'class_0_pct': (val_dist.get(0, 0) / len(y_val)) * 100,
            'class_1_pct': (val_dist.get(1, 0) / len(y_val)) * 100
        })
    
    # Test set
    test_dist = y_test.value_counts().sort_index()
    summary_data.append({
        'split': 'test',
        'n_samples': len(y_test),
        'class_0': test_dist.get(0, 0),
        'class_1': test_dist.get(1, 0),
        'class_0_pct': (test_dist.get(0, 0) / len(y_test)) * 100,
        'class_1_pct': (test_dist.get(1, 0) / len(y_test)) * 100
    })
    
    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    print("=" * 100)
    print("Data Splitting Module - Example Usage")
    print("=" * 100)
    print()
    
    # Example: Split data from file
    data_path = "data/processed/processed_data_with_target.csv"
    
    if Path(data_path).exists():
        print("Example 1: Split data from file")
        print("-" * 100)
        
        splits = split_data_from_file(
            data_path,
            target_col='is_high_risk',
            test_size=0.2,
            random_state=42,
            stratify=True,
            save_splits=True
        )
        
        X_train, X_test, y_train, y_test = splits
        
        print()
        print("Split Summary:")
        summary = get_split_summary(y_train, y_test)
        print(summary)
        print()
        
        print("=" * 100)
        print("Data splitting complete!")
        print("=" * 100)
    else:
        print(f"Data file not found: {data_path}")
        print("Please ensure the processed data with target exists.")
        print()
        print("Example usage:")
        print("""
        from src.data_splitting import split_data_from_file
        
        # Split data
        X_train, X_test, y_train, y_test = split_data_from_file(
            'data/processed/processed_data_with_target.csv',
            target_col='is_high_risk',
            test_size=0.2,
            random_state=42,
            stratify=True
        )
        """)

