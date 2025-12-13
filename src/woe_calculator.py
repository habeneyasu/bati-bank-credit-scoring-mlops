"""
Weight of Evidence (WoE) and Information Value (IV) calculator.

This module provides functions to calculate WoE and IV for feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def calculate_woe_iv(
    feature: pd.Series,
    target: pd.Series,
    bins: Optional[int] = None
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV) for a feature.
    
    Args:
        feature: Feature values (can be numerical or categorical)
        target: Binary target variable (0/1)
        bins: Number of bins for numerical features (None for categorical)
    
    Returns:
        Tuple of (WoE dataframe, IV value)
    """
    # Create a dataframe for calculation
    df = pd.DataFrame({
        'feature': feature,
        'target': target
    })
    
    # Handle missing values
    df = df.dropna()
    
    if len(df) == 0:
        return pd.DataFrame(), 0.0
    
    # Bin numerical features if needed
    if bins and pd.api.types.is_numeric_dtype(feature):
        df['binned'] = pd.cut(df['feature'], bins=bins, duplicates='drop')
        group_col = 'binned'
    else:
        group_col = 'feature'
    
    # Calculate counts
    grouped = df.groupby(group_col).agg({
        'target': ['count', 'sum']
    })
    grouped.columns = ['total', 'good']
    grouped['bad'] = grouped['total'] - grouped['good']
    
    # Calculate percentages
    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    
    if total_good == 0 or total_bad == 0:
        return pd.DataFrame(), 0.0
    
    grouped['pct_good'] = grouped['good'] / total_good
    grouped['pct_bad'] = grouped['bad'] / total_bad
    
    # Avoid division by zero
    grouped['pct_good'] = grouped['pct_good'].replace(0, 0.0001)
    grouped['pct_bad'] = grouped['pct_bad'].replace(0, 0.0001)
    
    # Calculate WoE
    grouped['woe'] = np.log(grouped['pct_good'] / grouped['pct_bad'])
    
    # Calculate IV
    grouped['iv'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['woe']
    iv_total = grouped['iv'].sum()
    
    return grouped, iv_total


def apply_woe_transformation(
    feature: pd.Series,
    woe_mapping: Dict
) -> pd.Series:
    """
    Apply WoE transformation to a feature using a pre-calculated mapping.
    
    Args:
        feature: Feature values to transform
        woe_mapping: Dictionary mapping feature values/bins to WoE values
    
    Returns:
        Series with WoE transformed values
    """
    return feature.map(woe_mapping).fillna(0)


def calculate_iv_for_features(
    df: pd.DataFrame,
    target: pd.Series,
    features: Optional[list] = None,
    bins: int = 10
) -> pd.DataFrame:
    """
    Calculate Information Value for multiple features.
    
    Args:
        df: DataFrame with features
        target: Target variable
        features: List of feature names (None for all)
        bins: Number of bins for numerical features
    
    Returns:
        DataFrame with IV values for each feature
    """
    if features is None:
        features = df.columns.tolist()
    
    iv_results = []
    
    for feature_name in features:
        if feature_name in df.columns:
            feature = df[feature_name]
            _, iv = calculate_woe_iv(feature, target, bins=bins)
            
            # Interpret IV
            if iv < 0.02:
                strength = "Not useful"
            elif iv < 0.1:
                strength = "Weak"
            elif iv < 0.3:
                strength = "Medium"
            elif iv < 0.5:
                strength = "Strong"
            else:
                strength = "Very Strong"
            
            iv_results.append({
                'feature': feature_name,
                'iv': iv,
                'strength': strength
            })
    
    iv_df = pd.DataFrame(iv_results)
    iv_df = iv_df.sort_values('iv', ascending=False)
    
    return iv_df

