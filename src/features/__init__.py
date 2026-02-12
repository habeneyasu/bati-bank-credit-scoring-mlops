"""
Feature Engineering Module

This module provides all feature engineering capabilities including:
- RFM metrics calculation
- Customer clustering
- High-risk labeling
- Data processing and transformation
- WoE calculation
- Data splitting
"""

from src.features.rfm import RFMCalculator, calculate_rfm_metrics
from src.features.clustering import CustomerClustering, cluster_customers
from src.features.labeling import HighRiskLabeler, create_high_risk_target
from src.features.processing import (
    DataProcessor,
    TemporalFeatureExtractor,
    CustomerAggregator,
    WoETransformer
)
from src.features.woe import (
    calculate_woe_iv,
    apply_woe_transformation,
    calculate_iv_for_features
)
from src.features.splitting import (
    split_data,
    split_data_from_file,
    load_splits,
    get_split_summary
)

__all__ = [
    # RFM
    "RFMCalculator",
    "calculate_rfm_metrics",
    # Clustering
    "CustomerClustering",
    "cluster_customers",
    # Labeling
    "HighRiskLabeler",
    "create_high_risk_target",
    # Processing
    "DataProcessor",
    "TemporalFeatureExtractor",
    "CustomerAggregator",
    "WoETransformer",
    # WoE
    "calculate_woe_iv",
    "apply_woe_transformation",
    "calculate_iv_for_features",
    # Splitting
    "split_data",
    "split_data_from_file",
    "load_splits",
    "get_split_summary",
]
