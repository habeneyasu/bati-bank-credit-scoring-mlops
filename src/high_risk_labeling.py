"""
Step 3: Define and Assign High-Risk Label

This module identifies the least engaged (highest-risk) customer cluster and creates
a binary target variable for credit risk modeling.

The process:
1. Analyze cluster characteristics to identify least engaged cluster
2. Create binary is_high_risk target variable
3. Assign 1 to high-risk cluster, 0 to all others
4. Add target to RFM data and transaction data
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class HighRiskLabeler:
    """
    Identify high-risk customer cluster and create binary target variable.
    
    This class implements Step 3 of the Proxy Target Variable Engineering process.
    It analyzes cluster characteristics to identify the least engaged (highest-risk)
    customer segment and creates a binary is_high_risk target variable.
    
    High-risk customers are typically characterized by:
    - High recency (long time since last transaction)
    - Low frequency (few transactions)
    - Low monetary value (low total spending)
    """
    
    def __init__(
        self,
        rfm_features: Optional[list] = None,
        risk_criteria: str = 'engagement_score'  # 'engagement_score' or 'manual'
    ):
        """
        Initialize High-Risk Labeler.
        
        Args:
            rfm_features: List of RFM feature names (default: ['recency', 'frequency', 'monetary'])
            risk_criteria: Method for identifying high-risk cluster:
                          - 'engagement_score': Automatic based on engagement score
                          - 'manual': Manual cluster ID specification
        
        Example:
            >>> labeler = HighRiskLabeler()
            >>> rfm_with_target = labeler.identify_high_risk(rfm_with_clusters)
        """
        self.rfm_features = rfm_features or ['recency', 'frequency', 'monetary']
        self.risk_criteria = risk_criteria
        
        # Fitted attributes
        self.high_risk_cluster_id_ = None
        self.cluster_stats_ = None
        self.engagement_scores_ = None
    
    def _calculate_engagement_score(
        self,
        cluster_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate engagement score for each cluster.
        
        Engagement score combines normalized RFM metrics:
        - Higher recency = worse (less engaged)
        - Lower frequency = worse (less engaged)
        - Lower monetary = worse (less engaged)
        
        Higher engagement_score = worse engagement = higher risk
        
        Args:
            cluster_stats: DataFrame with cluster statistics
            
        Returns:
            DataFrame with engagement scores added
        """
        stats = cluster_stats.copy()
        
        # Normalize each RFM metric to [0, 1] range
        for feature in self.rfm_features:
            if f'mean_{feature}' in stats.columns:
                max_val = stats[f'mean_{feature}'].max()
                min_val = stats[f'mean_{feature}'].min()
                
                if max_val > min_val:
                    if feature == 'recency':
                        # For recency: higher is worse
                        # Normalize so higher values = higher score
                        stats[f'normalized_{feature}'] = (
                            (stats[f'mean_{feature}'] - min_val) / (max_val - min_val)
                        )
                    else:
                        # For frequency and monetary: lower is worse
                        # Normalize and invert so lower values = higher score
                        stats[f'normalized_{feature}'] = (
                            1 - (stats[f'mean_{feature}'] - min_val) / (max_val - min_val)
                        )
                else:
                    stats[f'normalized_{feature}'] = 0.5  # Default if all same
        
        # Calculate engagement score (weighted sum)
        # Higher score = worse engagement = higher risk
        normalized_cols = [f'normalized_{f}' for f in self.rfm_features]
        stats['engagement_score'] = stats[normalized_cols].sum(axis=1)
        
        return stats
    
    def identify_high_risk_cluster(
        self,
        rfm_with_clusters: pd.DataFrame,
        cluster_stats: Optional[pd.DataFrame] = None
    ) -> int:
        """
        Identify the high-risk cluster based on engagement characteristics.
        
        High-risk cluster is identified as the cluster with:
        - Highest recency (longest time since last transaction)
        - Lowest frequency (fewest transactions)
        - Lowest monetary value (lowest total spending)
        
        Args:
            rfm_with_clusters: DataFrame with RFM metrics and cluster labels
            cluster_stats: Optional pre-calculated cluster statistics
        
        Returns:
            Cluster ID identified as high-risk
        """
        if 'cluster' not in rfm_with_clusters.columns:
            raise ValueError(
                "RFM data must contain 'cluster' column. "
                "Please run clustering first (Step 2)."
            )
        
        # Calculate cluster statistics if not provided
        if cluster_stats is None:
            cluster_stats = self._calculate_cluster_statistics(rfm_with_clusters)
        
        self.cluster_stats_ = cluster_stats
        
        # Calculate engagement scores
        stats_with_scores = self._calculate_engagement_score(cluster_stats)
        self.engagement_scores_ = stats_with_scores
        
        # High-risk cluster is the one with highest engagement_score
        # (worst engagement = highest risk)
        high_risk_cluster = stats_with_scores.loc[
            stats_with_scores['engagement_score'].idxmax(),
            'cluster_id'
        ]
        
        self.high_risk_cluster_id_ = int(high_risk_cluster)
        
        return self.high_risk_cluster_id_
    
    def _calculate_cluster_statistics(
        self,
        rfm_with_clusters: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate summary statistics for each cluster.
        
        Args:
            rfm_with_clusters: DataFrame with RFM metrics and cluster labels
        
        Returns:
            DataFrame with cluster statistics
        """
        cluster_stats = []
        
        for cluster_id in sorted(rfm_with_clusters['cluster'].unique()):
            cluster_mask = rfm_with_clusters['cluster'] == cluster_id
            cluster_rfm = rfm_with_clusters[cluster_mask]
            
            stats = {
                'cluster_id': cluster_id,
                'n_customers': cluster_mask.sum(),
                'pct_customers': (cluster_mask.sum() / len(rfm_with_clusters)) * 100,
            }
            
            # Add mean for each RFM feature
            for feature in self.rfm_features:
                if feature in cluster_rfm.columns:
                    stats[f'mean_{feature}'] = cluster_rfm[feature].mean()
                    stats[f'median_{feature}'] = cluster_rfm[feature].median()
            
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)
    
    def create_target_variable(
        self,
        rfm_with_clusters: pd.DataFrame,
        high_risk_cluster_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create binary is_high_risk target variable.
        
        Args:
            rfm_with_clusters: DataFrame with RFM metrics and cluster labels
            high_risk_cluster_id: Cluster ID to mark as high-risk.
                                 If None, automatically identifies it
        
        Returns:
            DataFrame with is_high_risk column added
        """
        rfm_with_target = rfm_with_clusters.copy()
        
        # Identify high-risk cluster if not provided
        if high_risk_cluster_id is None:
            if self.high_risk_cluster_id_ is None:
                self.identify_high_risk_cluster(rfm_with_clusters)
            high_risk_cluster_id = self.high_risk_cluster_id_
        else:
            self.high_risk_cluster_id_ = high_risk_cluster_id
        
        # Create binary target: 1 if high-risk cluster, 0 otherwise
        rfm_with_target['is_high_risk'] = (
            (rfm_with_target['cluster'] == high_risk_cluster_id).astype(int)
        )
        
        return rfm_with_target
    
    def add_target_to_transactions(
        self,
        transactions_df: pd.DataFrame,
        rfm_with_target: pd.DataFrame,
        customer_col: str = 'CustomerId'
    ) -> pd.DataFrame:
        """
        Add is_high_risk target variable to transaction-level DataFrame.
        
        Args:
            transactions_df: Original transaction DataFrame
            rfm_with_target: RFM DataFrame with is_high_risk target
            customer_col: Name of customer ID column
        
        Returns:
            Transaction DataFrame with is_high_risk column added
        """
        # Extract target mapping
        target_mapping = rfm_with_target[[customer_col, 'is_high_risk', 'cluster']].drop_duplicates()
        
        # Merge target back to transactions
        transactions_with_target = transactions_df.merge(
            target_mapping,
            on=customer_col,
            how='left'
        )
        
        # Fill any missing values (shouldn't happen, but safety check)
        if transactions_with_target['is_high_risk'].isna().any():
            warnings.warn(
                f"Some transactions have missing is_high_risk labels. "
                f"Filling with 0 (low-risk)."
            )
            transactions_with_target['is_high_risk'] = (
                transactions_with_target['is_high_risk'].fillna(0).astype(int)
            )
        
        return transactions_with_target
    
    def get_high_risk_summary(self) -> pd.DataFrame:
        """
        Get summary of high-risk cluster identification.
        
        Returns:
            DataFrame with cluster analysis and engagement scores
        
        Raises:
            ValueError: If high-risk cluster has not been identified yet
        """
        if self.engagement_scores_ is None:
            raise ValueError(
                "High-risk cluster has not been identified yet. "
                "Call identify_high_risk_cluster() first."
            )
        
        summary = self.engagement_scores_.copy()
        summary['is_high_risk'] = (
            summary['cluster_id'] == self.high_risk_cluster_id_
        )
        
        return summary


def create_high_risk_target(
    rfm_with_clusters: pd.DataFrame,
    transactions_df: Optional[pd.DataFrame] = None,
    customer_col: str = 'CustomerId',
    return_summary: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Convenience function to create high-risk target variable.
    
    Args:
        rfm_with_clusters: DataFrame with RFM metrics and cluster labels
        transactions_df: Optional transaction DataFrame to add target to
        customer_col: Name of customer ID column
        return_summary: If True, also return cluster analysis summary
    
    Returns:
        Tuple of:
        - rfm_with_target: RFM DataFrame with is_high_risk column
        - transactions_with_target: Transaction DataFrame with is_high_risk (if provided)
        - summary: Cluster analysis summary (if return_summary=True)
    
    Example:
        >>> rfm_with_clusters = pd.read_csv('data/processed/rfm_with_clusters.csv')
        >>> transactions_df = pd.read_csv('data/raw/data.csv')
        >>> rfm_target, trans_target, summary = create_high_risk_target(
        ...     rfm_with_clusters,
        ...     transactions_df,
        ...     return_summary=True
        ... )
    """
    labeler = HighRiskLabeler()
    
    # Identify high-risk cluster
    high_risk_cluster_id = labeler.identify_high_risk_cluster(rfm_with_clusters)
    print(f"Identified Cluster {high_risk_cluster_id} as high-risk (least engaged)")
    
    # Create target variable
    rfm_with_target = labeler.create_target_variable(rfm_with_clusters)
    
    # Add target to transactions if provided
    transactions_with_target = None
    if transactions_df is not None:
        transactions_with_target = labeler.add_target_to_transactions(
            transactions_df,
            rfm_with_target,
            customer_col=customer_col
        )
    
    # Get summary if requested
    summary = None
    if return_summary:
        summary = labeler.get_high_risk_summary()
    
    # Print target distribution
    print("\nTarget Variable Distribution:")
    target_dist = rfm_with_target['is_high_risk'].value_counts()
    print(f"  High-Risk (1): {target_dist.get(1, 0):,} customers ({target_dist.get(1, 0)/len(rfm_with_target)*100:.1f}%)")
    print(f"  Low-Risk (0):  {target_dist.get(0, 0):,} customers ({target_dist.get(0, 0)/len(rfm_with_target)*100:.1f}%)")
    
    return rfm_with_target, transactions_with_target, summary


if __name__ == "__main__":
    # Example usage
    print("=" * 100)
    print("High-Risk Labeling - Step 3 of Proxy Target Variable Engineering")
    print("=" * 100)
    print()
    
    # Check if clustered RFM data exists
    rfm_path = Path("data/processed/rfm_with_clusters.csv")
    
    if rfm_path.exists():
        print(f"Loading clustered RFM data from: {rfm_path}")
        rfm_with_clusters = pd.read_csv(rfm_path)
        print(f"Loaded RFM data for {len(rfm_with_clusters)} customers")
        print()
        
        # Create high-risk target
        labeler = HighRiskLabeler()
        
        # Identify high-risk cluster
        high_risk_cluster_id = labeler.identify_high_risk_cluster(rfm_with_clusters)
        print(f"\nHigh-Risk Cluster ID: {high_risk_cluster_id}")
        print()
        
        # Get summary
        summary = labeler.get_high_risk_summary()
        print("Cluster Analysis:")
        print(summary[['cluster_id', 'n_customers', 'mean_recency', 'mean_frequency', 
                       'mean_monetary', 'engagement_score', 'is_high_risk']])
        print()
        
        # Create target variable
        rfm_with_target = labeler.create_target_variable(rfm_with_clusters)
        
        # Display target distribution
        print("Target Variable Distribution:")
        target_dist = rfm_with_target['is_high_risk'].value_counts()
        print(f"  High-Risk (1): {target_dist.get(1, 0):,} customers")
        print(f"  Low-Risk (0):  {target_dist.get(0, 0):,} customers")
        print()
        
        print("=" * 100)
        print("Step 3 Complete: High-Risk Target Variable Created!")
        print("=" * 100)
    else:
        print(f"Clustered RFM data not found: {rfm_path}")
        print("Please run Step 2 first to cluster customers.")
        print()

