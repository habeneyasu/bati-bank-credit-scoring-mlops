"""
Unit tests for High-Risk Labeling (Step 3 of Proxy Target Variable Engineering).

Run with: pytest tests/test_high_risk_labeling.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.high_risk_labeling import HighRiskLabeler, create_high_risk_target


class TestHighRiskLabeler:
    """Tests for HighRiskLabeler class."""
    
    @pytest.fixture
    def sample_rfm_with_clusters(self):
        """Create sample RFM data with cluster labels."""
        np.random.seed(42)
        
        # Create 3 distinct clusters
        data = []
        
        # Cluster 0: High engagement (low recency, high frequency, high monetary)
        for i in range(10):
            data.append({
                'CustomerId': f'C0_{i}',
                'recency': np.random.uniform(0, 10),
                'frequency': np.random.uniform(20, 50),
                'monetary': np.random.uniform(50000, 100000),
                'cluster': 0
            })
        
        # Cluster 1: Low engagement (high recency, low frequency, low monetary)
        for i in range(10):
            data.append({
                'CustomerId': f'C1_{i}',
                'recency': np.random.uniform(60, 90),
                'frequency': np.random.uniform(1, 5),
                'monetary': np.random.uniform(100, 5000),
                'cluster': 1
            })
        
        # Cluster 2: Medium engagement
        for i in range(10):
            data.append({
                'CustomerId': f'C2_{i}',
                'recency': np.random.uniform(20, 40),
                'frequency': np.random.uniform(5, 15),
                'monetary': np.random.uniform(10000, 30000),
                'cluster': 2
            })
        
        return pd.DataFrame(data)
    
    def test_identify_high_risk_cluster(self, sample_rfm_with_clusters):
        """Test high-risk cluster identification."""
        labeler = HighRiskLabeler()
        
        high_risk_cluster_id = labeler.identify_high_risk_cluster(sample_rfm_with_clusters)
        
        # Should identify Cluster 1 as high-risk (highest recency, lowest frequency/monetary)
        assert high_risk_cluster_id == 1
        assert labeler.high_risk_cluster_id_ == 1
    
    def test_create_target_variable(self, sample_rfm_with_clusters):
        """Test binary target variable creation."""
        labeler = HighRiskLabeler()
        
        # Identify high-risk cluster
        labeler.identify_high_risk_cluster(sample_rfm_with_clusters)
        
        # Create target variable
        rfm_with_target = labeler.create_target_variable(sample_rfm_with_clusters)
        
        # Check structure
        assert 'is_high_risk' in rfm_with_target.columns
        assert rfm_with_target['is_high_risk'].isin([0, 1]).all()
        
        # Check that high-risk cluster has value 1
        high_risk_cluster = labeler.high_risk_cluster_id_
        high_risk_customers = rfm_with_target[rfm_with_target['cluster'] == high_risk_cluster]
        assert all(high_risk_customers['is_high_risk'] == 1)
        
        # Check that other clusters have value 0
        other_clusters = rfm_with_target[rfm_with_target['cluster'] != high_risk_cluster]
        assert all(other_clusters['is_high_risk'] == 0)
    
    def test_create_target_with_manual_cluster_id(self, sample_rfm_with_clusters):
        """Test creating target with manually specified cluster ID."""
        labeler = HighRiskLabeler()
        
        # Manually specify high-risk cluster
        rfm_with_target = labeler.create_target_variable(
            sample_rfm_with_clusters,
            high_risk_cluster_id=1
        )
        
        assert 'is_high_risk' in rfm_with_target.columns
        assert all(rfm_with_target[rfm_with_target['cluster'] == 1]['is_high_risk'] == 1)
        assert all(rfm_with_target[rfm_with_target['cluster'] != 1]['is_high_risk'] == 0)
    
    def test_add_target_to_transactions(self, sample_rfm_with_clusters):
        """Test adding target to transaction-level data."""
        labeler = HighRiskLabeler()
        
        # Create target variable
        labeler.identify_high_risk_cluster(sample_rfm_with_clusters)
        rfm_with_target = labeler.create_target_variable(sample_rfm_with_clusters)
        
        # Create sample transaction data
        transactions = []
        for _, row in sample_rfm_with_clusters.iterrows():
            for i in range(int(row['frequency'])):
                transactions.append({
                    'CustomerId': row['CustomerId'],
                    'TransactionId': f'T_{row["CustomerId"]}_{i}',
                    'Amount': row['monetary'] / row['frequency']
                })
        
        transactions_df = pd.DataFrame(transactions)
        
        # Add target to transactions
        transactions_with_target = labeler.add_target_to_transactions(
            transactions_df,
            rfm_with_target,
            customer_col='CustomerId'
        )
        
        # Check that target is added
        assert 'is_high_risk' in transactions_with_target.columns
        assert 'cluster' in transactions_with_target.columns
        
        # Check that all transactions for a customer have the same target
        for customer_id in transactions_with_target['CustomerId'].unique():
            customer_targets = transactions_with_target[
                transactions_with_target['CustomerId'] == customer_id
            ]['is_high_risk'].unique()
            assert len(customer_targets) == 1
    
    def test_get_high_risk_summary(self, sample_rfm_with_clusters):
        """Test cluster analysis summary generation."""
        labeler = HighRiskLabeler()
        
        # Identify high-risk cluster
        labeler.identify_high_risk_cluster(sample_rfm_with_clusters)
        
        # Get summary
        summary = labeler.get_high_risk_summary()
        
        # Check structure
        assert 'cluster_id' in summary.columns
        assert 'engagement_score' in summary.columns
        assert 'is_high_risk' in summary.columns
        
        # Check that exactly one cluster is marked as high-risk
        assert summary['is_high_risk'].sum() == 1
    
    def test_missing_cluster_column(self):
        """Test that missing cluster column raises error."""
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C2'],
            'recency': [10, 20],
            'frequency': [5, 10],
            'monetary': [1000, 2000]
            # Missing 'cluster' column
        })
        
        labeler = HighRiskLabeler()
        
        with pytest.raises(ValueError, match="cluster"):
            labeler.identify_high_risk_cluster(df)
    
    def test_summary_before_identification(self):
        """Test that summary raises error if cluster not identified."""
        labeler = HighRiskLabeler()
        
        with pytest.raises(ValueError, match="not been identified"):
            labeler.get_high_risk_summary()
    
    def test_engagement_score_calculation(self, sample_rfm_with_clusters):
        """Test engagement score calculation."""
        labeler = HighRiskLabeler()
        
        # Identify high-risk cluster
        labeler.identify_high_risk_cluster(sample_rfm_with_clusters)
        
        # Get summary with engagement scores
        summary = labeler.get_high_risk_summary()
        
        # High-risk cluster should have highest engagement_score
        high_risk_cluster = summary[summary['is_high_risk']]['cluster_id'].iloc[0]
        high_risk_score = summary[summary['is_high_risk']]['engagement_score'].iloc[0]
        
        # All other clusters should have lower scores
        other_scores = summary[~summary['is_high_risk']]['engagement_score']
        assert all(high_risk_score >= other_scores)


class TestCreateHighRiskTargetFunction:
    """Tests for convenience function create_high_risk_target."""
    
    def test_convenience_function(self):
        """Test the convenience function works correctly."""
        np.random.seed(42)
        
        # Create sample data
        data = []
        for cluster_id in [0, 1, 2]:
            for i in range(5):
                data.append({
                    'CustomerId': f'C{cluster_id}_{i}',
                    'recency': np.random.uniform(0, 90),
                    'frequency': np.random.uniform(1, 50),
                    'monetary': np.random.uniform(100, 100000),
                    'cluster': cluster_id
                })
        
        rfm_with_clusters = pd.DataFrame(data)
        
        # Create target
        rfm_with_target, transactions_with_target, summary = create_high_risk_target(
            rfm_with_clusters,
            return_summary=True
        )
        
        assert 'is_high_risk' in rfm_with_target.columns
        assert rfm_with_target['is_high_risk'].isin([0, 1]).all()
        assert summary is not None
        assert transactions_with_target is None  # No transactions provided
    
    def test_with_transactions(self):
        """Test convenience function with transaction data."""
        np.random.seed(42)
        
        # Create RFM data
        rfm_data = []
        for cluster_id in [0, 1]:
            for i in range(3):
                rfm_data.append({
                    'CustomerId': f'C{cluster_id}_{i}',
                    'recency': 10 if cluster_id == 0 else 80,
                    'frequency': 20 if cluster_id == 0 else 3,
                    'monetary': 50000 if cluster_id == 0 else 2000,
                    'cluster': cluster_id
                })
        
        rfm_with_clusters = pd.DataFrame(rfm_data)
        
        # Create transaction data
        transactions = []
        for _, row in rfm_with_clusters.iterrows():
            for i in range(int(row['frequency'])):
                transactions.append({
                    'CustomerId': row['CustomerId'],
                    'TransactionId': f'T_{row["CustomerId"]}_{i}',
                    'Amount': 100
                })
        
        transactions_df = pd.DataFrame(transactions)
        
        # Create target
        rfm_with_target, transactions_with_target, _ = create_high_risk_target(
            rfm_with_clusters,
            transactions_df=transactions_df
        )
        
        assert transactions_with_target is not None
        assert 'is_high_risk' in transactions_with_target.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

