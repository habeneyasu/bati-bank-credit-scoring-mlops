"""
Unit tests for Customer Clustering (Step 2 of Proxy Target Variable Engineering).

Run with: pytest tests/test_customer_clustering.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.customer_clustering import CustomerClustering, cluster_customers


class TestCustomerClustering:
    """Tests for CustomerClustering class."""
    
    @pytest.fixture
    def sample_rfm_data(self):
        """Create sample RFM data for testing."""
        np.random.seed(42)
        
        # Create 3 distinct customer groups with different RFM patterns
        data = []
        
        # Group 1: High engagement (low recency, high frequency, high monetary)
        for i in range(10):
            data.append({
                'CustomerId': f'C1_{i}',
                'recency': np.random.uniform(0, 10),
                'frequency': np.random.uniform(20, 50),
                'monetary': np.random.uniform(50000, 100000)
            })
        
        # Group 2: Medium engagement
        for i in range(10):
            data.append({
                'CustomerId': f'C2_{i}',
                'recency': np.random.uniform(20, 40),
                'frequency': np.random.uniform(5, 15),
                'monetary': np.random.uniform(10000, 30000)
            })
        
        # Group 3: Low engagement (high recency, low frequency, low monetary)
        for i in range(10):
            data.append({
                'CustomerId': f'C3_{i}',
                'recency': np.random.uniform(60, 90),
                'frequency': np.random.uniform(1, 5),
                'monetary': np.random.uniform(100, 5000)
            })
        
        return pd.DataFrame(data)
    
    def test_basic_clustering(self, sample_rfm_data):
        """Test basic clustering functionality."""
        clusterer = CustomerClustering(
            n_clusters=3,
            scaling_method='standardize',
            random_state=42
        )
        
        rfm_with_clusters = clusterer.fit_predict(sample_rfm_data)
        
        # Check structure
        assert 'cluster' in rfm_with_clusters.columns
        assert len(rfm_with_clusters) == len(sample_rfm_data)
        
        # Check that we have the right number of clusters
        assert rfm_with_clusters['cluster'].nunique() == 3
        
        # Check that all customers are assigned to a cluster
        assert rfm_with_clusters['cluster'].notna().all()
        assert all(rfm_with_clusters['cluster'].isin([0, 1, 2]))
    
    def test_different_scaling_methods(self, sample_rfm_data):
        """Test different scaling methods."""
        for scaling_method in ['standardize', 'robust', None]:
            clusterer = CustomerClustering(
                n_clusters=3,
                scaling_method=scaling_method,
                random_state=42
            )
            
            rfm_with_clusters = clusterer.fit_predict(sample_rfm_data)
            
            # Should work with all scaling methods
            assert 'cluster' in rfm_with_clusters.columns
            assert rfm_with_clusters['cluster'].nunique() == 3
    
    def test_reproducibility(self, sample_rfm_data):
        """Test that random_state ensures reproducibility."""
        clusterer1 = CustomerClustering(n_clusters=3, random_state=42)
        clusterer2 = CustomerClustering(n_clusters=3, random_state=42)
        
        rfm1 = clusterer1.fit_predict(sample_rfm_data)
        rfm2 = clusterer2.fit_predict(sample_rfm_data)
        
        # Should produce identical results
        assert (rfm1['cluster'].values == rfm2['cluster'].values).all()
    
    def test_different_random_states(self, sample_rfm_data):
        """Test that different random states produce different results."""
        clusterer1 = CustomerClustering(n_clusters=3, random_state=42)
        clusterer2 = CustomerClustering(n_clusters=3, random_state=123)
        
        rfm1 = clusterer1.fit_predict(sample_rfm_data)
        rfm2 = clusterer2.fit_predict(sample_rfm_data)
        
        # Results might be different (though could be same by chance)
        # At least verify both work
        assert rfm1['cluster'].nunique() == 3
        assert rfm2['cluster'].nunique() == 3
    
    def test_different_cluster_numbers(self, sample_rfm_data):
        """Test with different numbers of clusters."""
        for n_clusters in [2, 3, 4, 5]:
            clusterer = CustomerClustering(
                n_clusters=n_clusters,
                random_state=42
            )
            
            rfm_with_clusters = clusterer.fit_predict(sample_rfm_data)
            
            assert rfm_with_clusters['cluster'].nunique() == n_clusters
    
    def test_missing_features(self, sample_rfm_data):
        """Test that missing features raise appropriate errors."""
        # Remove a required feature
        incomplete_data = sample_rfm_data.drop(columns=['frequency'])
        
        clusterer = CustomerClustering()
        
        with pytest.raises(ValueError, match="Missing required RFM features"):
            clusterer.fit(incomplete_data)
    
    def test_missing_values(self):
        """Test that missing values raise appropriate errors."""
        data = pd.DataFrame({
            'CustomerId': ['C1', 'C2', 'C3'],
            'recency': [10, 20, np.nan],
            'frequency': [5, 10, 15],
            'monetary': [1000, 2000, 3000]
        })
        
        clusterer = CustomerClustering()
        
        with pytest.raises(ValueError, match="missing values"):
            clusterer.fit(data)
    
    def test_cluster_statistics(self, sample_rfm_data):
        """Test cluster statistics generation."""
        clusterer = CustomerClustering(n_clusters=3, random_state=42)
        clusterer.fit(sample_rfm_data)
        
        stats = clusterer.get_cluster_statistics()
        
        # Check structure
        assert 'cluster_id' in stats.columns
        assert 'n_customers' in stats.columns
        assert 'pct_customers' in stats.columns
        assert 'mean_recency' in stats.columns
        assert 'mean_frequency' in stats.columns
        assert 'mean_monetary' in stats.columns
        
        # Check that we have stats for all clusters
        assert len(stats) == 3
        
        # Check that percentages sum to 100
        assert abs(stats['pct_customers'].sum() - 100.0) < 0.01
    
    def test_cluster_centroids(self, sample_rfm_data):
        """Test cluster centroids calculation."""
        clusterer = CustomerClustering(n_clusters=3, random_state=42)
        clusterer.fit(sample_rfm_data)
        
        centroids = clusterer.get_cluster_centroids()
        
        # Check structure
        assert len(centroids) == 3
        assert 'recency' in centroids.columns
        assert 'frequency' in centroids.columns
        assert 'monetary' in centroids.columns
    
    def test_inertia(self, sample_rfm_data):
        """Test inertia calculation."""
        clusterer = CustomerClustering(n_clusters=3, random_state=42)
        clusterer.fit(sample_rfm_data)
        
        inertia = clusterer.get_inertia()
        
        # Inertia should be a positive number
        assert inertia > 0
        assert isinstance(inertia, (int, float))
    
    def test_predict_before_fit(self, sample_rfm_data):
        """Test that predict raises error if not fitted."""
        clusterer = CustomerClustering()
        
        with pytest.raises(ValueError, match="not been fitted"):
            clusterer.predict(sample_rfm_data)
    
    def test_statistics_before_fit(self):
        """Test that statistics raise error if not fitted."""
        clusterer = CustomerClustering()
        
        with pytest.raises(ValueError, match="not been fitted"):
            clusterer.get_cluster_statistics()
    
    def test_custom_rfm_features(self):
        """Test with custom RFM feature names."""
        data = pd.DataFrame({
            'CustomerId': ['C1', 'C2', 'C3'],
            'R': [10, 20, 30],  # Recency
            'F': [5, 10, 15],   # Frequency
            'M': [1000, 2000, 3000]  # Monetary
        })
        
        clusterer = CustomerClustering(
            n_clusters=2,
            rfm_features=['R', 'F', 'M'],
            random_state=42
        )
        
        rfm_with_clusters = clusterer.fit_predict(data)
        
        assert 'cluster' in rfm_with_clusters.columns
        assert rfm_with_clusters['cluster'].nunique() == 2


class TestClusterCustomersFunction:
    """Tests for convenience function cluster_customers."""
    
    def test_convenience_function(self):
        """Test the convenience function works correctly."""
        np.random.seed(42)
        data = pd.DataFrame({
            'CustomerId': [f'C{i}' for i in range(20)],
            'recency': np.random.uniform(0, 90, 20),
            'frequency': np.random.uniform(1, 50, 20),
            'monetary': np.random.uniform(100, 100000, 20)
        })
        
        rfm_with_clusters, clusterer = cluster_customers(
            data,
            n_clusters=3,
            scaling_method='standardize',
            random_state=42
        )
        
        assert 'cluster' in rfm_with_clusters.columns
        assert rfm_with_clusters['cluster'].nunique() == 3
        assert clusterer is not None
        assert clusterer.n_clusters == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

