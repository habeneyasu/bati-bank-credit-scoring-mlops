"""
Step 2: Customer Clustering using K-Means

This module clusters customers into distinct groups based on their RFM (Recency, Frequency, Monetary) profiles.

The process:
1. Load or calculate RFM metrics
2. Scale RFM features appropriately (StandardScaler or RobustScaler)
3. Apply K-Means clustering with 3 clusters
4. Analyze and visualize cluster characteristics
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings('ignore')


class CustomerClustering:
    """
    Cluster customers into distinct groups based on RFM metrics using K-Means.
    
    This class implements Step 2 of the Proxy Target Variable Engineering process.
    It scales RFM features and applies K-Means clustering to segment customers.
    
    Features:
    - Automatic feature scaling (StandardScaler or RobustScaler)
    - K-Means clustering with configurable number of clusters
    - Reproducible results with random_state
    - Cluster analysis and statistics
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        scaling_method: str = 'standardize',  # 'standardize', 'robust', or None
        random_state: int = 42,
        rfm_features: Optional[list] = None
    ):
        """
        Initialize Customer Clustering.
        
        Args:
            n_clusters: Number of clusters for K-Means (default: 3)
            scaling_method: Method for scaling RFM features:
                          - 'standardize': StandardScaler (mean=0, std=1)
                          - 'robust': RobustScaler (median, IQR-based, robust to outliers)
                          - None: No scaling
            random_state: Random state for K-Means reproducibility (default: 42)
            rfm_features: List of RFM feature names to use for clustering.
                         If None, uses ['recency', 'frequency', 'monetary'] (default: None)
        
        Example:
            >>> clusterer = CustomerClustering(
            ...     n_clusters=3,
            ...     scaling_method='standardize',
            ...     random_state=42
            ... )
        """
        self.n_clusters = n_clusters
        self.scaling_method = scaling_method
        self.random_state = random_state
        self.rfm_features = rfm_features or ['recency', 'frequency', 'monetary']
        
        # Components (fitted during fit())
        self.scaler_ = None
        self.kmeans_ = None
        self.cluster_labels_ = None
        self.rfm_data_ = None
        self.rfm_scaled_ = None
    
    def _validate_rfm_data(self, rfm_df: pd.DataFrame):
        """
        Validate that RFM DataFrame contains required features.
        
        Args:
            rfm_df: DataFrame with RFM metrics
            
        Raises:
            ValueError: If required features are missing
        """
        missing_features = [f for f in self.rfm_features if f not in rfm_df.columns]
        if missing_features:
            raise ValueError(
                f"Missing required RFM features: {missing_features}. "
                f"Available columns: {list(rfm_df.columns)}"
            )
    
    def _scale_features(self, rfm_features: pd.DataFrame) -> np.ndarray:
        """
        Scale RFM features based on the specified scaling method.
        
        Args:
            rfm_features: DataFrame with RFM features to scale
            
        Returns:
            Scaled features as numpy array
        """
        if self.scaling_method == 'standardize':
            if self.scaler_ is None:
                self.scaler_ = StandardScaler()
                scaled = self.scaler_.fit_transform(rfm_features)
            else:
                scaled = self.scaler_.transform(rfm_features)
        elif self.scaling_method == 'robust':
            if self.scaler_ is None:
                self.scaler_ = RobustScaler()
                scaled = self.scaler_.fit_transform(rfm_features)
            else:
                scaled = self.scaler_.transform(rfm_features)
        else:
            # No scaling
            scaled = rfm_features.values
        
        return scaled
    
    def fit(self, rfm_df: pd.DataFrame):
        """
        Fit the clustering model on RFM data.
        
        This method:
        1. Validates RFM data
        2. Extracts RFM features
        3. Scales features
        4. Fits K-Means clustering model
        
        Args:
            rfm_df: DataFrame with RFM metrics. Must contain columns:
                   - CustomerId (or similar identifier)
                   - recency, frequency, monetary (or specified rfm_features)
        
        Returns:
            self (for method chaining)
        
        Example:
            >>> rfm_df = pd.read_csv('data/processed/rfm_metrics.csv')
            >>> clusterer = CustomerClustering(n_clusters=3, random_state=42)
            >>> clusterer.fit(rfm_df)
        """
        # Validate input
        self._validate_rfm_data(rfm_df)
        
        # Store RFM data
        self.rfm_data_ = rfm_df.copy()
        
        # Extract RFM features
        rfm_features = rfm_df[self.rfm_features].copy()
        
        # Check for missing values
        if rfm_features.isnull().any().any():
            raise ValueError(
                "RFM features contain missing values. "
                "Please handle missing values before clustering."
            )
        
        # Scale features
        print(f"Scaling RFM features using: {self.scaling_method or 'no scaling'}")
        rfm_scaled = self._scale_features(rfm_features)
        self.rfm_scaled_ = rfm_scaled
        
        # Fit K-Means clustering
        print(f"Fitting K-Means clustering with {self.n_clusters} clusters...")
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,  # Number of times to run with different centroid seeds
            max_iter=300
        )
        
        self.cluster_labels_ = self.kmeans_.fit_predict(rfm_scaled)
        
        print(f"Clustering complete. Customers assigned to {self.n_clusters} clusters.")
        
        return self
    
    def predict(self, rfm_df: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new RFM data.
        
        Args:
            rfm_df: DataFrame with RFM metrics
            
        Returns:
            Array of cluster labels
            
        Raises:
            ValueError: If model has not been fitted yet
        """
        if self.kmeans_ is None or self.scaler_ is None:
            raise ValueError(
                "Clustering model has not been fitted yet. Call fit() first."
            )
        
        # Validate input
        self._validate_rfm_data(rfm_df)
        
        # Extract RFM features
        rfm_features = rfm_df[self.rfm_features].copy()
        
        # Scale features
        rfm_scaled = self._scale_features(rfm_features)
        
        # Predict clusters
        cluster_labels = self.kmeans_.predict(rfm_scaled)
        
        return cluster_labels
    
    def fit_predict(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit clustering model and add cluster labels to RFM DataFrame.
        
        Args:
            rfm_df: DataFrame with RFM metrics
            
        Returns:
            DataFrame with cluster labels added
        """
        self.fit(rfm_df)
        
        # Add cluster labels to RFM data
        rfm_with_clusters = self.rfm_data_.copy()
        rfm_with_clusters['cluster'] = self.cluster_labels_
        
        return rfm_with_clusters
    
    def get_cluster_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.
        
        Returns:
            DataFrame with cluster statistics including:
            - cluster_id: Cluster identifier
            - n_customers: Number of customers in cluster
            - pct_customers: Percentage of total customers
            - mean_recency, mean_frequency, mean_monetary: Mean RFM values
            - median_recency, median_frequency, median_monetary: Median RFM values
        
        Raises:
            ValueError: If model has not been fitted yet
        """
        if self.rfm_data_ is None or self.cluster_labels_ is None:
            raise ValueError(
                "Clustering model has not been fitted yet. Call fit() first."
            )
        
        cluster_stats = []
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels_ == cluster_id
            cluster_rfm = self.rfm_data_[cluster_mask]
            
            stats = {
                'cluster_id': cluster_id,
                'n_customers': cluster_mask.sum(),
                'pct_customers': (cluster_mask.sum() / len(self.rfm_data_)) * 100,
            }
            
            # Add mean and median for each RFM feature
            for feature in self.rfm_features:
                stats[f'mean_{feature}'] = cluster_rfm[feature].mean()
                stats[f'median_{feature}'] = cluster_rfm[feature].median()
                stats[f'std_{feature}'] = cluster_rfm[feature].std()
                stats[f'min_{feature}'] = cluster_rfm[feature].min()
                stats[f'max_{feature}'] = cluster_rfm[feature].max()
            
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)
    
    def get_cluster_centroids(self) -> pd.DataFrame:
        """
        Get cluster centroids in original (unscaled) RFM space.
        
        Returns:
            DataFrame with cluster centroids for each RFM feature
        
        Raises:
            ValueError: If model has not been fitted yet
        """
        if self.kmeans_ is None:
            raise ValueError(
                "Clustering model has not been fitted yet. Call fit() first."
            )
        
        # Get centroids in scaled space
        centroids_scaled = self.kmeans_.cluster_centers_
        
        # Inverse transform to original space
        if self.scaler_ is not None:
            centroids = self.scaler_.inverse_transform(centroids_scaled)
        else:
            centroids = centroids_scaled
        
        # Create DataFrame
        centroids_df = pd.DataFrame(
            centroids,
            columns=self.rfm_features,
            index=[f'Cluster_{i}' for i in range(self.n_clusters)]
        )
        
        return centroids_df
    
    def get_inertia(self) -> float:
        """
        Get K-Means inertia (sum of squared distances to centroids).
        
        Lower inertia indicates better clustering.
        
        Returns:
            Inertia value
        """
        if self.kmeans_ is None:
            raise ValueError(
                "Clustering model has not been fitted yet. Call fit() first."
            )
        
        return self.kmeans_.inertia_


def cluster_customers(
    rfm_df: pd.DataFrame,
    n_clusters: int = 3,
    scaling_method: str = 'standardize',
    random_state: int = 42,
    customer_col: str = 'CustomerId'
) -> Tuple[pd.DataFrame, CustomerClustering]:
    """
    Convenience function to cluster customers based on RFM metrics.
    
    Args:
        rfm_df: DataFrame with RFM metrics
        n_clusters: Number of clusters (default: 3)
        scaling_method: Scaling method ('standardize', 'robust', or None)
        random_state: Random state for reproducibility (default: 42)
        customer_col: Name of customer ID column
    
    Returns:
        Tuple of (rfm_df_with_clusters, clusterer)
        - rfm_df_with_clusters: RFM DataFrame with cluster labels added
        - clusterer: Fitted CustomerClustering object
    
    Example:
        >>> rfm_df = pd.read_csv('data/processed/rfm_metrics.csv')
        >>> rfm_with_clusters, clusterer = cluster_customers(rfm_df, n_clusters=3)
        >>> print(clusterer.get_cluster_statistics())
    """
    clusterer = CustomerClustering(
        n_clusters=n_clusters,
        scaling_method=scaling_method,
        random_state=random_state
    )
    
    rfm_with_clusters = clusterer.fit_predict(rfm_df)
    
    return rfm_with_clusters, clusterer


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 100)
    print("Customer Clustering - Step 2 of Proxy Target Variable Engineering")
    print("=" * 100)
    print()
    
    # Check if RFM data exists
    rfm_path = Path("data/processed/rfm_metrics.csv")
    
    if rfm_path.exists():
        print(f"Loading RFM metrics from: {rfm_path}")
        rfm_df = pd.read_csv(rfm_path)
        print(f"Loaded RFM metrics for {len(rfm_df)} customers")
        print()
        
        # Cluster customers
        rfm_with_clusters, clusterer = cluster_customers(
            rfm_df,
            n_clusters=3,
            scaling_method='standardize',
            random_state=42
        )
        
        print("\nCluster Statistics:")
        print(clusterer.get_cluster_statistics())
        print()
        
        print("Cluster Centroids (in original RFM space):")
        print(clusterer.get_cluster_centroids())
        print()
        
        print(f"Clustering Inertia: {clusterer.get_inertia():.2f}")
        print()
        
        print("=" * 100)
        print("Step 2 Complete: Customer Clustering Successful!")
        print("=" * 100)
    else:
        print(f"RFM metrics file not found: {rfm_path}")
        print("Please run Step 1 first to calculate RFM metrics.")
        print()
        print("Example usage:")
        print("""
        from src.customer_clustering import CustomerClustering
        
        # Load RFM metrics
        rfm_df = pd.read_csv('data/processed/rfm_metrics.csv')
        
        # Create clusterer
        clusterer = CustomerClustering(
            n_clusters=3,
            scaling_method='standardize',
            random_state=42
        )
        
        # Fit and predict
        rfm_with_clusters = clusterer.fit_predict(rfm_df)
        
        # Get statistics
        stats = clusterer.get_cluster_statistics()
        print(stats)
        """)

