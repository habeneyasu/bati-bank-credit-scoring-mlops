"""
Example: Step 2 - Cluster Customers using K-Means

This script demonstrates how to cluster customers into distinct groups based on
their RFM (Recency, Frequency, Monetary) profiles.

This is Step 2 of the Proxy Target Variable Engineering process.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.customer_clustering import CustomerClustering, cluster_customers


def main():
    """Main function to demonstrate customer clustering."""
    
    print("=" * 100)
    print("Step 2: Cluster Customers using K-Means")
    print("=" * 100)
    print()
    
    # Load RFM metrics from Step 1
    rfm_path = project_root / "data" / "processed" / "rfm_metrics.csv"
    
    if not rfm_path.exists():
        print(f"RFM metrics file not found: {rfm_path}")
        print("Please run Step 1 first to calculate RFM metrics.")
        print("Run: python examples/step1_calculate_rfm.py")
        return
    
    print(f"Loading RFM metrics from: {rfm_path}")
    rfm_df = pd.read_csv(rfm_path)
    print(f"Loaded RFM metrics for {len(rfm_df):,} customers")
    print()
    
    # Display RFM data summary
    print("RFM Metrics Summary:")
    print(rfm_df[['recency', 'frequency', 'monetary']].describe())
    print()
    
    # Cluster customers
    print("Clustering customers...")
    print("-" * 100)
    print()
    
    rfm_with_clusters, clusterer = cluster_customers(
        rfm_df,
        n_clusters=3,
        scaling_method='standardize',  # Use StandardScaler for scaling
        random_state=42  # For reproducibility
    )
    
    print()
    print("=" * 100)
    print("Clustering Results")
    print("=" * 100)
    print()
    
    # Display cluster statistics
    print("Cluster Statistics:")
    print("-" * 100)
    cluster_stats = clusterer.get_cluster_statistics()
    print(cluster_stats.round(2))
    print()
    
    # Display cluster centroids
    print("Cluster Centroids (in original RFM space):")
    print("-" * 100)
    centroids = clusterer.get_cluster_centroids()
    print(centroids.round(2))
    print()
    
    # Display cluster distribution
    print("Cluster Distribution:")
    print("-" * 100)
    cluster_counts = rfm_with_clusters['cluster'].value_counts().sort_index()
    cluster_pct = rfm_with_clusters['cluster'].value_counts(normalize=True).sort_index() * 100
    
    for cluster_id in sorted(rfm_with_clusters['cluster'].unique()):
        count = cluster_counts[cluster_id]
        pct = cluster_pct[cluster_id]
        print(f"  Cluster {cluster_id}: {count:,} customers ({pct:.1f}%)")
    print()
    
    # Display clustering quality metric
    print(f"Clustering Inertia: {clusterer.get_inertia():.2f}")
    print("(Lower inertia indicates tighter clusters)")
    print()
    
    # Display sample customers from each cluster
    print("Sample Customers from Each Cluster:")
    print("-" * 100)
    for cluster_id in sorted(rfm_with_clusters['cluster'].unique()):
        cluster_sample = rfm_with_clusters[
            rfm_with_clusters['cluster'] == cluster_id
        ].head(5)
        
        print(f"\nCluster {cluster_id} (sample of 5 customers):")
        print(cluster_sample[['CustomerId', 'recency', 'frequency', 'monetary', 'cluster']])
    print()
    
    # Save clustered RFM data
    output_path = project_root / "data" / "processed" / "rfm_with_clusters.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rfm_with_clusters.to_csv(output_path, index=False)
    print(f"Clustered RFM data saved to: {output_path}")
    print()
    
    print("=" * 100)
    print("Step 2 Complete: Customer Clustering Successful!")
    print("=" * 100)
    print()
    print("Next Steps:")
    print("  1. Review cluster characteristics to understand customer segments")
    print("  2. Proceed to Step 3: Identify high-risk cluster and create target variable")
    print()


if __name__ == "__main__":
    main()

