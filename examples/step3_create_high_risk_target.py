"""
Example: Step 3 - Create High-Risk Target Variable

This script demonstrates how to identify the high-risk customer cluster and create
a binary target variable for credit risk modeling.

This is Step 3 of the Proxy Target Variable Engineering process.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.high_risk_labeling import HighRiskLabeler, create_high_risk_target


def main():
    """Main function to demonstrate high-risk target creation."""
    
    print("=" * 100)
    print("Step 3: Create High-Risk Target Variable")
    print("=" * 100)
    print()
    
    # Load clustered RFM data from Step 2
    rfm_path = project_root / "data" / "processed" / "rfm_with_clusters.csv"
    
    if not rfm_path.exists():
        print(f"Clustered RFM data not found: {rfm_path}")
        print("Please run Step 2 first to cluster customers.")
        print("Run: python examples/step2_cluster_customers.py")
        return
    
    print(f"Loading clustered RFM data from: {rfm_path}")
    rfm_with_clusters = pd.read_csv(rfm_path)
    print(f"Loaded RFM data for {len(rfm_with_clusters):,} customers")
    print()
    
    # Create high-risk labeler
    print("Identifying high-risk cluster...")
    print("-" * 100)
    print()
    
    labeler = HighRiskLabeler()
    
    # Identify high-risk cluster
    high_risk_cluster_id = labeler.identify_high_risk_cluster(rfm_with_clusters)
    
    print(f"High-Risk Cluster ID: {high_risk_cluster_id}")
    print()
    
    # Get cluster analysis summary
    summary = labeler.get_high_risk_summary()
    
    print("Cluster Analysis:")
    print("-" * 100)
    display_cols = ['cluster_id', 'n_customers', 'pct_customers', 
                    'mean_recency', 'mean_frequency', 'mean_monetary', 
                    'engagement_score', 'is_high_risk']
    print(summary[display_cols].round(2))
    print()
    
    # Explain why this cluster is high-risk
    high_risk_stats = summary[summary['is_high_risk']].iloc[0]
    print("High-Risk Cluster Characteristics:")
    print(f"  - Recency: {high_risk_stats['mean_recency']:.1f} days (long time since last transaction)")
    print(f"  - Frequency: {high_risk_stats['mean_frequency']:.1f} transactions (low activity)")
    print(f"  - Monetary: ${high_risk_stats['mean_monetary']:,.2f} (low spending)")
    print(f"  - Engagement Score: {high_risk_stats['engagement_score']:.3f} (worst engagement)")
    print()
    
    # Create target variable
    print("Creating binary target variable...")
    print("-" * 100)
    print()
    
    rfm_with_target = labeler.create_target_variable(rfm_with_clusters)
    
    # Display target distribution
    print("Target Variable Distribution:")
    print("-" * 100)
    target_dist = rfm_with_target['is_high_risk'].value_counts().sort_index()
    target_pct = rfm_with_target['is_high_risk'].value_counts(normalize=True).sort_index() * 100
    
    for target_value in sorted(target_dist.index):
        count = target_dist[target_value]
        pct = target_pct[target_value]
        label = "High-Risk" if target_value == 1 else "Low-Risk"
        print(f"  {label} ({target_value}): {count:,} customers ({pct:.1f}%)")
    print()
    
    # Display sample customers
    print("Sample High-Risk Customers:")
    print("-" * 100)
    high_risk_sample = rfm_with_target[rfm_with_target['is_high_risk'] == 1].head(10)
    print(high_risk_sample[['CustomerId', 'recency', 'frequency', 'monetary', 'cluster', 'is_high_risk']])
    print()
    
    print("Sample Low-Risk Customers:")
    print("-" * 100)
    low_risk_sample = rfm_with_target[rfm_with_target['is_high_risk'] == 0].head(10)
    print(low_risk_sample[['CustomerId', 'recency', 'frequency', 'monetary', 'cluster', 'is_high_risk']])
    print()
    
    # Save RFM data with target
    output_path = project_root / "data" / "processed" / "rfm_with_target.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rfm_with_target.to_csv(output_path, index=False)
    print(f"RFM data with target saved to: {output_path}")
    print()
    
    # Optionally add target to transaction data
    transactions_path = project_root / "data" / "raw" / "data.csv"
    if transactions_path.exists():
        print(f"Loading transaction data from: {transactions_path}")
        transactions_df = pd.read_csv(transactions_path)
        print(f"Loaded {len(transactions_df):,} transactions")
        print()
        
        print("Adding target variable to transaction data...")
        transactions_with_target = labeler.add_target_to_transactions(
            transactions_df,
            rfm_with_target,
            customer_col='CustomerId'
        )
        
        # Save transactions with target
        transactions_output = project_root / "data" / "processed" / "transactions_with_target.csv"
        transactions_with_target.to_csv(transactions_output, index=False)
        print(f"Transactions with target saved to: {transactions_output}")
        print()
        
        # Display target distribution in transactions
        print("Target Distribution in Transactions:")
        print("-" * 100)
        trans_target_dist = transactions_with_target['is_high_risk'].value_counts().sort_index()
        trans_target_pct = transactions_with_target['is_high_risk'].value_counts(normalize=True).sort_index() * 100
        
        for target_value in sorted(trans_target_dist.index):
            count = trans_target_dist[target_value]
            pct = trans_target_pct[target_value]
            label = "High-Risk" if target_value == 1 else "Low-Risk"
            print(f"  {label} ({target_value}): {count:,} transactions ({pct:.1f}%)")
        print()
    
    print("=" * 100)
    print("Step 3 Complete: High-Risk Target Variable Created Successfully!")
    print("=" * 100)
    print()
    print("Summary:")
    print(f"  - High-Risk Cluster: {high_risk_cluster_id}")
    print(f"  - High-Risk Customers: {target_dist.get(1, 0):,} ({target_pct.get(1, 0):.1f}%)")
    print(f"  - Low-Risk Customers: {target_dist.get(0, 0):,} ({target_pct.get(0, 0):.1f}%)")
    print()
    print("Next Steps:")
    print("  1. Review the target variable distribution")
    print("  2. Use this target variable for model training (Task 5)")
    print()


if __name__ == "__main__":
    main()

