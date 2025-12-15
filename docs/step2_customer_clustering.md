# Step 2: Cluster Customers using K-Means - Implementation Summary

## Overview

This document summarizes the implementation of **Step 2: Cluster Customers using K-Means** for the Proxy Target Variable Engineering process.

## What Was Implemented

### 1. Customer Clustering Module (`src/customer_clustering.py`)

A comprehensive module that clusters customers into distinct groups based on their RFM (Recency, Frequency, Monetary) profiles using K-Means clustering.

#### Key Features:

- **Feature Scaling**: Pre-processes RFM features using StandardScaler or RobustScaler
- **K-Means Clustering**: Segments customers into 3 distinct groups (configurable)
- **Reproducibility**: Uses random_state to ensure consistent results
- **Cluster Analysis**: Provides statistics and centroids for each cluster
- **Quality Metrics**: Calculates inertia to assess clustering quality

### 2. Feature Scaling

Before clustering, RFM features are scaled to ensure all features contribute equally to the clustering process:

- **StandardScaler** (default): Standardizes features to mean=0, std=1
  - Good for normally distributed data
  - Sensitive to outliers
  
- **RobustScaler**: Uses median and IQR (Interquartile Range)
  - Robust to outliers
  - Better for skewed data with extreme values
  
- **No Scaling**: Can be disabled if features are already on similar scales

### 3. K-Means Clustering

- **Algorithm**: K-Means clustering from scikit-learn
- **Number of Clusters**: 3 (configurable)
- **Random State**: Set to 42 for reproducibility
- **Initialization**: Uses 10 different centroid seeds (n_init=10)
- **Max Iterations**: 300 iterations maximum

### 4. Cluster Analysis

The module provides comprehensive cluster analysis:

- **Cluster Statistics**: Mean, median, std, min, max for each RFM feature per cluster
- **Cluster Centroids**: Representative RFM values for each cluster
- **Cluster Distribution**: Number and percentage of customers per cluster
- **Inertia**: Sum of squared distances to centroids (lower is better)

## Usage

### Basic Usage

```python
from src.customer_clustering import CustomerClustering

# Initialize clusterer
clusterer = CustomerClustering(
    n_clusters=3,
    scaling_method='standardize',
    random_state=42
)

# Load RFM metrics from Step 1
rfm_df = pd.read_csv('data/processed/rfm_metrics.csv')

# Fit and predict clusters
rfm_with_clusters = clusterer.fit_predict(rfm_df)

# Get cluster statistics
stats = clusterer.get_cluster_statistics()
print(stats)
```

### Convenience Function

```python
from src.customer_clustering import cluster_customers

# Quick clustering
rfm_with_clusters, clusterer = cluster_customers(
    rfm_df,
    n_clusters=3,
    scaling_method='standardize',
    random_state=42
)
```

### Example Output

```
Cluster Statistics:
   cluster_id  n_customers  pct_customers  mean_recency  mean_frequency  mean_monetary
0           0         1248           33.4          5.2            45.3        250000.0
1           1         1247           33.3         25.1            12.5         50000.0
2           2         1247           33.3         60.8             3.2          5000.0
```

## Files Created

1. **`src/customer_clustering.py`**: Main clustering module
2. **`tests/test_customer_clustering.py`**: Comprehensive unit tests
3. **`examples/step2_cluster_customers.py`**: Example script demonstrating usage
4. **`docs/step2_customer_clustering.md`**: This documentation

## Testing

Run the unit tests to verify the implementation:

```bash
pytest tests/test_customer_clustering.py -v
```

## Example Script

Run the example script to see clustering in action:

```bash
python examples/step2_cluster_customers.py
```

## Key Design Decisions

1. **Feature Scaling**: StandardScaler by default to ensure all RFM features contribute equally
2. **Random State**: Set to 42 for reproducibility across runs
3. **3 Clusters**: Default number aligns with typical customer segmentation (high/medium/low engagement)
4. **Robust Scaling Option**: Available for datasets with extreme outliers
5. **Comprehensive Analysis**: Provides detailed statistics for understanding clusters

## Cluster Interpretation

After clustering, you can interpret clusters based on their RFM characteristics:

- **High Engagement Cluster**: Low recency, high frequency, high monetary
- **Medium Engagement Cluster**: Moderate recency, frequency, and monetary
- **Low Engagement Cluster**: High recency, low frequency, low monetary (likely high-risk)

## Next Steps

After reviewing Step 2, proceed to **Step 3: Identify High-Risk Cluster** to:
1. Analyze cluster characteristics
2. Identify the least engaged cluster (high-risk)
3. Create binary `is_high_risk` target variable

## Validation

The implementation has been validated with:
- Unit tests covering various scenarios
- Different scaling methods
- Reproducibility checks
- Cluster statistics validation
- Edge cases (missing values, missing features, etc.)

---

**Status**: ✅ Step 2 Complete - Ready for Review

