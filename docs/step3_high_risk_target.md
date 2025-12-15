# Step 3: Create High-Risk Target Variable - Implementation Summary

## Overview

This document summarizes the implementation of **Step 3: Create High-Risk Target Variable** for the Proxy Target Variable Engineering process.

## What Was Implemented

### 1. High-Risk Labeling Module (`src/high_risk_labeling.py`)

A comprehensive module that identifies the least engaged (highest-risk) customer cluster and creates a binary target variable for credit risk modeling.

#### Key Features:

- **Automatic Cluster Analysis**: Analyzes cluster characteristics to identify high-risk segment
- **Engagement Score Calculation**: Combines normalized RFM metrics to score engagement
- **Binary Target Variable**: Creates `is_high_risk` column (1 for high-risk, 0 for low-risk)
- **Transaction-Level Integration**: Adds target to both RFM and transaction data

### 2. High-Risk Cluster Identification

The module automatically identifies the high-risk cluster based on engagement characteristics:

- **High Recency**: Long time since last transaction (worse engagement)
- **Low Frequency**: Few transactions (worse engagement)
- **Low Monetary**: Low total spending (worse engagement)

The cluster with the **highest engagement score** (worst engagement) is identified as high-risk.

### 3. Engagement Score Calculation

Engagement score is calculated by:

1. Normalizing each RFM metric to [0, 1] range
2. For Recency: Higher values = higher score (worse)
3. For Frequency/Monetary: Lower values = higher score (worse)
4. Summing normalized scores: Higher total = worse engagement = higher risk

### 4. Binary Target Variable

- **`is_high_risk = 1`**: Customers in the high-risk cluster
- **`is_high_risk = 0`**: Customers in all other clusters

## Usage

### Basic Usage

```python
from src.high_risk_labeling import HighRiskLabeler

# Initialize labeler
labeler = HighRiskLabeler()

# Load clustered RFM data
rfm_with_clusters = pd.read_csv('data/processed/rfm_with_clusters.csv')

# Identify high-risk cluster
high_risk_cluster_id = labeler.identify_high_risk_cluster(rfm_with_clusters)
print(f"High-Risk Cluster: {high_risk_cluster_id}")

# Create target variable
rfm_with_target = labeler.create_target_variable(rfm_with_clusters)

# Get summary
summary = labeler.get_high_risk_summary()
print(summary)
```

### Convenience Function

```python
from src.high_risk_labeling import create_high_risk_target

# Quick target creation
rfm_with_target, transactions_with_target, summary = create_high_risk_target(
    rfm_with_clusters,
    transactions_df=transactions_df,  # Optional
    return_summary=True
)
```

### Example Output

```
High-Risk Cluster ID: 1

Target Variable Distribution:
  High-Risk (1): 1,428 customers (38.2%)
  Low-Risk (0):  2,314 customers (61.8%)
```

## Files Created

1. **`src/high_risk_labeling.py`**: Main high-risk labeling module
2. **`tests/test_high_risk_labeling.py`**: Comprehensive unit tests
3. **`examples/step3_create_high_risk_target.py`**: Example script demonstrating usage
4. **`docs/step3_high_risk_target.md`**: This documentation

## Testing

Run the unit tests to verify the implementation:

```bash
pytest tests/test_high_risk_labeling.py -v
```

## Example Script

Run the example script to see high-risk target creation in action:

```bash
python examples/step3_create_high_risk_target.py
```

## Key Design Decisions

1. **Automatic Identification**: Automatically identifies high-risk cluster based on engagement score
2. **Engagement Score**: Combines all RFM metrics for comprehensive risk assessment
3. **Binary Target**: Simple binary classification (1 = high-risk, 0 = low-risk)
4. **Flexible Integration**: Can add target to both RFM and transaction-level data
5. **Comprehensive Analysis**: Provides detailed cluster analysis and summary

## Target Variable Interpretation

- **`is_high_risk = 1`**: Customers in the least engaged cluster
  - High recency (long time since last transaction)
  - Low frequency (few transactions)
  - Low monetary value (low spending)
  - **Interpretation**: Higher likelihood of default/credit risk

- **`is_high_risk = 0`**: Customers in other clusters
  - Better engagement patterns
  - **Interpretation**: Lower likelihood of default/credit risk

## Output Files

After running Step 3, the following files are created:

1. **`data/processed/rfm_with_target.csv`**: RFM metrics with `is_high_risk` target
2. **`data/processed/transactions_with_target.csv`**: Transaction data with `is_high_risk` target (if transaction data provided)

## Next Steps

After completing Step 3, you have:

✅ **RFM Metrics** calculated for each customer  
✅ **Customer Clusters** identified (3 distinct groups)  
✅ **High-Risk Target Variable** created (`is_high_risk`)

You can now proceed to **Task 5: Model Training** using this target variable to build credit risk prediction models.

## Validation

The implementation has been validated with:
- Unit tests covering various scenarios
- Automatic high-risk cluster identification
- Target variable creation and distribution
- Integration with transaction data
- Edge cases and error handling

---

**Status**: ✅ Step 3 Complete - Ready for Review

