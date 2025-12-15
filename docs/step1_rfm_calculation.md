# Step 1: Calculate RFM Metrics - Implementation Summary

## Overview

This document summarizes the implementation of **Step 1: Calculate RFM Metrics** for the Proxy Target Variable Engineering process.

## What Was Implemented

### 1. RFM Calculator Module (`src/rfm_calculator.py`)

A comprehensive module that calculates RFM (Recency, Frequency, Monetary) metrics for each customer from transaction history data.

#### Key Features:

- **Recency Calculation**: Days since last transaction (calculated from a snapshot date)
- **Frequency Calculation**: Total number of transactions per customer
- **Monetary Calculation**: Total transaction amount per customer (sum of all transaction amounts)
- **Snapshot Date**: Configurable reference date for consistent Recency calculation across all customers

### 2. RFM Metrics Explained

#### Recency
- **Definition**: Number of days between the snapshot date and the customer's last transaction
- **Interpretation**: Lower values indicate more recent activity (better engagement)
- **Calculation**: `(snapshot_date - last_transaction_date).days`

#### Frequency
- **Definition**: Total number of transactions per customer
- **Interpretation**: Higher values indicate more frequent activity (better engagement)
- **Calculation**: Count of transactions per customer

#### Monetary
- **Definition**: Total transaction amount per customer
- **Interpretation**: Higher values indicate higher spending (better engagement)
- **Calculation**: Sum of absolute values of all transaction amounts per customer
- **Note**: Uses absolute value to handle refunds (negative amounts)

### 3. Snapshot Date

The snapshot date is a reference point used to ensure consistent Recency calculation across all customers. 

- **If provided**: Uses the specified date
- **If not provided**: Automatically uses the maximum date in the transaction data

This ensures all customers are evaluated at the same point in time.

## Usage

### Basic Usage

```python
from src.rfm_calculator import RFMCalculator
from datetime import datetime

# Initialize calculator
calculator = RFMCalculator(
    customer_col='CustomerId',
    datetime_col='TransactionStartTime',
    amount_col='Amount',
    snapshot_date=datetime(2019, 2, 13)  # Optional
)

# Calculate RFM metrics
rfm_df = calculator.calculate_rfm(df)

# Get summary statistics
summary = calculator.get_summary_statistics()
```

### Convenience Function

```python
from src.rfm_calculator import calculate_rfm_metrics

# Quick calculation
rfm_df = calculate_rfm_metrics(
    df,
    snapshot_date=datetime(2019, 2, 13)
)
```

### Example Output

```
CustomerId  recency  frequency  monetary
C1              5          2       300
C2             20          1       150
C3             10          3       450
```

## Files Created

1. **`src/rfm_calculator.py`**: Main RFM calculator module
2. **`tests/test_rfm_calculator.py`**: Comprehensive unit tests
3. **`examples/step1_calculate_rfm.py`**: Example script demonstrating usage
4. **`docs/step1_rfm_calculation.md`**: This documentation

## Testing

Run the unit tests to verify the implementation:

```bash
pytest tests/test_rfm_calculator.py -v
```

## Example Script

Run the example script to see RFM calculation in action:

```bash
python examples/step1_calculate_rfm.py
```

## Next Steps

After reviewing Step 1, proceed to **Step 2: Clustering customers based on RFM metrics** to identify high-risk customer segments.

## Key Design Decisions

1. **Absolute Value for Monetary**: Uses `abs()` to handle refunds (negative amounts) consistently
2. **Snapshot Date Flexibility**: Allows manual specification or automatic detection
3. **Data Validation**: Validates required columns before processing
4. **Comprehensive Logging**: Provides informative output during calculation
5. **Summary Statistics**: Includes helper method for quick analysis

## Validation

The implementation has been validated with:
- Unit tests covering various scenarios
- Edge cases (negative amounts, missing columns, etc.)
- Multiple customers and transaction patterns
- Custom column names support

---

**Status**: ✅ Step 1 Complete - Ready for Review

