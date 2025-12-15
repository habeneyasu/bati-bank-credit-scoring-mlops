# API Input Features Documentation

The `/predict` endpoint requires **exactly 26 features** in a specific order. This document explains what each feature represents.

---

## Feature List (In Order)

### 1-5: Original Transaction Features

| # | Feature Name | Type | Description | Example Value |
|---|--------------|------|-------------|---------------|
| 1 | `CountryCode` | Numerical | Numerical geographical code of the country | 231 (Ethiopia) |
| 2 | `Amount` | Numerical | Transaction amount (positive for debits, negative for credits) | 1000.0 |
| 3 | `Value` | Numerical | Absolute value of the transaction amount | 1000.0 |
| 4 | `PricingStrategy` | Numerical | Category of pricing structure (1 or 2) | 1 |
| 5 | `FraudResult` | Binary | Fraud status (1 = fraud detected, 0 = no fraud) | 0 |

### 6-11: Temporal Features (Extracted from TransactionStartTime)

| # | Feature Name | Type | Description | Range |
|---|--------------|------|-------------|-------|
| 6 | `transaction_hour` | Numerical | Hour of day when transaction occurred | 0-23 |
| 7 | `transaction_day` | Numerical | Day of month when transaction occurred | 1-31 |
| 8 | `transaction_month` | Numerical | Month when transaction occurred | 1-12 |
| 9 | `transaction_year` | Numerical | Year when transaction occurred | 2018, 2019, etc. |
| 10 | `transaction_dayofweek` | Numerical | Day of week (0=Monday, 6=Sunday) | 0-6 |
| 11 | `transaction_week` | Numerical | Week number of the year | 1-52 |

### 12-18: Customer Aggregate Features

These are calculated per customer from their transaction history:

| # | Feature Name | Type | Description |
|---|--------------|------|-------------|
| 12 | `total_transaction_amount` | Numerical | Sum of all transaction amounts for the customer |
| 13 | `avg_transaction_amount` | Numerical | Average transaction amount for the customer |
| 14 | `transaction_count` | Numerical | Total number of transactions for the customer |
| 15 | `std_transaction_amount` | Numerical | Standard deviation of transaction amounts (variability) |
| 16 | `min_transaction_amount` | Numerical | Minimum transaction amount for the customer |
| 17 | `max_transaction_amount` | Numerical | Maximum transaction amount for the customer |
| 18 | `median_transaction_amount` | Numerical | Median transaction amount for the customer |

### 19-26: Product Category Features (One-Hot Encoded)

These are binary features (0 or 1) indicating which product category the transaction belongs to:

| # | Feature Name | Type | Description |
|---|--------------|------|-------------|
| 19 | `ProductCategory_data_bundles` | Binary | 1 if transaction is for data bundles, else 0 |
| 20 | `ProductCategory_financial_services` | Binary | 1 if transaction is for financial services, else 0 |
| 21 | `ProductCategory_movies` | Binary | 1 if transaction is for movies, else 0 |
| 22 | `ProductCategory_other` | Binary | 1 if transaction is in other category, else 0 |
| 23 | `ProductCategory_ticket` | Binary | 1 if transaction is for tickets, else 0 |
| 24 | `ProductCategory_transport` | Binary | 1 if transaction is for transport, else 0 |
| 25 | `ProductCategory_tv` | Binary | 1 if transaction is for TV, else 0 |
| 26 | `ProductCategory_utility_bill` | Binary | 1 if transaction is for utility bills, else 0 |

**Note**: For one-hot encoded features, only one should be 1 (the category the transaction belongs to), and all others should be 0.

---

## Important Notes

### Feature Scaling

**All features should be in their processed/scaled form**, not raw values. The model expects:
- Numerical features to be standardized (mean=0, std=1) or scaled
- Categorical features to be one-hot encoded (0 or 1)
- Temporal features as extracted integers

### Feature Order

**The order matters!** Features must be provided in the exact order shown above (1-26).

### Getting Real Feature Values

To get properly processed feature values from your data:

```python
import pandas as pd
import requests

# Load processed data
df = pd.read_csv('data/processed/processed_data_with_target.csv')

# Get features (exclude target)
features = df.drop(columns=['is_high_risk']).iloc[0].values.tolist()

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": features}
)
print(response.json())
```

---

## Example Request

```json
{
  "features": [
    231,           // CountryCode
    1000.0,        // Amount
    1000.0,        // Value
    1,             // PricingStrategy
    0,             // FraudResult
    14,            // transaction_hour
    15,            // transaction_day
    12,            // transaction_month
    2018,          // transaction_year
    2,             // transaction_dayofweek (Wednesday)
    50,            // transaction_week
    5000.0,        // total_transaction_amount
    1000.0,        // avg_transaction_amount
    5,             // transaction_count
    200.0,         // std_transaction_amount
    500.0,         // min_transaction_amount
    1500.0,        // max_transaction_amount
    1000.0,        // median_transaction_amount
    0,             // ProductCategory_data_bundles
    1,             // ProductCategory_financial_services
    0,             // ProductCategory_movies
    0,             // ProductCategory_other
    0,             // ProductCategory_ticket
    0,             // ProductCategory_transport
    0,             // ProductCategory_tv
    0              // ProductCategory_utility_bill
  ]
}
```

---

## Feature Engineering Pipeline

These features are created through the following pipeline:

1. **Temporal Extraction**: Extract hour, day, month, year, day of week, week from `TransactionStartTime`
2. **Customer Aggregation**: Calculate statistics (sum, mean, count, std, min, max, median) per customer
3. **Categorical Encoding**: One-hot encode `ProductCategory` into 8 binary features
4. **Scaling**: Standardize/normalize numerical features

The processed features are stored in `data/processed/processed_data_with_target.csv`.

---

## Validation

The API validates that:
- Exactly 26 features are provided
- All features are numeric (float)
- Features are in the correct order

If validation fails, you'll receive a 400 error with details about what's wrong.

---

For more information, see:
- `src/data_processing.py` - Feature engineering implementation
- `data/processed/processed_data_with_target.csv` - Example processed data
- `src/api/pydantic_models.py` - API request/response models

