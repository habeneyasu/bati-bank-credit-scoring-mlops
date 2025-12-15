# Bati Bank Credit Scoring MLOps

An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

---

## Task 1: Credit Scoring Business Understanding

This section addresses three critical questions about credit risk modeling in a regulated financial environment.

### 1. Basel II Accord's Emphasis on Risk Measurement and Model Interpretability

The Basel II Capital Accord fundamentally transformed banking regulation by introducing a risk-based approach to capital adequacy. Under Basel II, banks are required to:

- **Quantify credit risk** with sufficient precision to determine appropriate capital reserves
- **Document model methodologies** comprehensively for regulatory review and audit
- **Demonstrate model validation** through backtesting, stress testing, and ongoing monitoring
- **Maintain transparency** in risk assessment processes to satisfy both regulators and internal risk management

This regulatory framework directly influences our model development approach:

**Interpretability Requirement**: Basel II mandates that banks understand and explain their risk models. An interpretable model (e.g., Logistic Regression with Weight of Evidence transformations) allows risk officers, auditors, and regulators to:
- Trace how individual features contribute to risk predictions
- Validate that model behavior aligns with business logic and domain expertise
- Identify potential biases or discriminatory patterns
- Explain credit decisions to customers and stakeholders

**Documentation Necessity**: Comprehensive documentation is not optional—it's a regulatory requirement. Our model must include:
- Clear feature definitions and transformations
- Model performance metrics and validation results
- Documentation of assumptions and limitations
- Procedures for model monitoring and recalibration

**Capital Adequacy Implications**: The accuracy and reliability of our risk probability estimates directly impact Bati Bank's capital allocation. Overestimating risk leads to excessive capital reserves (reducing profitability), while underestimating risk exposes the bank to regulatory penalties and financial losses.

### 2. Proxy Variable Necessity and Associated Business Risks

**Why Proxy Variables Are Necessary**:

In this project, we lack direct historical default data because:
- The eCommerce partner is a new entity without established credit history
- Traditional credit bureau data may not be available for all customers
- The buy-now-pay-later service is novel, so historical performance data doesn't exist

We must create a **proxy variable** using available behavioral data (RFM patterns—Recency, Frequency, Monetary) to approximate credit risk. This proxy might be defined as:
- **High Risk (Bad)**: Customers with patterns indicating potential default (e.g., irregular payment behavior, declining transaction frequency, increasing transaction amounts relative to historical patterns)
- **Low Risk (Good)**: Customers with stable, positive behavioral patterns

**Potential Business Risks of Proxy-Based Predictions**:

1. **Model Risk**: The proxy may not accurately reflect true credit risk. RFM patterns from eCommerce transactions may correlate with creditworthiness, but the relationship is indirect and may break down under different economic conditions or customer lifecycle stages.

2. **Concept Drift**: Customer behavior in eCommerce may change over time, causing the proxy relationship to degrade. A model trained on pre-pandemic behavior may not generalize to post-pandemic patterns.

3. **Selection Bias**: The proxy definition itself introduces bias. If we define "bad" customers too narrowly, we may miss emerging risk patterns. If defined too broadly, we may reject creditworthy customers.

4. **Regulatory Scrutiny**: Regulators may question the validity of proxy-based models, especially if they cannot be validated against actual default outcomes. This could lead to:
   - Rejection of the model for regulatory capital calculations
   - Requirement for additional validation studies
   - Mandatory conservative adjustments to risk estimates

5. **Business Impact**: Incorrect proxy definitions can lead to:
   - **Type I Errors**: Rejecting creditworthy customers (lost revenue, customer dissatisfaction)
   - **Type II Errors**: Approving high-risk customers (increased defaults, financial losses)

6. **Limited Validation**: Without true default labels, we cannot directly validate model performance on the actual outcome of interest. We must rely on indirect validation methods, which are inherently less reliable.

**Mitigation Strategies**:
- Use multiple validation techniques (cross-validation, temporal validation, business logic checks)
- Implement conservative risk estimates initially, then refine based on observed outcomes
- Establish continuous monitoring to detect when proxy relationships degrade
- Document all assumptions and limitations transparently

### 3. Trade-offs: Simple Interpretable Models vs. Complex High-Performance Models

In a regulated financial context, the choice between Logistic Regression with WoE (Weight of Evidence) and Gradient Boosting involves critical trade-offs:

#### Logistic Regression with WoE - Advantages:
- **Regulatory Compliance**: Highly interpretable—each feature's contribution is transparent and explainable
- **Audit Trail**: Clear documentation of how each variable affects the risk score
- **Business Alignment**: WoE transformations align with credit scoring industry standards
- **Stability**: Less prone to overfitting, more stable predictions across different data distributions
- **Regulatory Acceptance**: Well-understood by regulators and risk officers, easier to gain approval
- **Feature Engineering Discipline**: WoE requires careful binning and transformation, enforcing domain expertise

#### Logistic Regression with WoE - Disadvantages:
- **Limited Complexity**: Cannot capture complex non-linear interactions automatically
- **Lower Performance**: May achieve lower AUC/accuracy compared to ensemble methods
- **Manual Feature Engineering**: Requires significant domain expertise and time investment
- **Limited Scalability**: Adding many features can become unwieldy

#### Gradient Boosting (XGBoost/LightGBM) - Advantages:
- **Higher Performance**: Typically achieves superior predictive accuracy and AUC scores
- **Automatic Feature Interactions**: Captures complex non-linear relationships without explicit engineering
- **Handles Non-linearity**: Better at modeling complex patterns in data
- **Feature Importance**: Provides feature importance metrics (though less interpretable than coefficients)

#### Gradient Boosting - Disadvantages:
- **Black Box Nature**: Difficult to explain individual predictions or feature contributions
- **Regulatory Challenges**: May face resistance from regulators who require model interpretability
- **Overfitting Risk**: More prone to overfitting, especially with limited data
- **Validation Complexity**: Harder to validate that model behavior aligns with business logic
- **Bias Detection**: More difficult to identify and mitigate potential discriminatory patterns

#### Key Trade-offs in Regulated Context:

1. **Regulatory Approval**: 
   - Simple models: Easier to gain regulatory approval, faster time-to-market
   - Complex models: May require extensive documentation, additional validation, potential rejection

2. **Business Trust**:
   - Simple models: Risk officers and business stakeholders can understand and trust the model
   - Complex models: May face skepticism and require extensive education

3. **Performance vs. Interpretability**:
   - Simple models: Lower performance but higher trust and compliance
   - Complex models: Higher performance but potential regulatory and trust issues

4. **Operational Risk**:
   - Simple models: Easier to monitor, debug, and maintain
   - Complex models: Harder to diagnose issues, more complex monitoring requirements

5. **Customer Communication**:
   - Simple models: Can explain credit decisions to customers ("Your score was low due to X, Y, Z")
   - Complex models: Difficult to provide meaningful explanations

#### Recommended Approach for This Project:

Given the regulatory context and business requirements, a **hybrid approach** may be optimal:

1. **Primary Model**: Start with Logistic Regression + WoE for regulatory compliance and interpretability
2. **Validation Model**: Use Gradient Boosting as a benchmark to validate that the simpler model isn't missing critical patterns
3. **Ensemble Consideration**: If performance gap is significant, consider a carefully documented ensemble with clear interpretability components
4. **Documentation**: Regardless of model choice, maintain comprehensive documentation of:
   - Model selection rationale
   - Performance metrics
   - Interpretability analysis
   - Validation procedures
   - Limitations and assumptions

**Conclusion**: In regulated financial contexts, interpretability and regulatory compliance often outweigh marginal performance gains. However, the optimal choice depends on the specific regulatory environment, business requirements, and the magnitude of the performance difference between models.

---

## Task 4: Proxy Target Variable Engineering

**Objective**: Create a credit risk target variable since there is no pre-existing "credit risk" column in the data. Programmatically identify "disengaged" customers and label them as high-risk proxies.

### Overview

Since direct default labels are unavailable, we use RFM (Recency, Frequency, Monetary) analysis to create a proxy target variable. This approach identifies high-risk customers based on behavioral patterns that correlate with credit risk.

### Implementation Steps

#### Step 1: Calculate RFM Metrics

For each `CustomerId`, calculate their Recency, Frequency, and Monetary (RFM) values from transaction history.

**Key Features:**
- **Recency**: Days since last transaction (calculated from a snapshot date)
- **Frequency**: Total number of transactions per customer
- **Monetary**: Total transaction amount per customer (sum of all transaction amounts)
- **Snapshot Date**: Configurable reference date for consistent Recency calculation

**Implementation:**
- Module: `src/rfm_calculator.py`
- Example: `examples/step1_calculate_rfm.py`
- Documentation: `docs/step1_rfm_calculation.md`

**Usage:**
```python
from src.rfm_calculator import RFMCalculator

calculator = RFMCalculator(
    customer_col='CustomerId',
    datetime_col='TransactionStartTime',
    amount_col='Amount',
    snapshot_date='2024-12-31'  # Optional, defaults to max date
)

rfm_metrics = calculator.calculate_rfm(transactions_df)
```

**Output:**
- `data/processed/rfm_metrics.csv`: RFM metrics for each customer

#### Step 2: Cluster Customers using K-Means

Use K-Means clustering algorithm to segment customers into 3 distinct groups based on their RFM profiles.

**Key Features:**
- **Feature Scaling**: Pre-processes RFM features using StandardScaler or RobustScaler
- **K-Means Clustering**: Segments customers into 3 distinct groups (configurable)
- **Reproducibility**: Uses `random_state=42` to ensure consistent results
- **Cluster Analysis**: Provides statistics and centroids for each cluster

**Implementation:**
- Module: `src/customer_clustering.py`
- Example: `examples/step2_cluster_customers.py`
- Documentation: `docs/step2_customer_clustering.md`

**Usage:**
```python
from src.customer_clustering import CustomerClustering

clustering = CustomerClustering(
    n_clusters=3,
    scaling_method='standardize',  # 'standardize', 'robust', or None
    random_state=42
)

rfm_with_clusters = clustering.cluster_customers(rfm_metrics)
```

**Output:**
- `data/processed/rfm_with_clusters.csv`: RFM metrics with cluster assignments

#### Step 3: Define and Assign High-Risk Label

Analyze the resulting clusters to determine which one represents the least engaged (highest-risk) customer segment and create a binary target variable.

**Key Features:**
- **Automatic Cluster Analysis**: Analyzes cluster characteristics to identify high-risk segment
- **Engagement Score**: Combines normalized RFM metrics to score engagement
- **Binary Target Variable**: Creates `is_high_risk` column (1 for high-risk, 0 for low-risk)
- **High-Risk Identification**: Cluster with highest recency, lowest frequency, and lowest monetary value

**Implementation:**
- Module: `src/high_risk_labeling.py`
- Example: `examples/step3_create_high_risk_target.py`
- Documentation: `docs/step3_high_risk_target.md`

**Usage:**
```python
from src.high_risk_labeling import HighRiskLabeler

labeler = HighRiskLabeler()

# Identify high-risk cluster
high_risk_cluster_id = labeler.identify_high_risk_cluster(rfm_with_clusters)

# Create target variable
rfm_with_target = labeler.create_target_variable(rfm_with_clusters)
```

**Output:**
- `data/processed/rfm_with_target.csv`: RFM metrics with `is_high_risk` target
- `data/processed/transactions_with_target.csv`: Transaction data with `is_high_risk` target

#### Step 4: Integrate Target Variable

Merge the `is_high_risk` column back into the main processed dataset for model training.

**Implementation:**
- Script: `examples/integrate_target_to_processed_data.py`

**Usage:**
```python
python examples/integrate_target_to_processed_data.py
```

**Output:**
- `data/processed/processed_data_with_target.csv`: Feature-engineered data with `is_high_risk` target

### Target Variable Interpretation

- **`is_high_risk = 1`**: Customers in the least engaged cluster
  - High recency (long time since last transaction)
  - Low frequency (few transactions)
  - Low monetary value (low spending)
  - **Interpretation**: Higher likelihood of default/credit risk

- **`is_high_risk = 0`**: Customers in other clusters
  - Better engagement patterns
  - **Interpretation**: Lower likelihood of default/credit risk

### Testing

All modules include comprehensive unit tests:

```bash
# Test RFM calculation
pytest tests/test_rfm_calculator.py -v

# Test customer clustering
pytest tests/test_customer_clustering.py -v

# Test high-risk labeling
pytest tests/test_high_risk_labeling.py -v
```

### Complete Workflow

Run all steps sequentially:

```bash
# Step 1: Calculate RFM metrics
python examples/step1_calculate_rfm.py

# Step 2: Cluster customers
python examples/step2_cluster_customers.py

# Step 3: Create high-risk target
python examples/step3_create_high_risk_target.py

# Step 4: Integrate target with processed data
python examples/integrate_target_to_processed_data.py
```

### Key Design Decisions

1. **RFM Analysis**: Industry-standard approach for customer segmentation and risk assessment
2. **K-Means Clustering**: Unsupervised learning to identify natural customer segments
3. **Automatic High-Risk Identification**: Data-driven approach based on engagement scores
4. **Binary Classification**: Simple binary target for model training
5. **Reproducibility**: All steps use fixed random states for consistent results

### Deliverables

✅ **RFM Metrics Calculator** (`src/rfm_calculator.py`)  
✅ **Customer Clustering Module** (`src/customer_clustering.py`)  
✅ **High-Risk Labeling Module** (`src/high_risk_labeling.py`)  
✅ **Integration Script** (`examples/integrate_target_to_processed_data.py`)  
✅ **Comprehensive Unit Tests** (`tests/test_*.py`)  
✅ **Example Scripts** (`examples/step*.py`)  
✅ **Documentation** (`docs/step*.md`)  
✅ **Processed Data with Target** (`data/processed/processed_data_with_target.csv`)

### Next Steps

After completing Task 4, you have:
- ✅ RFM metrics calculated for each customer
- ✅ Customers clustered into 3 distinct groups
- ✅ High-risk target variable created (`is_high_risk`)
- ✅ Target integrated with feature-engineered data

**Ready for Task 5: Model Training and Tracking**

---

## Task 5: Model Training and Tracking

**Objective**: Develop a structured model training process that includes experiment tracking, model versioning, and unit testing.

### Overview

Task 5 implements a comprehensive model training pipeline with MLflow integration for experiment tracking, hyperparameter tuning, and model registry management.

### Implementation Components

#### 1. Data Preparation

Split the data into training and testing sets with reproducibility.

**Implementation:**
- Module: `src/data_splitting.py`
- Example: `examples/prepare_data_splits.py`

**Usage:**
```python
from src.data_splitting import DataSplitter

splitter = DataSplitter(test_size=0.2, random_state=42, stratify=True)
X_train, X_test, y_train, y_test = splitter.split_data(df, target_col='is_high_risk')
```

#### 2. Model Selection and Training

Train multiple models: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting (XGBoost/LightGBM).

**Implementation:**
- Module: `src/model_training.py`
- Example: `examples/train_models.py`

**Usage:**
```python
from src.model_training import train_models

models, metrics = train_models(
    X_train, y_train, X_test, y_test,
    model_names=['logistic_regression', 'random_forest', 'xgboost'],
    random_state=42
)
```

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC, PR-AUC
- Confusion Matrix

#### 3. Hyperparameter Tuning

Improve model performance using Grid Search and Random Search.

**Implementation:**
- Module: `src/hyperparameter_tuning.py`
- Example: `examples/tune_hyperparameters.py`

**Usage:**
```python
from src.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(search_method='grid', cv=5, scoring='roc_auc')
best_model = tuner.tune_model(X_train, y_train, model_type='random_forest')
```

#### 4. Experiment Tracking with MLflow

Log all experiments including parameters, metrics, and model artifacts.

**Implementation:**
- Module: `src/mlflow_tracking.py`
- Example: `examples/train_with_mlflow.py`

**Features:**
- Automatic experiment tracking
- Model parameter logging
- Evaluation metrics tracking
- Model artifact storage
- Model registry integration

**Usage:**
```python
from src.mlflow_tracking import MLflowTracker

tracker = MLflowTracker(experiment_name="credit_scoring")
tracker.log_model_training(model, X_train, y_train, X_test, y_test, 
                           model_name="random_forest", params={...})
```

**View Experiments:**
```bash
mlflow ui --backend-store-uri file:./mlruns
# Open http://localhost:5000
```

#### 5. Model Registry

Register and manage model versions in MLflow Model Registry.

**Usage:**
```python
# Register best model
tracker.register_model(run_id, model_name="credit_scoring_model", stage="Production")

# Load from registry
model = mlflow.sklearn.load_model("models:/credit_scoring_model/Production")
```

#### 6. Unit Testing

Comprehensive unit tests for all helper functions.

**Test Files:**
- `tests/test_data_splitting.py`
- `tests/test_model_training.py`
- `tests/test_hyperparameter_tuning.py`
- `tests/test_mlflow_tracking.py`
- `tests/test_data_processing.py` (helper functions)

**Run Tests:**
```bash
pytest tests/ -v
```

### Complete Workflow

```bash
# 1. Prepare data splits
python examples/prepare_data_splits.py

# 2. Train models with MLflow tracking
python examples/train_with_mlflow.py

# 3. Hyperparameter tuning with MLflow
python examples/tune_with_mlflow.py

# 4. Register best model
python examples/register_best_model.py

# 5. View experiments
mlflow ui
```

### Deliverables

✅ **Data Splitting Module** (`src/data_splitting.py`)  
✅ **Model Training Module** (`src/model_training.py`)  
✅ **Hyperparameter Tuning Module** (`src/hyperparameter_tuning.py`)  
✅ **MLflow Tracking Module** (`src/mlflow_tracking.py`)  
✅ **Comprehensive Unit Tests** (`tests/test_*.py`)  
✅ **Example Scripts** (`examples/train_*.py`, `examples/tune_*.py`)  
✅ **Documentation** (`docs/mlflow_usage.md`)  
✅ **Trained Models** (saved in `models/` and MLflow registry)

### Next Steps

After completing Task 5, you have:
- ✅ Multiple trained models with evaluation metrics
- ✅ Hyperparameter-tuned models
- ✅ MLflow experiment tracking and model registry
- ✅ Best model identified and registered

**Ready for Task 6: Model Deployment and CI/CD**

---
