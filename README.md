# Bati Bank Credit Scoring MLOps

An End-to-End MLOps Implementation for Building, Deploying, and Automating a Credit Risk Model

---

## 📋 Overview

This project implements a complete MLOps pipeline for credit risk scoring in a regulated banking environment. Since direct default labels are unavailable, we use RFM (Recency, Frequency, Monetary) analysis to create a proxy target variable and build predictive models for credit risk assessment.

### Key Features

- **Proxy Target Engineering**: RFM-based customer segmentation and high-risk identification
- **Feature Engineering**: Automated pipeline with temporal, aggregate, and WoE transformations
- **Model Training**: Multiple algorithms with MLflow experiment tracking
- **Hyperparameter Tuning**: Grid Search and Random Search optimization
- **Model Registry**: MLflow-based model versioning and management
- **REST API**: FastAPI-based production-ready API
- **Containerization**: Docker and Docker Compose deployment
- **CI/CD**: Automated testing and code quality checks with GitHub Actions

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker and Docker Compose (for deployment)
- MLflow (for experiment tracking)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd bati-bank-credit-scoring-mlops

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
# 1. Calculate RFM metrics and create target variable
python examples/step1_calculate_rfm.py
python examples/step2_cluster_customers.py
python examples/step3_create_high_risk_target.py
python examples/integrate_target_to_processed_data.py

# 2. Prepare data splits
python examples/prepare_data_splits.py

# 3. Train models with MLflow tracking
python examples/train_with_mlflow.py

# 4. Deploy API
docker-compose up --build
```

---

## 📚 Project Structure

```
bati-bank-credit-scoring-mlops/
├── src/                    # Source code modules
│   ├── api/               # FastAPI REST API
│   ├── data_processing.py  # Feature engineering pipeline
│   ├── rfm_calculator.py  # RFM metrics calculation
│   ├── customer_clustering.py  # K-Means clustering
│   ├── high_risk_labeling.py   # Target variable creation
│   ├── model_training.py       # Model training
│   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   └── mlflow_tracking.py      # MLflow integration
├── examples/              # Example scripts and workflows
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── data/                  # Data files
│   ├── raw/              # Raw transaction data
│   └── processed/       # Processed features and splits
├── mlruns/               # MLflow experiment tracking
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
└── requirements.txt      # Python dependencies
```

---

## 🎯 Tasks Overview

### Task 1: Business Understanding

**Objective**: Understand regulatory requirements and model trade-offs in credit risk modeling.

**Key Deliverables**:
- Basel II Accord compliance analysis
- Proxy variable necessity and risks assessment
- Trade-off analysis: Interpretable vs. High-performance models

**Documentation**: See `README.md` Task 1 section for detailed analysis.

---

### Task 2: Exploratory Data Analysis

**Objective**: Analyze transaction data to understand patterns and inform feature engineering.

**Key Deliverables**:
- Comprehensive EDA notebook with visualizations
- Data quality assessment
- Feature correlation analysis

**Files**: `notebooks/eda.ipynb`

---

### Task 3: Feature Engineering

**Objective**: Build automated feature engineering pipeline for credit risk modeling.

**Key Features**:
- Temporal feature extraction (hour, day, month, etc.)
- Customer aggregation (sum, mean, count, etc.)
- Categorical encoding (one-hot, label, WoE)
- Feature scaling (standardize, normalize, robust)
- Weight of Evidence (WoE) transformation

**Usage**:
```python
from src.data_processing import DataProcessor

processor = DataProcessor(
    customer_col='CustomerId',
    datetime_col='TransactionStartTime',
    amount_col='Amount'
)
processed_df = processor.process_step_by_step(df)
```

**Files**: `src/data_processing.py`, `tests/test_data_processing.py`

---

### Task 4: Proxy Target Variable Engineering

**Objective**: Create credit risk target variable using RFM analysis and clustering.

**Challenge Requirement**: Compute RFM per customer, perform 3-cluster K-Means, derive `is_high_risk` flag, and merge target into processed modeling dataset.

#### Step 1: Calculate RFM Metrics Per Customer

**Implementation**: `src/rfm_calculator.py` - Calculates Recency, Frequency, and Monetary metrics **for each CustomerId**.

**Concrete Implementation**:
```python
from src.rfm_calculator import RFMCalculator
import pandas as pd

# Load transaction data
df = pd.read_csv('data/raw/data.csv')

# Calculate RFM per customer
calculator = RFMCalculator(
    customer_col='CustomerId',
    datetime_col='TransactionStartTime',
    amount_col='Amount',
    snapshot_date=None  # Uses max date in data
)

# This computes RFM for EACH CustomerId
rfm_metrics = calculator.calculate_rfm(df)
# Output: DataFrame with columns [CustomerId, recency, frequency, monetary]
# Each row = one customer with their RFM metrics
```

**What it does**:
- **Recency**: Days since last transaction (per customer)
- **Frequency**: Total transaction count (per customer)
- **Monetary**: Sum of transaction amounts (per customer)

**Run**: `python examples/step1_calculate_rfm.py` → Creates `data/processed/rfm_metrics.csv`

#### Step 2: 3-Cluster K-Means Clustering

**Implementation**: `src/customer_clustering.py` - Performs **exactly 3-cluster K-Means** on scaled RFM features.

**Concrete Implementation**:
```python
from src.customer_clustering import CustomerClustering
import pandas as pd

# Load RFM metrics
rfm_metrics = pd.read_csv('data/processed/rfm_metrics.csv')

# Perform 3-cluster K-Means
clustering = CustomerClustering(
    n_clusters=3,  # Exactly 3 clusters as required
    scaling_method='standardize',  # Pre-process RFM features
    random_state=42  # Reproducibility
)

# Fit and assign clusters
rfm_with_clusters = clustering.cluster_customers(rfm_metrics)
# Output: DataFrame with columns [CustomerId, recency, frequency, monetary, cluster]
# Each customer assigned to cluster 0, 1, or 2
```

**What it does**:
- Scales RFM features (standardize/robust)
- Fits K-Means with `n_clusters=3`
- Assigns each customer to one of 3 clusters

**Run**: `python examples/step2_cluster_customers.py` → Creates `data/processed/rfm_with_clusters.csv`

#### Step 3: Derive `is_high_risk` Flag

**Implementation**: `src/high_risk_labeling.py` - Creates binary `is_high_risk` column (1 = high-risk, 0 = low-risk).

**Concrete Implementation**:
```python
from src.high_risk_labeling import HighRiskLabeler
import pandas as pd

# Load clustered RFM data
rfm_with_clusters = pd.read_csv('data/processed/rfm_with_clusters.csv')

# Identify high-risk cluster (least engaged)
labeler = HighRiskLabeler()
high_risk_cluster = labeler.identify_high_risk_cluster(rfm_with_clusters)

# Create is_high_risk flag
rfm_with_target = labeler.create_target_variable(rfm_with_clusters)
# Output: DataFrame with columns [CustomerId, recency, frequency, monetary, cluster, is_high_risk]
# is_high_risk = 1 for high-risk cluster, 0 for others
```

**What it does**:
- Analyzes clusters to find least engaged (high recency, low frequency, low monetary)
- Creates `is_high_risk` binary column:
  - `is_high_risk = 1`: Customers in high-risk cluster
  - `is_high_risk = 0`: Customers in other clusters

**Run**: `python examples/step3_create_high_risk_target.py` → Creates `data/processed/rfm_with_target.csv` and `data/processed/transactions_with_target.csv`

#### Step 4: Merge Target into Processed Modeling Dataset

**Implementation**: `examples/integrate_target_to_processed_data.py` - Merges `is_high_risk` into feature-engineered dataset.

**Concrete Implementation**:
```python
import pandas as pd

# Load processed features (from Task 3)
processed_df = pd.read_csv('data/processed/processed_data.csv')

# Load transactions with target (from Step 3)
transactions_with_target = pd.read_csv('data/processed/transactions_with_target.csv')

# Merge is_high_risk into processed dataset
processed_df['is_high_risk'] = transactions_with_target['is_high_risk'].values

# Save final dataset ready for modeling
processed_df.to_csv('data/processed/processed_data_with_target.csv', index=False)
```

**What it does**:
- Takes feature-engineered data from Task 3
- Adds `is_high_risk` target column
- Creates final dataset: `processed_data_with_target.csv` ready for model training

**Run**: `python examples/integrate_target_to_processed_data.py` → Creates `data/processed/processed_data_with_target.csv`

#### Complete Workflow

```bash
# Step 1: Calculate RFM per customer
python examples/step1_calculate_rfm.py
# Output: data/processed/rfm_metrics.csv (RFM per CustomerId)

# Step 2: 3-cluster K-Means
python examples/step2_cluster_customers.py
# Output: data/processed/rfm_with_clusters.csv (3 clusters assigned)

# Step 3: Create is_high_risk flag
python examples/step3_create_high_risk_target.py
# Output: data/processed/rfm_with_target.csv (with is_high_risk column)

# Step 4: Merge into processed dataset
python examples/integrate_target_to_processed_data.py
# Output: data/processed/processed_data_with_target.csv (ready for training)
```

#### Verification

**Check RFM per customer**:
```python
rfm = pd.read_csv('data/processed/rfm_metrics.csv')
print(rfm.head())  # Shows CustomerId, recency, frequency, monetary
print(f"Total customers: {len(rfm)}")  # One row per customer
```

**Check 3 clusters**:
```python
clustered = pd.read_csv('data/processed/rfm_with_clusters.csv')
print(clustered['cluster'].value_counts())  # Should show 3 clusters
print(f"Clusters: {clustered['cluster'].unique()}")  # [0, 1, 2]
```

**Check is_high_risk flag**:
```python
target = pd.read_csv('data/processed/rfm_with_target.csv')
print(target['is_high_risk'].value_counts())  # Binary: 0 and 1
```

**Check merged dataset**:
```python
final = pd.read_csv('data/processed/processed_data_with_target.csv')
print('is_high_risk' in final.columns)  # True
print(final['is_high_risk'].value_counts())  # Target distribution
```

**Files**: 
- `src/rfm_calculator.py` - RFM calculation per customer
- `src/customer_clustering.py` - 3-cluster K-Means implementation
- `src/high_risk_labeling.py` - is_high_risk flag creation
- `examples/integrate_target_to_processed_data.py` - Target merging
- `examples/step1_calculate_rfm.py` - RFM calculation script
- `examples/step2_cluster_customers.py` - Clustering script
- `examples/step3_create_high_risk_target.py` - Target creation script

**Documentation**: `docs/step1_rfm_calculation.md`, `docs/step2_customer_clustering.md`, `docs/step3_high_risk_target.md`

---

### Task 5: Model Training and Tracking

**Objective**: Train multiple models with experiment tracking and hyperparameter tuning.

**Key Components**:

1. **Data Splitting**: Train/test split with stratification
   ```python
   from src.data_splitting import DataSplitter
   splitter = DataSplitter(test_size=0.2, random_state=42, stratify=True)
   X_train, X_test, y_train, y_test = splitter.split_data(df, target_col='is_high_risk')
   ```

2. **Model Training**: Multiple algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost)
   ```python
   from src.model_training import train_models
   models, metrics = train_models(X_train, y_train, X_test, y_test)
   ```

3. **Hyperparameter Tuning**: Grid Search and Random Search
   ```python
   from src.hyperparameter_tuning import HyperparameterTuner
   tuner = HyperparameterTuner(search_method='grid', cv=5)
   best_model = tuner.tune_model(X_train, y_train, model_type='random_forest')
   ```

4. **MLflow Tracking**: Experiment tracking, metrics logging, model registry
   ```python
   from src.mlflow_tracking import MLflowTracker
   tracker = MLflowTracker(experiment_name="credit_scoring")
   tracker.log_model_training(model, X_train, y_train, X_test, y_test, ...)
   ```

**Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC, PR-AUC

**Files**: 
- `src/model_training.py`
- `src/hyperparameter_tuning.py`
- `src/mlflow_tracking.py`
- `examples/train_with_mlflow.py`
- `examples/tune_with_mlflow.py`

**Documentation**: `docs/mlflow_usage.md`

**View Experiments**:
```bash
mlflow ui --backend-store-uri file:./mlruns
# Open http://localhost:5000
```

---

### Task 6: Model Deployment and CI/CD

**Objective**: Deploy model as containerized REST API with automated CI/CD pipeline.

**Key Components**:

1. **REST API**: FastAPI with `/predict` and `/health` endpoints
   ```bash
   uvicorn src.api.main:app --reload
   # API docs: http://localhost:8000/docs
   ```

2. **Docker Containerization**: Multi-stage build with Docker Compose
   ```bash
   docker-compose up --build
   ```

3. **CI/CD Pipeline**: GitHub Actions with flake8 linting and pytest testing
   - Automatically runs on push to `main` branch
   - Code quality checks and unit test execution

**API Endpoints**:
- `GET /health`: Health check and model status
- `POST /predict`: Credit risk prediction (requires 26 feature values)

**Files**:
- `src/api/main.py` - FastAPI application
- `src/api/pydantic_models.py` - Request/response validation
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Service orchestration
- `.github/workflows/ci.yml` - CI/CD pipeline

**Documentation**: `docs/api_deployment.md`, `docs/docker_testing_guide.md`

**Test API**:
```bash
python examples/test_api.py
# Or
./scripts/test_docker_compose.sh
```

---

## 🧪 Testing

Run all unit tests:

```bash
pytest tests/ -v
```

Test specific modules:

```bash
pytest tests/test_data_processing.py -v
pytest tests/test_rfm_calculator.py -v
pytest tests/test_model_training.py -v
```

---

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:

- **RFM Calculation**: `docs/step1_rfm_calculation.md`
- **Customer Clustering**: `docs/step2_customer_clustering.md`
- **High-Risk Target**: `docs/step3_high_risk_target.md`
- **MLflow Usage**: `docs/mlflow_usage.md`
- **API Deployment**: `docs/api_deployment.md`
- **Docker Testing**: `docs/docker_testing_guide.md`

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Raw Data       │
│  (Transactions) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │
│  Engineering    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│  RFM Metrics     │─────▶│  Clustering     │
│  Calculation     │      │  (K-Means)      │
└─────────────────┘      └────────┬────────┘
                                  │
                                  ▼
┌─────────────────┐      ┌─────────────────┐
│  High-Risk       │      │  Model          │
│  Target Creation │─────▶│  Training       │
└─────────────────┘      └────────┬────────┘
                                  │
                                  ▼
┌─────────────────┐      ┌─────────────────┐
│  MLflow         │      │  FastAPI        │
│  Registry       │─────▶│  REST API       │
└─────────────────┘      └─────────────────┘
```

---

## 🔧 Configuration

### Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow tracking URI (default: `file:./mlruns`)
- `MODEL_NAME`: Registered model name (default: `credit_scoring_model`)
- `MODEL_STAGE`: Model stage (default: `Production`)
- `PORT`: API port (default: `8000`)

### Model Configuration

Models are configured in:
- `src/model_training.py` - Model parameters
- `src/hyperparameter_tuning.py` - Tuning configurations
- `examples/train_with_mlflow.py` - Training scripts

---

## 📊 Model Performance

Models are evaluated using:
- **Classification Metrics**: Accuracy, Precision, Recall, F1 Score
- **Ranking Metrics**: ROC-AUC, PR-AUC
- **Business Metrics**: Confusion Matrix (Type I/II errors)

View experiment results in MLflow UI:
```bash
mlflow ui
```

---

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Run linter: `flake8 src/`
5. Submit a pull request

---

## 👥 Authors

Haben Eyasu

---

## 🙏 Acknowledgments

- Basel II Accord compliance considerations
- RFM analysis methodology
- MLflow for experiment tracking
- FastAPI for API development

---

**Status**: ✅ All Tasks Complete - Production Ready
