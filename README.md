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

**Objective**: Develop a structured model training process with experiment tracking, model versioning, and unit testing.

**Challenge Requirement**: Implement explicit training code with reproducible train/test split, at least two models, hyperparameter search, MLflow experiment logging, metric calculations, and at least two unit tests.

#### Step 1: Reproducible Train/Test Split

**Implementation**: `src/data_splitting.py` - Creates reproducible train/test split with `random_state=42`.

**Concrete Implementation**:
```python
from src.data_splitting import DataSplitter
import pandas as pd

# Load processed data with target
df = pd.read_csv('data/processed/processed_data_with_target.csv')

# Create reproducible split with random_state=42
splitter = DataSplitter(
    test_size=0.2,        # 80% train, 20% test
    random_state=42,      # Ensures reproducibility
    stratify=True         # Maintains class distribution
)

# Split data
X_train, X_test, y_train, y_test = splitter.split_data(
    df, 
    target_col='is_high_risk'
)

# Save splits for reproducibility
splitter.save_splits('data/processed/splits', X_train, X_test, y_train, y_test)
```

**What it does**:
- Uses `random_state=42` for reproducibility
- Stratified split maintains class distribution
- Saves splits to disk for consistent training

**Run**: `python examples/prepare_data_splits.py` → Creates `data/processed/splits/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

#### Step 2: Train At Least Two Models

**Implementation**: `src/model_training.py` - Trains multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost).

**Concrete Implementation**:
```python
from src.model_training import train_models, ModelTrainer
import pandas as pd

# Load data splits
X_train = pd.read_csv('data/processed/splits/X_train.csv')
X_test = pd.read_csv('data/processed/splits/X_test.csv')
y_train = pd.read_csv('data/processed/splits/y_train.csv')['is_high_risk']
y_test = pd.read_csv('data/processed/splits/y_test.csv')['is_high_risk']

# Train at least 2 models (actually trains 4)
models, metrics = train_models(
    X_train, y_train, X_test, y_test,
    model_names=['logistic_regression', 'random_forest'],  # At least 2 models
    random_state=42  # Reproducibility
)

# Models trained:
# 1. Logistic Regression
# 2. Random Forest
# (Also supports: Decision Tree, XGBoost, LightGBM)
```

**What it does**:
- Trains **Logistic Regression** model
- Trains **Random Forest** model
- Calculates evaluation metrics for each model
- Returns trained models and metrics dictionary

**Run**: `python examples/train_models.py` → Trains models and saves to `models/` directory

#### Step 3: Hyperparameter Search (Grid Search / Random Search)

**Implementation**: `src/hyperparameter_tuning.py` - Performs Grid Search and Random Search.

**Concrete Implementation**:
```python
from src.hyperparameter_tuning import HyperparameterTuner
import pandas as pd

# Load training data
X_train = pd.read_csv('data/processed/splits/X_train.csv')
y_train = pd.read_csv('data/processed/splits/y_train.csv')['is_high_risk']

# Grid Search hyperparameter tuning
tuner_grid = HyperparameterTuner(
    search_method='grid',  # Grid Search
    cv=5,                  # 5-fold cross-validation
    scoring='roc_auc',     # Metric to optimize
    random_state=42
)

# Tune Random Forest with Grid Search
best_model_grid = tuner_grid.tune_model(
    X_train, y_train,
    model_type='random_forest'
)

# Random Search hyperparameter tuning
tuner_random = HyperparameterTuner(
    search_method='random',  # Random Search
    cv=5,
    n_iter=20,              # Number of iterations
    random_state=42
)

# Tune Logistic Regression with Random Search
best_model_random = tuner_random.tune_model(
    X_train, y_train,
    model_type='logistic_regression'
)
```

**What it does**:
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling of parameter space
- Uses cross-validation (cv=5) for robust evaluation
- Optimizes ROC-AUC score

**Run**: `python examples/tune_hyperparameters.py` → Tuned models saved to `models/tuned/`

#### Step 4: MLflow Experiment Logging

**Implementation**: `src/mlflow_tracking.py` - Logs all experiments to MLflow.

**Concrete Implementation**:
```python
from src.mlflow_tracking import MLflowTracker
from src.model_training import train_models
import pandas as pd

# Initialize MLflow tracker
tracker = MLflowTracker(
    experiment_name="credit_scoring",
    tracking_uri="file:./mlruns"
)

# Load data
X_train = pd.read_csv('data/processed/splits/X_train.csv')
X_test = pd.read_csv('data/processed/splits/X_test.csv')
y_train = pd.read_csv('data/processed/splits/y_train.csv')['is_high_risk']
y_test = pd.read_csv('data/processed/splits/y_test.csv')['is_high_risk']

# Train models
trainer = ModelTrainer(random_state=42)
trainer.train(X_train, y_train, model_names=['logistic_regression', 'random_forest'])

# Log each model to MLflow
for model_name in ['logistic_regression', 'random_forest']:
    model = trainer.models_[model_name]
    
    # Log model training with MLflow
    run_id = tracker.log_model_training(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        params=trainer.get_model_params(model_name),
        artifact_path="model"
    )
    
    print(f"Logged {model_name} to MLflow: run_id={run_id}")
```

**What it logs**:
- **Parameters**: Model hyperparameters
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Artifacts**: Trained model files
- **Tags**: Model name, timestamp, etc.

**Run**: `python examples/train_with_mlflow.py` → Logs experiments to `mlruns/`

**View in MLflow UI**:
```bash
mlflow ui --backend-store-uri file:./mlruns
# Open http://localhost:5000
```

#### Step 5: Metric Calculations

**Implementation**: `src/model_training.py` - Calculates comprehensive evaluation metrics.

**Concrete Implementation**:
```python
from src.model_training import ModelTrainer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Train model
trainer = ModelTrainer(random_state=42)
trainer.train(X_train, y_train, model_names=['logistic_regression'])

# Evaluate and get metrics
model = trainer.models_['logistic_regression']
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate all required metrics
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}

# Also calculates confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

**Metrics Calculated**:
- ✅ **Accuracy**: Ratio of correct predictions
- ✅ **Precision**: True positives / (True positives + False positives)
- ✅ **Recall**: True positives / (True positives + False negatives)
- ✅ **F1 Score**: Harmonic mean of Precision and Recall
- ✅ **ROC-AUC**: Area Under ROC Curve

#### Step 6: Unit Tests (At Least Two)

**Implementation**: `tests/test_model_training.py` - Unit tests for training and evaluation.

**Concrete Implementation**:

**Test 1: Train and Evaluate Model**
```python
# tests/test_model_training.py
def test_train_and_evaluate(self, sample_data):
    """Test that model training and evaluation works correctly."""
    X_train, X_test, y_train, y_test = sample_data
    
    trainer = ModelTrainer(random_state=42)
    trainer.train(X_train, y_train, model_names=['logistic_regression'])
    
    # Verify model was trained
    assert 'logistic_regression' in trainer.models_
    
    # Evaluate model
    metrics = trainer.evaluate(trainer.models_['logistic_regression'], 
                               X_test, y_test)
    
    # Verify metrics are calculated
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics
    
    # Verify metrics are valid
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1
```

**Test 2: Train Multiple Models**
```python
# tests/test_model_training.py
def test_train_multiple_models(self, sample_data):
    """Test training multiple models."""
    X_train, X_test, y_train, y_test = sample_data
    
    trainer = ModelTrainer(random_state=42)
    trainer.train(
        X_train, y_train,
        model_names=['logistic_regression', 'random_forest']  # At least 2
    )
    
    # Verify both models were trained
    assert 'logistic_regression' in trainer.models_
    assert 'random_forest' in trainer.models_
    
    # Verify both models have metrics
    assert 'logistic_regression' in trainer.metrics_
    assert 'random_forest' in trainer.metrics_
```

**Additional Tests**:
- `test_train_logistic_regression` - Tests Logistic Regression training
- `test_train_random_forest` - Tests Random Forest training
- `test_data_splitting` - Tests reproducible data splitting
- `test_hyperparameter_tuning` - Tests Grid Search and Random Search

**Run Tests**:
```bash
# Run all model training tests
pytest tests/test_model_training.py -v

# Run specific tests
pytest tests/test_model_training.py::TestModelTrainer::test_train_and_evaluate -v
pytest tests/test_model_training.py::TestModelTrainer::test_train_multiple_models -v
```

#### Complete Workflow

```bash
# 1. Prepare data splits (reproducible with random_state=42)
python examples/prepare_data_splits.py

# 2. Train models with MLflow tracking
python examples/train_with_mlflow.py
# Trains at least 2 models, logs to MLflow, calculates metrics

# 3. Hyperparameter tuning with MLflow
python examples/tune_with_mlflow.py
# Performs Grid Search and Random Search

# 4. View experiments
mlflow ui
# Open http://localhost:5000

# 5. Run unit tests
pytest tests/test_model_training.py -v
pytest tests/test_data_splitting.py -v
pytest tests/test_hyperparameter_tuning.py -v
```

#### Verification

**Check train/test split**:
```python
X_train = pd.read_csv('data/processed/splits/X_train.csv')
X_test = pd.read_csv('data/processed/splits/X_test.csv')
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
# Should show 80/20 split
```

**Check models trained**:
```python
import joblib
lr = joblib.load('models/logistic_regression.pkl')
rf = joblib.load('models/random_forest.pkl')
# Both models exist
```

**Check MLflow experiments**:
```bash
mlflow ui
# View logged experiments, parameters, metrics
```

**Check unit tests**:
```bash
pytest tests/test_model_training.py -v
# Should show at least 2 tests passing
```

**Files**: 
- `src/data_splitting.py` - Reproducible train/test split
- `src/model_training.py` - Model training (at least 2 models)
- `src/hyperparameter_tuning.py` - Grid Search and Random Search
- `src/mlflow_tracking.py` - MLflow experiment logging
- `examples/train_with_mlflow.py` - Complete training workflow
- `examples/tune_with_mlflow.py` - Hyperparameter tuning workflow
- `tests/test_model_training.py` - Unit tests (at least 2 tests)
- `tests/test_data_splitting.py` - Data splitting tests
- `tests/test_hyperparameter_tuning.py` - Tuning tests

**Documentation**: `docs/mlflow_usage.md`

---

### Task 6: Model Deployment and Continuous Integration

**Objective**: Package the trained model into a containerized API and set up a CI/CD pipeline.

**Challenge Requirement**: Add a real FastAPI app that loads the best MLflow model, exposes a `/predict` endpoint with Pydantic schemas, provide working Docker and docker-compose definitions, and a CI workflow that runs linting and pytest and fails on errors.

#### Step 1: FastAPI App Loading Best MLflow Model

**Implementation**: `src/api/main.py` - FastAPI application that loads model from MLflow registry.

**Concrete Implementation**:
```python
# src/api/main.py
from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
import os
import numpy as np

app = FastAPI(title="Credit Scoring API")

# Global model variable
model = None
model_name = None

def load_model_from_mlflow(
    model_name: str = "credit_scoring_model",
    stage: str = "Production"
):
    """Load best model from MLflow Model Registry."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    
    # Load model from registry
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model

@app.on_event("startup")
async def startup_event():
    """Load best model on application startup."""
    global model, model_name
    
    model_name = os.getenv("MODEL_NAME", "credit_scoring_model")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    
    # Load best model from MLflow registry
    model = load_model_from_mlflow(model_name, model_stage)
    print(f"Model loaded: {model_name} ({model_stage})")

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict credit risk using loaded MLflow model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Make prediction
    features_array = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0][1]
    
    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability),
        risk_level="high" if prediction == 1 else "low"
    )
```

**What it does**:
- ✅ Loads best model from MLflow registry (`models:/credit_scoring_model/Production`)
- ✅ Loads on startup using `@app.on_event("startup")`
- ✅ Uses MLflow's `mlflow.sklearn.load_model()` to load registered model

**File**: `src/api/main.py`

#### Step 2: /predict Endpoint with Pydantic Schemas

**Implementation**: `src/api/pydantic_models.py` - Pydantic models for request/response validation.

**Concrete Implementation**:
```python
# src/api/pydantic_models.py
from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    """Request model with Pydantic validation."""
    features: List[float] = Field(
        ...,
        description="List of 26 feature values",
        min_length=26,
        max_length=26,
        example=[0.0, -0.046, -0.072, -0.349, -0.045, -2.156, ...]
    )

class PredictionResponse(BaseModel):
    """Response model with Pydantic validation."""
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction")
    probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability")
    risk_level: str = Field(..., description="Risk level: 'low' or 'high'")
```

**Endpoint Implementation**:
```python
# In src/api/main.py
from src.api.pydantic_models import PredictionRequest, PredictionResponse

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict endpoint with Pydantic request/response validation."""
    # Pydantic automatically validates request.features
    # Returns validated PredictionResponse
    ...
```

**What it does**:
- ✅ `/predict` endpoint accepts `PredictionRequest` (validated by Pydantic)
- ✅ Returns `PredictionResponse` (validated by Pydantic)
- ✅ Automatic validation of feature count, types, and ranges

**Files**: 
- `src/api/pydantic_models.py` - Pydantic schemas
- `src/api/main.py` - Endpoint implementation

#### Step 3: Working Docker Definition

**Implementation**: `Dockerfile` - Multi-stage build for optimized container.

**Concrete Implementation**:
```dockerfile
# Dockerfile
FROM python:3.12-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY src/ ./src/
RUN mkdir -p ./mlruns

# Environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:./mlruns
ENV MODEL_NAME=credit_scoring_model
ENV MODEL_STAGE=Production
ENV PORT=8000

EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**What it does**:
- ✅ Multi-stage build for smaller image
- ✅ Installs all dependencies from `requirements.txt`
- ✅ Copies FastAPI application
- ✅ Sets up MLflow model registry path
- ✅ Runs FastAPI with uvicorn

**File**: `Dockerfile`

#### Step 4: Working docker-compose Definition

**Implementation**: `docker-compose.yml` - Service orchestration.

**Concrete Implementation**:
```yaml
# docker-compose.yml
version: '3.8'

services:
  credit-scoring-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: credit-scoring-api
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=file:./mlruns
      - MODEL_NAME=credit_scoring_model
      - MODEL_STAGE=Production
      - PORT=8000
    volumes:
      - ./mlruns:/app/mlruns  # Mount MLflow registry
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**What it does**:
- ✅ Builds Docker image from Dockerfile
- ✅ Exposes port 8000
- ✅ Mounts MLflow registry volume
- ✅ Sets environment variables for model loading
- ✅ Health check configured

**File**: `docker-compose.yml`

**Usage**:
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

#### Step 5: CI Workflow with Linting and pytest (Fails on Errors)

**Implementation**: `.github/workflows/ci.yml` - GitHub Actions workflow.

**Concrete Implementation**:
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches:
      - main
      - master
  pull_request:
    branches:
      - main
      - master

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run flake8 linter
        run: |
          # Fail on syntax errors
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          # Check code style
          flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
      
      - name: Run pytest
        run: |
          pytest tests/ -v --tb=short
```

**What it does**:
- ✅ Triggers on push to `main` branch
- ✅ Runs **flake8 linter** - **FAILS on syntax errors** (E9, F63, F7, F82)
- ✅ Runs **pytest** - **FAILS if tests fail**
- ✅ Build fails if either step fails (no `continue-on-error: true`)

**File**: `.github/workflows/ci.yml`

**Verification**:
- Push code with syntax error → CI fails ✅
- Push code with failing test → CI fails ✅
- Push code with linting issues → CI fails ✅

#### Complete Workflow

```bash
# 1. Build and run API with Docker
docker-compose up --build

# 2. Test API endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994, -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}'

# 3. View API documentation
# Open http://localhost:8000/docs

# 4. CI/CD automatically runs on push
git push origin main
# GitHub Actions runs flake8 and pytest, fails on errors
```

#### Verification

**Check FastAPI app loads MLflow model**:
```python
# Test that model loads from MLflow
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
model = mlflow.sklearn.load_model("models:/credit_scoring_model/Production")
assert model is not None
```

**Check /predict endpoint**:
```bash
# Test endpoint
curl http://localhost:8000/predict \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
# Should return: {"prediction": 0, "probability": 0.15, "risk_level": "low"}
```

**Check Docker build**:
```bash
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api
# Should start FastAPI app
```

**Check CI/CD workflow**:
```bash
# View workflow runs in GitHub
# .github/workflows/ci.yml should show:
# - flake8 step (fails on errors)
# - pytest step (fails on errors)
```

**Files**: 
- `src/api/main.py` - FastAPI app loading MLflow model
- `src/api/pydantic_models.py` - Pydantic request/response schemas
- `Dockerfile` - Working Docker definition
- `docker-compose.yml` - Working docker-compose definition
- `.github/workflows/ci.yml` - CI workflow with linting and pytest (fails on errors)

**Documentation**: `docs/api_deployment.md`, `docs/docker_testing_guide.md`

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
