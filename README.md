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

**Implementation Steps**:

1. **Calculate RFM Metrics**: Recency, Frequency, Monetary for each customer
   ```python
   from src.rfm_calculator import RFMCalculator
   calculator = RFMCalculator(customer_col='CustomerId', ...)
   rfm_metrics = calculator.calculate_rfm(transactions_df)
   ```

2. **Cluster Customers**: K-Means clustering (3 clusters) on scaled RFM features
   ```python
   from src.customer_clustering import CustomerClustering
   clustering = CustomerClustering(n_clusters=3, random_state=42)
   rfm_with_clusters = clustering.cluster_customers(rfm_metrics)
   ```

3. **Create High-Risk Label**: Identify least engaged cluster as high-risk
   ```python
   from src.high_risk_labeling import HighRiskLabeler
   labeler = HighRiskLabeler()
   rfm_with_target = labeler.create_target_variable(rfm_with_clusters)
   ```

4. **Integrate Target**: Merge `is_high_risk` into processed dataset

**Files**: 
- `src/rfm_calculator.py`
- `src/customer_clustering.py`
- `src/high_risk_labeling.py`
- `examples/step1_calculate_rfm.py`
- `examples/step2_cluster_customers.py`
- `examples/step3_create_high_risk_target.py`

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
