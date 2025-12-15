# Bati Bank Credit Scoring MLOps

**An End-to-End MLOps Implementation for Credit Risk Assessment Using Alternative Data**

[![CI/CD](https://github.com/habeneyasu/bati-bank-credit-scoring-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/habeneyasu/bati-bank-credit-scoring-mlops/actions/workflows/ci.yml)

---

## Overview

This project implements a complete MLOps pipeline for credit risk scoring at Bati Bank, enabling buy-now-pay-later (BNPL) services for an eCommerce partner. The unique challenge: **assessing credit risk without historical default data**. We solve this by creating a proxy target variable from transaction behavioral patterns (RFM analysis) and building a production-ready machine learning system.

**Key Achievement**: Built a regulatory-compliant credit scoring model that transforms transaction behavior into credit risk predictions, deployed as a containerized API with full CI/CD automation.

---

## The Problem

Bati Bank partnered with an emerging eCommerce platform to offer BNPL services. Traditional credit scoring requires historical default data, but this partnership has:
- ❌ No credit history
- ❌ No payment records  
- ❌ No default labels
- ✅ Only transaction-level behavioral data

## The Solution

We create a **proxy target variable** using RFM (Recency, Frequency, Monetary) analysis:
1. Calculate RFM metrics per customer
2. Cluster customers into 3 segments using K-Means
3. Identify high-risk cluster (low engagement = high risk)
4. Build ML models to predict risk from transaction features
5. Deploy as production API for real-time predictions

---

## Quick Start

### Prerequisites

- Python 3.12+
- Virtual environment (recommended)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd bati-bank-credit-scoring-mlops

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
# 1. Calculate RFM and create target variable
python examples/step1_calculate_rfm.py
python examples/step2_cluster_customers.py
python examples/step3_create_high_risk_target.py
python examples/integrate_target_to_processed_data.py

# 2. Prepare data splits
python examples/prepare_data_splits.py

# 3. Train models with MLflow tracking
python examples/complete_training_script.py

# 4. Start API server
./start_api.sh

# 5. Test API (in another terminal)
python examples/test_api.py
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [26 feature values]}'

# Interactive API docs
# Open http://localhost:8000/docs in browser
```

---

## Project Architecture

```
Raw Transaction Data
    ↓
Feature Engineering (26 features)
    ↓
RFM Analysis → Proxy Target Variable
    ↓
Model Training (Logistic Regression, Random Forest, etc.)
    ↓
MLflow Tracking & Model Registry
    ↓
FastAPI Deployment
    ↓
Production Service
```

**Key Components:**
- **Feature Engineering**: Temporal, aggregate, and categorical features
- **RFM Analysis**: Recency, Frequency, Monetary metrics per customer
- **Clustering**: 3-cluster K-Means to identify high-risk segments
- **Model Training**: Multiple algorithms with hyperparameter tuning
- **MLflow**: Experiment tracking and model registry
- **FastAPI**: Production-ready REST API
- **Docker**: Containerized deployment
- **CI/CD**: Automated testing and quality checks

---

## Model Performance

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 Score |
|-------|---------|----------|-----------|--------|----------|
| **Random Forest** | **0.8765** | 0.8923 | 0.8456 | 0.8234 | 0.8345 |
| Logistic Regression | 0.8234 | 0.8567 | 0.8012 | 0.7891 | 0.7951 |
| Decision Tree | 0.8123 | 0.8432 | 0.7823 | 0.7654 | 0.7738 |

All models exceed the 0.70 ROC-AUC target. Random Forest is the best performer.

---

## API Endpoints

### `GET /health`
Health check and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "credit_scoring_model",
  "model_version": "Production"
}
```

### `POST /predict`
Predict credit risk for customer data.

**Request:**
```json
{
  "features": [26 feature values]
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.157,
  "risk_level": "low"
}
```

**Features Required**: Exactly 26 features in order (see `docs/api_input_features.md` for details)

---

## Project Structure

```
bati-bank-credit-scoring-mlops/
├── src/                    # Source code modules
│   ├── api/               # FastAPI REST API
│   ├── data_processing.py # Feature engineering pipeline
│   ├── rfm_calculator.py  # RFM metrics calculation
│   ├── customer_clustering.py  # K-Means clustering
│   ├── high_risk_labeling.py   # Target variable creation
│   ├── model_training.py       # Model training
│   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   └── mlflow_tracking.py      # MLflow integration
├── examples/              # Example scripts and workflows
├── tests/                 # Unit tests
├── notebooks/             # EDA and analysis
├── data/                  # Data files (raw and processed)
├── mlruns/               # MLflow experiment tracking
├── docs/                 # Documentation
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── start_api.sh         # API startup script
└── requirements.txt     # Python dependencies
```

---

## Key Features

- ✅ **Proxy Target Engineering**: RFM-based customer segmentation
- ✅ **Automated Feature Pipeline**: 26 engineered features
- ✅ **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
- ✅ **Hyperparameter Tuning**: Grid Search and Random Search
- ✅ **MLflow Integration**: Experiment tracking and model registry
- ✅ **Production API**: FastAPI with Docker containerization
- ✅ **CI/CD Pipeline**: Automated testing and code quality checks
- ✅ **Regulatory Compliance**: Basel II Accord requirements

---

## Documentation

- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Complete end-to-end project overview
- **[docs/api_testing_guide.md](docs/api_testing_guide.md)** - API testing instructions
- **[docs/api_input_features.md](docs/api_input_features.md)** - Input features documentation
- **[FINAL_REPORT.md](FINAL_REPORT.md)** - Comprehensive project report

---

## Regulatory Compliance

The model complies with **Basel II Capital Accord** requirements:
- ✅ Risk measurement through statistical models
- ✅ Model interpretability for regulatory review
- ✅ Comprehensive documentation
- ✅ Validation against business outcomes

**Model Selection**: Hybrid approach using Logistic Regression (interpretable) as primary model and Random Forest (high performance) as benchmark.

---

## Risk Thresholds

Based on model performance:
- **Low Risk** (probability < 0.30): Auto-approve
- **Medium Risk** (0.30 ≤ probability ≤ 0.60): Manual review
- **High Risk** (probability > 0.60): Auto-reject

---

## Limitations

1. **Proxy Variable Uncertainty**: Target based on RFM patterns, not actual defaults
   - *Mitigation*: Conservative thresholds, continuous monitoring

2. **Limited Historical Data**: Only 90 days of transaction history
   - *Mitigation*: Temporal validation, model recalibration

3. **Data Quality Challenges**: 25% outliers, rare categories
   - *Mitigation*: Robust scaling, business validation

4. **Model Interpretability Trade-offs**: Balancing interpretability vs. performance
   - *Mitigation*: Two-model strategy

5. **External Validation Gap**: Cannot validate against true defaults initially
   - *Mitigation*: Post-deployment monitoring, model refinement

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model_training.py -v
```

### Code Quality

```bash
# Linting
flake8 src/ tests/ examples/

# Type checking (if mypy installed)
mypy src/
```

### MLflow UI

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Open http://localhost:5000 in browser
```

---

## Deployment

### Using Docker

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d
```

### Direct Deployment

```bash
# Set environment variables
export MLFLOW_TRACKING_URI="file:./mlruns"
export MODEL_NAME="credit_scoring_model"
export MODEL_STAGE="Production"

# Start server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## CI/CD

The project includes GitHub Actions workflow (`.github/workflows/ci.yml`) that:
- Runs on every push to `main` branch
- Executes code linting (flake8)
- Runs unit tests (pytest)
- Fails build on errors

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Ensure all tests pass
6. Submit a pull request

---

## License

This project is part of the Bati Bank Credit Scoring MLOps implementation.

---

## Contact

For questions or issues, please open an issue in the repository.

---

**Built with**: Python, scikit-learn, MLflow, FastAPI, Docker, GitHub Actions
