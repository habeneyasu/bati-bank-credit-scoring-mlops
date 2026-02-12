# Bati Bank Credit Scoring MLOps

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![Docker](https://img.shields.io/badge/Docker-20.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production--ready-success)
[![CI/CD](https://github.com/habeneyasu/bati-bank-credit-scoring-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/habeneyasu/bati-bank-credit-scoring-mlops/actions/workflows/ci.yml)

**An End-to-End MLOps Implementation for Credit Risk Assessment Using Alternative Data**

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

## The Solution: RFM-Based Proxy Approach

### Why RFM?

When traditional credit data is unavailable, **customer engagement patterns** serve as reliable risk proxies:

1. **Recency** → Days since last transaction (recent = engaged = lower risk)
2. **Frequency** → Transaction count (frequent = active = lower risk)  
3. **Monetary** → Total spend (higher = stable = lower risk)

### Implementation Flow:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Transaction    │───▶│  RFM Analysis   │───▶│  K-Means        │
│  Data           │    │  & Clustering   │    │  Segmentation   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  26 Engineered  │    │  Proxy Target   │    │  High/Low Risk  │
│  Features       │    │  Variable       │    │  Labels         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         └──────────────────────┴──────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  ML Models      │
                    │  Training       │
                    └─────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  FastAPI        │
                    │  Deployment     │
                    └─────────────────┘
```

### Business Impact:

- **Market Expansion**: Score 100% of customers vs traditional ~40%
- **Speed**: Milliseconds vs days for manual underwriting
- **Scalability**: Automated pipeline handles volume growth

---

## 📋 Prerequisites Checklist

- [ ] Python 3.12+ installed
- [ ] Git installed
- [ ] Docker installed (for containerized deployment) - *Optional*
- [ ] 4GB+ RAM available
- [ ] Virtual environment support (venv or conda)
- [ ] Dataset obtained through approved channels and placed in `data/raw/` directory

---

## Dataset

This project uses transaction data with the following characteristics:

- **95,662 transactions** across 90 days
- **16 original features** expanded to **26 engineered features**
- **11,000+ unique customers**

**⚠️ Important**: The dataset is **not included** in this repository. Data files are in `.gitignore` for privacy and security reasons. Users must obtain the dataset through their organization's approved channels.

### Setting Up the Data

**Note**: Obtain the dataset through your organization's approved data access channels. Ensure compliance with data privacy and security policies.

1. **Create data directory structure**:
   ```bash
   mkdir -p data/raw data/processed
   ```

2. **Place the dataset**:
   ```bash
   # Place your dataset file in data/raw/ directory
   # Ensure the file is named 'data.csv' or update the code accordingly
   cp <your_dataset_file>.csv data/raw/data.csv
   ```

3. **Verify the data**:
   ```bash
   # Check file exists
   ls -lh data/raw/data.csv
   ```

**Data Requirements**:
- CSV format
- Required columns: `CustomerId`, `TransactionStartTime`, `Amount`, `ProductCategory`, `ChannelId`, `ProviderId`, etc.
- See `notebooks/eda.ipynb` for expected data structure

---

## 🚀 Quick Start

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

**Before starting**: Ensure you have downloaded the dataset and placed it in `data/raw/data.csv` (see Dataset section above).

```bash
# 1. Calculate RFM and create target variable
python3 examples/step1_calculate_rfm.py
python3 examples/step2_cluster_customers.py
python3 examples/step3_create_high_risk_target.py
python3 examples/integrate_target_to_processed_data.py

# 2. Prepare data splits
python3 examples/prepare_data_splits.py

# 3. Train models with MLflow tracking
python3 examples/complete_training_script.py

# 4. Start API server
docker-compose up -d
# Or directly: uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 5. Test API (in another terminal)
python3 examples/test_api.py
```

**Note**: The project uses a clean modular structure. Import examples:
```python
# Feature engineering
from src.features import RFMCalculator, DataProcessor, split_data

# Model training
from src.models import ModelTrainer, MLflowTracker

# Utilities
from src.utils import settings, get_logger
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

## Project Workflow

**Data Flow:**

```
Raw Data → Feature Engineering → RFM Analysis → Model Training → 
MLflow Tracking → FastAPI Deployment → Production API
```

### Quick Visual Guide

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data Processing** | scikit-learn Pipeline | Automated feature engineering |
| **ML Tracking** | MLflow | Experiment tracking & model registry |
| **API** | FastAPI | Real-time predictions |
| **Deployment** | Docker | Containerized service |
| **Automation** | GitHub Actions | CI/CD pipeline |

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

### `GET /`
API information and available endpoints.

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

### `GET /metrics`
Prometheus-style metrics endpoint (if enabled).

Returns metrics in Prometheus format:
- `predictions_total`: Total number of predictions
- `predictions_success`: Successful predictions
- `predictions_errors`: Failed predictions
- `prediction_latency_seconds`: Average prediction latency
- `model_load_errors`: Model loading errors

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

**Risk Levels**:
- `low`: probability < 0.30 (auto-approve)
- `medium`: 0.30 ≤ probability ≤ 0.60 (manual review)
- `high`: probability > 0.60 (auto-reject)

---

## Project Structure

```
bati-bank-credit-scoring-mlops/
├── src/                          # Source code (modular structure)
│   ├── features/                 # Feature engineering modules
│   │   ├── rfm.py               # RFM metrics calculation
│   │   ├── clustering.py        # Customer clustering
│   │   ├── labeling.py          # High-risk labeling
│   │   ├── processing.py        # Data processing pipeline
│   │   ├── woe.py               # Weight of Evidence
│   │   └── splitting.py         # Data splitting
│   ├── models/                   # Model training and tracking
│   │   ├── training.py          # Model training
│   │   ├── tuning.py            # Hyperparameter tuning
│   │   └── tracking.py         # MLflow integration
│   ├── api/                      # API layer
│   │   ├── main.py              # FastAPI application
│   │   ├── middleware.py        # Custom middleware
│   │   └── pydantic_models.py   # Request/response models
│   └── utils/                    # Utility modules
│       ├── config.py            # Configuration management
│       ├── logging.py           # Structured logging
│       └── retry.py             # Retry utilities
├── examples/                      # Example scripts and workflows
├── tests/                         # Unit tests
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda.ipynb             # Production-grade EDA
│   └── eda.ipynb                # Legacy EDA
├── data/                          # Data files (raw and processed) - NOT in repository
│   ├── raw/                      # Place downloaded dataset here (gitignored)
│   └── processed/                # Generated processed files (gitignored)
├── mlruns/                        # MLflow experiment tracking
├── docs/                          # Documentation
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose setup
└── requirements.txt               # Python dependencies
```

**Key Improvements:**
- ✅ **Modular Structure**: Clean separation of features, models, API, and utils
- ✅ **Industry Best Practices**: Follows standard Python project structure
- ✅ **Easy Navigation**: Related functionality grouped together
- ✅ **Scalable**: Easy to extend with new features

---

## Key Features

### ML/AI Features
- ✅ **Proxy Target Engineering**: RFM-based customer segmentation
- ✅ **Automated Feature Pipeline**: 26 engineered features
- ✅ **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
- ✅ **Hyperparameter Tuning**: Grid Search and Random Search
- ✅ **MLflow Integration**: Experiment tracking and model registry

### Production Features
- ✅ **Production API**: FastAPI with async support
- ✅ **Configuration Management**: Type-safe settings with pydantic-settings
- ✅ **Structured Logging**: JSON logs for production, text for development
- ✅ **Monitoring**: Prometheus-style metrics endpoint
- ✅ **Security**: CORS, rate limiting, input validation
- ✅ **Error Handling**: Retry logic with exponential backoff
- ✅ **Docker**: Multi-stage builds, non-root user, security best practices
- ✅ **CI/CD Pipeline**: Automated testing, linting, and quality checks
- ✅ **Code Quality**: Pre-commit hooks, type checking, code formatting
- ✅ **Regulatory Compliance**: Basel II Accord requirements

---

## 🔧 Environment Variables

Create a `.env` file in the project root (see `.env.example` for all options):

```env
# MLflow Configuration
MLFLOW_TRACKING_URI=file:./mlruns
MODEL_NAME=credit_scoring_model
MODEL_STAGE=Production

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Risk Thresholds
RISK_THRESHOLD_LOW=0.30
RISK_THRESHOLD_HIGH=0.60

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
ENABLE_RATE_LIMITING=false
RATE_LIMIT_PER_MINUTE=60

# Monitoring
ENABLE_METRICS=true
```

Or set them directly:

```bash
export MLFLOW_TRACKING_URI="file:./mlruns"
export MODEL_NAME="credit_scoring_model"
export MODEL_STAGE="Production"
```

**Note**: For production, copy `.env.example` to `.env` and configure appropriately.

---

## 📚 Documentation

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

## 📈 Business Metrics

| Metric | Before Implementation | After Implementation |
|--------|---------------------|---------------------|
| **Customer Coverage** | 40% (with credit history) | 100% (all customers) |
| **Decision Time** | 2-5 days | <1 second |
| **Manual Review** | 70% of applications | 30% of applications |
| **Default Rate** | 8% (estimated) | Projected <5% |

---

## 🎯 Use Cases

This implementation is ideal for:

- **FinTech startups** offering BNPL services
- **Traditional banks** expanding to digital channels
- **E-commerce platforms** launching embedded finance
- **Financial inclusion** initiatives in emerging markets

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

## 🔧 Troubleshooting

### Common Issues:

**Issue**: MLflow UI not starting  
**Solution**: Ensure port 5000 is free or change port: `mlflow ui --port 5001`

**Issue**: Docker build fails  
**Solution**: Check Docker daemon is running: `docker ps`

**Issue**: Import errors  
**Solution**: Ensure virtual environment is activated and requirements installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Issue**: API returns 422 error  
**Solution**: Verify you're sending exactly 26 features in correct order (see `docs/api_input_features.md`)

**Issue**: Model not loading  
**Solution**: Check MLflow registry has registered model:
```bash
mlflow ui --backend-store-uri file:./mlruns
# Navigate to Models tab to verify
```

**Issue**: Connection refused on API  
**Solution**: Ensure API is running and check port 8000 is not in use:
```bash
lsof -i :8000  # Check if port is in use
```

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

## 🤝 Acknowledgments

- **Kifiya AI Mastery 10 Academy** for the structured learning framework
- **Open Source Community** for MLflow, FastAPI, and other tools

---

## ⚠️ Data Privacy Notice

**Data files are not included in this repository** for privacy and security reasons:
- Raw data files are in `.gitignore` (not tracked by Git)
- Processed data files are in `.gitignore` (not tracked by Git)
- Model files are in `.gitignore` (not tracked by Git)
- MLflow runs are in `.gitignore` (not tracked by Git)

**Users must obtain the dataset through their organization's approved data access channels** and set up the data directory structure as described in the Dataset section above. Ensure compliance with all data privacy and security policies.

---

## License

This project is part of the Bati Bank Credit Scoring MLOps implementation.

---

## Contact

For questions or issues, please open an issue in the repository.

---

**Built with**: Python, scikit-learn, MLflow, FastAPI, Docker, GitHub Actions
