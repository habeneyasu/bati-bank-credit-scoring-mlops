# FastAPI Testing Guide

This guide shows you how to test the Credit Scoring API endpoints.

---

## Prerequisites

1. **Model must be registered in MLflow**: Ensure you've trained models and registered the best one in MLflow Model Registry
2. **MLflow runs directory**: The `mlruns/` directory should contain your experiments

---

## Method 1: Using Docker (Recommended)

### Step 1: Start the API

```bash
# Build and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

The API will be available at: `http://localhost:8000`

### Step 2: Check if the API is running

```bash
# Check container status
docker ps

# View logs
docker-compose logs -f
```

---

## Method 2: Running Directly (Without Docker)

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set environment variables

```bash
export MLFLOW_TRACKING_URI="file:./mlruns"
export MODEL_NAME="credit_scoring_model"
export MODEL_STAGE="Production"
```

### Step 3: Start the API

```bash
# From project root
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

---

## Testing the Endpoints

### Option 1: Using the Test Script (Easiest)

```bash
# Make sure API is running first
python examples/test_api.py
```

This script will:
- Test the `/health` endpoint
- Test the `/predict` endpoint with sample data
- Test error handling

### Option 2: Using curl

#### Test Health Endpoint

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "credit_scoring_model",
  "model_version": "1"
}
```

#### Test Predict Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994, -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }'
```

**Expected Response:**
```json
{
  "prediction": 0,
  "probability": 0.234,
  "risk_level": "low"
}
```

### Option 3: Using Python requests

```python
import requests
import json

# API base URL
API_URL = "http://localhost:8000"

# Test health endpoint
response = requests.get(f"{API_URL}/health")
print("Health Check:", response.json())

# Test predict endpoint
payload = {
    "features": [
        0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994,
        -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
}

response = requests.post(
    f"{API_URL}/predict",
    json=payload,
    headers={"Content-Type": "application/json"}
)

print("Prediction:", response.json())
```

### Option 4: Using FastAPI Interactive Docs

The FastAPI automatically generates interactive API documentation:

1. Start the API (Docker or direct)
2. Open your browser and go to: `http://localhost:8000/docs`
3. You'll see the Swagger UI with all endpoints
4. Click "Try it out" on any endpoint to test it interactively

**Alternative docs**: `http://localhost:8000/redoc` (ReDoc format)

---

## Getting Real Feature Data

To test with actual data from your processed dataset:

```python
import pandas as pd
import requests

# Load processed data
df = pd.read_csv('data/processed/processed_data_with_target.csv')

# Get features (exclude target column)
features_df = df.drop(columns=['is_high_risk'])

# Get first row as example
sample_features = features_df.iloc[0].values.tolist()

# Make prediction request
payload = {"features": sample_features}

response = requests.post(
    "http://localhost:8000/predict",
    json=payload
)

print("Prediction:", response.json())
print("Actual target:", df.iloc[0]['is_high_risk'])
```

---

## Understanding the Request Format

### Prediction Request

The `/predict` endpoint expects:
- **26 features** (matching your processed feature set)
- Features should be in the same order as your training data
- All features should be numeric (float)

**Request Structure:**
```json
{
  "features": [float, float, float, ...]  // 26 features
}
```

### Response Format

**Success Response (200):**
```json
{
  "prediction": 0,           // 0 = low-risk, 1 = high-risk
  "probability": 0.234,      // Probability of high-risk [0, 1]
  "risk_level": "low"        // "low" or "high"
}
```

**Error Response (400):**
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Common Issues and Solutions

### Issue 1: Model Not Loaded

**Error**: `"model_loaded": false` in health check

**Solutions**:
1. Ensure model is registered in MLflow:
   ```bash
   # Check MLflow registry
   mlflow ui --backend-store-uri file:./mlruns
   # Navigate to Models tab
   ```

2. Check environment variables:
   ```bash
   echo $MLFLOW_TRACKING_URI
   echo $MODEL_NAME
   echo $MODEL_STAGE
   ```

3. Verify model exists:
   ```bash
   ls -la mlruns/models/credit_scoring_model/
   ```

### Issue 2: Wrong Number of Features

**Error**: `"ensure this value has at least 26 items"`

**Solution**: Ensure your feature vector has exactly 26 features matching the processed data format.

### Issue 3: Connection Refused

**Error**: `Connection refused` or `Could not connect`

**Solutions**:
1. Check if API is running:
   ```bash
   docker ps  # For Docker
   # Or check if uvicorn process is running
   ```

2. Check port 8000 is not in use:
   ```bash
   lsof -i :8000
   ```

3. Try different port:
   ```bash
   uvicorn src.api.main:app --port 8001
   ```

### Issue 4: Model Loading Error

**Error**: `Error loading model from MLflow`

**Solutions**:
1. Verify MLflow tracking URI is correct
2. Check model is registered in the specified stage
3. Ensure model files exist in `mlruns/` directory

---

## Testing Workflow

1. **Start the API** (Docker or direct)
2. **Test health endpoint** - Verify API is running and model is loaded
3. **Test predict endpoint** - Use sample or real data
4. **Verify predictions** - Check that predictions make sense
5. **Test error handling** - Try invalid requests

---

## Example: Complete Testing Session

```bash
# 1. Start API
docker-compose up -d

# 2. Wait a few seconds for startup
sleep 5

# 3. Test health
curl http://localhost:8000/health

# 4. Run test script
python examples/test_api.py

# 5. Test with real data (Python)
python -c "
import pandas as pd
import requests

df = pd.read_csv('data/processed/processed_data_with_target.csv')
features = df.drop(columns=['is_high_risk']).iloc[0].values.tolist()

response = requests.post(
    'http://localhost:8000/predict',
    json={'features': features}
)
print('Prediction:', response.json())
print('Actual:', df.iloc[0]['is_high_risk'])
"

# 6. View API docs
# Open http://localhost:8000/docs in browser
```

---

## API Endpoints Summary

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/health` | GET | Health check | None | Status, model info |
| `/predict` | POST | Risk prediction | `{"features": [26 floats]}` | Prediction, probability, risk level |
| `/docs` | GET | API documentation | None | Swagger UI |
| `/redoc` | GET | API documentation | None | ReDoc UI |

---

## Next Steps

- Integrate API into your application
- Set up monitoring and logging
- Configure production environment variables
- Implement authentication if needed
- Set up load balancing for production

---

For more details, see:
- `src/api/main.py` - API implementation
- `src/api/pydantic_models.py` - Request/response models
- `examples/test_api.py` - Test script

