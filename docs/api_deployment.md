# API Deployment Guide

## Overview

This guide explains how to deploy the Credit Scoring API using Docker and Docker Compose.

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Build and start the service
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

The API will be available at: http://localhost:8000

### 2. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994, -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }'
```

### 3. View API Documentation

Open in browser: http://localhost:8000/docs

FastAPI automatically generates interactive API documentation.

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "credit_scoring_model",
  "model_version": "Production"
}
```

### POST /predict
Predict credit risk for customer data.

**Request:**
```json
{
  "features": [0.0, -0.046, -0.072, ...]  // 26 feature values
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.15,
  "risk_level": "low"
}
```

## Environment Variables

Configure via environment variables or docker-compose.yml:

- `MLFLOW_TRACKING_URI`: MLflow tracking URI (default: `file:./mlruns`)
- `MODEL_NAME`: Registered model name (default: `credit_scoring_model`)
- `MODEL_STAGE`: Model stage (default: `Production`)
- `PORT`: API port (default: `8000`)

## Docker Commands

### Build Image
```bash
docker build -t credit-scoring-api .
```

### Run Container
```bash
docker run -p 8000:8000 \
  -v $(pwd)/mlruns:/app/mlruns \
  -e MODEL_NAME=credit_scoring_model \
  -e MODEL_STAGE=Production \
  credit-scoring-api
```

### View Logs
```bash
docker-compose logs -f
```

### Stop Service
```bash
docker-compose down
```

## Troubleshooting

### Model Not Loading

1. Check MLflow model registry:
   ```bash
   mlflow models list -r models:/credit_scoring_model
   ```

2. Verify model exists in `mlruns/` directory

3. Check API logs:
   ```bash
   docker-compose logs credit-scoring-api
   ```

### Port Already in Use

Change port in docker-compose.yml:
```yaml
ports:
  - "8001:8000"  # Use different host port
```

### Model Registry Not Found

Ensure `mlruns/` directory is mounted:
```yaml
volumes:
  - ./mlruns:/app/mlruns
```

## Production Deployment

For production:

1. Use environment-specific model stages
2. Set up proper authentication
3. Configure CORS for specific origins
4. Use HTTPS
5. Set up monitoring and logging
6. Use orchestration (Kubernetes, ECS, etc.)

---

**Status**: ✅ API Deployment Ready

