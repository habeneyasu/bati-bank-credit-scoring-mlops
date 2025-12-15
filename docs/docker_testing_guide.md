# Docker Compose Testing Guide

This guide walks you through testing the Docker Compose setup for the Credit Scoring API.

## Prerequisites

1. **Docker and Docker Compose installed**
   ```bash
   docker --version
   docker-compose --version
   ```

2. **MLflow Model Registered** (optional but recommended)
   - Ensure you have trained and registered a model in MLflow
   - The model should be registered as `credit_scoring_model` in `Production` stage
   - If not, the API will start but predictions will fail

## Quick Test

### Option 1: Automated Test Script

```bash
# Run the automated test script
./scripts/test_docker_compose.sh
```

### Option 2: Manual Testing

#### Step 1: Build and Start

```bash
# Build the Docker image
docker-compose build

# Start the container in detached mode
docker-compose up -d

# Or run in foreground to see logs
docker-compose up
```

#### Step 2: Check Container Status

```bash
# Check if container is running
docker ps | grep credit-scoring-api

# Or use docker-compose
docker-compose ps
```

#### Step 3: Test Health Endpoint

```bash
# Test health check
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "model_name": "credit_scoring_model",
#   "model_version": "Production"
# }
```

#### Step 4: Test Predict Endpoint

```bash
# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994, -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }'

# Expected response:
# {
#   "prediction": 0,
#   "probability": 0.15,
#   "risk_level": "low"
# }
```

#### Step 5: View API Documentation

Open in browser: http://localhost:8000/docs

FastAPI automatically generates interactive Swagger documentation.

#### Step 6: Test with Python Script

```bash
# Install requests if needed
pip install requests

# Run the test script
python examples/test_api.py
```

## Viewing Logs

```bash
# View all logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# View logs for specific service
docker-compose logs credit-scoring-api

# View last 100 lines
docker-compose logs --tail=100
```

## Troubleshooting

### Container Won't Start

1. **Check logs:**
   ```bash
   docker-compose logs
   ```

2. **Check if port 8000 is already in use:**
   ```bash
   lsof -i :8000
   # Or
   netstat -tulpn | grep 8000
   ```
   
   If port is in use, change it in `docker-compose.yml`:
   ```yaml
   ports:
     - "8001:8000"  # Use different host port
   ```

### Model Not Loading

1. **Check if model exists:**
   ```bash
   # Check MLflow registry
   mlflow models list -r models:/credit_scoring_model
   ```

2. **Verify mlruns directory:**
   ```bash
   ls -la mlruns/
   ```

3. **Check container logs for errors:**
   ```bash
   docker-compose logs credit-scoring-api | grep -i error
   ```

4. **Verify volume mount:**
   ```bash
   docker-compose exec credit-scoring-api ls -la /app/mlruns
   ```

### Health Check Fails

1. **Check if API is responding:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check container health:**
   ```bash
   docker ps  # Check STATUS column
   ```

3. **Restart container:**
   ```bash
   docker-compose restart
   ```

### Predictions Return 503 Error

This means the model is not loaded. Check:
1. Model is registered in MLflow
2. `mlruns/` directory is mounted correctly
3. Model name and stage match environment variables

## Stopping the Service

```bash
# Stop containers (keeps them)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop, remove containers, and remove volumes
docker-compose down -v
```

## Rebuilding After Code Changes

```bash
# Rebuild and restart
docker-compose up -d --build

# Or force rebuild
docker-compose build --no-cache
docker-compose up -d
```

## Environment Variables

You can override environment variables:

```bash
# Using .env file
echo "MODEL_STAGE=Staging" > .env
docker-compose up -d

# Or inline
MODEL_STAGE=Staging docker-compose up -d
```

## Production Considerations

For production deployment:

1. **Use environment-specific configurations**
2. **Set up proper logging** (e.g., ELK stack)
3. **Add monitoring** (e.g., Prometheus, Grafana)
4. **Use HTTPS** with reverse proxy (nginx, traefik)
5. **Set resource limits** in docker-compose.yml:
   ```yaml
   services:
     credit-scoring-api:
       deploy:
         resources:
           limits:
             cpus: '2'
             memory: 2G
   ```

## Next Steps

- ✅ API is running and accessible
- ✅ Health checks passing
- ✅ Predictions working
- 🔄 Set up CI/CD pipeline
- 🔄 Deploy to cloud (AWS, GCP, Azure)
- 🔄 Add monitoring and alerting

---

**Status**: ✅ Docker Compose Testing Ready

