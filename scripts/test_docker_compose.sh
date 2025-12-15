#!/bin/bash
# Test script for Docker Compose setup

set -e

echo "=========================================="
echo "Testing Docker Compose Setup"
echo "=========================================="
echo

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose not found${NC}"
    echo "Please install docker-compose"
    exit 1
fi

# Check if MLflow model exists
echo "Step 1: Checking MLflow model registry..."
if [ -d "mlruns" ]; then
    echo -e "${GREEN}✓ mlruns directory exists${NC}"
else
    echo -e "${YELLOW}⚠ mlruns directory not found${NC}"
    echo "   The API will start but model loading may fail."
    echo "   Make sure you have trained and registered a model first."
    echo
fi

# Build and start containers
echo
echo "Step 2: Building Docker image..."
docker-compose build

echo
echo "Step 3: Starting containers..."
docker-compose up -d

echo
echo "Step 4: Waiting for API to be ready..."
sleep 5

# Check if container is running
if docker ps | grep -q credit-scoring-api; then
    echo -e "${GREEN}✓ Container is running${NC}"
else
    echo -e "${RED}✗ Container is not running${NC}"
    echo "Check logs with: docker-compose logs"
    exit 1
fi

# Test health endpoint
echo
echo "Step 5: Testing /health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health || echo "FAILED")

if [ "$HEALTH_RESPONSE" != "FAILED" ]; then
    echo -e "${GREEN}✓ Health endpoint is responding${NC}"
    echo "Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}✗ Health endpoint failed${NC}"
    echo "Check logs with: docker-compose logs"
    exit 1
fi

# Test predict endpoint
echo
echo "Step 6: Testing /predict endpoint..."
PREDICT_RESPONSE=$(curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "features": [0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994, -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }' || echo "FAILED")

if [ "$PREDICT_RESPONSE" != "FAILED" ]; then
    echo -e "${GREEN}✓ Predict endpoint is responding${NC}"
    echo "Response: $PREDICT_RESPONSE"
else
    echo -e "${YELLOW}⚠ Predict endpoint failed (may be due to missing model)${NC}"
    echo "Check logs with: docker-compose logs"
fi

echo
echo "=========================================="
echo "Testing Complete!"
echo "=========================================="
echo
echo "Useful commands:"
echo "  View logs:        docker-compose logs -f"
echo "  Stop containers:  docker-compose down"
echo "  Restart:          docker-compose restart"
echo "  View API docs:    http://localhost:8000/docs"
echo

