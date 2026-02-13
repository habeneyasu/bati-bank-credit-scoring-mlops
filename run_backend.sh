#!/bin/bash
# Script to run the backend API directly without Docker

# Activate virtual environment
source venv/bin/activate

# Set environment variables (optional - defaults are provided in config.py)
export MLFLOW_TRACKING_URI="file:./mlruns"
export MODEL_NAME="credit_scoring_model"
export MODEL_STAGE="Production"
export API_PORT=8000
export API_HOST="0.0.0.0"

# Run the backend
echo "Starting Credit Scoring API backend..."
echo "API will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
