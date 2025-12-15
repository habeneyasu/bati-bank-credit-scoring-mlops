#!/bin/bash
# Script to start the FastAPI server

# Set environment variables
export MLFLOW_TRACKING_URI="file:./mlruns"
export MODEL_NAME="credit_scoring_model"
export MODEL_STAGE="Production"

# Start the API server
echo "Starting FastAPI server..."
echo "API will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

