#!/bin/bash
# Start MLflow UI to view experiments

echo "Starting MLflow UI..."
echo "MLflow runs directory: ./mlruns"
echo ""
echo "Access MLflow UI at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

mlflow ui --backend-store-uri file:./mlruns --port 5000

