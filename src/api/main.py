"""
FastAPI REST API for Credit Scoring Model

This API provides endpoints for credit risk prediction using the trained model
loaded from MLflow Model Registry.

Endpoints:
- GET /health: Health check
- POST /predict: Predict credit risk for customer data
"""

import os
import sys
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Scoring API",
    description="API for credit risk prediction using MLflow-registered models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
model_name = None
model_version = None


def load_model_from_mlflow(
    model_name: str = "credit_scoring_model",
    stage: str = "Production"
):
    """
    Load model from MLflow Model Registry.
    
    Args:
        model_name: Name of the registered model
        stage: Model stage (Production, Staging, or version number)
    
    Returns:
        Loaded model object
    """
    try:
        # Set MLflow tracking URI
        mlflow_tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "file:./mlruns"
        )
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Load model from registry
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading model from: {model_uri}")
        
        model = mlflow.sklearn.load_model(model_uri)
        
        logger.info(f"Model loaded successfully: {model_name} ({stage})")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global model, model_name, model_version
    
    try:
        model_name = os.getenv("MODEL_NAME", "credit_scoring_model")
        model_stage = os.getenv("MODEL_STAGE", "Production")
        
        model = load_model_from_mlflow(model_name, model_stage)
        model_version = model_stage
        
        logger.info("Model loaded successfully on startup")
        
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        logger.warning("API will start but /predict endpoint will not work until model is loaded")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Scoring API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and model.
    """
    # Access global variables (no assignment, so no 'global' needed)
    current_model = model
    current_model_name = model_name
    current_model_version = model_version
    
    status = "healthy" if current_model is not None else "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=current_model is not None,
        model_name=current_model_name if current_model is not None else None,
        model_version=current_model_version if current_model is not None else None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict credit risk for customer data.
    
    Args:
        request: PredictionRequest containing feature values
    
    Returns:
        PredictionResponse with prediction, probability, and risk level
    
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Access global model (no assignment, so no 'global' needed)
    current_model = model
    
    if current_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert features to numpy array
        features_array = np.array(request.features).reshape(1, -1)
        
        # Validate feature count
        expected_features = 26  # Based on processed data features
        if len(request.features) != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_features} features, got {len(request.features)}"
            )
        
        # Make prediction
        prediction = current_model.predict(features_array)[0]
        
        # Get prediction probability
        if hasattr(current_model, 'predict_proba'):
            probabilities = current_model.predict_proba(features_array)[0]
            probability = float(probabilities[1])  # Probability of high-risk class
        else:
            # Fallback if model doesn't support predict_proba
            probability = float(prediction)
        
        # Determine risk level
        risk_level = "high" if prediction == 1 else "low"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
