"""
FastAPI REST API for Credit Scoring Model

This API provides endpoints for credit risk prediction using the trained model
loaded from MLflow Model Registry.

Endpoints:
- GET /: API information
- GET /health: Health check
- GET /metrics: Prometheus-style metrics
- POST /predict: Predict credit risk for customer data
"""

import sys
from pathlib import Path
from typing import Optional
import time
import numpy as np
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import settings
from src.utils.logging import get_logger
from src.utils.retry import retry
from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)

# Initialize logger
logger = get_logger(__name__)

# Metrics storage (in production, use Prometheus client)
metrics = {
    "predictions_total": 0,
    "predictions_success": 0,
    "predictions_errors": 0,
    "prediction_latency_seconds": [],
    "model_load_errors": 0,
}

# Global model variable
model: Optional[object] = None
model_name: Optional[str] = None
model_version: Optional[str] = None
model_load_time: Optional[float] = None


@retry(max_attempts=3, delay=2.0, backoff=2.0, exceptions=(Exception,))
def load_model_from_mlflow(
    model_name: str,
    stage: str
) -> object:
    """
    Load model from MLflow Model Registry.
    
    Args:
        model_name: Name of the registered model
        stage: Model stage (Production, Staging, or version number)
    
    Returns:
        Loaded model object
    
    Raises:
        Exception: If model loading fails
    """
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        # Load model from registry
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(
            "Loading model from MLflow",
            extra={"model_uri": model_uri, "model_name": model_name, "stage": stage}
        )
        
        model = mlflow.sklearn.load_model(model_uri)
        
        logger.info(
            "Model loaded successfully",
            extra={"model_name": model_name, "stage": stage}
        )
        
        return model
        
    except Exception as e:
        logger.error(
            "Error loading model from MLflow",
            extra={"error": str(e), "model_name": model_name, "stage": stage},
            exc_info=True
        )
        metrics["model_load_errors"] += 1
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global model, model_name, model_version, model_load_time
    
    # Startup
    logger.info("Starting Credit Scoring API", extra={"version": "1.0.0"})
    
    try:
        model_name = settings.model_name
        model_stage = settings.model_stage
        
        start_time = time.time()
        model = load_model_from_mlflow(model_name, model_stage)
        model_load_time = time.time() - start_time
        model_version = model_stage
        
        logger.info(
            "Model loaded successfully on startup",
            extra={
                "model_name": model_name,
                "model_version": model_version,
                "load_time_seconds": model_load_time
            }
        )
        
    except Exception as e:
        logger.error(
            "Failed to load model on startup",
            extra={"error": str(e)},
            exc_info=True
        )
        logger.warning(
            "API will start but /predict endpoint will not work until model is loaded"
        )
    
    yield
    
    # Shutdown
    logger.info("Shutting down Credit Scoring API")


# Initialize FastAPI app
app = FastAPI(
    title="Credit Scoring API",
    description="API for credit risk prediction using MLflow-registered models",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Add custom middleware
from src.api.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Add error handling middleware
app.add_middleware(ErrorHandlingMiddleware)

# Add rate limiting middleware if enabled
if settings.enable_rate_limiting:
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_per_minute
    )


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Scoring API",
        "version": "1.0.0",
        "environment": settings.environment,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics" if settings.enable_metrics else None,
            "docs": "/docs" if not settings.is_production else None
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and model.
    """
    global model, model_name, model_version
    
    status_value = "healthy" if model is not None else "degraded"
    
    health_data = {
        "status": status_value,
        "model_loaded": model is not None,
        "model_name": model_name if model is not None else None,
        "model_version": model_version if model is not None else None,
    }
    
    if model_load_time:
        health_data["model_load_time_seconds"] = model_load_time
    
    return HealthResponse(**health_data)


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Prometheus-style metrics endpoint.
    
    Returns application metrics in a format compatible with Prometheus.
    """
    if not settings.enable_metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics endpoint is disabled"
        )
    
    # Calculate average latency
    avg_latency = (
        sum(metrics["prediction_latency_seconds"]) / len(metrics["prediction_latency_seconds"])
        if metrics["prediction_latency_seconds"]
        else 0.0
    )
    
    # Format as Prometheus metrics
    metrics_text = f"""# HELP predictions_total Total number of predictions
# TYPE predictions_total counter
predictions_total {metrics['predictions_total']}

# HELP predictions_success Total number of successful predictions
# TYPE predictions_success counter
predictions_success {metrics['predictions_success']}

# HELP predictions_errors Total number of prediction errors
# TYPE predictions_errors counter
predictions_errors {metrics['predictions_errors']}

# HELP prediction_latency_seconds Average prediction latency in seconds
# TYPE prediction_latency_seconds gauge
prediction_latency_seconds {avg_latency}

# HELP model_load_errors Total number of model load errors
# TYPE model_load_errors counter
model_load_errors {metrics['model_load_errors']}
"""
    
    return metrics_text


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
    global model
    
    if model is None:
        logger.error("Prediction attempted but model is not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    start_time = time.time()
    metrics["predictions_total"] += 1
    
    try:
        # Validate feature count
        if len(request.features) != settings.expected_features:
            logger.warning(
                "Invalid feature count",
                extra={
                    "expected": settings.expected_features,
                    "received": len(request.features)
                }
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected {settings.expected_features} features, got {len(request.features)}"
            )
        
        # Convert features to numpy array
        try:
            features_array = np.array(request.features, dtype=np.float64).reshape(1, -1)
        except (ValueError, TypeError) as e:
            logger.warning("Invalid feature values", extra={"error": str(e)})
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid feature values: {str(e)}"
            )
        
        # Validate feature values (check for NaN, Inf)
        if not np.isfinite(features_array).all():
            logger.warning("Non-finite feature values detected")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Feature values must be finite numbers"
            )
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Get prediction probability
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)[0]
            probability = float(probabilities[1])  # Probability of high-risk class
        else:
            # Fallback if model doesn't support predict_proba
            probability = float(prediction)
        
        # Determine risk level based on thresholds
        if probability < settings.risk_threshold_low:
            risk_level = "low"
        elif probability > settings.risk_threshold_high:
            risk_level = "high"
        else:
            risk_level = "medium"
        
        # Calculate latency
        latency = time.time() - start_time
        metrics["prediction_latency_seconds"].append(latency)
        metrics["predictions_success"] += 1
        
        # Keep only last 1000 latency measurements
        if len(metrics["prediction_latency_seconds"]) > 1000:
            metrics["prediction_latency_seconds"] = metrics["prediction_latency_seconds"][-1000:]
        
        logger.info(
            "Prediction completed",
            extra={
                "prediction": int(prediction),
                "probability": probability,
                "risk_level": risk_level,
                "latency_seconds": latency
            }
        )
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level
        )
        
    except HTTPException:
        metrics["predictions_errors"] += 1
        raise
    except Exception as e:
        metrics["predictions_errors"] += 1
        logger.error(
            "Prediction error",
            extra={"error": str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please try again later."
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers if not settings.api_reload else 1,
        reload=settings.api_reload,
        log_config=None  # Use our custom logging
    )
