"""
FastAPI REST API for Credit Scoring Model

This API provides endpoints for credit risk prediction using the trained model
loaded from MLflow Model Registry.

Endpoints:
- GET /: API information
- GET /health: Health check
- GET /metrics: Prometheus-style metrics
- POST /predict: Predict credit risk for customer data
- POST /explain: Get SHAP-based explanation for a prediction
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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
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
    HealthResponse,
    ExplanationResponse,
    FeatureImportance
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

# Global explainer variable
explainer: Optional[object] = None
background_data: Optional[np.ndarray] = None
feature_names: Optional[list] = None


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


def load_background_data() -> Optional[np.ndarray]:
    """
    Load background data for SHAP explainer from training data splits.
    
    Returns:
        Background data array or None if not available
    """
    try:
        from src.features.splitting import load_splits
        import pandas as pd
        
        splits_dir = project_root / "data" / "processed" / "splits"
        
        if not splits_dir.exists():
            logger.warning("Training data splits not found, explainability will use model defaults")
            return None
        
        # Load training data
        X_train, _, _, _ = load_splits(str(splits_dir))
        
        # Sample a subset for background (SHAP works better with smaller samples)
        n_samples = min(100, len(X_train))
        if len(X_train) > n_samples:
            X_train_sample = X_train.sample(n=n_samples, random_state=42)
        else:
            X_train_sample = X_train
        
        # Convert to numpy array
        background = X_train_sample.values.astype(np.float64)
        
        logger.info(
            f"Loaded background data for SHAP: {background.shape[0]} samples, {background.shape[1]} features"
        )
        
        return background
        
    except Exception as e:
        logger.warning(
            f"Could not load background data for SHAP: {e}. Explainability may be limited."
        )
        return None


def get_feature_names() -> list:
    """
    Get feature names for model interpretability.
    
    Returns:
        List of feature names
    """
    try:
        from src.features.splitting import load_splits
        
        splits_dir = project_root / "data" / "processed" / "splits"
        
        if splits_dir.exists():
            X_train, _, _, _ = load_splits(str(splits_dir))
            return list(X_train.columns)
    except Exception:
        pass
    
    # Fallback to generic feature names
    return [f"feature_{i}" for i in range(settings.expected_features)]


def initialize_explainer():
    """Initialize the SHAP explainer if model is loaded."""
    global explainer, background_data, feature_names
    
    if model is None:
        return
    
    try:
        from src.models.explainability import ModelExplainer
        
        # Get feature names
        feature_names = get_feature_names()
        
        # Load background data
        background_data = load_background_data()
        
        # Initialize explainer
        explainer = ModelExplainer(
            model=model,
            background_data=background_data,
            feature_names=feature_names,
            explainer_type="auto"
        )
        
        logger.info("SHAP explainer initialized successfully")
        
    except ImportError as e:
        logger.warning(
            f"SHAP not available, explainability endpoints will not work: {e}"
        )
    except Exception as e:
        logger.error(
            f"Failed to initialize SHAP explainer: {e}",
            exc_info=True
        )


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
        
        # Initialize explainer after model is loaded
        initialize_explainer()
        
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

# Mount static files for dashboard
static_dir = project_root / "src" / "api" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


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
            "explain": "/explain",
            "dashboard": "/dashboard",
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
        
        response_data = {
            "prediction": int(prediction),
            "probability": probability,
            "risk_level": risk_level
        }
        
        # Add explanation if requested
        if request.include_explanation:
            if explainer is None:
                logger.warning("Explanation requested but explainer not available")
                response_data["explanation"] = None
            else:
                try:
                    explanation = explainer.explain_instance(features_array)
                    # Convert to API format
                    response_data["explanation"] = {
                        "base_value": explanation["base_value"],
                        "explanation_summary": explanation["explanation_summary"],
                        "feature_importance": [
                            {
                                "feature": feat["feature"],
                                "shap_value": feat["shap_value"],
                                "feature_value": feat["feature_value"]
                            }
                            for feat in explanation["feature_importance"]
                        ],
                        "shap_values": explanation["shap_values"],
                        "feature_names": explanation["feature_names"]
                    }
                except Exception as e:
                    logger.error(f"Error generating explanation: {e}", exc_info=True)
                    response_data["explanation"] = None
        
        return PredictionResponse(**response_data)
        
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


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def explain_prediction(
    request: PredictionRequest,
    include_plot: bool = False
):
    """
    Generate SHAP-based explanation for a credit risk prediction.
    
    This endpoint provides model interpretability to meet regulatory requirements:
    - CFPB adverse action notifications
    - EU AI Act transparency obligations for high-risk AI systems
    
    Args:
        request: PredictionRequest containing feature values
        include_plot: Whether to include base64-encoded waterfall plot
    
    Returns:
        ExplanationResponse with SHAP values, feature importance, and explanation summary
    
    Raises:
        HTTPException: If model/explainer not loaded or explanation fails
    """
    global model, explainer
    
    if model is None:
        logger.error("Explanation attempted but model is not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    if explainer is None:
        logger.error("Explanation attempted but explainer is not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model explainer not available. SHAP may not be installed or initialization failed."
        )
    
    try:
        # Validate feature count
        if len(request.features) != settings.expected_features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected {settings.expected_features} features, got {len(request.features)}"
            )
        
        # Convert features to numpy array
        try:
            features_array = np.array(request.features, dtype=np.float64).reshape(1, -1)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid feature values: {str(e)}"
            )
        
        # Validate feature values
        if not np.isfinite(features_array).all():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Feature values must be finite numbers"
            )
        
        # Generate explanation
        explanation = explainer.explain_instance(features_array)
        
        # Generate waterfall plot if requested
        waterfall_plot = None
        if include_plot:
            try:
                waterfall_plot = explainer.plot_waterfall(features_array)
            except Exception as e:
                logger.warning(f"Could not generate waterfall plot: {e}")
        
        # Format feature importance
        feature_importance = [
            FeatureImportance(
                feature=feat["feature"],
                shap_value=feat["shap_value"],
                feature_value=feat["feature_value"]
            )
            for feat in explanation["feature_importance"]
        ]
        
        logger.info(
            "Explanation generated successfully",
            extra={
                "prediction": explanation["prediction"],
                "probability": explanation["probability"]
            }
        )
        
        return ExplanationResponse(
            prediction=explanation["prediction"],
            probability=explanation["probability"],
            base_value=explanation["base_value"],
            explanation_summary=explanation["explanation_summary"],
            feature_importance=feature_importance,
            shap_values=explanation["shap_values"],
            feature_names=explanation["feature_names"],
            waterfall_plot=waterfall_plot
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Explanation error",
            extra={"error": str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


@app.get("/dashboard", tags=["Dashboard"])
async def dashboard():
    """
    Serve the interactive credit scoring dashboard.
    
    This endpoint provides a user-friendly web interface for loan officers
    and credit analysts to explore risk profiles, test scenarios, and
    understand predictions without writing code.
    """
    dashboard_path = project_root / "src" / "api" / "templates" / "dashboard.html"
    
    if not dashboard_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard not found"
        )
    
    return FileResponse(dashboard_path)


@app.get("/api/feature-names", tags=["Dashboard"])
async def get_feature_names_endpoint():
    """
    Get feature names for the dashboard.
    
    Returns the list of feature names used by the model for display
    in the interactive dashboard.
    """
    try:
        feature_names_list = get_feature_names()
        return JSONResponse({
            "feature_names": feature_names_list,
            "count": len(feature_names_list)
        })
    except Exception as e:
        logger.error(f"Error getting feature names: {e}", exc_info=True)
        # Return default feature names
        return JSONResponse({
            "feature_names": [f"feature_{i}" for i in range(settings.expected_features)],
            "count": settings.expected_features
        })


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
