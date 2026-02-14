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
import uuid
from datetime import datetime, timezone
import numpy as np
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, status, Depends, Body, Cookie, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
import bcrypt
import secrets
import csv
import json
import io
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import settings
from src.utils.logging import get_logger
from src.utils.retry import retry
from src.utils.cache import get_cache_manager
from src.utils.performance import get_performance_monitor, PerformanceTimer
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
        
        # #region agent log
        try:
            import json
            from pathlib import Path
            log_path = Path("/home/haben/Project/KAIM-Training-Portfolio/bati-bank-credit-scoring-mlops/.cursor/debug.log")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "id": f"log_model_load_start_{int(time.time() * 1000)}",
                    "timestamp": int(time.time() * 1000),
                    "location": "main.py:237",
                    "message": "Model loading started",
                    "data": {"model_name": model_name, "model_stage": model_stage},
                    "runId": "debug_run_1",
                    "hypothesisId": "B"
                }) + "\n")
        except: pass
        # #endregion
        
        start_time = time.time()
        model = load_model_from_mlflow(model_name, model_stage)
        model_load_time = time.time() - start_time
        model_version = model_stage
        
        # #region agent log
        try:
            import json
            from pathlib import Path
            log_path = Path("/home/haben/Project/KAIM-Training-Portfolio/bati-bank-credit-scoring-mlops/.cursor/debug.log")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "id": f"log_model_load_success_{int(time.time() * 1000)}",
                    "timestamp": int(time.time() * 1000),
                    "location": "main.py:245",
                    "message": "Model loaded successfully",
                    "data": {
                        "model_name": model_name,
                        "model_version": model_version,
                        "load_time_seconds": model_load_time,
                        "model_type": type(model).__name__ if model else None
                    },
                    "runId": "debug_run_1",
                    "hypothesisId": "B"
                }) + "\n")
        except: pass
        # #endregion
        
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
        # #region agent log
        try:
            import json, traceback
            from pathlib import Path
            log_path = Path("/home/haben/Project/KAIM-Training-Portfolio/bati-bank-credit-scoring-mlops/.cursor/debug.log")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "id": f"log_model_load_failure_{int(time.time() * 1000)}",
                    "timestamp": int(time.time() * 1000),
                    "location": "main.py:257",
                    "message": "Model loading failed",
                    "data": {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc()[:1000],
                        "model_name": model_name if 'model_name' in locals() else None,
                        "model_stage": model_stage if 'model_stage' in locals() else None
                    },
                    "runId": "debug_run_1",
                    "hypothesisId": "B"
                }) + "\n")
        except: pass
        # #endregion
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

# In-memory session store (in production, use Redis or database)
session_store = {}

# OAuth2 scheme for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


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
    
    Optimized for sub-200ms latency with caching and performance monitoring.
    
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
    
    # Performance monitoring
    perf_monitor = get_performance_monitor() if settings.enable_performance_monitoring else None
    
    # Check cache if enabled
    cache = None
    cache_key = None
    if settings.enable_prediction_cache:
        cache = get_cache_manager()
        # Generate cache key from features (excluding explanation flag)
        cache_key = cache._generate_key("prediction", request.features)
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.debug("Cache hit for prediction")
            if perf_monitor:
                perf_monitor.record_latency(0.001, "predict")  # Cache hit is very fast
            return PredictionResponse(**cached_result)
    
    # Start performance timer
    start_time = time.time()
    metrics["predictions_total"] += 1
    
    try:
        with PerformanceTimer(perf_monitor, "predict") as timer:
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
            
            # Check for active A/B testing experiments
            prediction_model = model
            experiment_id = None
            variant_name = None
            variant_model_version = model_version
            
            if request.customer_id:
                try:
                    from src.experimentation.ab_testing import get_ab_testing_framework
                    from src.database.connection import get_db_session
                    from src.database.repositories import ExperimentRepository
                    
                    ab_framework = get_ab_testing_framework()
                    
                    # Get running experiments
                    with get_db_session() as session:
                        experiment_repo = ExperimentRepository(session)
                        running_experiments = experiment_repo.get_running_experiments()
                        
                        # Check if customer is in any experiment
                        for exp in running_experiments:
                            variant = ab_framework.get_assignment(
                                exp.experiment_id,
                                request.customer_id,
                                "customer"
                            )
                            
                            if variant:
                                # Get model for this variant
                                variant_model = ab_framework.get_model_for_variant(variant)
                                if variant_model:
                                    prediction_model = variant_model
                                    experiment_id = exp.experiment_id
                                    variant_name = variant
                                    
                                    # Get model version from variant config
                                    variants = exp.variants if isinstance(exp.variants, list) else []
                                    variant_config = next((v for v in variants if v["name"] == variant), None)
                                    if variant_config:
                                        variant_model_version = variant_config.get("model_version", model_version)
                                    
                                    logger.info(
                                        f"Using variant {variant} (model v{variant_model_version}) for customer {request.customer_id} in experiment {exp.experiment_id}"
                                    )
                                    break
                except Exception as e:
                    logger.warning(f"Error checking A/B test assignment: {e}", exc_info=True)
                    # Continue with default model if A/B test check fails
            
            # Make prediction (optimized: single call to predict_proba if available)
            if hasattr(prediction_model, 'predict_proba'):
                # Use predict_proba to get both prediction and probability in one call
                probabilities = prediction_model.predict_proba(features_array)[0]
                probability = float(probabilities[1])  # Probability of high-risk class
                prediction = int(np.argmax(probabilities))  # Class with highest probability
            else:
                # Fallback if model doesn't support predict_proba
                prediction = prediction_model.predict(features_array)[0]
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
            if timer:
                timer.latency = latency  # Update timer with actual latency
            metrics["prediction_latency_seconds"].append(latency)
            metrics["predictions_success"] += 1
            
            # Keep only last 1000 latency measurements
            if len(metrics["prediction_latency_seconds"]) > 1000:
                metrics["prediction_latency_seconds"] = metrics["prediction_latency_seconds"][-1000:]
            
            # Generate prediction ID and timestamp for tracking
            prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
            timestamp = datetime.now(timezone.utc)
            timestamp_iso = timestamp.isoformat()
            
            # Save prediction to database
            try:
                from src.database.connection import get_db_session
                from src.database.services import PredictionService
                
                with get_db_session() as session:
                    prediction_service = PredictionService(session)
                    
                    # Calculate customer score (0-100 scale)
                    customer_score = int((1 - probability) * 100)
                    
                    # Store A/B test info in request_metadata
                    request_metadata = {}
                    if experiment_id:
                        request_metadata["experiment_id"] = experiment_id
                        request_metadata["variant_name"] = variant_name
                    
                    prediction_service.save_prediction(
                        prediction_id=prediction_id,
                        customer_id=request.customer_id,
                        prediction=int(prediction),
                        probability=probability,
                        risk_level=risk_level,
                        customer_score=customer_score,
                        latency_ms=latency * 1000,
                        model_name=model_name or "credit_scoring_model",
                        model_version=variant_model_version or model_version or "unknown",
                        model_stage=settings.model_stage or "Production",
                        features=request.features,
                        request_metadata=request_metadata if request_metadata else None
                    )
                    session.commit()
                    logger.debug(f"Prediction saved to database: {prediction_id}")
            except Exception as db_error:
                # Log error but don't fail the prediction
                logger.warning(
                    f"Failed to save prediction to database: {db_error}",
                    exc_info=True
                )
            
            # Log prediction with customer identification
            logger.info(
                "Prediction completed",
                extra={
                    "customer_id": request.customer_id or "unknown",
                    "prediction_id": prediction_id,
                    "prediction": int(prediction),
                    "probability": probability,
                    "risk_level": risk_level,
                    "latency_seconds": latency,
                    "latency_ms": latency * 1000,
                    "timestamp": timestamp_iso
                }
            )
            
            response_data = {
                "customer_id": request.customer_id,
                "prediction": int(prediction),
                "probability": probability,
                "risk_level": risk_level,
                "prediction_id": prediction_id,
                "timestamp": timestamp_iso
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
            
            # Cache result if caching enabled (only cache without explanation for performance)
            if cache and cache_key and not request.include_explanation:
                cache.set(cache_key, response_data, ttl=settings.cache_ttl_seconds)
            
            return PredictionResponse(**response_data)
        
    except HTTPException:
        metrics["predictions_errors"] += 1
        if perf_monitor:
            perf_monitor.record_error()
        raise
    except Exception as e:
        metrics["predictions_errors"] += 1
        if perf_monitor:
            perf_monitor.record_error()
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


@app.get("/api/fairness", tags=["Governance"])
async def get_fairness_analysis():
    """
    Get model fairness analysis metrics.
    
    Returns fairness metrics including demographic parity, equalized odds,
    calibration, and disparate impact ratio for regulatory compliance.
    """
    global model
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Cannot perform fairness analysis."
        )
    
    try:
        from src.models.fairness import FairnessAnalyzer
        from src.features.splitting import load_splits
        
        # Load test data for fairness analysis
        splits_dir = project_root / "data" / "processed" / "splits"
        
        if not splits_dir.exists():
            # Return mock data if splits not available
            logger.warning("Test data not available, returning mock fairness metrics")
            return JSONResponse({
                "demographic_parity": {
                    "value": 0.85,
                    "threshold": 0.80,
                    "status": "compliant"
                },
                "equalized_odds": {
                    "value": 0.82,
                    "threshold": 0.75,
                    "status": "compliant"
                },
                "calibration": {
                    "value": 0.88,
                    "threshold": 0.85,
                    "status": "compliant"
                },
                "disparate_impact": {
                    "value": 0.92,
                    "threshold": 0.80,
                    "status": "compliant"
                },
                "overall_status": "compliant",
                "note": "Mock data - actual analysis requires test data"
            })
        
        # Load test data
        X_test, _, y_test, _ = load_splits(str(splits_dir))
        
        # Create groups based on customer segments (if available)
        # For now, use a simple grouping based on feature values
        # In production, this would use actual protected attributes or segments
        groups = np.array([0] * len(X_test))  # Placeholder - would use actual groups
        
        # Perform fairness analysis
        analyzer = FairnessAnalyzer()
        y_pred = model.predict(X_test.values)
        y_pred_proba = model.predict_proba(X_test.values)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        results = analyzer.comprehensive_analysis(
            y_test.values,
            y_pred,
            y_pred_proba,
            groups
        )
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results = convert_types(results)
        
        return JSONResponse(results)
        
    except Exception as e:
        logger.error(f"Error performing fairness analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fairness analysis failed: {str(e)}"
        )


@app.get("/api/versions", tags=["Versioning"])
async def get_versions():
    """
    Get comprehensive version information for models, data, and system.
    
    Returns version information including:
    - Model versions (from MLflow)
    - Data versions
    - System information
    """
    try:
        from src.utils.versioning import get_system_versions
        
        versions = get_system_versions()
        return JSONResponse(versions)
        
    except Exception as e:
        logger.error(f"Error getting versions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get version information: {str(e)}"
        )


@app.get("/api/versions/model", tags=["Versioning"])
async def get_model_versions():
    """
    Get model version information.
    
    Returns all versions of the registered model with metrics and metadata.
    """
    try:
        from src.utils.versioning import ModelVersioner
        
        model_versioner = ModelVersioner()
        model_name = settings.model_name
        
        versions = {
            "model_name": model_name,
            "current_production": model_versioner.get_current_production_model(model_name),
            "all_versions": model_versioner.list_model_versions(model_name),
            "current_stage": settings.model_stage
        }
        
        return JSONResponse(versions)
        
    except Exception as e:
        logger.error(f"Error getting model versions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model versions: {str(e)}"
        )


@app.get("/api/versions/data", tags=["Versioning"])
async def get_data_versions(
    data_type: Optional[str] = None,
    token: str = Depends(oauth2_scheme)
):
    """
    Get data version information from both file-based and database storage.
    
    Returns all data versions including datasets, features, and splits.
    Combines file-based versioning (legacy) with database versions (new).
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.utils.versioning import DataVersioner
        from src.database.connection import get_db_session
        from src.database.services import DataVersionService
        
        # Get file-based versions (legacy)
        data_versioner = DataVersioner()
        file_versions = data_versioner.list_versions()
        
        # Get database versions (new)
        db_versions = {}
        with get_db_session() as session:
            version_service = DataVersionService(session)
            
            # Get all data types or filter by requested type
            data_types = [data_type] if data_type else ["raw_transactions", "processed", "features", "splits", "artifacts"]
            
            for dt in data_types:
                versions_list = version_service.repository.list_by_type(dt)
                if versions_list:
                    db_versions[dt] = [
                        {
                            "id": v.id,
                            "version": v.version,
                            "data_type": v.data_type,
                            "file_path": v.file_path,
                            "file_size": v.file_size,
                            "checksum": v.checksum_sha256,
                            "created": v.created_at.isoformat() if v.created_at else None,
                            "metadata": v.data_metadata or {},
                            "dependencies": v.dependencies or [],
                            "source": "database"
                        }
                        for v in versions_list
                    ]
        
        # Merge file and database versions (database takes precedence)
        all_versions = {}
        for dt in set(list(file_versions.keys()) + list(db_versions.keys())):
            file_v = file_versions.get(dt, {})
            db_v = db_versions.get(dt, [])
            
            # Convert file versions to list format
            file_list = [
                {**v, "source": "file", "id": None}
                for v in file_v.values()
            ] if isinstance(file_v, dict) else []
            
            # Combine and deduplicate by version string
            combined = {}
            for v in file_list + db_v:
                key = v.get("version", "")
                if key and (key not in combined or v.get("source") == "database"):
                    combined[key] = v
            
            all_versions[dt] = list(combined.values())
        
        # Get latest versions for each type
        latest_versions = {}
        for dt, versions_list in all_versions.items():
            if versions_list:
                # Sort by created date (newest first)
                sorted_versions = sorted(
                    versions_list,
                    key=lambda x: x.get("created", "") or "",
                    reverse=True
                )
                latest_versions[dt] = sorted_versions[0]
        
        return JSONResponse({
            "all_versions": all_versions,
            "latest_versions": latest_versions,
            "database_versions": db_versions
        })
        
    except Exception as e:
        logger.error(f"Error getting data versions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get data versions: {str(e)}"
        )


@app.get("/api/versions/current", tags=["Versioning"])
async def get_current_versions():
    """
    Get current production versions.
    
    Returns the currently deployed model version and latest data versions.
    """
    try:
        from src.utils.versioning import ModelVersioner, DataVersioner
        from datetime import datetime
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "model": {},
            "data": {}
        }
        
        # Get current model
        model_versioner = ModelVersioner()
        if model_versioner.mlflow_available:
            model_info = model_versioner.get_current_production_model(settings.model_name)
            result["model"] = model_info or {"status": "no_production_model"}
        else:
            result["model"] = {"status": "mlflow_not_available"}
        
        # Get latest data versions
        data_versioner = DataVersioner()
        data_types = ["dataset", "features", "splits", "artifacts"]
        for data_type in data_types:
            latest = data_versioner.get_latest_version(data_type)
            if latest:
                result["data"][data_type] = {
                    "version": latest["version"],
                    "created": latest["created"],
                    "checksum": latest["checksum"][:16] + "..."  # Truncate for display
                }
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Error getting current versions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get current versions: {str(e)}"
        )


@app.get("/api/lineage", tags=["Versioning", "Lineage"])
async def get_lineage(
    data_version_id: Optional[int] = None,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    token: str = Depends(oauth2_scheme)
):
    """
    Get data lineage information.
    
    Args:
        data_version_id: Filter by source data version ID
        target_type: Filter by target type ('model', 'prediction', 'feature_set', etc.)
        target_id: Filter by target ID
        token: Authentication token
    
    Returns:
        Lineage graph with relationships
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.services import DataLineageService, DataVersionService
        
        with get_db_session() as session:
            lineage_service = DataLineageService(session)
            version_service = DataVersionService(session)
            
            # Get lineage records
            lineage_records = lineage_service.get_lineage_graph(
                data_version_id=data_version_id,
                target_type=target_type,
                target_id=target_id
            )
            
            # Format response
            lineage_data = []
            for record in lineage_records:
                # Get source version details
                source_version = version_service.repository.get_by_id(record.source_data_version_id)
                
                lineage_data.append({
                    "id": record.id,
                    "source": {
                        "data_version_id": record.source_data_version_id,
                        "data_type": record.source_data_type,
                        "version": record.source_version,
                        "checksum": source_version.checksum_sha256[:16] + "..." if source_version else None
                    },
                    "target": {
                        "type": record.target_type,
                        "id": record.target_id,
                        "name": record.target_name
                    },
                    "relationship": {
                        "type": record.relationship_type,
                        "operation": record.operation
                    },
                    "metadata": record.lineage_metadata or {},
                    "created_at": record.created_at.isoformat() if record.created_at else None
                })
            
            return JSONResponse({
                "lineage": lineage_data,
                "total": len(lineage_data)
            })
            
    except Exception as e:
        logger.error(f"Error getting lineage: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get lineage: {str(e)}"
        )


@app.get("/api/lineage/data/{data_version_id}", tags=["Versioning", "Lineage"])
async def get_lineage_by_data_version(
    data_version_id: int,
    token: str = Depends(oauth2_scheme)
):
    """
    Get all lineage records for a specific data version.
    
    Shows what was created from this data version (models, predictions, etc.).
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.services import DataLineageService, DataVersionService
        
        with get_db_session() as session:
            lineage_service = DataLineageService(session)
            version_service = DataVersionService(session)
            
            # Get data version
            data_version = version_service.repository.get_by_id(data_version_id)
            if not data_version:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Data version {data_version_id} not found"
                )
            
            # Get lineage records
            lineage_records = lineage_service.get_lineage_by_source(data_version_id)
            
            # Format response
            lineage_data = []
            for record in lineage_records:
                lineage_data.append({
                    "id": record.id,
                    "target": {
                        "type": record.target_type,
                        "id": record.target_id,
                        "name": record.target_name
                    },
                    "relationship": {
                        "type": record.relationship_type,
                        "operation": record.operation
                    },
                    "metadata": record.lineage_metadata or {},
                    "created_at": record.created_at.isoformat() if record.created_at else None
                })
            
            return JSONResponse({
                "data_version": {
                    "id": data_version.id,
                    "data_type": data_version.data_type,
                    "version": data_version.version,
                    "checksum": data_version.checksum_sha256[:16] + "...",
                    "created_at": data_version.created_at.isoformat() if data_version.created_at else None
                },
                "downstream": lineage_data,
                "total": len(lineage_data)
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting lineage by data version: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get lineage: {str(e)}"
        )


@app.get("/api/lineage/target/{target_type}/{target_id}", tags=["Versioning", "Lineage"])
async def get_lineage_by_target(
    target_type: str,
    target_id: str,
    token: str = Depends(oauth2_scheme)
):
    """
    Get lineage records for a specific target (e.g., prediction, model).
    
    Shows what data versions were used to create this target.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.services import DataLineageService, DataVersionService
        
        with get_db_session() as session:
            lineage_service = DataLineageService(session)
            version_service = DataVersionService(session)
            
            # Get lineage records
            lineage_records = lineage_service.get_lineage_by_target(target_type, target_id)
            
            # Format response
            upstream_data = []
            for record in lineage_records:
                # Get source version details
                source_version = version_service.repository.get_by_id(record.source_data_version_id)
                
                upstream_data.append({
                    "id": record.id,
                    "source": {
                        "data_version_id": record.source_data_version_id,
                        "data_type": record.source_data_type,
                        "version": record.source_version,
                        "checksum": source_version.checksum_sha256[:16] + "..." if source_version else None,
                        "created_at": source_version.created_at.isoformat() if source_version and source_version.created_at else None
                    },
                    "relationship": {
                        "type": record.relationship_type,
                        "operation": record.operation
                    },
                    "metadata": record.lineage_metadata or {},
                    "created_at": record.created_at.isoformat() if record.created_at else None
                })
            
            return JSONResponse({
                "target": {
                    "type": target_type,
                    "id": target_id
                },
                "upstream": upstream_data,
                "total": len(upstream_data)
            })
            
    except Exception as e:
        logger.error(f"Error getting lineage by target: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get lineage: {str(e)}"
        )


@app.get("/api/versions/compare", tags=["Versioning"])
async def compare_versions(
    version1: str = Query(..., description="First version to compare (e.g., 'raw_transactions:v1')"),
    version2: str = Query(..., description="Second version to compare (e.g., 'raw_transactions:v2')"),
    token: str = Depends(oauth2_scheme)
):
    """
    Compare two data versions.
    
    Shows differences in checksums, metadata, file sizes, and other attributes.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.services import DataVersionService
        
        # Parse version strings (format: "data_type:version")
        try:
            data_type1, ver1 = version1.split(":", 1)
            data_type2, ver2 = version2.split(":", 1)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid version format. Use 'data_type:version' (e.g., 'raw_transactions:v1')"
            )
        
        with get_db_session() as session:
            version_service = DataVersionService(session)
            
            # Get versions
            v1 = version_service.get_version(data_type1, ver1)
            v2 = version_service.get_version(data_type2, ver2)
            
            if not v1:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Version {version1} not found"
                )
            if not v2:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Version {version2} not found"
                )
            
            # Compare versions
            comparison = {
                "version1": {
                    "data_type": v1.data_type,
                    "version": v1.version,
                    "file_path": v1.file_path,
                    "file_size": v1.file_size,
                    "checksum": v1.checksum_sha256,
                    "created_at": v1.created_at.isoformat() if v1.created_at else None,
                    "metadata": v1.data_metadata or {}
                },
                "version2": {
                    "data_type": v2.data_type,
                    "version": v2.version,
                    "file_path": v2.file_path,
                    "file_size": v2.file_size,
                    "checksum": v2.checksum_sha256,
                    "created_at": v2.created_at.isoformat() if v2.created_at else None,
                    "metadata": v2.data_metadata or {}
                },
                "differences": {
                    "checksum_match": v1.checksum_sha256 == v2.checksum_sha256,
                    "size_difference": v2.file_size - v1.file_size if v1.file_size and v2.file_size else None,
                    "size_percent_change": (
                        ((v2.file_size - v1.file_size) / v1.file_size * 100)
                        if v1.file_size and v2.file_size and v1.file_size > 0
                        else None
                    ),
                    "metadata_differences": {}
                }
            }
            
            # Compare metadata
            meta1 = v1.data_metadata or {}
            meta2 = v2.data_metadata or {}
            
            all_keys = set(meta1.keys()) | set(meta2.keys())
            for key in all_keys:
                val1 = meta1.get(key)
                val2 = meta2.get(key)
                if val1 != val2:
                    comparison["differences"]["metadata_differences"][key] = {
                        "version1": val1,
                        "version2": val2
                    }
            
            return JSONResponse(comparison)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing versions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare versions: {str(e)}"
        )


@app.post("/api/versions/{data_version_id}/rollback", tags=["Versioning"])
async def rollback_data_version(
    data_version_id: int,
    create_new_version: bool = Body(default=True, description="Create a new version from rolled back data"),
    token: str = Depends(oauth2_scheme)
):
    """
    Rollback to a specific data version.
    
    This creates a new version from the rolled back data, preserving history.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "data:rollback" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to rollback data versions"
        )
    
    username = session_data.get("username", "unknown")
    
    try:
        import shutil
        from pathlib import Path
        from src.database.connection import get_db_session
        from src.database.services import DataVersionService
        from src.utils.versioning import DataVersioner
        
        with get_db_session() as session:
            version_service = DataVersionService(session)
            
            # Get the version to rollback to
            data_version = version_service.repository.get_by_id(data_version_id)
            if not data_version:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Data version {data_version_id} not found"
                )
            
            # Check if file exists
            file_path = Path(data_version.file_path)
            if not file_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Data file not found: {file_path}"
                )
            
            # Verify checksum
            data_versioner = DataVersioner()
            if file_path.is_file():
                current_checksum = data_versioner._calculate_checksum(file_path)
            else:
                current_checksum = data_versioner._calculate_directory_checksum(file_path)
            
            if current_checksum != data_version.checksum_sha256:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File checksum mismatch. Data may have been modified."
                )
            
            if create_new_version:
                # Create a new version from the rolled back data
                version_info = data_versioner.version_data(
                    data_path=file_path,
                    data_type=data_version.data_type,
                    metadata={
                        "rolled_back_from": data_version.version,
                        "rolled_back_by": username,
                        "original_version_id": data_version_id,
                        "rollback_timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                # Store in database
                new_db_version = version_service.create_version(
                    data_type=version_info["data_type"],
                    version=version_info["version"],
                    file_path=str(file_path.absolute()),
                    file_size=version_info["file_info"]["size"],
                    checksum_sha256=version_info["checksum"],
                    metadata=version_info["metadata"],
                    dependencies=[data_version.version]
                )
                session.commit()
                
                logger.info(
                    f"Data version rolled back: {data_version.version} -> {version_info['version']}",
                    extra={
                        "username": username,
                        "original_version": data_version.version,
                        "new_version": version_info["version"]
                    }
                )
                
                return JSONResponse({
                    "message": "Data version rolled back successfully",
                    "original_version": {
                        "id": data_version.id,
                        "data_type": data_version.data_type,
                        "version": data_version.version
                    },
                    "new_version": {
                        "id": new_db_version.id,
                        "data_type": new_db_version.data_type,
                        "version": new_db_version.version,
                        "checksum": new_db_version.checksum_sha256[:16] + "..."
                    }
                })
            else:
                # Just return the version info for manual rollback
                return JSONResponse({
                    "message": "Rollback information",
                    "version": {
                        "id": data_version.id,
                        "data_type": data_version.data_type,
                        "version": data_version.version,
                        "file_path": str(file_path),
                        "checksum": data_version.checksum_sha256,
                        "file_size": data_version.file_size
                    },
                    "note": "Set create_new_version=true to automatically create a new version"
                })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back data version: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rollback data version: {str(e)}"
        )


@app.post("/api/training/link-data-version", tags=["Versioning", "Training"])
async def link_training_to_data_version(
    mlflow_run_id: str = Body(..., description="MLflow run ID"),
    data_version_id: int = Body(..., description="Data version ID used for training"),
    model_name: str = Body(..., description="Model name"),
    model_version: str = Body(..., description="Model version"),
    token: str = Depends(oauth2_scheme)
):
    """
    Link a model training run to a data version.
    
    Creates lineage tracking between data version and model training.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.services import DataVersionService, DataLineageService, ModelMetadata
        
        with get_db_session() as session:
            version_service = DataVersionService(session)
            lineage_service = DataLineageService(session)
            
            # Get data version
            data_version = version_service.repository.get_by_id(data_version_id)
            if not data_version:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Data version {data_version_id} not found"
                )
            
            # Create or update model metadata
            model_target_id = f"{model_name}:{model_version}"
            
            # Check if model metadata exists
            from src.database.models import ModelMetadata
            existing_model = session.query(ModelMetadata).filter(
                ModelMetadata.model_name == model_name,
                ModelMetadata.model_version == model_version
            ).first()
            
            if existing_model:
                # Update existing model metadata
                existing_model.training_data_version = data_version.version
                existing_model.mlflow_run_id = mlflow_run_id
            else:
                # Create new model metadata
                new_model = ModelMetadata(
                    model_name=model_name,
                    model_version=model_version,
                    model_stage="Staging",
                    mlflow_run_id=mlflow_run_id,
                    training_data_version=data_version.version,
                    is_active=False
                )
                session.add(new_model)
            
            # Create lineage record
            lineage_service.create_lineage(
                source_data_version_id=data_version.id,
                source_data_type=data_version.data_type,
                source_version=data_version.version,
                target_type="model",
                target_id=model_target_id,
                target_name=f"{model_name} v{model_version}",
                relationship_type="trained_on",
                operation="training",
                metadata={
                    "mlflow_run_id": mlflow_run_id,
                    "model_name": model_name,
                    "model_version": model_version
                }
            )
            
            session.commit()
            
            logger.info(
                f"Linked training run to data version: {mlflow_run_id} -> {data_version.version}",
                extra={
                    "mlflow_run_id": mlflow_run_id,
                    "data_version": data_version.version,
                    "model": model_target_id
                }
            )
            
            return JSONResponse({
                "message": "Training run linked to data version successfully",
                "lineage": {
                    "source": {
                        "data_type": data_version.data_type,
                        "version": data_version.version
                    },
                    "target": {
                        "type": "model",
                        "id": model_target_id,
                        "name": f"{model_name} v{model_version}"
                    }
                }
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error linking training to data version: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to link training: {str(e)}"
        )


@app.get("/api/performance", tags=["Monitoring"])
async def get_performance_metrics():
    """
    Get performance metrics including latency percentiles.
    
    Returns performance statistics to monitor SLA compliance
    (95th percentile latency under 200ms for real-time lending decisions).
    
    Optimized for fast response - returns cached metrics when available.
    """
    try:
        if not settings.enable_performance_monitoring:
            return JSONResponse({
                "status": "disabled",
                "message": "Performance monitoring is disabled",
                "stats": {},
                "sla": {"compliant": False, "message": "Monitoring disabled"},
                "target_p95_ms": settings.target_p95_latency_ms
            })
        
        perf_monitor = get_performance_monitor()
        
        # Get stats (this should be fast as it's just reading from memory)
        stats = perf_monitor.get_all_stats()
        
        # Check SLA (also fast - just percentile calculation)
        sla_check = perf_monitor.check_sla(
            percentile=95,
            threshold_ms=settings.target_p95_latency_ms
        )
        
        # Check for SLA violations and send alerts
        if not sla_check.get("compliant", True):
            try:
                from src.monitoring.alerts import get_alert_manager
                alert_manager = get_alert_manager()
                alert = alert_manager.check_sla_violation(
                    p95_latency_ms=sla_check.get("p95_ms", 0),
                    threshold_ms=settings.target_p95_latency_ms
                )
                if alert:
                    alert_manager.send_alert(alert)
            except Exception as e:
                logger.warning(f"Error sending SLA alert: {e}")
        
        return JSONResponse({
            "status": "enabled",
            "stats": stats,
            "sla": sla_check,
            "target_p95_ms": settings.target_p95_latency_ms
        })
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}", exc_info=True)
        # Return error response instead of raising exception for better UX
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "stats": {},
            "sla": {"compliant": False, "message": f"Error: {str(e)}"},
            "target_p95_ms": settings.target_p95_latency_ms
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ============================================================================
# Database Service Endpoints
# ============================================================================

@app.get("/api/predictions", tags=["Database"])
async def get_predictions(
    customer_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Get predictions from database.
    
    Args:
        customer_id: Optional customer ID to filter by
        limit: Maximum number of predictions to return (default: 100)
        offset: Number of predictions to skip (default: 0)
    
    Returns:
        List of predictions with pagination metadata
    """
    try:
        from src.database.connection import get_db_session
        from src.database.services import PredictionService
        from src.database.repositories import PredictionRepository
        
        with get_db_session() as session:
            repository = PredictionRepository(session)
            
            if customer_id:
                # Get customer-specific predictions
                predictions = repository.get_by_customer_id(
                    customer_id=customer_id,
                    limit=limit
                )
                # Count total for customer
                from sqlalchemy import func
                from src.database.models import Prediction
                total = session.query(func.count(Prediction.prediction_id)).filter(
                    Prediction.customer_id_indexed == customer_id
                ).scalar() or 0
            else:
                # Get all predictions with pagination
                predictions = repository.get_all(limit=limit, offset=offset)
                # Count total
                from sqlalchemy import func
                from src.database.models import Prediction
                total = session.query(func.count(Prediction.prediction_id)).scalar() or 0
            
            # Convert to dict format
            predictions_data = []
            for pred in predictions:
                predictions_data.append({
                    "prediction_id": pred.prediction_id,
                    "customer_id": pred.customer_id,
                    "prediction": pred.prediction,
                    "probability": float(pred.probability),
                    "risk_level": pred.risk_level,
                    "customer_score": pred.customer_score,
                    "latency_ms": float(pred.latency_ms) if pred.latency_ms else None,
                    "model_version": pred.model_version,
                    "created_at": pred.created_at.isoformat() if pred.created_at else None,
                    "created_at_date": pred.created_at_date.isoformat() if pred.created_at_date else None
                })
            
            # Log statistics for debugging
            predictions_with_customer = [p for p in predictions_data if p.get("customer_id")]
            predictions_with_score = [p for p in predictions_data if p.get("customer_score") is not None]
            predictions_with_both = [p for p in predictions_data if p.get("customer_id") and p.get("customer_score") is not None]
            
            logger.debug(
                f"Returning predictions",
                extra={
                    "total": total,
                    "returned": len(predictions_data),
                    "with_customer_id": len(predictions_with_customer),
                    "with_customer_score": len(predictions_with_score),
                    "with_both": len(predictions_with_both),
                    "offset": offset,
                    "limit": limit
                }
            )
            
            return JSONResponse({
                "predictions": predictions_data,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total
            })
            
    except Exception as e:
        logger.error(f"Error getting predictions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get predictions: {str(e)}"
        )


@app.get("/api/predictions/{prediction_id}", tags=["Database"])
async def get_prediction_by_id(prediction_id: str):
    """
    Get a specific prediction by ID.
    
    Args:
        prediction_id: Prediction identifier
    
    Returns:
        Prediction details
    """
    try:
        from src.database.connection import get_db_session
        from src.database.services import PredictionService
        
        with get_db_session() as session:
            prediction_service = PredictionService(session)
            prediction = prediction_service.get_prediction_by_id(prediction_id)
            
            return JSONResponse({
                "prediction_id": prediction.prediction_id,
                "customer_id": prediction.customer_id,
                "prediction": prediction.prediction,
                "probability": float(prediction.probability),
                "risk_level": prediction.risk_level,
                "customer_score": prediction.customer_score,
                "latency_ms": float(prediction.latency_ms) if prediction.latency_ms else None,
                "model_version": prediction.model_version,
                "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
                "created_at_date": prediction.created_at_date.isoformat() if prediction.created_at_date else None
            })
            
    except Exception as e:
        logger.error(f"Error getting prediction: {e}", exc_info=True)
        from src.database.exceptions import RecordNotFoundError
        if isinstance(e, RecordNotFoundError) or "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prediction not found: {prediction_id}"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get prediction: {str(e)}"
        )


@app.get("/api/predictions/customer/{customer_id}", tags=["Database"])
async def get_customer_predictions(
    customer_id: str,
    limit: int = 100
):
    """
    Get all predictions for a specific customer.
    
    Args:
        customer_id: Customer identifier
        limit: Maximum number of predictions to return
    
    Returns:
        List of predictions for the customer
    """
    try:
        from src.database.connection import get_db_session
        from src.database.services import PredictionService
        
        with get_db_session() as session:
            prediction_service = PredictionService(session)
            predictions = prediction_service.get_customer_predictions(
                customer_id=customer_id,
                limit=limit
            )
            
            predictions_data = []
            for pred in predictions:
                predictions_data.append({
                    "prediction_id": pred.prediction_id,
                    "customer_id": pred.customer_id,
                    "prediction": pred.prediction,
                    "probability": float(pred.probability),
                    "risk_level": pred.risk_level,
                    "customer_score": pred.customer_score,
                    "latency_ms": float(pred.latency_ms) if pred.latency_ms else None,
                    "model_version": pred.model_version,
                    "created_at": pred.created_at.isoformat() if pred.created_at else None
                })
            
            return JSONResponse({
                "customer_id": customer_id,
                "predictions": predictions_data,
                "count": len(predictions_data)
            })
            
    except Exception as e:
        logger.error(f"Error getting customer predictions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get customer predictions: {str(e)}"
        )


@app.get("/api/debug/predictions-with-scores", tags=["Debug"])
async def debug_predictions_with_scores():
    """
    Debug endpoint to check predictions with customer_id and customer_score.
    This helps diagnose why customer scores might not be showing up.
    """
    try:
        from src.database.connection import get_db_session
        from src.database.models import Prediction
        
        with get_db_session() as session:
            # Get all predictions
            all_predictions = session.query(Prediction).all()
            
            # Count statistics
            total = len(all_predictions)
            with_customer_id = len([p for p in all_predictions if p.customer_id])
            with_customer_score = len([p for p in all_predictions if p.customer_score is not None])
            with_both = len([p for p in all_predictions if p.customer_id and p.customer_score is not None])
            
            # Get sample predictions with both
            sample_predictions = [
                {
                    "prediction_id": p.prediction_id,
                    "customer_id": p.customer_id,
                    "customer_score": p.customer_score,
                    "risk_level": p.risk_level,
                    "created_at": p.created_at.isoformat() if p.created_at else None
                }
                for p in all_predictions[:10] if p.customer_id and p.customer_score is not None
            ]
            
            return JSONResponse({
                "statistics": {
                    "total_predictions": total,
                    "with_customer_id": with_customer_id,
                    "with_customer_score": with_customer_score,
                    "with_both_customer_id_and_score": with_both
                },
                "sample_predictions": sample_predictions
            })
            
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug endpoint error: {str(e)}"
        )


@app.get("/api/kpis", tags=["Database"])
async def get_kpis(
    period_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Get business KPIs from database.
    
    Args:
        period_type: Filter by period type (hourly, daily, weekly, monthly)
        limit: Maximum number of KPIs to return
        offset: Number of KPIs to skip
    
    Returns:
        List of business KPIs
    """
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import BusinessKPIRepository
        
        with get_db_session() as session:
            repository = BusinessKPIRepository(session)
            
            if period_type:
                # Filter by period type
                from sqlalchemy import and_
                from src.database.models import BusinessKPI
                query = session.query(BusinessKPI).filter(
                    BusinessKPI.period_type == period_type
                ).order_by(BusinessKPI.period_start.desc())
                if limit:
                    query = query.limit(limit)
                if offset:
                    query = query.offset(offset)
                kpis = query.all()
            else:
                kpis = repository.get_all(limit=limit, offset=offset)
            
            kpis_data = []
            for kpi in kpis:
                kpis_data.append({
                    "id": kpi.id,
                    "period_start": kpi.period_start.isoformat() if kpi.period_start else None,
                    "period_end": kpi.period_end.isoformat() if kpi.period_end else None,
                    "period_type": kpi.period_type,
                    "total_predictions": kpi.total_predictions,
                    "approval_count": kpi.approval_count,
                    "rejection_count": kpi.rejection_count,
                    "review_count": kpi.review_count,
                    "approval_rate": float(kpi.approval_rate) if kpi.approval_rate else None,
                    "rejection_rate": float(kpi.rejection_rate) if kpi.rejection_rate else None,
                    "review_rate": float(kpi.review_rate) if kpi.review_rate else None,
                    "avg_risk_score": float(kpi.avg_risk_score) if kpi.avg_risk_score else None,
                    "unique_customers": kpi.unique_customers,
                    "avg_latency_ms": float(kpi.avg_latency_ms) if kpi.avg_latency_ms else None,
                    "p95_latency_ms": float(kpi.p95_latency_ms) if kpi.p95_latency_ms else None,
                    "created_at": kpi.created_at.isoformat() if kpi.created_at else None
                })
            
            return JSONResponse({
                "kpis": kpis_data,
                "limit": limit,
                "offset": offset
            })
            
    except Exception as e:
        logger.error(f"Error getting KPIs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get KPIs: {str(e)}"
        )


@app.get("/api/kpis/latest", tags=["Database"])
async def get_latest_kpis(period_type: str = "daily"):
    """
    Get the latest business KPIs for a period type.
    
    Args:
        period_type: Period type (hourly, daily, weekly, monthly)
    
    Returns:
        Latest KPI data
    """
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import BusinessKPIRepository
        
        with get_db_session() as session:
            repository = BusinessKPIRepository(session)
            kpi = repository.get_latest(period_type)
            
            if not kpi:
                return JSONResponse({
                    "message": f"No KPIs found for period type: {period_type}",
                    "kpi": None
                })
            
            return JSONResponse({
                "kpi": {
                    "id": kpi.id,
                    "period_start": kpi.period_start.isoformat() if kpi.period_start else None,
                    "period_end": kpi.period_end.isoformat() if kpi.period_end else None,
                    "period_type": kpi.period_type,
                    "total_predictions": kpi.total_predictions,
                    "approval_count": kpi.approval_count,
                    "rejection_count": kpi.rejection_count,
                    "review_count": kpi.review_count,
                    "approval_rate": float(kpi.approval_rate) if kpi.approval_rate else None,
                    "rejection_rate": float(kpi.rejection_rate) if kpi.rejection_rate else None,
                    "review_rate": float(kpi.review_rate) if kpi.review_rate else None,
                    "avg_risk_score": float(kpi.avg_risk_score) if kpi.avg_risk_score else None,
                    "unique_customers": kpi.unique_customers,
                    "avg_latency_ms": float(kpi.avg_latency_ms) if kpi.avg_latency_ms else None,
                    "p95_latency_ms": float(kpi.p95_latency_ms) if kpi.p95_latency_ms else None
                }
            })
            
    except Exception as e:
        logger.error(f"Error getting latest KPIs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get latest KPIs: {str(e)}"
        )


@app.post("/api/kpis/calculate", tags=["Database"])
async def calculate_kpis(
    period_type: str = "daily",
    hours_back: int = 24
):
    """
    Calculate and save business KPIs for a time period.
    
    Args:
        period_type: Period type (hourly, daily, weekly, monthly)
        hours_back: Number of hours to look back for calculations
    
    Returns:
        Calculated KPI data
    """
    try:
        from src.database.connection import get_db_session
        from src.database.services import BusinessKPIService
        from datetime import timedelta
        
        with get_db_session() as session:
            kpi_service = BusinessKPIService(session)
            
            # Calculate period
            period_end = datetime.now(timezone.utc)
            if period_type == "hourly":
                period_start = period_end - timedelta(hours=1)
            elif period_type == "daily":
                period_start = period_end - timedelta(hours=hours_back)
            elif period_type == "weekly":
                period_start = period_end - timedelta(days=7)
            elif period_type == "monthly":
                period_start = period_end - timedelta(days=30)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid period_type: {period_type}"
                )
            
            kpi = kpi_service.calculate_and_save_kpis(
                period_start=period_start,
                period_end=period_end,
                period_type=period_type
            )
            session.commit()
            
            return JSONResponse({
                "message": "KPIs calculated and saved successfully",
                "kpi": {
                    "id": kpi.id,
                    "period_type": kpi.period_type,
                    "total_predictions": kpi.total_predictions,
                    "approval_rate": float(kpi.approval_rate) if kpi.approval_rate else None,
                    "rejection_rate": float(kpi.rejection_rate) if kpi.rejection_rate else None
                }
            })
            
    except Exception as e:
        logger.error(f"Error calculating KPIs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate KPIs: {str(e)}"
        )


# ============================================================================
# User & Role Management Endpoints
# ============================================================================

@app.get("/api/users", tags=["Users & Roles"])
async def get_users(
    limit: int = 100,
    offset: int = 0,
    is_active: Optional[bool] = None
):
    """Get all users from database."""
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import UserRepository
        from src.database.models import User
        
        with get_db_session() as session:
            query = session.query(User).filter(User.deleted_at.is_(None))
            if is_active is not None:
                query = query.filter(User.is_active == is_active)
            
            total = query.count()
            users = query.order_by(User.created_at.desc()).offset(offset).limit(limit).all()
            
            users_data = []
            for user in users:
                user_roles = []
                for user_role in user.user_roles:
                    if user_role.role:
                        user_roles.append({
                            "role_id": user_role.role.role_id,
                            "role_name": user_role.role.role_name,
                            "role_code": user_role.role.role_code
                        })
                
                users_data.append({
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "department": user.department,
                    "position": user.position,
                    "is_active": user.is_active,
                    "is_verified": user.is_verified,
                    "is_superuser": user.is_superuser,
                    "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
                    "roles": user_roles,
                    "created_at": user.created_at.isoformat() if user.created_at else None
                })
            
            return JSONResponse({
                "users": users_data,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total
            })
    except Exception as e:
        logger.error(f"Error getting users: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get users: {str(e)}"
        )


@app.get("/api/users/{user_id}", tags=["Users & Roles"])
async def get_user_by_id(user_id: int):
    """Get a specific user by ID."""
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import UserRepository
        
        with get_db_session() as session:
            repository = UserRepository(session)
            user = repository.get_by_id(user_id)
            
            if not user or user.deleted_at:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User not found: {user_id}"
                )
            
            user_roles = []
            for user_role in user.user_roles:
                if user_role.role:
                    user_roles.append({
                        "role_id": user_role.role.role_id,
                        "role_name": user_role.role.role_name,
                        "role_code": user_role.role.role_code,
                        "assigned_at": user_role.assigned_at.isoformat() if user_role.assigned_at else None
                    })
            
            return JSONResponse({
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "department": user.department,
                "position": user.position,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "is_superuser": user.is_superuser,
                "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
                "roles": user_roles,
                "created_at": user.created_at.isoformat() if user.created_at else None
            })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user: {str(e)}"
        )


@app.get("/api/roles", tags=["Users & Roles"])
async def get_roles(
    limit: int = 100,
    offset: int = 0,
    is_active: Optional[bool] = None
):
    """Get all roles from database."""
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import RoleRepository
        from src.database.models import Role
        
        with get_db_session() as session:
            query = session.query(Role)
            if is_active is not None:
                query = query.filter(Role.is_active == is_active)
            
            total = query.count()
            roles = query.order_by(Role.role_name).offset(offset).limit(limit).all()
            
            roles_data = []
            for role in roles:
                role_permissions = []
                for role_perm in role.role_permissions:
                    if role_perm.permission:
                        role_permissions.append({
                            "permission_id": role_perm.permission.permission_id,
                            "permission_name": role_perm.permission.permission_name,
                            "permission_code": role_perm.permission.permission_code,
                            "resource_type": role_perm.permission.resource_type,
                            "action": role_perm.permission.action
                        })
                
                roles_data.append({
                    "role_id": role.role_id,
                    "role_name": role.role_name,
                    "role_code": role.role_code,
                    "description": role.description,
                    "is_active": role.is_active,
                    "permissions": role_permissions,
                    "user_count": len(role.user_roles),
                    "created_at": role.created_at.isoformat() if role.created_at else None
                })
            
            return JSONResponse({
                "roles": roles_data,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total
            })
    except Exception as e:
        logger.error(f"Error getting roles: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get roles: {str(e)}"
        )


@app.get("/api/roles/{role_id}", tags=["Users & Roles"])
async def get_role_by_id(role_id: int):
    """Get a specific role by ID with permissions."""
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import RoleRepository
        
        with get_db_session() as session:
            repository = RoleRepository(session)
            role = repository.get_by_id(role_id)
            
            if not role:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Role not found: {role_id}"
                )
            
            role_permissions = []
            for role_perm in role.role_permissions:
                if role_perm.permission:
                    role_permissions.append({
                        "permission_id": role_perm.permission.permission_id,
                        "permission_name": role_perm.permission.permission_name,
                        "permission_code": role_perm.permission.permission_code,
                        "resource_type": role_perm.permission.resource_type,
                        "action": role_perm.permission.action,
                        "description": role_perm.permission.description
                    })
            
            users_with_role = []
            for user_role in role.user_roles:
                if user_role.user and not user_role.user.deleted_at:
                    users_with_role.append({
                        "user_id": user_role.user.user_id,
                        "username": user_role.user.username,
                        "email": user_role.user.email,
                        "full_name": user_role.user.full_name,
                        "assigned_at": user_role.assigned_at.isoformat() if user_role.assigned_at else None
                    })
            
            return JSONResponse({
                "role_id": role.role_id,
                "role_name": role.role_name,
                "role_code": role.role_code,
                "description": role.description,
                "is_active": role.is_active,
                "permissions": role_permissions,
                "users": users_with_role,
                "user_count": len(users_with_role),
                "created_at": role.created_at.isoformat() if role.created_at else None
            })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting role: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get role: {str(e)}"
        )


@app.get("/api/permissions", tags=["Users & Roles"])
async def get_permissions():
    """Get all permissions from database."""
    try:
        from src.database.connection import get_db_session
        from src.database.models import Permission
        
        with get_db_session() as session:
            permissions = session.query(Permission).order_by(Permission.resource_type, Permission.action).all()
            
            permissions_data = []
            for perm in permissions:
                permissions_data.append({
                    "permission_id": perm.permission_id,
                    "permission_name": perm.permission_name,
                    "permission_code": perm.permission_code,
                    "resource_type": perm.resource_type,
                    "action": perm.action,
                    "description": perm.description,
                    "created_at": perm.created_at.isoformat() if perm.created_at else None
                })
            
            return JSONResponse({
                "permissions": permissions_data,
                "total": len(permissions_data)
            })
    except Exception as e:
        logger.error(f"Error getting permissions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get permissions: {str(e)}"
        )


# ============================================================================
# Authentication Endpoints
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def create_session_token() -> str:
    """Create a new session token."""
    return secrets.token_urlsafe(32)


@app.post("/api/auth/login", tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint for user authentication.
    
    Returns:
        Access token and user information
    """
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import UserRepository
        from datetime import datetime, timezone
        
        with get_db_session() as session:
            user_repo = UserRepository(session)
            user = user_repo.get_by_username(form_data.username)
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password"
                )
            
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is inactive"
                )
            
            if user.deleted_at:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account has been deleted"
                )
            
            # Verify password
            if not user.password_hash:
                logger.error(f"User {form_data.username} has no password hash")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="User account configuration error"
                )
            
            if not verify_password(form_data.password, user.password_hash):
                # Increment failed login attempts
                user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
                session.commit()
                
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password"
                )
            
            # Reset failed login attempts
            user.failed_login_attempts = 0
            user.last_login_at = datetime.now(timezone.utc)
            session.commit()
            
            # Get user roles and permissions
            user_roles = []
            all_permissions = set()
            
            # Safely access user_roles relationship
            if hasattr(user, 'user_roles') and user.user_roles:
                for user_role in user.user_roles:
                    if user_role and user_role.role and user_role.role.is_active:
                        role_data = {
                            "role_id": user_role.role.role_id,
                            "role_name": user_role.role.role_name,
                            "role_code": user_role.role.role_code
                        }
                        user_roles.append(role_data)
                        
                        # Collect permissions from all roles
                        if hasattr(user_role.role, 'role_permissions') and user_role.role.role_permissions:
                            for role_perm in user_role.role.role_permissions:
                                if role_perm and role_perm.permission:
                                    all_permissions.add(role_perm.permission.permission_code)
            
            # Create session token
            session_token = create_session_token()
            session_store[session_token] = {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": user_roles,
                "permissions": list(all_permissions),
                "is_superuser": user.is_superuser
            }
            
            logger.info(
                f"User logged in: {user.username}",
                extra={
                    "user_id": user.user_id,
                    "username": user.username,
                    "roles": [r["role_code"] for r in user_roles]
                }
            )
            
            return JSONResponse({
                "access_token": session_token,
                "token_type": "bearer",
                "user": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "department": user.department,
                    "position": user.position,
                    "roles": user_roles,
                    "permissions": list(all_permissions),
                    "is_superuser": user.is_superuser
                }
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@app.post("/api/auth/logout", tags=["Authentication"])
async def logout(token: str = Depends(oauth2_scheme)):
    """Logout endpoint to invalidate session token."""
    if token and token in session_store:
        del session_store[token]
        logger.info(f"User logged out: token invalidated")
    
    return JSONResponse({"message": "Logged out successfully"})


@app.get("/api/auth/me", tags=["Authentication"])
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Get current authenticated user information.
    
    Returns:
        Current user details with roles and permissions
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import UserRepository
        
        with get_db_session() as session:
            user_repo = UserRepository(session)
            user = user_repo.get_by_id(session_data["user_id"])
            
            if not user or not user.is_active or user.deleted_at:
                # Remove invalid session
                if token in session_store:
                    del session_store[token]
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User account is inactive or deleted"
                )
            
            # Refresh user roles and permissions
            user_roles = []
            all_permissions = set()
            
            for user_role in user.user_roles:
                if user_role.role and user_role.role.is_active:
                    role_data = {
                        "role_id": user_role.role.role_id,
                        "role_name": user_role.role.role_name,
                        "role_code": user_role.role.role_code
                    }
                    user_roles.append(role_data)
                    
                    for role_perm in user_role.role.role_permissions:
                        if role_perm.permission:
                            all_permissions.add(role_perm.permission.permission_code)
            
            # Update session
            session_store[token] = {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": user_roles,
                "permissions": list(all_permissions),
                "is_superuser": user.is_superuser
            }
            
            return JSONResponse({
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "department": user.department,
                "position": user.position,
                "roles": user_roles,
                "permissions": list(all_permissions),
                "is_superuser": user.is_superuser,
                "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user: {str(e)}"
        )


# ============================================================================
# Raw Data Upload Endpoints
# ============================================================================

@app.post("/api/data/upload", tags=["Data Management"])
async def upload_raw_data(
    file: UploadFile = File(...),
    data_source: str = Form(default="manual_upload"),
    data_version: Optional[str] = Form(default=None),
    token: str = Depends(oauth2_scheme)
):
    """
    Upload raw transaction data from CSV or JSON file.
    
    Args:
        file: CSV or JSON file containing transaction data
        data_source: Source system identifier
        data_version: Optional data version identifier
        token: Authentication token
    
    Returns:
        Upload summary with count and status
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "data:upload" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to upload data"
        )
    
        username = session_data.get("username", "unknown")
    
    try:
        import tempfile
        from pathlib import Path
        from src.database.connection import get_db_session
        from src.database.services import RawTransactionService, DataVersionService
        from src.utils.versioning import DataVersioner
        
        # Read file content
        content = await file.read()
        file_extension = file.filename.split('.')[-1].lower() if file.filename else ''
        
        # Save file temporarily for versioning
        temp_file = None
        temp_path = None
        auto_version = None
        
        try:
            # Create temporary file
            suffix = f'.{file_extension}' if file_extension else ''
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp_file:
                tmp_file.write(content)
                temp_path = Path(tmp_file.name)
            
            # Auto-version the dataset (unless user provided version)
            if not data_version:
                data_versioner = DataVersioner()
                version_info = data_versioner.version_data(
                    data_path=temp_path,
                    data_type="raw_transactions",
                    metadata={
                        "uploaded_by": username,
                        "data_source": data_source,
                        "file_name": file.filename or "unknown",
                        "row_count": 0  # Will update after validation
                    }
                )
                auto_version = version_info["version"]
                
                # Store version in database
                with get_db_session() as version_session:
                    version_service = DataVersionService(version_session)
                    db_version = version_service.create_version(
                        data_type="raw_transactions",
                        version=version_info["version"],
                        file_path=str(temp_path.absolute()),
                        file_size=version_info["file_info"]["size"],
                        checksum_sha256=version_info["checksum"],
                        metadata=version_info["metadata"],
                        dependencies=version_info.get("dependencies", [])
                    )
                    version_session.commit()
                    logger.info(f"Auto-created data version: {auto_version} (ID: {db_version.id})")
                
                # Use auto-generated version
                data_version = auto_version
            else:
                # User provided version - still create version record if it doesn't exist
                with get_db_session() as version_session:
                    version_service = DataVersionService(version_session)
                    existing = version_service.get_version("raw_transactions", data_version)
                    if not existing:
                        # Create version record for user-provided version
                        data_versioner = DataVersioner()
                        version_info = data_versioner.version_data(
                            data_path=temp_path,
                            data_type="raw_transactions",
                            version=data_version,
                            metadata={
                                "uploaded_by": username,
                                "data_source": data_source,
                                "file_name": file.filename or "unknown",
                                "user_provided": True
                            }
                        )
                        db_version = version_service.create_version(
                            data_type="raw_transactions",
                            version=version_info["version"],
                            file_path=str(temp_path.absolute()),
                            file_size=version_info["file_info"]["size"],
                            checksum_sha256=version_info["checksum"],
                            metadata=version_info["metadata"],
                            dependencies=version_info.get("dependencies", [])
                        )
                        version_session.commit()
                        logger.info(f"Created data version record for user-provided version: {data_version}")
        
        except Exception as version_error:
            logger.warning(f"Error during auto-versioning (continuing with upload): {version_error}", exc_info=True)
            # Continue with upload even if versioning fails
        
        # Parse file based on extension
        transactions = []
        
        if file_extension == 'csv':
            # Parse CSV
            content_str = content.decode('utf-8')
            csv_reader = csv.DictReader(io.StringIO(content_str))
            transactions = list(csv_reader)
        elif file_extension == 'json':
            # Parse JSON
            content_str = content.decode('utf-8')
            data = json.loads(content_str)
            # Handle both array and object formats
            if isinstance(data, list):
                transactions = data
            elif isinstance(data, dict) and 'transactions' in data:
                transactions = data['transactions']
            else:
                raise ValueError("JSON must be an array or object with 'transactions' key")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Please upload CSV or JSON file."
            )
        
        if not transactions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No transactions found in file"
            )
        
        # Validate and normalize transaction data
        normalized_transactions = []
        validation_errors = []
        
        # Common field name variations - based on actual raw data format
        # Actual format uses PascalCase: TransactionId, CustomerId, TransactionStartTime, etc.
        field_mappings = {
            "customer_id": ["CustomerId", "customer_id", "customerId", "CustomerID", "customer", "CUSTOMER_ID", "cust_id"],
            "transaction_id": ["TransactionId", "transaction_id", "transactionId", "TransactionID", "transaction", "TRANSACTION_ID", "txn_id", "id"],
            "amount": ["Amount", "amount", "AMOUNT", "amt", "transaction_amount"],
            "transaction_start_time": ["TransactionStartTime", "transaction_start_time", "transactionStartTime", "TransactionStartTime", 
                                      "start_time", "timestamp", "date", "transaction_date", "time", "TransactionStart"],
            "batch_id": ["BatchId", "batch_id", "batchId", "BatchID", "batch"],
            "account_id": ["AccountId", "account_id", "accountId", "AccountID", "account"],
            "subscription_id": ["SubscriptionId", "subscription_id", "subscriptionId", "SubscriptionID", "subscription"],
            "currency_code": ["CurrencyCode", "currency_code", "currencyCode", "CurrencyCode", "currency", "curr"],
            "country_code": ["CountryCode", "country_code", "countryCode", "CountryCode", "country"],
            "provider_id": ["ProviderId", "provider_id", "providerId", "ProviderID", "provider"],
            "product_id": ["ProductId", "product_id", "productId", "ProductID", "product"],
            "product_category": ["ProductCategory", "product_category", "productCategory", "ProductCategory", "category", "product_type"],
            "channel_id": ["ChannelId", "channel_id", "channelId", "ChannelID", "channel"],
            "value": ["Value", "value", "VALUE"],
            "pricing_strategy": ["PricingStrategy", "pricing_strategy", "pricingStrategy", "PricingStrategy", "pricing"],
            "fraud_result": ["FraudResult", "fraud_result", "fraudResult", "FraudResult", "fraud"],
        }
        
        def get_field_value(txn, field_name):
            """Get field value with case-insensitive and variation matching."""
            # Try exact match first
            if field_name in txn:
                return txn[field_name]
            
            # Try case-insensitive match
            for key in txn.keys():
                if key.lower() == field_name.lower():
                    return txn[key]
            
            # Try variations
            if field_name in field_mappings:
                for variation in field_mappings[field_name]:
                    if variation in txn:
                        return txn[variation]
                    # Case-insensitive variation match
                    for key in txn.keys():
                        if key.lower() == variation.lower():
                            return txn[key]
            
            return None
        
        for idx, txn in enumerate(transactions):
            try:
                # Get customer_id with flexible matching
                customer_id = get_field_value(txn, "customer_id")
                if customer_id is None or str(customer_id).strip() == "":
                    validation_errors.append(f"Row {idx + 1}: Missing customer_id (tried: customer_id, customerId, CustomerID, customer, cust_id)")
                    continue
                
                # Get amount with flexible matching
                amount_value = get_field_value(txn, "amount")
                if amount_value is None:
                    validation_errors.append(f"Row {idx + 1}: Missing amount")
                    continue
                
                try:
                    amount = float(amount_value)
                except (ValueError, TypeError):
                    validation_errors.append(f"Row {idx + 1}: Invalid amount value: {amount_value}")
                    continue
                
                if amount == 0:
                    validation_errors.append(f"Row {idx + 1}: Amount cannot be zero")
                    continue
                
                # Get transaction_id
                transaction_id = get_field_value(txn, "transaction_id")
                if not transaction_id:
                    transaction_id = f"TXN_{idx}_{int(datetime.now().timestamp())}"
                
                # Get transaction_start_time
                transaction_time = get_field_value(txn, "transaction_start_time")
                if not transaction_time:
                    transaction_time = datetime.now(timezone.utc).isoformat()
                else:
                    # Try to parse and format the time
                    try:
                        if isinstance(transaction_time, str):
                            # Remove timezone suffix if present (e.g., 'Z' or '+00:00')
                            time_str = transaction_time.replace('Z', '').split('+')[0].split('-')[0] if '+' in transaction_time else transaction_time.replace('Z', '')
                            
                            # Try parsing various date formats (actual format: 2018-11-15T02:18:49Z)
                            date_formats = [
                                "%Y-%m-%dT%H:%M:%S",  # ISO format with T
                                "%Y-%m-%dT%H:%M:%SZ",  # ISO format with T and Z
                                "%Y-%m-%d %H:%M:%S",   # Space separated
                                "%Y-%m-%d",            # Date only
                                "%Y/%m/%d %H:%M:%S",   # Slash separated
                            ]
                            
                            parsed = False
                            for fmt in date_formats:
                                try:
                                    dt = datetime.strptime(time_str, fmt)
                                    transaction_time = dt.replace(tzinfo=timezone.utc).isoformat()
                                    parsed = True
                                    break
                                except ValueError:
                                    continue
                            
                            if not parsed:
                                # If parsing fails, use current time
                                transaction_time = datetime.now(timezone.utc).isoformat()
                    except Exception as e:
                        logger.warning(f"Error parsing transaction time: {e}, using current time")
                        transaction_time = datetime.now(timezone.utc).isoformat()
                
                # Normalize transaction data
                normalized = {
                    "transaction_id": str(transaction_id),
                    "customer_id": str(customer_id).strip(),
                    "amount": amount,
                    "transaction_start_time": transaction_time,
                }
                
                # Add optional fields with flexible matching
                optional_fields = {
                    "batch_id": get_field_value(txn, "batch_id"),
                    "account_id": get_field_value(txn, "account_id"),
                    "subscription_id": get_field_value(txn, "subscription_id"),
                    "currency_code": get_field_value(txn, "currency_code"),
                    "country_code": get_field_value(txn, "country_code"),
                    "provider_id": get_field_value(txn, "provider_id"),
                    "product_id": get_field_value(txn, "product_id"),
                    "product_category": get_field_value(txn, "product_category"),
                    "channel_id": get_field_value(txn, "channel_id"),
                    "value": get_field_value(txn, "value"),
                    "pricing_strategy": get_field_value(txn, "pricing_strategy"),
                    "fraud_result": get_field_value(txn, "fraud_result"),
                }
                
                for field, value in optional_fields.items():
                    if value is not None:
                        normalized[field] = value
                
                normalized_transactions.append(normalized)
                
            except Exception as e:
                validation_errors.append(f"Row {idx + 1}: {str(e)}")
                continue
        
        if not normalized_transactions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No valid transactions found. Errors: {validation_errors[:5]}"
            )
        
        # Upload to database
        with get_db_session() as session:
            service = RawTransactionService(session)
            count = service.upload_transactions(
                transactions=normalized_transactions,
                uploaded_by=username,
                data_source=data_source,
                file_name=file.filename or "unknown",
                data_version=data_version
            )
            session.commit()
        
        # Update version metadata with actual row count
        if auto_version:
            try:
                with get_db_session() as version_session:
                    version_service = DataVersionService(version_session)
                    db_version = version_service.get_version("raw_transactions", auto_version)
                    if db_version:
                        # Update metadata with actual row count
                        metadata = db_version.data_metadata or {}
                        metadata["row_count"] = count
                        metadata["validated_rows"] = count
                        metadata["validation_errors"] = len(validation_errors)
                        db_version.data_metadata = metadata
                        version_session.commit()
                        logger.info(f"Updated version metadata for {auto_version} with row count: {count}")
            except Exception as update_error:
                logger.warning(f"Error updating version metadata: {update_error}", exc_info=True)
        
        # Clean up temporary file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temp file: {cleanup_error}")
        
        logger.info(
            f"Raw data uploaded successfully",
            extra={
                "username": username,
                "file_name": file.filename,
                "count": count,
                "total_rows": len(transactions),
                "validation_errors": len(validation_errors),
                "data_version": data_version
            }
        )
        
        return JSONResponse({
            "message": "Data uploaded successfully",
            "uploaded_count": count,
            "total_rows": len(transactions),
            "validation_errors": len(validation_errors),
            "file_name": file.filename,
            "data_source": data_source,
            "data_version": data_version,
            "auto_versioned": auto_version is not None
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading raw data: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload data: {str(e)}"
        )


@app.get("/api/data/transactions", tags=["Data Management"])
async def get_transactions(
    customer_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    token: str = Depends(oauth2_scheme)
):
    """
    Get raw transactions from database.
    
    Args:
        customer_id: Filter by customer ID
        limit: Maximum number of records
        offset: Number of records to skip
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        token: Authentication token
    
    Returns:
        List of transactions with pagination
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "data:read" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to view data"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import RawTransactionRepository
        from src.database.models import RawTransaction
        from sqlalchemy import and_
        
        with get_db_session() as session:
            repository = RawTransactionRepository(session)
            
            # Build query
            query = session.query(RawTransaction)
            
            if customer_id:
                query = query.filter(RawTransaction.customer_id == customer_id)
            
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    query = query.filter(RawTransaction.transaction_start_time >= start_dt)
                except:
                    pass
            
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    query = query.filter(RawTransaction.transaction_start_time <= end_dt)
                except:
                    pass
            
            # Get total count
            total = query.count()
            
            # Get transactions with pagination
            transactions = query.order_by(
                RawTransaction.transaction_start_time.desc()
            ).offset(offset).limit(limit).all()
            
            transactions_data = []
            for txn in transactions:
                transactions_data.append({
                    "transaction_id": txn.transaction_id,
                    "customer_id": txn.customer_id,
                    "amount": float(txn.amount) if txn.amount else 0,
                    "currency_code": txn.currency_code,
                    "product_category": txn.product_category,
                    "channel_id": txn.channel_id,
                    "transaction_start_time": txn.transaction_start_time.isoformat() if txn.transaction_start_time else None,
                    "uploaded_at": txn.uploaded_at.isoformat() if txn.uploaded_at else None,
                    "uploaded_by": txn.uploaded_by,
                    "data_source": txn.data_source,
                    "file_name": txn.file_name
                })
            
            return JSONResponse({
                "transactions": transactions_data,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total
            })
            
    except Exception as e:
        logger.error(f"Error getting transactions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transactions: {str(e)}"
        )


# ============================================================================
# Monitoring & Drift Detection Endpoints
# ============================================================================

@app.post("/api/monitoring/drift/detect", tags=["Monitoring", "Drift Detection"])
async def detect_drift(
    feature_name: Optional[str] = None,
    model_version: Optional[str] = None,
    token: str = Depends(oauth2_scheme)
):
    """
    Detect drift for features or predictions.
    
    Args:
        feature_name: Specific feature to check (optional, checks all if not provided)
        model_version: Model version to check
        token: Authentication token
    
    Returns:
        Drift detection results
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "model:performance" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access drift detection"
        )
    
    try:
        from src.monitoring.drift_detection import DriftDetector, DriftMonitor
        from src.database.connection import get_db_session
        from src.database.repositories import PredictionRepository
        from src.database.models import Prediction
        import numpy as np
        
        with get_db_session() as session:
            # Get recent predictions for drift analysis
            pred_repo = PredictionRepository(session)
            recent_predictions = session.query(Prediction).order_by(
                Prediction.created_at.desc()
            ).limit(1000).all()
            
            if not recent_predictions:
                return JSONResponse({
                    "message": "No predictions available for drift detection",
                    "drift_detected": False
                })
            
            # Extract prediction probabilities
            current_probs = np.array([float(p.probability) for p in recent_predictions])
            
            # Try to load actual training data for reference distribution
            reference_probs = None
            try:
                from src.database.models import ModelMetadata, DataSplit
                from src.database.services import DataVersionService
                
                # Get the active model to find its training data version
                active_model = session.query(ModelMetadata).filter(
                    ModelMetadata.is_active == True
                ).order_by(ModelMetadata.created_at.desc()).first()
                
                if active_model and active_model.training_data_version:
                    # Try to get training predictions from the same model version
                    # Use older predictions as reference (from when model was trained)
                    training_predictions = session.query(Prediction).filter(
                        Prediction.model_version == active_model.model_version,
                        Prediction.created_at < recent_predictions[-1].created_at if recent_predictions else None
                    ).order_by(Prediction.created_at.asc()).limit(1000).all()
                    
                    if training_predictions and len(training_predictions) >= 100:
                        # Use first 1000 training predictions as reference
                        reference_probs = np.array([float(p.probability) for p in training_predictions])
                        logger.info(f"Using {len(reference_probs)} training predictions as reference for drift detection")
                    else:
                        # Fallback: use first half of recent predictions as reference
                        # (assuming they're from before any potential drift)
                        mid_point = len(recent_predictions) // 2
                        if mid_point >= 50:
                            reference_probs = np.array([float(p.probability) for p in recent_predictions[mid_point:]])
                            logger.info(f"Using {len(reference_probs)} older predictions as reference (fallback)")
            except Exception as ref_error:
                logger.warning(f"Could not load training reference data: {ref_error}", exc_info=True)
            
            # Final fallback: use statistical estimate based on current data
            if reference_probs is None or len(reference_probs) < 50:
                # Estimate reference from current data (assuming it's close to training)
                # Use a smoothed distribution based on current data
                if len(current_probs) >= 100:
                    # Use first 30% as reference (assuming it's closer to training)
                    ref_size = max(100, int(len(current_probs) * 0.3))
                    reference_probs = current_probs[:ref_size]
                    logger.info(f"Using first {len(reference_probs)} predictions as reference (statistical fallback)")
                else:
                    # Last resort: use beta distribution estimate from current data
                    mean_prob = np.mean(current_probs)
                    std_prob = np.std(current_probs)
                    if std_prob > 0:
                        # Estimate beta parameters from mean and variance
                        alpha = mean_prob * (mean_prob * (1 - mean_prob) / (std_prob ** 2) - 1)
                        beta = (1 - mean_prob) * (mean_prob * (1 - mean_prob) / (std_prob ** 2) - 1)
                        alpha = max(0.1, min(alpha, 100))
                        beta = max(0.1, min(beta, 100))
                        reference_probs = np.random.beta(alpha, beta, size=min(1000, len(current_probs) * 2))
                        logger.info(f"Using estimated beta distribution as reference (alpha={alpha:.2f}, beta={beta:.2f})")
                    else:
                        reference_probs = np.random.beta(2, 5, size=1000)
                        logger.warning("Using default beta(2,5) distribution as reference - no training data available")
            
            # Initialize drift detector with reference data
            detector = DriftDetector(reference_data=reference_probs)
            monitor = DriftMonitor(detector)
            
            # Detect drift in predictions
            drift_result = monitor.monitor_predictions(
                predictions=current_probs,
                reference_predictions=reference_probs,
                model_version=model_version or "latest"
            )
            
            # Also detect drift on features if available
            feature_drift_results = {}
            try:
                # Extract features from predictions
                predictions_with_features = [p for p in recent_predictions if p.features and isinstance(p.features, dict)]
                
                if predictions_with_features and len(predictions_with_features) >= 50:
                    # Get feature names from first prediction
                    sample_features = predictions_with_features[0].features
                    if isinstance(sample_features, dict):
                        feature_names = list(sample_features.keys())[:10]  # Check top 10 features
                        
                        # Extract feature values
                        for feat_name in feature_names:
                            try:
                                current_feat_values = np.array([
                                    float(p.features.get(feat_name, 0)) 
                                    for p in predictions_with_features 
                                    if isinstance(p.features, dict) and feat_name in p.features
                                ])
                                
                                if len(current_feat_values) >= 50:
                                    # Get reference feature values from older predictions
                                    ref_feat_values = None
                                    if reference_probs is not None and len(training_predictions) >= 50:
                                        ref_feat_predictions = [p for p in training_predictions if p.features and isinstance(p.features, dict)]
                                        if ref_feat_predictions:
                                            ref_feat_values = np.array([
                                                float(p.features.get(feat_name, 0))
                                                for p in ref_feat_predictions
                                                if isinstance(p.features, dict) and feat_name in p.features
                                            ])
                                    
                                    if ref_feat_values is None or len(ref_feat_values) < 50:
                                        # Use first 30% as reference
                                        ref_size = max(50, int(len(current_feat_values) * 0.3))
                                        ref_feat_values = current_feat_values[:ref_size]
                                    
                                    if len(ref_feat_values) >= 50 and len(current_feat_values) >= 50:
                                        # Detect drift on this feature
                                        feat_detector = DriftDetector(reference_data=ref_feat_values)
                                        feat_result = feat_detector.detect_drift(
                                            feature_name=feat_name,
                                            current_data=current_feat_values,
                                            model_version=model_version or "latest"
                                        )
                                        
                                        feature_drift_results[feat_name] = feat_result
                                        
                                        # Save to database if drift detected
                                        if feat_result.get("drift_detected"):
                                            monitor.save_drift_metric(
                                                feature_name=feat_name,
                                                psi=feat_result.get("psi", 0.0),
                                                ks_statistic=feat_result.get("ks_statistic"),
                                                chi_square=feat_result.get("chi_square"),
                                                is_drifted=True,
                                                drift_severity=feat_result.get("drift_severity", "minor"),
                                                model_version=model_version or "latest"
                                            )
                            except Exception as feat_error:
                                logger.warning(f"Error detecting drift for feature {feat_name}: {feat_error}")
            except Exception as feat_drift_error:
                logger.warning(f"Error in feature-level drift detection: {feat_drift_error}", exc_info=True)
            
            # Save prediction probability drift to database if detected
            if drift_result.get("drift_detected"):
                monitor.save_drift_metric(
                    feature_name="prediction_probability",
                    psi=drift_result.get("psi", 0.0),
                    ks_statistic=drift_result.get("ks_statistic"),
                    is_drifted=True,
                    drift_severity=drift_result.get("drift_severity", "minor"),
                    model_version=model_version or "latest"
                )
                
                # Send alert
                from src.monitoring.alerts import get_alert_manager, AlertSeverity
                alert_manager = get_alert_manager()
                alert = alert_manager.check_drift_alert(
                    feature_name="prediction_probability",
                    psi=drift_result.get("psi", 0.0),
                    drift_severity=drift_result.get("drift_severity", "minor")
                )
                if alert:
                    alert_manager.send_alert(alert)
                
                # Auto-trigger retraining if drift severity is major
                drift_severity = drift_result.get("drift_severity", "minor")
                if drift_severity in ["major", "critical"]:
                    try:
                        from src.pipelines.retraining import RetrainingScheduler
                        scheduler = RetrainingScheduler()
                        job_id = scheduler.trigger_on_drift(
                            model_name=settings.model_name,
                            drift_metadata={
                                "psi": drift_result.get("psi", 0.0),
                                "drift_severity": drift_severity,
                                "ks_statistic": drift_result.get("ks_statistic"),
                                "detected_at": datetime.now(timezone.utc).isoformat(),
                                "model_version": model_version or "latest"
                            }
                        )
                        if job_id:
                            logger.info(f"Auto-triggered retraining job {job_id} due to {drift_severity} drift")
                    except Exception as retrain_error:
                        logger.warning(f"Failed to auto-trigger retraining on drift: {retrain_error}", exc_info=True)
            
            # Combine results
            combined_result = {
                **drift_result,
                "feature_drift": feature_drift_results,
                "features_checked": len(feature_drift_results),
                "features_with_drift": len([r for r in feature_drift_results.values() if r.get("drift_detected")])
            }
            
            return JSONResponse(combined_result)
            
    except Exception as e:
        logger.error(f"Error detecting drift: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect drift: {str(e)}"
        )


@app.get("/api/monitoring/drift/metrics", tags=["Monitoring", "Drift Detection"])
async def get_drift_metrics(
    feature_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    token: str = Depends(oauth2_scheme)
):
    """
    Get drift metrics from database.
    
    Args:
        feature_name: Filter by feature name
        start_date: Start date filter
        end_date: End date filter
        limit: Maximum number of records
        token: Authentication token
    
    Returns:
        List of drift metrics
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.models import DriftMetric
        
        with get_db_session() as session:
            query = session.query(DriftMetric)
            
            if feature_name:
                query = query.filter(DriftMetric.feature_name == feature_name)
            
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    query = query.filter(DriftMetric.time >= start_dt)
                except:
                    pass
            
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    query = query.filter(DriftMetric.time <= end_dt)
                except:
                    pass
            
            metrics = query.order_by(DriftMetric.time.desc()).limit(limit).all()
            
            metrics_data = []
            for metric in metrics:
                metrics_data.append({
                    "id": metric.id,
                    "time": metric.time.isoformat() if metric.time else None,
                    "feature_name": metric.feature_name,
                    "psi": float(metric.psi) if metric.psi else None,
                    "ks_statistic": float(metric.ks_statistic) if metric.ks_statistic else None,
                    "chi_square": float(metric.chi_square) if metric.chi_square else None,
                    "is_drifted": metric.is_drifted,
                    "drift_severity": metric.drift_severity,
                    "model_version": metric.model_version
                })
            
            return JSONResponse({
                "metrics": metrics_data,
                "total": len(metrics_data)
            })
            
    except Exception as e:
        logger.error(f"Error getting drift metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get drift metrics: {str(e)}"
        )


@app.post("/api/monitoring/data-quality/check", tags=["Monitoring", "Data Quality"])
async def check_data_quality(
    token: str = Depends(oauth2_scheme)
):
    """
    Check data quality for uploaded transactions.
    
    Args:
        token: Authentication token
    
    Returns:
        Data quality report
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.monitoring.data_quality import DataQualityChecker
        from src.database.connection import get_db_session
        from src.database.models import RawTransaction
        
        with get_db_session() as session:
            # Get recent transactions
            recent_txns = session.query(RawTransaction).order_by(
                RawTransaction.uploaded_at.desc()
            ).limit(1000).all()
            
            if not recent_txns:
                return JSONResponse({
                    "message": "No transactions available for quality check",
                    "quality_score": 0.0
                })
            
            # Convert to dictionaries
            txn_data = []
            for txn in recent_txns:
                txn_dict = {
                    "transaction_id": txn.transaction_id,
                    "customer_id": txn.customer_id,
                    "amount": float(txn.amount) if txn.amount else None,
                    "transaction_start_time": txn.transaction_start_time.isoformat() if txn.transaction_start_time else None,
                }
                if txn.batch_id:
                    txn_dict["batch_id"] = txn.batch_id
                if txn.currency_code:
                    txn_dict["currency_code"] = txn.currency_code
                if txn.product_category:
                    txn_dict["product_category"] = txn.product_category
                txn_data.append(txn_dict)
            
            # Check quality
            checker = DataQualityChecker()
            report = checker.generate_quality_report(txn_data)
            
            return JSONResponse(report)
            
    except Exception as e:
        logger.error(f"Error checking data quality: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check data quality: {str(e)}"
        )


@app.get("/api/monitoring/alerts", tags=["Monitoring", "Alerts"])
async def get_alerts(
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    limit: int = 50,
    token: str = Depends(oauth2_scheme)
):
    """
    Get recent alerts.
    
    Args:
        severity: Filter by severity (info, warning, error, critical)
        alert_type: Filter by alert type
        limit: Maximum number of alerts
        token: Authentication token
    
    Returns:
        List of alerts
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.monitoring.alerts import get_alert_manager
        
        alert_manager = get_alert_manager()
        alerts = alert_manager.alert_history
        
        # Apply filters
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        # Sort by timestamp (newest first) and limit
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        alerts = alerts[:limit]
        
        return JSONResponse({
            "alerts": [a.to_dict() for a in alerts],
            "total": len(alerts)
        })
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alerts: {str(e)}"
        )


# ============================================================================
# Customer Scoring Endpoint
# ============================================================================

@app.post("/api/customers/score", tags=["Customer Scoring"])
async def score_customer(
    customer_id: str = Body(...),
    transactions: List[Dict[str, Any]] = Body(...),
    token: str = Depends(oauth2_scheme)
):
    """
    Score a customer based on their transaction data.
    
    This endpoint:
    1. Takes a customer_id and their transactions
    2. Generates features from the transactions
    3. Makes a prediction with the customer_id
    4. Returns the score and prediction details
    
    Args:
        customer_id: Customer ID to score
        transactions: List of transaction dictionaries
        token: Authentication token
    
    Returns:
        Prediction response with customer score
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.features.store import get_feature_store
        from datetime import datetime, timezone
        
        global model
        
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please check server logs."
            )
        
        # Initialize feature store
        feature_store = get_feature_store()
        
        # Try to get features from feature store first
        feature_vector = feature_store.get_feature_vector(customer_id, use_cache=True)
        features_from_store = feature_vector is not None
        
        # If not in store, compute from transactions
        if not features_from_store:
            if not transactions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No transactions provided and features not found in store"
                )
            
            logger.info(f"Computing features for customer {customer_id} (not in feature store)")
            
            # Compute and store features
            feature_data = feature_store.compute_and_store_features(
                customer_id=customer_id,
                transactions=transactions,
                feature_version="v1.0",
                store_features=True
            )
            feature_vector = feature_data["feature_vector"]
        else:
            logger.info(f"Using cached features for customer {customer_id}")
        
        # Ensure we have the right number of features
        if len(feature_vector) < settings.expected_features:
            feature_vector.extend([0.0] * (settings.expected_features - len(feature_vector)))
        elif len(feature_vector) > settings.expected_features:
            feature_vector = feature_vector[:settings.expected_features]
        
        # Set features and transaction_count for later use
        features = feature_vector
        transaction_count = len(transactions) if transactions else 0
        
        # Make prediction
        features_array = np.array(features, dtype=np.float64).reshape(1, -1)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)[0]
            probability = float(probabilities[1])
            prediction = int(np.argmax(probabilities))
        else:
            prediction = model.predict(features_array)[0]
            probability = float(prediction)
        
        # Determine risk level
        if probability < settings.risk_threshold_low:
            risk_level = "low"
        elif probability > settings.risk_threshold_high:
            risk_level = "high"
        else:
            risk_level = "medium"
        
        # Calculate customer score (0-100 scale)
        customer_score = int((1 - probability) * 100)
        
        # Generate prediction ID and timestamp
        prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc)
        timestamp_iso = timestamp.isoformat()
        
        # Save prediction to database and track lineage
        try:
            from src.database.connection import get_db_session
            from src.database.services import PredictionService, DataVersionService, DataLineageService
            
            with get_db_session() as session:
                prediction_service = PredictionService(session)
                
                # Log what we're about to save
                logger.info(
                    f"Saving prediction for customer {customer_id}",
                    extra={
                        "prediction_id": prediction_id,
                        "customer_id": customer_id,
                        "customer_score": customer_score,
                        "model_name": model_name or "credit_scoring_model",
                        "model_version": model_version or "unknown",
                        "model_stage": settings.model_stage or "Production"
                    }
                )
                
                saved_prediction = prediction_service.save_prediction(
                    prediction_id=prediction_id,
                    customer_id=customer_id,
                    prediction=int(prediction),
                    probability=probability,
                    risk_level=risk_level,
                    customer_score=customer_score,
                    latency_ms=0,  # Will be calculated if needed
                    model_name=model_name or "credit_scoring_model",
                    model_version=model_version or "unknown",
                    model_stage=settings.model_stage or "Production",
                    features=features
                )
                
                # Verify the prediction was saved correctly
                logger.info(
                    f"Prediction saved successfully",
                    extra={
                        "prediction_id": saved_prediction.prediction_id,
                        "customer_id": saved_prediction.customer_id,
                        "customer_score": saved_prediction.customer_score,
                        "has_customer_id": saved_prediction.customer_id is not None,
                        "has_customer_score": saved_prediction.customer_score is not None
                    }
                )
                
                # Track lineage: link to latest raw_transactions data version
                # This is optional and won't fail the prediction if it errors
                try:
                    version_service = DataVersionService(session)
                    lineage_service = DataLineageService(session)
                    
                    # Get latest raw_transactions version
                    try:
                        latest_version = version_service.get_latest_version("raw_transactions")
                    except Exception as version_error:
                        # If data_versions table doesn't have the right schema, skip lineage
                        logger.debug(f"Could not get latest version (table may need migration): {version_error}")
                        latest_version = None
                    
                    if latest_version:
                        lineage_service.create_lineage(
                            source_data_version_id=latest_version.id,
                            source_data_type=latest_version.data_type,
                            source_version=latest_version.version,
                            target_type="prediction",
                            target_id=prediction_id,
                            target_name=f"Customer Score: {customer_id}",
                            relationship_type="used_for",
                            operation="prediction",
                            metadata={
                                "customer_id": customer_id,
                                "model_version": model_version or "unknown",
                                "transaction_count": transaction_count,
                                "features_from_store": features_from_store
                            }
                        )
                        logger.info(f"Created lineage for prediction {prediction_id} from data version {latest_version.version}")
                except Exception as lineage_error:
                    logger.warning(f"Failed to create lineage (non-critical): {lineage_error}", exc_info=True)
                    # Don't fail the prediction if lineage tracking fails
                
                session.commit()
                logger.info(f"Customer scored and saved: {customer_id}, score: {customer_score}, prediction_id: {prediction_id}")
        except Exception as db_error:
            logger.error(
                f"Failed to save prediction to database: {db_error}",
                extra={
                    "prediction_id": prediction_id,
                    "customer_id": customer_id,
                    "customer_score": customer_score
                },
                exc_info=True
            )
            # Re-raise the error so the user knows something went wrong
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save prediction to database: {str(db_error)}"
            )
        
        return JSONResponse({
            "customer_id": customer_id,
            "prediction": int(prediction),
            "probability": probability,
            "customer_score": customer_score,
            "risk_level": risk_level,
            "prediction_id": prediction_id,
            "timestamp": timestamp_iso,
            "features_used": len(features),
            "transaction_count": transaction_count,
            "features_from_store": features_from_store
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scoring customer: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to score customer: {str(e)}"
        )


# ============================================================================
# Feature Store Endpoints
# ============================================================================

@app.get("/api/features/stats", tags=["Feature Store"])
async def get_feature_store_stats(
    token: str = Depends(oauth2_scheme)
):
    """
    Get feature store statistics.
    
    Args:
        token: Authentication token
    
    Returns:
        Feature store statistics including total features, cache hit rate, etc.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.models import CustomerFeature, Prediction
        from sqlalchemy import func
        from datetime import datetime, timezone, timedelta
        
        with get_db_session() as session:
            # Get total features in store
            total_features = session.query(func.count(CustomerFeature.customer_id)).scalar() or 0
            
            # Get features updated in last 24 hours
            last_24h = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_features = session.query(func.count(CustomerFeature.customer_id)).filter(
                CustomerFeature.last_updated >= last_24h
            ).scalar() or 0
            
            # Get features updated in last 7 days
            last_7d = datetime.now(timezone.utc) - timedelta(days=7)
            weekly_features = session.query(func.count(CustomerFeature.customer_id)).filter(
                CustomerFeature.last_updated >= last_7d
            ).scalar() or 0
            
            # Get oldest and newest feature timestamps
            oldest_feature = session.query(func.min(CustomerFeature.last_updated)).scalar()
            newest_feature = session.query(func.max(CustomerFeature.last_updated)).scalar()
            
            # Count predictions that used features from store (if tracking available)
            # This is an approximation - we can track this better in the future
            total_predictions = session.query(func.count(Prediction.prediction_id)).scalar() or 0
            
            # Get feature version distribution
            version_counts = session.query(
                CustomerFeature.feature_version,
                func.count(CustomerFeature.customer_id)
            ).group_by(CustomerFeature.feature_version).all()
            
            version_distribution = {
                version: count for version, count in version_counts if version
            }
            
            # Get cache stats from cache manager
            from src.utils.cache import get_cache_manager
            cache_manager = get_cache_manager()
            cached_items_count = cache_manager.size() if hasattr(cache_manager, 'size') else 0
            cache_max_size = cache_manager.max_size if hasattr(cache_manager, 'max_size') else 0
            
            # Calculate cache coverage (approximate)
            cache_coverage = (cached_items_count / max(total_features, 1)) * 100 if total_features > 0 else 0
            
            return JSONResponse({
                "total_features": total_features,
                "recent_features_24h": recent_features,
                "recent_features_7d": weekly_features,
                "oldest_feature": oldest_feature.isoformat() if oldest_feature else None,
                "newest_feature": newest_feature.isoformat() if newest_feature else None,
                "version_distribution": version_distribution,
                "total_predictions": total_predictions,
                "cache_coverage": cache_coverage,
                "cached_items_count": cached_items_count,
                "cache_max_size": cache_max_size,
                "features_updated_24h": recent_features,
                "features_updated_7d": weekly_features,
                "feature_version_distribution": version_distribution
            })
            
    except Exception as e:
        logger.error(f"Error getting feature store stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feature store stats: {str(e)}"
        )


@app.get("/api/features/{customer_id}", tags=["Feature Store"])
async def get_customer_features(
    customer_id: str,
    token: str = Depends(oauth2_scheme)
):
    """
    Get features for a customer from feature store.
    
    Args:
        customer_id: Customer identifier
        token: Authentication token
    
    Returns:
        Customer features from feature store
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.features.store import get_feature_store
        
        feature_store = get_feature_store()
        features = feature_store.get_features(customer_id, use_cache=True)
        
        if features is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Features not found for customer {customer_id}"
            )
        
        return JSONResponse(features)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting features for customer {customer_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get features: {str(e)}"
        )


@app.post("/api/features/{customer_id}", tags=["Feature Store"])
async def compute_and_store_features(
    customer_id: str,
    transactions: List[Dict[str, Any]] = Body(...),
    feature_version: Optional[str] = Body(default=None),
    data_version: Optional[str] = Body(default=None),
    token: str = Depends(oauth2_scheme)
):
    """
    Compute and store features for a customer.
    
    Args:
        customer_id: Customer identifier
        transactions: List of transaction dictionaries
        feature_version: Version of feature engineering pipeline
        data_version: Version of source data
        token: Authentication token
    
    Returns:
        Computed and stored features
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "data:write" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to store features"
        )
    
    try:
        from src.features.store import get_feature_store
        
        feature_store = get_feature_store()
        features = feature_store.compute_and_store_features(
            customer_id=customer_id,
            transactions=transactions,
            feature_version=feature_version,
            data_version=data_version,
            store_features=True
        )
        
        return JSONResponse({
            "customer_id": customer_id,
            "features": features,
            "stored": True,
            "message": "Features computed and stored successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing and storing features for customer {customer_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute and store features: {str(e)}"
        )


@app.put("/api/features/{customer_id}", tags=["Feature Store"])
async def update_customer_features(
    customer_id: str,
    feature_vector: List[float] = Body(...),
    recency_normalized: Optional[float] = Body(default=None),
    frequency_normalized: Optional[float] = Body(default=None),
    monetary_normalized: Optional[float] = Body(default=None),
    feature_version: Optional[str] = Body(default=None),
    data_version: Optional[str] = Body(default=None),
    token: str = Depends(oauth2_scheme)
):
    """
    Update features for a customer in feature store.
    
    Args:
        customer_id: Customer identifier
        feature_vector: List of 26 feature values
        recency_normalized: RFM recency feature
        frequency_normalized: RFM frequency feature
        monetary_normalized: RFM monetary feature
        feature_version: Version of feature engineering pipeline
        data_version: Version of source data
        token: Authentication token
    
    Returns:
        Updated features
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "data:write" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to update features"
        )
    
    try:
        from src.features.store import get_feature_store
        
        if len(feature_vector) != settings.expected_features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected {settings.expected_features} features, got {len(feature_vector)}"
            )
        
        feature_store = get_feature_store()
        feature = feature_store.store_features(
            customer_id=customer_id,
            feature_vector=feature_vector,
            recency_normalized=recency_normalized,
            frequency_normalized=frequency_normalized,
            monetary_normalized=monetary_normalized,
            feature_version=feature_version,
            data_version=data_version
        )
        
        return JSONResponse({
            "customer_id": customer_id,
            "stored": True,
            "last_updated": feature.last_updated.isoformat() if feature.last_updated else None,
            "message": "Features updated successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating features for customer {customer_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update features: {str(e)}"
        )


@app.post("/api/features/batch", tags=["Feature Store"])
async def batch_get_features(
    customer_ids: List[str] = Body(...),
    token: str = Depends(oauth2_scheme)
):
    """
    Get features for multiple customers.
    
    Args:
        customer_ids: List of customer identifiers
        token: Authentication token
    
    Returns:
        Dictionary mapping customer_id to features
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.features.store import get_feature_store
        
        if len(customer_ids) > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 1000 customer IDs allowed per request"
            )
        
        feature_store = get_feature_store()
        features = feature_store.batch_get_features(customer_ids)
        
        return JSONResponse({
            "features": features,
            "count": len([f for f in features.values() if f is not None]),
            "total_requested": len(customer_ids)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error batch getting features: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch get features: {str(e)}"
        )


# ============================================================================
# A/B Testing Endpoints
# ============================================================================

@app.post("/api/experiments", tags=["A/B Testing"])
async def create_experiment(
    experiment_name: str = Body(...),
    variants: List[Dict[str, Any]] = Body(...),
    description: Optional[str] = Body(default=None),
    traffic_percentage: int = Body(default=100),
    assignment_method: str = Body(default="hash"),
    primary_metric: str = Body(default="accuracy"),
    minimum_sample_size: int = Body(default=1000),
    significance_level: float = Body(default=0.05),
    minimum_improvement: float = Body(default=0.01),
    token: str = Depends(oauth2_scheme)
):
    """
    Create a new A/B testing experiment.
    
    Args:
        experiment_name: Unique experiment name
        variants: List of variant configurations
        description: Experiment description
        traffic_percentage: Percentage of traffic to include (0-100)
        assignment_method: Assignment method ('hash', 'random')
        primary_metric: Primary metric to compare
        minimum_sample_size: Minimum samples per variant
        significance_level: Statistical significance level
        minimum_improvement: Minimum improvement to declare winner
        token: Authentication token
    
    Returns:
        Created experiment
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "model:write" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to create experiments"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import ExperimentRepository
        
        with get_db_session() as session:
            experiment_repo = ExperimentRepository(session)
            
            # Check if experiment name already exists
            existing = experiment_repo.get_by_name(experiment_name)
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Experiment '{experiment_name}' already exists"
                )
            
            # Validate variants
            total_percentage = sum(v.get("traffic_percentage", 0) for v in variants)
            if total_percentage != 100:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Variant traffic percentages must sum to 100, got {total_percentage}"
                )
            
            # Create experiment
            experiment = experiment_repo.create_experiment(
                experiment_name=experiment_name,
                variants=variants,
                description=description,
                traffic_percentage=traffic_percentage,
                assignment_method=assignment_method,
                primary_metric=primary_metric,
                minimum_sample_size=minimum_sample_size,
                significance_level=significance_level,
                minimum_improvement=minimum_improvement,
                created_by=session_data.get("username")
            )
            
            return JSONResponse({
                "experiment_id": experiment.experiment_id,
                "experiment_name": experiment.experiment_name,
                "status": experiment.status,
                "message": "Experiment created successfully"
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating experiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create experiment: {str(e)}"
        )


@app.get("/api/experiments", tags=["A/B Testing"])
async def list_experiments(
    status_filter: Optional[str] = Query(None, description="Filter by status ('draft', 'running', 'paused', 'completed', 'cancelled')"),
    token: str = Depends(oauth2_scheme)
):
    """
    List all experiments.
    
    Args:
        status_filter: Filter by status ('draft', 'running', 'paused', 'completed', 'cancelled')
        token: Authentication token
    
    Returns:
        List of experiments
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import ExperimentRepository
        from src.database.models import Experiment
        
        with get_db_session() as session:
            experiment_repo = ExperimentRepository(session)
            
            if status_filter:
                experiments = session.query(Experiment).filter(
                    Experiment.status == status_filter
                ).all()
            else:
                experiments = experiment_repo.get_all()
            
            experiments_data = []
            for exp in experiments:
                experiments_data.append({
                    "experiment_id": exp.experiment_id,
                    "experiment_name": exp.experiment_name,
                    "description": exp.description,
                    "status": exp.status,
                    "variants": exp.variants,
                    "traffic_percentage": exp.traffic_percentage,
                    "primary_metric": exp.primary_metric,
                    "winner_variant": exp.winner_variant,
                    "statistical_significance": float(exp.statistical_significance) if exp.statistical_significance else None,
                    "start_date": exp.start_date.isoformat() if exp.start_date else None,
                    "end_date": exp.end_date.isoformat() if exp.end_date else None,
                    "actual_started_at": exp.actual_started_at.isoformat() if exp.actual_started_at else None,
                    "actual_ended_at": exp.actual_ended_at.isoformat() if exp.actual_ended_at else None,
                    "created_at": exp.created_at.isoformat() if exp.created_at else None
                })
            
            return JSONResponse({
                "experiments": experiments_data,
                "total": len(experiments_data)
            })
            
    except Exception as e:
        logger.error(f"Error listing experiments: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list experiments: {str(e)}"
        )


@app.get("/api/experiments/{experiment_id}", tags=["A/B Testing"])
async def get_experiment(
    experiment_id: int,
    token: str = Depends(oauth2_scheme)
):
    """
    Get experiment details.
    
    Args:
        experiment_id: Experiment ID
        token: Authentication token
    
    Returns:
        Experiment details
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import ExperimentRepository
        
        with get_db_session() as session:
            experiment_repo = ExperimentRepository(session)
            experiment = experiment_repo.get_by_id(experiment_id)
            
            if not experiment:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Experiment {experiment_id} not found"
                )
            
            return JSONResponse({
                "experiment_id": experiment.experiment_id,
                "experiment_name": experiment.experiment_name,
                "description": experiment.description,
                "status": experiment.status,
                "variants": experiment.variants,
                "traffic_percentage": experiment.traffic_percentage,
                "assignment_method": experiment.assignment_method,
                "primary_metric": experiment.primary_metric,
                "minimum_sample_size": experiment.minimum_sample_size,
                "significance_level": float(experiment.significance_level),
                "minimum_improvement": float(experiment.minimum_improvement),
                "winner_variant": experiment.winner_variant,
                "statistical_significance": float(experiment.statistical_significance) if experiment.statistical_significance else None,
                "confidence_interval": experiment.confidence_interval,
                "conclusion": experiment.conclusion,
                "start_date": experiment.start_date.isoformat() if experiment.start_date else None,
                "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
                "actual_started_at": experiment.actual_started_at.isoformat() if experiment.actual_started_at else None,
                "actual_ended_at": experiment.actual_ended_at.isoformat() if experiment.actual_ended_at else None,
                "created_by": experiment.created_by,
                "created_at": experiment.created_at.isoformat() if experiment.created_at else None
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiment: {str(e)}"
        )


@app.post("/api/experiments/{experiment_id}/start", tags=["A/B Testing"])
async def start_experiment(
    experiment_id: int,
    token: str = Depends(oauth2_scheme)
):
    """
    Start an experiment.
    
    Args:
        experiment_id: Experiment ID
        token: Authentication token
    
    Returns:
        Updated experiment
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "model:write" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to start experiments"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import ExperimentRepository
        from src.experimentation.ab_testing import get_ab_testing_framework
        from datetime import datetime, timezone
        
        with get_db_session() as session:
            experiment_repo = ExperimentRepository(session)
            experiment = experiment_repo.get_by_id(experiment_id)
            
            if not experiment:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Experiment {experiment_id} not found"
                )
            
            if experiment.status == "running":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Experiment is already running"
                )
            
            # Load models for all variants
            ab_framework = get_ab_testing_framework()
            variants = experiment.variants if isinstance(experiment.variants, list) else []
            
            for variant in variants:
                model_name = variant.get("model_name", settings.model_name)
                model_version = variant.get("model_version")
                
                if model_version:
                    success = ab_framework.load_model_for_variant(
                        variant["name"],
                        model_name,
                        model_version
                    )
                    if not success:
                        logger.warning(f"Failed to load model for variant {variant['name']}")
            
            # Update experiment status
            experiment.status = "running"
            experiment.actual_started_at = datetime.now(timezone.utc)
            if not experiment.start_date:
                experiment.start_date = experiment.actual_started_at
            
            session.commit()
            
            return JSONResponse({
                "experiment_id": experiment.experiment_id,
                "status": experiment.status,
                "actual_started_at": experiment.actual_started_at.isoformat(),
                "message": "Experiment started successfully"
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting experiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start experiment: {str(e)}"
        )


@app.post("/api/experiments/{experiment_id}/stop", tags=["A/B Testing"])
async def stop_experiment(
    experiment_id: int,
    token: str = Depends(oauth2_scheme)
):
    """
    Stop an experiment.
    
    Args:
        experiment_id: Experiment ID
        token: Authentication token
    
    Returns:
        Updated experiment
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "model:write" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to stop experiments"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import ExperimentRepository
        from datetime import datetime, timezone
        
        with get_db_session() as session:
            experiment_repo = ExperimentRepository(session)
            experiment = experiment_repo.get_by_id(experiment_id)
            
            if not experiment:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Experiment {experiment_id} not found"
                )
            
            if experiment.status != "running":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Experiment is not running (current status: {experiment.status})"
                )
            
            # Update experiment status
            experiment.status = "completed"
            experiment.actual_ended_at = datetime.now(timezone.utc)
            if not experiment.end_date:
                experiment.end_date = experiment.actual_ended_at
            
            session.commit()
            
            return JSONResponse({
                "experiment_id": experiment.experiment_id,
                "status": experiment.status,
                "actual_ended_at": experiment.actual_ended_at.isoformat(),
                "message": "Experiment stopped successfully"
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping experiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop experiment: {str(e)}"
        )


@app.get("/api/experiments/{experiment_id}/results", tags=["A/B Testing"])
async def get_experiment_results(
    experiment_id: int,
    token: str = Depends(oauth2_scheme)
):
    """
    Get experiment results and metrics.
    
    Args:
        experiment_id: Experiment ID
        token: Authentication token
    
    Returns:
        Experiment results with variant metrics and statistical analysis
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import ExperimentRepository, ExperimentMetricRepository
        from src.experimentation.ab_testing import get_ab_testing_framework
        
        with get_db_session() as session:
            experiment_repo = ExperimentRepository(session)
            experiment = experiment_repo.get_by_id(experiment_id)
            
            if not experiment:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Experiment {experiment_id} not found"
                )
            
            # Calculate current metrics
            ab_framework = get_ab_testing_framework()
            variant_metrics = ab_framework.calculate_experiment_metrics(experiment_id)
            
            # Get stored metrics
            metric_repo = ExperimentMetricRepository(session)
            stored_metrics = metric_repo.get_by_experiment(experiment_id)
            
            # Determine winner if experiment is completed
            winner_info = None
            if experiment.status == "completed":
                winner_info = ab_framework.determine_winner(
                    experiment_id,
                    experiment.primary_metric,
                    float(experiment.significance_level),
                    float(experiment.minimum_improvement)
                )
            
            return JSONResponse({
                "experiment_id": experiment_id,
                "experiment_name": experiment.experiment_name,
                "status": experiment.status,
                "variant_metrics": variant_metrics,
                "stored_metrics": [
                    {
                        "variant_name": m.variant_name,
                        "sample_size": m.sample_size,
                        "accuracy": float(m.accuracy) if m.accuracy else None,
                        "roc_auc": float(m.roc_auc) if m.roc_auc else None,
                        "precision": float(m.precision) if m.precision else None,
                        "recall": float(m.recall) if m.recall else None,
                        "f1_score": float(m.f1_score) if m.f1_score else None,
                        "avg_latency_ms": float(m.avg_latency_ms) if m.avg_latency_ms else None,
                        "calculated_at": m.calculated_at.isoformat() if m.calculated_at else None
                    }
                    for m in stored_metrics
                ],
                "winner": winner_info,
                "primary_metric": experiment.primary_metric,
                "statistical_significance": float(experiment.statistical_significance) if experiment.statistical_significance else None,
                "confidence_interval": experiment.confidence_interval,
                "conclusion": experiment.conclusion
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment results: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiment results: {str(e)}"
        )


@app.post("/api/experiments/{experiment_id}/promote", tags=["A/B Testing"])
async def promote_winner(
    experiment_id: int,
    variant_name: Optional[str] = Body(default=None),
    token: str = Depends(oauth2_scheme)
):
    """
    Promote winning variant to production.
    
    Args:
        experiment_id: Experiment ID
        variant_name: Variant to promote (if None, uses experiment winner)
        token: Authentication token
    
    Returns:
        Promotion result
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "model:write" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to promote models"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import ExperimentRepository
        
        with get_db_session() as session:
            experiment_repo = ExperimentRepository(session)
            experiment = experiment_repo.get_by_id(experiment_id)
            
            if not experiment:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Experiment {experiment_id} not found"
                )
            
            # Determine variant to promote
            if not variant_name:
                variant_name = experiment.winner_variant
            
            if not variant_name:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No winner determined. Please specify variant_name or complete experiment analysis."
                )
            
            # Get variant configuration
            variants = experiment.variants if isinstance(experiment.variants, list) else []
            variant_config = next((v for v in variants if v["name"] == variant_name), None)
            
            if not variant_config:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Variant '{variant_name}' not found in experiment"
                )
            
            model_version = variant_config.get("model_version")
            model_name = variant_config.get("model_name", settings.model_name)
            
            # In production, would promote model in MLflow registry
            # For now, just return success message
            logger.info(
                f"Promoting variant {variant_name} (model {model_name} v{model_version}) to production",
                extra={
                    "experiment_id": experiment_id,
                    "variant_name": variant_name,
                    "model_name": model_name,
                    "model_version": model_version
                }
            )
            
            return JSONResponse({
                "experiment_id": experiment_id,
                "variant_name": variant_name,
                "model_name": model_name,
                "model_version": model_version,
                "message": f"Variant {variant_name} promoted to production. Please update model configuration manually."
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error promoting winner: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to promote winner: {str(e)}"
        )


# ============================================================================
# Model Retraining Endpoints
# ============================================================================

@app.post("/api/retraining/jobs", tags=["Model Retraining"])
async def create_retraining_job(
    job_name: str = Body(...),
    model_name: str = Body(...),
    model_type: str = Body(default="random_forest"),
    trigger_type: str = Body(default="manual"),
    hyperparameters: Optional[Dict[str, Any]] = Body(default=None),
    training_config: Optional[Dict[str, Any]] = Body(default=None),
    token: str = Depends(oauth2_scheme)
):
    """
    Create a new retraining job.
    
    Args:
        job_name: Name for the retraining job
        model_name: Name of the model to retrain
        model_type: Type of model ('logistic_regression', 'random_forest', etc.)
        trigger_type: Trigger type ('manual', 'scheduled', 'drift', 'new_data', 'performance_degradation')
        hyperparameters: Model hyperparameters
        training_config: Additional training configuration
        token: Authentication token
    
    Returns:
        Created retraining job
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "model:write" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to create retraining jobs"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import RetrainingJobRepository
        
        with get_db_session() as session:
            job_repo = RetrainingJobRepository(session)
            
            job = job_repo.create_job(
                job_name=job_name,
                model_name=model_name,
                model_type=model_type,
                trigger_type=trigger_type,
                hyperparameters=hyperparameters,
                training_config=training_config,
                created_by=session_data.get("username")
            )
            
            session.commit()
            
            return JSONResponse({
                "job_id": job.job_id,
                "job_name": job.job_name,
                "status": job.status,
                "message": "Retraining job created successfully"
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating retraining job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create retraining job: {str(e)}"
        )


@app.post("/api/retraining/jobs/{job_id}/run", tags=["Model Retraining"])
async def run_retraining_job(
    job_id: int,
    token: str = Depends(oauth2_scheme)
):
    """
    Execute a retraining job.
    
    Args:
        job_id: Retraining job ID
        token: Authentication token
    
    Returns:
        Job execution result
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "model:write" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to run retraining jobs"
        )
    
    try:
        from src.pipelines.retraining import get_retraining_pipeline
        
        pipeline = get_retraining_pipeline()
        result = pipeline.run_retraining_job(job_id, trigger_type="manual")
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running retraining job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run retraining job: {str(e)}"
        )


@app.get("/api/retraining/jobs", tags=["Model Retraining"])
async def list_retraining_jobs(
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, description="Maximum number of jobs to return"),
    token: str = Depends(oauth2_scheme)
):
    """
    List retraining jobs.
    
    Args:
        status_filter: Filter by status ('pending', 'running', 'completed', 'failed', 'cancelled')
        limit: Maximum number of jobs to return
        token: Authentication token
    
    Returns:
        List of retraining jobs
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import RetrainingJobRepository
        from src.database.models import RetrainingJob
        
        with get_db_session() as session:
            job_repo = RetrainingJobRepository(session)
            
            if status_filter:
                jobs = job_repo.get_by_status(status_filter)
            else:
                jobs = job_repo.get_recent_jobs(limit=limit)
            
            jobs_data = []
            for job in jobs:
                jobs_data.append({
                    "job_id": job.job_id,
                    "job_name": job.job_name,
                    "model_name": job.model_name,
                    "model_type": job.model_type,
                    "status": job.status,
                    "trigger_type": job.trigger_type,
                    "validation_passed": job.validation_passed,
                    "promotion_status": job.promotion_status,
                    "promoted_to_stage": job.promoted_to_stage,
                    "model_version": job.model_version,
                    "mlflow_run_id": job.mlflow_run_id,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "error_message": job.error_message
                })
            
            return JSONResponse({
                "jobs": jobs_data,
                "total": len(jobs_data)
            })
            
    except Exception as e:
        logger.error(f"Error listing retraining jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list retraining jobs: {str(e)}"
        )


@app.get("/api/retraining/jobs/{job_id}", tags=["Model Retraining"])
async def get_retraining_job(
    job_id: int,
    token: str = Depends(oauth2_scheme)
):
    """
    Get retraining job details.
    
    Args:
        job_id: Retraining job ID
        token: Authentication token
    
    Returns:
        Retraining job details
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import RetrainingJobRepository
        
        with get_db_session() as session:
            job_repo = RetrainingJobRepository(session)
            job = job_repo.get_by_id(job_id)
            
            if not job:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Retraining job {job_id} not found"
                )
            
            return JSONResponse({
                "job_id": job.job_id,
                "job_name": job.job_name,
                "model_name": job.model_name,
                "model_type": job.model_type,
                "status": job.status,
                "trigger_type": job.trigger_type,
                "trigger_metadata": job.trigger_metadata,
                "training_metrics": job.training_metrics,
                "test_metrics": job.test_metrics,
                "validation_passed": job.validation_passed,
                "validation_errors": job.validation_errors,
                "baseline_comparison": job.baseline_comparison,
                "promotion_status": job.promotion_status,
                "promoted_to_stage": job.promoted_to_stage,
                "promotion_timestamp": job.promotion_timestamp.isoformat() if job.promotion_timestamp else None,
                "model_version": job.model_version,
                "mlflow_run_id": job.mlflow_run_id,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "error_message": job.error_message
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting retraining job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get retraining job: {str(e)}"
        )


@app.post("/api/retraining/schedules", tags=["Model Retraining"])
async def create_retraining_schedule(
    schedule_name: str = Body(...),
    model_name: str = Body(...),
    schedule_type: str = Body(...),
    schedule_config: Dict[str, Any] = Body(...),
    training_config: Optional[Dict[str, Any]] = Body(default=None),
    description: Optional[str] = Body(default=None),
    token: str = Depends(oauth2_scheme)
):
    """
    Create a retraining schedule.
    
    Args:
        schedule_name: Unique schedule name
        model_name: Name of the model to retrain
        schedule_type: Schedule type ('daily', 'weekly', 'monthly', 'cron')
        schedule_config: Schedule configuration (cron expression or schedule details)
        training_config: Training configuration
        description: Schedule description
        token: Authentication token
    
    Returns:
        Created schedule
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "model:write" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to create retraining schedules"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import RetrainingScheduleRepository
        from src.database.models import RetrainingSchedule
        from datetime import datetime, timezone, timedelta
        
        with get_db_session() as session:
            schedule_repo = RetrainingScheduleRepository(session)
            
            # Check if schedule name already exists
            existing = schedule_repo.get_by_name(schedule_name)
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Schedule '{schedule_name}' already exists"
                )
            
            # Calculate next run time
            now = datetime.now(timezone.utc)
            if schedule_type == "daily":
                next_run = now + timedelta(days=1)
            elif schedule_type == "weekly":
                next_run = now + timedelta(weeks=1)
            elif schedule_type == "monthly":
                next_run = now + timedelta(days=30)
            else:
                next_run = now + timedelta(days=1)  # Default for cron
            
            schedule = RetrainingSchedule(
                schedule_name=schedule_name,
                model_name=model_name,
                schedule_type=schedule_type,
                schedule_config=schedule_config,
                training_config=training_config,
                description=description,
                is_active=True,
                next_run_at=next_run,
                created_by=session_data.get("username")
            )
            
            session.add(schedule)
            session.commit()
            
            return JSONResponse({
                "schedule_id": schedule.schedule_id,
                "schedule_name": schedule.schedule_name,
                "is_active": schedule.is_active,
                "next_run_at": schedule.next_run_at.isoformat() if schedule.next_run_at else None,
                "message": "Retraining schedule created successfully"
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating retraining schedule: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create retraining schedule: {str(e)}"
        )


@app.get("/api/retraining/schedules", tags=["Model Retraining"])
async def list_retraining_schedules(
    token: str = Depends(oauth2_scheme)
):
    """
    List retraining schedules.
    
    Args:
        token: Authentication token
    
    Returns:
        List of retraining schedules
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import RetrainingScheduleRepository
        
        with get_db_session() as session:
            schedule_repo = RetrainingScheduleRepository(session)
            schedules = schedule_repo.get_active_schedules()
            
            schedules_data = []
            for schedule in schedules:
                schedules_data.append({
                    "schedule_id": schedule.schedule_id,
                    "schedule_name": schedule.schedule_name,
                    "model_name": schedule.model_name,
                    "schedule_type": schedule.schedule_type,
                    "is_active": schedule.is_active,
                    "last_run_at": schedule.last_run_at.isoformat() if schedule.last_run_at else None,
                    "next_run_at": schedule.next_run_at.isoformat() if schedule.next_run_at else None,
                    "created_at": schedule.created_at.isoformat() if schedule.created_at else None
                })
            
            return JSONResponse({
                "schedules": schedules_data,
                "total": len(schedules_data)
            })
            
    except Exception as e:
        logger.error(f"Error listing retraining schedules: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list retraining schedules: {str(e)}"
        )


@app.post("/api/retraining/trigger/drift", tags=["Model Retraining"])
async def trigger_retraining_on_drift(
    model_name: str = Body(...),
    drift_metadata: Optional[Dict[str, Any]] = Body(default=None),
    token: str = Depends(oauth2_scheme)
):
    """
    Trigger retraining based on drift detection.
    
    Args:
        model_name: Name of the model
        drift_metadata: Drift detection metadata
        token: Authentication token
    
    Returns:
        Created retraining job
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "model:write" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to trigger retraining"
        )
    
    try:
        from src.pipelines.retraining import RetrainingScheduler
        
        scheduler = RetrainingScheduler()
        job_id = scheduler.trigger_on_drift(model_name, drift_metadata or {})
        
        if not job_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create drift-triggered retraining job"
            )
        
        return JSONResponse({
            "job_id": job_id,
            "trigger_type": "drift",
            "model_name": model_name,
            "message": "Drift-triggered retraining job created successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering drift retraining: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger retraining: {str(e)}"
        )


# ============================================================================
# Batch Prediction Endpoints
# ============================================================================

@app.post("/api/batch-predictions/jobs", tags=["Batch Predictions"])
async def create_batch_prediction_job(
    job_name: str = Body(...),
    input_source: str = Body(...),
    input_config: Dict[str, Any] = Body(...),
    output_format: str = Body(...),
    output_config: Dict[str, Any] = Body(...),
    model_name: str = Body(...),
    batch_size: int = Body(default=1000),
    max_workers: int = Body(default=4),
    use_feature_store: bool = Body(default=True),
    model_version: Optional[str] = Body(default=None),
    model_stage: str = Body(default="Production"),
    token: str = Depends(oauth2_scheme)
):
    """
    Create a new batch prediction job.
    
    Args:
        job_name: Name for the batch prediction job
        input_source: Input source type ('database', 'file', 'api')
        input_config: Input source configuration
        output_format: Output format ('database', 'csv', 'parquet')
        output_config: Output configuration
        model_name: Name of the model to use
        batch_size: Number of records to process per batch
        max_workers: Maximum number of worker threads
        use_feature_store: Whether to use feature store for features
        model_version: Model version (optional)
        model_stage: Model stage ('Production', 'Staging')
        token: Authentication token
    
    Returns:
        Created batch prediction job
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "prediction:write" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to create batch prediction jobs"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import BatchPredictionJobRepository
        
        with get_db_session() as session:
            job_repo = BatchPredictionJobRepository(session)
            
            job = job_repo.create_job(
                job_name=job_name,
                input_source=input_source,
                input_config=input_config,
                output_format=output_format,
                output_config=output_config,
                model_name=model_name,
                batch_size=batch_size,
                max_workers=max_workers,
                use_feature_store=use_feature_store,
                model_version=model_version,
                model_stage=model_stage,
                created_by=session_data.get("username")
            )
            
            session.commit()
            
            return JSONResponse({
                "job_id": job.job_id,
                "job_name": job.job_name,
                "status": job.status,
                "message": "Batch prediction job created successfully"
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating batch prediction job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch prediction job: {str(e)}"
        )


@app.post("/api/batch-predictions/jobs/{job_id}/run", tags=["Batch Predictions"])
async def run_batch_prediction_job(
    job_id: int,
    token: str = Depends(oauth2_scheme)
):
    """
    Execute a batch prediction job.
    
    Args:
        job_id: Batch prediction job ID
        token: Authentication token
    
    Returns:
        Job execution result
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission
    user_permissions = session_data.get("permissions", [])
    if "prediction:write" not in user_permissions and not session_data.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to run batch prediction jobs"
        )
    
    try:
        from src.pipelines.batch_prediction import get_batch_prediction_processor
        
        processor = get_batch_prediction_processor()
        result = processor.process_batch_job(job_id)
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running batch prediction job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run batch prediction job: {str(e)}"
        )


@app.get("/api/batch-predictions/jobs", tags=["Batch Predictions"])
async def list_batch_prediction_jobs(
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, description="Maximum number of jobs to return"),
    token: str = Depends(oauth2_scheme)
):
    """
    List batch prediction jobs.
    
    Args:
        status_filter: Filter by status ('pending', 'running', 'completed', 'failed', 'cancelled', 'paused')
        limit: Maximum number of jobs to return
        token: Authentication token
    
    Returns:
        List of batch prediction jobs
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import BatchPredictionJobRepository
        
        with get_db_session() as session:
            job_repo = BatchPredictionJobRepository(session)
            
            if status_filter:
                jobs = job_repo.get_by_status(status_filter)
            else:
                jobs = job_repo.get_recent_jobs(limit=limit)
            
            jobs_data = []
            for job in jobs:
                jobs_data.append({
                    "job_id": job.job_id,
                    "job_name": job.job_name,
                    "input_source": job.input_source,
                    "output_format": job.output_format,
                    "status": job.status,
                    "total_records": job.total_records,
                    "processed_records": job.processed_records,
                    "failed_records": job.failed_records,
                    "progress_percentage": float(job.progress_percentage) if job.progress_percentage else 0.0,
                    "output_path": job.output_path,
                    "records_per_second": float(job.records_per_second) if job.records_per_second else None,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "error_message": job.error_message
                })
            
            return JSONResponse({
                "jobs": jobs_data,
                "total": len(jobs_data)
            })
            
    except Exception as e:
        logger.error(f"Error listing batch prediction jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list batch prediction jobs: {str(e)}"
        )


@app.get("/api/batch-predictions/jobs/{job_id}", tags=["Batch Predictions"])
async def get_batch_prediction_job(
    job_id: int,
    token: str = Depends(oauth2_scheme)
):
    """
    Get batch prediction job details.
    
    Args:
        job_id: Batch prediction job ID
        token: Authentication token
    
    Returns:
        Batch prediction job details
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    try:
        from src.database.connection import get_db_session
        from src.database.repositories import BatchPredictionJobRepository
        
        with get_db_session() as session:
            job_repo = BatchPredictionJobRepository(session)
            job = job_repo.get_by_id(job_id)
            
            if not job:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Batch prediction job {job_id} not found"
                )
            
            return JSONResponse({
                "job_id": job.job_id,
                "job_name": job.job_name,
                "input_source": job.input_source,
                "input_config": job.input_config,
                "output_format": job.output_format,
                "output_config": job.output_config,
                "model_name": job.model_name,
                "model_version": job.model_version,
                "model_stage": job.model_stage,
                "status": job.status,
                "total_records": job.total_records,
                "processed_records": job.processed_records,
                "failed_records": job.failed_records,
                "progress_percentage": float(job.progress_percentage) if job.progress_percentage else 0.0,
                "output_path": job.output_path,
                "output_file_size_bytes": job.output_file_size_bytes,
                "records_per_second": float(job.records_per_second) if job.records_per_second else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "error_message": job.error_message
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch prediction job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch prediction job: {str(e)}"
        )


# ============================================================================
# Load Testing & Performance Benchmarking Endpoints
# ============================================================================

@app.post("/api/testing/load-test", tags=["Testing", "Load Testing"])
async def run_load_test(
    scenarios: List[Dict[str, Any]] = Body(...),
    total_requests: int = Body(100),
    concurrent_users: int = Body(10),
    duration_seconds: Optional[int] = Body(None),
    token: str = Depends(oauth2_scheme)
):
    """
    Run a load test.
    
    Args:
        scenarios: List of test scenarios (endpoint, method, payload, weight)
        total_requests: Total number of requests to make
        concurrent_users: Number of concurrent users
        duration_seconds: Optional duration limit in seconds
        token: Authentication token
    
    Returns:
        Load test results
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission - allow users with monitoring or model performance permissions
    user_permissions = session_data.get("permissions", [])
    has_permission = (
        "model:performance" in user_permissions or
        "dashboard:monitoring" in user_permissions or
        "model:write" in user_permissions or
        session_data.get("is_superuser", False)
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to run load tests. Required: model:performance, dashboard:monitoring, or model:write"
        )
    
    try:
        from src.testing.load_testing import LoadTester, LoadTestScenario
        
        tester = LoadTester()
        
        # Authenticate
        username = session_data.get("username", "test_user")
        password = "test_password"  # In production, would get from request or config
        await tester.authenticate(username, password)
        
        # Convert scenarios
        test_scenarios = []
        for scenario_data in scenarios:
            scenario = LoadTestScenario(
                name=scenario_data.get("name", "unnamed"),
                endpoint=scenario_data.get("endpoint"),
                method=scenario_data.get("method", "POST"),
                payload=scenario_data.get("payload", {}),
                headers=scenario_data.get("headers", {}),
                weight=scenario_data.get("weight", 1)
            )
            test_scenarios.append(scenario)
        
        # Run load test
        results = await tester.run_load_test(
            scenarios=test_scenarios,
            total_requests=total_requests,
            concurrent_users=concurrent_users,
            duration_seconds=duration_seconds
        )
        
        return JSONResponse(results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running load test: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run load test: {str(e)}"
        )


@app.post("/api/testing/capacity-planning", tags=["Testing", "Capacity Planning"])
async def estimate_capacity(
    target_rps: float = Body(...),
    avg_latency_ms: float = Body(...),
    target_p95_ms: float = Body(default=200.0),
    token: str = Depends(oauth2_scheme)
):
    """
    Estimate required capacity for target load.
    
    Args:
        target_rps: Target requests per second
        avg_latency_ms: Average latency in milliseconds
        target_p95_ms: Target P95 latency in milliseconds
        token: Authentication token
    
    Returns:
        Capacity estimates and recommendations
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission - allow users with monitoring or model performance permissions
    user_permissions = session_data.get("permissions", [])
    has_permission = (
        "model:performance" in user_permissions or
        "dashboard:monitoring" in user_permissions or
        "model:write" in user_permissions or
        session_data.get("is_superuser", False)
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access capacity planning. Required: model:performance, dashboard:monitoring, or model:write"
        )
    
    try:
        from src.testing.load_testing import get_capacity_planner
        
        planner = get_capacity_planner()
        estimates = planner.estimate_capacity(
            target_rps=target_rps,
            avg_latency_ms=avg_latency_ms,
            target_p95_ms=target_p95_ms
        )
        
        return JSONResponse(estimates)
        
    except Exception as e:
        logger.error(f"Error estimating capacity: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to estimate capacity: {str(e)}"
        )


@app.get("/api/testing/benchmark", tags=["Testing", "Benchmarking"])
async def get_benchmark_results(
    limit: int = Query(10, description="Number of recent benchmarks to return"),
    token: str = Depends(oauth2_scheme)
):
    """
    Get recent benchmark results.
    
    Args:
        limit: Number of results to return
        token: Authentication token
    
    Returns:
        Recent benchmark results
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    session_data = session_store.get(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # Check permission - allow users with monitoring or model performance permissions
    user_permissions = session_data.get("permissions", [])
    has_permission = (
        "model:performance" in user_permissions or
        "dashboard:monitoring" in user_permissions or
        "model:write" in user_permissions or
        session_data.get("is_superuser", False)
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access benchmarks. Required: model:performance, dashboard:monitoring, or model:write"
        )
    
    try:
        # Get performance metrics as benchmark data
        perf_monitor = get_performance_monitor() if settings.enable_performance_monitoring else None
        
        if not perf_monitor:
            return JSONResponse({
                "message": "Performance monitoring is disabled",
                "benchmarks": []
            })
        
        stats = perf_monitor.get_all_stats()
        sla_check = perf_monitor.check_sla(
            percentile=95,
            threshold_ms=settings.target_p95_latency_ms
        )
        
        return JSONResponse({
            "current_benchmark": {
                "statistics": stats,
                "sla_compliance": sla_check,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "message": "Use /api/performance for detailed metrics"
        })
        
    except Exception as e:
        logger.error(f"Error getting benchmark results: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get benchmark results: {str(e)}"
        )


# ============================================================================
# MULTI-MODEL SERVING ENDPOINTS
# ============================================================================

@app.post("/api/multi-model/routing-rules", tags=["Multi-Model Serving"])
async def create_routing_rule(
    rule_name: str = Body(...),
    priority: int = Body(0),
    routing_criteria: Dict[str, Any] = Body(...),
    routing_type: str = Body(...),
    target_models: List[Dict[str, Any]] = Body(...),
    model_weights: Optional[Dict[str, float]] = Body(None),
    fallback_model_name: Optional[str] = Body(None),
    fallback_model_stage: str = Body("Production"),
    description: Optional[str] = Body(None),
    token: str = Depends(oauth2_scheme)
):
    """Create a model routing rule."""
    try:
        session_data = verify_token(token)
        has_permission = (
            check_permission(session_data, "model:write") or
            session_data.get("is_superuser", False)
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to create routing rules"
            )
        
        with get_db_session() as session:
            from src.database.repositories import ModelRoutingRuleRepository
            
            rule_repo = ModelRoutingRuleRepository(session)
            
            # Check if rule name already exists
            existing = rule_repo.get_by_name(rule_name)
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Routing rule '{rule_name}' already exists"
                )
            
            rule = ModelRoutingRule(
                rule_name=rule_name,
                priority=priority,
                routing_criteria=routing_criteria,
                routing_type=routing_type,
                target_models=target_models,
                model_weights=model_weights,
                fallback_model_name=fallback_model_name,
                fallback_model_stage=fallback_model_stage,
                description=description,
                created_by=session_data.get("username", "unknown"),
                is_active=True
            )
            
            session.add(rule)
            session.commit()
            
            return JSONResponse({
                "message": "Routing rule created successfully",
                "rule_id": rule.rule_id,
                "rule_name": rule.rule_name
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating routing rule: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create routing rule: {str(e)}"
        )


@app.get("/api/multi-model/routing-rules", tags=["Multi-Model Serving"])
async def list_routing_rules(
    active_only: bool = Query(True, description="Only return active rules"),
    token: str = Depends(oauth2_scheme)
):
    """List all routing rules."""
    try:
        session_data = verify_token(token)
        has_permission = (
            check_permission(session_data, "model:read") or
            check_permission(session_data, "dashboard:monitoring") or
            session_data.get("is_superuser", False)
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to view routing rules"
            )
        
        with get_db_session() as session:
            from src.database.repositories import ModelRoutingRuleRepository
            
            rule_repo = ModelRoutingRuleRepository(session)
            
            if active_only:
                rules = rule_repo.get_active_rules()
            else:
                rules = rule_repo.get_all()
            
            return JSONResponse({
                "rules": [
                    {
                        "rule_id": r.rule_id,
                        "rule_name": r.rule_name,
                        "priority": r.priority,
                        "is_active": r.is_active,
                        "routing_type": r.routing_type,
                        "target_models": r.target_models,
                        "description": r.description,
                        "created_at": r.created_at.isoformat() if r.created_at else None
                    }
                    for r in rules
                ]
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing routing rules: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list routing rules: {str(e)}"
        )


@app.post("/api/multi-model/registry/register", tags=["Multi-Model Serving"])
async def register_model(
    model_name: str = Body(...),
    model_version: str = Body(...),
    model_stage: str = Body("Production"),
    model_type: Optional[str] = Body(None),
    mlflow_run_id: Optional[str] = Body(None),
    mlflow_model_uri: Optional[str] = Body(None),
    performance_metrics: Optional[Dict[str, float]] = Body(None),
    token: str = Depends(oauth2_scheme)
):
    """Register a model in the multi-model registry."""
    try:
        session_data = verify_token(token)
        has_permission = (
            check_permission(session_data, "model:write") or
            session_data.get("is_superuser", False)
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to register models"
            )
        
        with get_db_session() as session:
            from src.database.repositories import ModelRegistryRepository
            
            registry_repo = ModelRegistryRepository(session)
            
            registry = registry_repo.register_model(
                model_name=model_name,
                model_version=model_version,
                model_stage=model_stage,
                model_type=model_type,
                mlflow_run_id=mlflow_run_id,
                mlflow_model_uri=mlflow_model_uri,
                performance_metrics=performance_metrics
            )
            
            session.commit()
            
            return JSONResponse({
                "message": "Model registered successfully",
                "registry_id": registry.registry_id,
                "model_name": registry.model_name,
                "model_version": registry.model_version
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register model: {str(e)}"
        )


@app.get("/api/multi-model/registry", tags=["Multi-Model Serving"])
async def list_registered_models(
    loaded_only: bool = Query(False, description="Only return loaded models"),
    token: str = Depends(oauth2_scheme)
):
    """List all registered models."""
    try:
        session_data = verify_token(token)
        has_permission = (
            check_permission(session_data, "model:read") or
            check_permission(session_data, "dashboard:monitoring") or
            session_data.get("is_superuser", False)
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to view model registry"
            )
        
        with get_db_session() as session:
            from src.database.repositories import ModelRegistryRepository
            
            registry_repo = ModelRegistryRepository(session)
            
            if loaded_only:
                models = registry_repo.get_loaded_models()
            else:
                models = registry_repo.get_all()
            
            return JSONResponse({
                "models": [
                    {
                        "registry_id": m.registry_id,
                        "model_name": m.model_name,
                        "model_version": m.model_version,
                        "model_stage": m.model_stage,
                        "model_type": m.model_type,
                        "is_loaded": m.is_loaded,
                        "status": m.status,
                        "accuracy": float(m.accuracy) if m.accuracy else None,
                        "roc_auc": float(m.roc_auc) if m.roc_auc else None,
                        "registered_at": m.registered_at.isoformat() if m.registered_at else None
                    }
                    for m in models
                ]
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing registered models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@app.post("/api/multi-model/compare", tags=["Multi-Model Serving"])
async def compare_models(
    model_1_name: str = Body(...),
    model_1_version: str = Body(...),
    model_2_name: str = Body(...),
    model_2_version: str = Body(...),
    test_data: List[Dict[str, Any]] = Body(...),
    comparison_type: str = Body("real_time"),
    token: str = Depends(oauth2_scheme)
):
    """Compare two models on test data."""
    try:
        session_data = verify_token(token)
        has_permission = (
            check_permission(session_data, "model:read") or
            check_permission(session_data, "model:write") or
            session_data.get("is_superuser", False)
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to compare models"
            )
        
        from src.serving.multi_model import ModelComparator
        import numpy as np
        
        comparator = ModelComparator()
        
        result = comparator.compare_models(
            model_1_name=model_1_name,
            model_1_version=model_1_version,
            model_2_name=model_2_name,
            model_2_version=model_2_version,
            test_data=test_data,
            comparison_type=comparison_type
        )
        
        return JSONResponse({
            "message": "Model comparison completed",
            "comparison": result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare models: {str(e)}"
        )


@app.post("/api/multi-model/ensembles", tags=["Multi-Model Serving"])
async def create_ensemble(
    ensemble_name: str = Body(...),
    ensemble_type: str = Body(...),
    model_configs: List[Dict[str, Any]] = Body(...),
    description: Optional[str] = Body(None),
    token: str = Depends(oauth2_scheme)
):
    """Create a model ensemble configuration."""
    try:
        session_data = verify_token(token)
        has_permission = (
            check_permission(session_data, "model:write") or
            session_data.get("is_superuser", False)
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to create ensembles"
            )
        
        with get_db_session() as session:
            from src.database.repositories import ModelEnsembleRepository
            
            ensemble_repo = ModelEnsembleRepository(session)
            
            # Check if ensemble name already exists
            existing = ensemble_repo.get_by_name(ensemble_name)
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Ensemble '{ensemble_name}' already exists"
                )
            
            ensemble = ModelEnsemble(
                ensemble_name=ensemble_name,
                ensemble_type=ensemble_type,
                model_configs=model_configs,
                description=description,
                created_by=session_data.get("username", "unknown"),
                is_active=True
            )
            
            session.add(ensemble)
            session.commit()
            
            return JSONResponse({
                "message": "Ensemble created successfully",
                "ensemble_id": ensemble.ensemble_id,
                "ensemble_name": ensemble.ensemble_name
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating ensemble: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create ensemble: {str(e)}"
        )


@app.get("/api/multi-model/ensembles", tags=["Multi-Model Serving"])
async def list_ensembles(
    active_only: bool = Query(True, description="Only return active ensembles"),
    token: str = Depends(oauth2_scheme)
):
    """List all model ensembles."""
    try:
        session_data = verify_token(token)
        has_permission = (
            check_permission(session_data, "model:read") or
            check_permission(session_data, "dashboard:monitoring") or
            session_data.get("is_superuser", False)
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to view ensembles"
            )
        
        with get_db_session() as session:
            from src.database.repositories import ModelEnsembleRepository
            
            ensemble_repo = ModelEnsembleRepository(session)
            
            if active_only:
                ensembles = ensemble_repo.get_active_ensembles()
            else:
                ensembles = ensemble_repo.get_all()
            
            return JSONResponse({
                "ensembles": [
                    {
                        "ensemble_id": e.ensemble_id,
                        "ensemble_name": e.ensemble_name,
                        "ensemble_type": e.ensemble_type,
                        "model_configs": e.model_configs,
                        "is_active": e.is_active,
                        "description": e.description,
                        "created_at": e.created_at.isoformat() if e.created_at else None
                    }
                    for e in ensembles
                ]
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing ensembles: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list ensembles: {str(e)}"
        )


@app.post("/api/multi-model/predict", tags=["Multi-Model Serving", "Prediction"])
async def predict_with_routing(
    features: List[float] = Body(...),
    customer_id: Optional[str] = Body(None),
    customer_segment: Optional[str] = Body(None),
    amount: Optional[float] = Body(None),
    token: str = Depends(oauth2_scheme)
):
    """Make prediction using multi-model routing."""
    try:
        session_data = verify_token(token)
        has_permission = (
            check_permission(session_data, "model:predict") or
            check_permission(session_data, "model:read") or
            session_data.get("is_superuser", False)
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to make predictions"
            )
        
        from src.serving.multi_model import get_multi_model_manager
        import numpy as np
        
        global model  # Default model
        
        multi_model_manager = get_multi_model_manager()
        
        # Prepare request data for routing
        request_data = {
            "features": features,
            "customer_id": customer_id,
            "customer_segment": customer_segment,
            "amount": amount
        }
        
        # Make prediction with routing
        features_array = np.array([features])
        result = multi_model_manager.predict_with_routing(
            features=features_array,
            request_data=request_data,
            default_model=model
        )
        
        return JSONResponse({
            "prediction": result["prediction"],
            "probability": result["probability"],
            "routing_metadata": result.get("routing_metadata", {}),
            "ensemble_details": result.get("individual_predictions") if "individual_predictions" in result else None
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making multi-model prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to make prediction: {str(e)}"
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
