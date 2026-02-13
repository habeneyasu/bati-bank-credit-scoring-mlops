"""
Pydantic models for API request and response validation.

These models ensure data integrity and provide clear API documentation.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any


class PredictionRequest(BaseModel):
    """
    Request model for credit risk prediction.
    
    Contains all features required by the trained model.
    Features should match the processed feature set (26 features after engineering).
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CUST-12345",
                "features": [0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 
                            0.849, -0.994, -0.006, 0.853, 0.170, -0.068, -0.312, 
                            -0.167, 0.164, -0.193, -0.025, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0],
                "include_explanation": False
            }
        }
    )
    
    # Customer identification (required for audit trail and compliance)
    customer_id: Optional[str] = Field(
        default=None,
        description="Unique customer identifier for tracking and audit purposes. "
                    "Recommended for production use to enable prediction tracking, "
                    "customer-level analytics, and regulatory compliance.",
        examples=["CUST-12345", "customer_abc123", "user_789"]
    )
    
    # Feature values as a list (for flexibility)
    features: List[float] = Field(
        ...,
        description="List of feature values matching the model's expected input",
        min_length=26,
        max_length=26,
        examples=[[0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994, 
                  -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    
    include_explanation: bool = Field(
        default=False,
        description="Whether to include SHAP explanation in the response"
    )


class PredictionResponse(BaseModel):
    """
    Response model for credit risk prediction.
    
    Contains the prediction results including risk probability and classification.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CUST-12345",
                "prediction": 0,
                "probability": 0.15,
                "risk_level": "low",
                "prediction_id": "pred_abc123xyz",
                "timestamp": "2026-02-12T18:30:00Z"
            }
        }
    )
    
    # Echo back customer_id for confirmation
    customer_id: Optional[str] = Field(
        default=None,
        description="Customer identifier (echoed from request for confirmation)"
    )
    
    prediction: int = Field(
        ...,
        description="Binary prediction: 0 (low-risk) or 1 (high-risk)",
        ge=0,
        le=1
    )
    
    probability: float = Field(
        ...,
        description="Probability of high-risk (is_high_risk=1), range [0, 1]",
        ge=0.0,
        le=1.0
    )
    
    risk_level: str = Field(
        ...,
        description="Human-readable risk level: 'low', 'medium', or 'high'"
    )
    
    prediction_id: Optional[str] = Field(
        default=None,
        description="Unique prediction identifier for tracking and audit purposes"
    )
    
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp of when the prediction was made"
    )
    
    explanation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="SHAP explanation of the prediction (if requested)"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "credit_scoring_model",
                "model_version": "2"
            }
        }
    )
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")
    model_version: Optional[str] = Field(None, description="Version of loaded model")


class FeatureImportance(BaseModel):
    """Model for feature importance in explanations."""
    
    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP value for this feature")
    feature_value: float = Field(..., description="Actual feature value")


class ExplanationResponse(BaseModel):
    """
    Response model for model explanation endpoint.
    
    Provides SHAP-based explanations for model predictions to meet
    regulatory transparency requirements (CFPB, EU AI Act).
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 0,
                "probability": 0.15,
                "base_value": 0.25,
                "explanation_summary": "Prediction: Low Risk (Probability: 15.00%). Key factors: ...",
                "feature_importance": [
                    {
                        "feature": "feature_1",
                        "shap_value": -0.05,
                        "feature_value": 0.5
                    }
                ]
            }
        }
    )
    
    prediction: int = Field(
        ...,
        description="Binary prediction: 0 (low-risk) or 1 (high-risk)",
        ge=0,
        le=1
    )
    
    probability: float = Field(
        ...,
        description="Probability of high-risk (is_high_risk=1), range [0, 1]",
        ge=0.0,
        le=1.0
    )
    
    base_value: float = Field(
        ...,
        description="Base/expected value from SHAP explainer"
    )
    
    explanation_summary: str = Field(
        ...,
        description="Human-readable explanation summary"
    )
    
    feature_importance: List[FeatureImportance] = Field(
        ...,
        description="List of features sorted by importance (absolute SHAP value)"
    )
    
    shap_values: List[float] = Field(
        ...,
        description="SHAP values for all features in original order"
    )
    
    feature_names: List[str] = Field(
        ...,
        description="Feature names in original order"
    )
    
    waterfall_plot: Optional[str] = Field(
        default=None,
        description="Base64-encoded waterfall plot image (if requested)"
    )