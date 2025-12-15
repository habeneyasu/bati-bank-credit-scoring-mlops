"""
Pydantic models for API request and response validation.

These models ensure data integrity and provide clear API documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np


class PredictionRequest(BaseModel):
    """
    Request model for credit risk prediction.
    
    Contains all features required by the trained model.
    Features should match the processed feature set (26 features after engineering).
    """
    
    # Feature values as a list (for flexibility)
    features: List[float] = Field(
        ...,
        description="List of feature values matching the model's expected input",
        min_length=26,
        max_length=26,
        example=[0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994, 
                -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 
                            0.849, -0.994, -0.006, 0.853, 0.170, -0.068, -0.312, 
                            -0.167, 0.164, -0.193, -0.025, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0]
            }
        }


class PredictionResponse(BaseModel):
    """
    Response model for credit risk prediction.
    
    Contains the prediction results including risk probability and classification.
    """
    
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
        description="Human-readable risk level: 'low' or 'high'"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.15,
                "risk_level": "low"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")
    model_version: Optional[str] = Field(None, description="Version of loaded model")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "credit_scoring_model",
                "model_version": "2"
            }
        }
