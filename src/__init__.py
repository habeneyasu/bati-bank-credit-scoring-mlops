"""
Bati Bank Credit Scoring MLOps

A production-grade MLOps implementation for credit risk assessment using alternative data.
"""

__version__ = "1.0.0"

# Import main components for easy access
from src.features import (
    RFMCalculator,
    CustomerClustering,
    HighRiskLabeler,
    DataProcessor,
)
from src.models import (
    ModelTrainer,
    HyperparameterTuner,
    MLflowTracker,
)
from src.utils import settings, get_logger
from src.api import app

__all__ = [
    "__version__",
    # Features
    "RFMCalculator",
    "CustomerClustering",
    "HighRiskLabeler",
    "DataProcessor",
    # Models
    "ModelTrainer",
    "HyperparameterTuner",
    "MLflowTracker",
    # Utils
    "settings",
    "get_logger",
    # API
    "app",
]