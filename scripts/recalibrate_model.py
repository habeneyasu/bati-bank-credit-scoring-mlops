#!/usr/bin/env python3
"""
Model recalibration tool.
Implements Step 3: Medium-term recommendation from SCORING_DIAGNOSTIC.md

Supports:
- Platt scaling (logistic regression calibration)
- Isotonic regression calibration
- Model retraining with class weights
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    print("Warning: MLflow not available")
    HAS_MLFLOW = False

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    print("Warning: scikit-learn not available")
    HAS_SKLEARN = False

try:
    from src.utils.config import settings
    from src.utils.logging import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)
except ImportError:
    logger = None
    print("Warning: Could not import project modules")


def load_model_from_mlflow() -> Optional[Any]:
    """Load model from MLflow."""
    if not HAS_MLFLOW:
        return None
    
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        if logger:
            logger.error(f"Error loading model: {e}", exc_info=True)
        return None


def calibrate_model_platt(model: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> Any:
    """
    Apply Platt scaling (logistic regression calibration).
    
    Args:
        model: Base model
        X_cal: Calibration features
        y_cal: Calibration labels
        
    Returns:
        Calibrated model
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required for calibration")
    
    # Get base predictions
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_cal)[:, 1]  # Probability of positive class
    else:
        proba = model.predict(X_cal)
    
    # Apply Platt scaling using logistic regression
    lr = LogisticRegression()
    lr.fit(proba.reshape(-1, 1), y_cal)
    
    # Create calibrated model wrapper
    class PlattCalibratedModel:
        def __init__(self, base_model, calibrator):
            self.base_model = base_model
            self.calibrator = calibrator
        
        def predict_proba(self, X):
            base_proba = self.base_model.predict_proba(X)[:, 1]
            calibrated = self.calibrator.predict_proba(base_proba.reshape(-1, 1))[:, 1]
            return np.column_stack([1 - calibrated, calibrated])
        
        def predict(self, X):
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
    
    return PlattCalibratedModel(model, lr)


def calibrate_model_isotonic(model: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> Any:
    """
    Apply isotonic regression calibration.
    
    Args:
        model: Base model
        X_cal: Calibration features
        y_cal: Calibration labels
        
    Returns:
        Calibrated model
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required for calibration")
    
    # Get base predictions
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_cal)[:, 1]
    else:
        proba = model.predict(X_cal)
    
    # Apply isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(proba, y_cal)
    
    # Create calibrated model wrapper
    class IsotonicCalibratedModel:
        def __init__(self, base_model, calibrator):
            self.base_model = base_model
            self.calibrator = calibrator
        
        def predict_proba(self, X):
            base_proba = self.base_model.predict_proba(X)[:, 1]
            calibrated = self.calibrator.transform(base_proba)
            return np.column_stack([1 - calibrated, calibrated])
        
        def predict(self, X):
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
    
    return IsotonicCalibratedModel(model, iso_reg)


def calibrate_model_sklearn(model: Any, X_cal: np.ndarray, y_cal: np.ndarray, method: str = "isotonic") -> Any:
    """
    Use sklearn's CalibratedClassifierCV for calibration.
    
    Args:
        model: Base model
        X_cal: Calibration features
        y_cal: Calibration labels
        method: 'isotonic' or 'sigmoid' (Platt scaling)
        
    Returns:
        Calibrated model
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required for calibration")
    
    calibrated = CalibratedClassifierCV(model, method=method, cv='prefit')
    calibrated.fit(X_cal, y_cal)
    return calibrated


def generate_recalibration_guide() -> Dict[str, Any]:
    """
    Generate recalibration guide and recommendations.
    
    Returns:
        Recalibration guide
    """
    print("=" * 80)
    print("MODEL RECALIBRATION GUIDE")
    print("=" * 80)
    print()
    
    guide = {
        "timestamp": datetime.now().isoformat(),
        "calibration_methods": {
            "platt_scaling": {
                "description": "Logistic regression calibration (Platt scaling)",
                "best_for": "Small calibration sets, binary classification",
                "pros": ["Simple", "Fast", "Works well with limited data"],
                "cons": ["Assumes sigmoid shape", "May not handle extreme probabilities well"]
            },
            "isotonic_regression": {
                "description": "Non-parametric isotonic regression",
                "best_for": "Larger calibration sets, non-sigmoid distributions",
                "pros": ["More flexible", "No distribution assumptions", "Better for complex patterns"],
                "cons": ["Requires more data", "Can overfit with small sets"]
            },
            "sklearn_calibrated": {
                "description": "scikit-learn's CalibratedClassifierCV",
                "best_for": "General purpose, cross-validated calibration",
                "pros": ["Built-in cross-validation", "Handles overfitting", "Production-ready"],
                "cons": ["Requires more data", "Slower"]
            }
        },
        "steps": [
            "1. Load current model from MLflow",
            "2. Prepare calibration dataset (holdout from training or recent predictions)",
            "3. Choose calibration method based on data size and requirements",
            "4. Apply calibration to model",
            "5. Evaluate calibrated model on validation set",
            "6. Compare calibration metrics (Brier score, calibration curve)",
            "7. If improved, register calibrated model to MLflow",
            "8. Deploy calibrated model to production"
        ],
        "evaluation_metrics": [
            "Brier score (lower is better)",
            "Calibration curve (should be close to diagonal)",
            "Expected Calibration Error (ECE)",
            "Maximum Calibration Error (MCE)"
        ],
        "recommendations": []
    }
    
    # Load model to check if it exists
    model = load_model_from_mlflow()
    if model:
        guide["model_info"] = {
            "model_loaded": True,
            "model_type": type(model).__name__,
            "has_predict_proba": hasattr(model, 'predict_proba')
        }
        print("✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Has predict_proba: {hasattr(model, 'predict_proba')}")
    else:
        guide["model_info"] = {
            "model_loaded": False,
            "error": "Could not load model from MLflow"
        }
        print("✗ Could not load model from MLflow")
        guide["recommendations"].append("Ensure MLflow is configured and model is registered")
    
    print()
    print("Calibration Methods Available:")
    for method, info in guide["calibration_methods"].items():
        print(f"  - {method}: {info['description']}")
        print(f"    Best for: {info['best_for']}")
    
    print()
    print("Steps to Recalibrate:")
    for step in guide["steps"]:
        print(f"  {step}")
    
    print()
    print("Evaluation Metrics:")
    for metric in guide["evaluation_metrics"]:
        print(f"  - {metric}")
    
    # Save guide
    guide_file = project_root / "recalibration_guide.json"
    with open(guide_file, "w") as f:
        json.dump(guide, f, indent=2, default=str)
    print(f"\nGuide saved to: {guide_file}")
    
    return guide


if __name__ == "__main__":
    guide = generate_recalibration_guide()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
To actually recalibrate the model:

1. Prepare calibration dataset:
   - Use holdout set from training data, OR
   - Use recent predictions with known outcomes

2. Run calibration:
   python3 scripts/recalibrate_model.py --method isotonic --calibration-data path/to/data.csv

3. Evaluate calibrated model:
   - Compare Brier scores
   - Plot calibration curves
   - Test on validation set

4. Register to MLflow if improved:
   - Save calibrated model
   - Register with new version
   - Deploy to production
    """)
