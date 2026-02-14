"""
Multi-Model Serving Module

Provides comprehensive multi-model serving capabilities:
- Multiple models simultaneously
- Model routing based on criteria
- Model ensemble serving
- Real-time model version comparison
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import defaultdict
import threading

from src.utils.logging import get_logger
from src.utils.config import settings
from src.database.connection import get_db_session
from src.database.models import (
    ModelRoutingRule, ModelRegistry, ModelComparisonResult, ModelEnsemble
)
from src.api.main import load_model_from_mlflow

logger = get_logger(__name__)


class ModelRouter:
    """
    Routes prediction requests to appropriate models based on routing rules.
    """
    
    def __init__(self):
        """Initialize model router."""
        self.logger = get_logger(f"{__name__}.ModelRouter")
        self._rules_cache = None
        self._cache_lock = threading.Lock()
    
    def get_model_for_request(
        self,
        request_data: Dict[str, Any],
        default_model: Any = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Get the appropriate model for a request based on routing rules.
        
        Args:
            request_data: Request data (features, customer_id, etc.)
            default_model: Default model to use if no rule matches
            
        Returns:
            Tuple of (model, routing_metadata)
        """
        try:
            with get_db_session() as session:
                # Get active routing rules (sorted by priority)
                rules = session.query(ModelRoutingRule).filter(
                    ModelRoutingRule.is_active == True
                ).order_by(ModelRoutingRule.priority.desc()).all()
                
                # Evaluate rules in priority order
                for rule in rules:
                    if self._evaluate_routing_criteria(rule.routing_criteria, request_data):
                        # Rule matches - get model(s) based on routing type
                        if rule.routing_type == "single":
                            model = self._get_single_model(rule.target_models, session)
                            if model:
                                return model, {
                                    "routing_rule": rule.rule_name,
                                    "routing_type": "single",
                                    "model_name": rule.target_models[0] if rule.target_models else None
                                }
                        elif rule.routing_type in ["ensemble", "weighted_ensemble"]:
                            models = self._get_ensemble_models(rule.target_models, rule.model_weights, session)
                            if models:
                                return models, {
                                    "routing_rule": rule.rule_name,
                                    "routing_type": rule.routing_type,
                                    "models": rule.target_models,
                                    "weights": rule.model_weights
                                }
                
                # No rule matched - use fallback or default
                if default_model:
                    return default_model, {"routing_type": "default", "model": "default"}
                
                return None, {"routing_type": "none", "error": "No model available"}
                
        except Exception as e:
            self.logger.error(f"Error routing model: {e}", exc_info=True)
            return default_model, {"routing_type": "error", "error": str(e)}
    
    def _evaluate_routing_criteria(
        self,
        criteria: Dict[str, Any],
        request_data: Dict[str, Any]
    ) -> bool:
        """Evaluate if request matches routing criteria."""
        try:
            # Check customer segment
            if "customer_segment" in criteria:
                if request_data.get("customer_segment") != criteria["customer_segment"]:
                    return False
            
            # Check amount range
            if "amount_range" in criteria:
                amount = request_data.get("amount", 0)
                min_amount = criteria["amount_range"].get("min", 0)
                max_amount = criteria["amount_range"].get("max", float("inf"))
                if not (min_amount <= amount <= max_amount):
                    return False
            
            # Check feature ranges
            if "feature_ranges" in criteria:
                features = request_data.get("features", [])
                for feature_idx, feature_range in criteria["feature_ranges"].items():
                    idx = int(feature_idx)
                    if idx < len(features):
                        value = features[idx]
                        min_val = feature_range.get("min", float("-inf"))
                        max_val = feature_range.get("max", float("inf"))
                        if not (min_val <= value <= max_val):
                            return False
            
            # Check customer ID patterns
            if "customer_id_pattern" in criteria:
                customer_id = request_data.get("customer_id", "")
                import re
                if not re.match(criteria["customer_id_pattern"], customer_id):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error evaluating routing criteria: {e}")
            return False
    
    def _get_single_model(
        self,
        target_models: List[Dict[str, Any]],
        session: Any
    ) -> Optional[Any]:
        """Get a single model from target models list."""
        if not target_models:
            return None
        
        model_config = target_models[0]
        model_name = model_config.get("model_name") or model_config.get("name")
        model_stage = model_config.get("model_stage") or model_config.get("stage", "Production")
        
        try:
            return load_model_from_mlflow(model_name, model_stage)
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def _get_ensemble_models(
        self,
        target_models: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]],
        session: Any
    ) -> Optional[List[Tuple[Any, float]]]:
        """Get models for ensemble."""
        models = []
        
        for model_config in target_models:
            model_name = model_config.get("model_name") or model_config.get("name")
            model_stage = model_config.get("model_stage") or model_config.get("stage", "Production")
            weight = weights.get(model_name, 1.0) if weights else 1.0
            
            try:
                model = load_model_from_mlflow(model_name, model_stage)
                if model:
                    models.append((model, weight))
            except Exception as e:
                self.logger.warning(f"Error loading model {model_name} for ensemble: {e}")
        
        return models if models else None


class ModelEnsemblePredictor:
    """
    Predicts using model ensembles (voting, weighted average, stacking).
    """
    
    def __init__(self):
        """Initialize ensemble predictor."""
        self.logger = get_logger(f"{__name__}.ModelEnsemblePredictor")
    
    def predict_ensemble(
        self,
        models: List[Tuple[Any, float]],
        features: np.ndarray,
        ensemble_type: str = "weighted_average"
    ) -> Dict[str, Any]:
        """
        Make prediction using ensemble of models.
        
        Args:
            models: List of (model, weight) tuples
            features: Feature array
            ensemble_type: Type of ensemble ('voting', 'weighted_average', 'stacking')
            
        Returns:
            Ensemble prediction results
        """
        if not models:
            raise ValueError("No models provided for ensemble")
        
        predictions = []
        probabilities = []
        weights = []
        
        for model, weight in models:
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)[0]
                    pred = int(np.argmax(proba))
                    prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    pred = int(model.predict(features)[0])
                    prob = float(pred)
                
                predictions.append(pred)
                probabilities.append(prob)
                weights.append(weight)
                
            except Exception as e:
                self.logger.warning(f"Error getting prediction from model: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from ensemble models")
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Combine predictions based on ensemble type
        if ensemble_type == "voting":
            # Majority vote
            final_prediction = max(set(predictions), key=predictions.count)
            final_probability = np.mean(probabilities)
            
        elif ensemble_type == "weighted_average":
            # Weighted average of probabilities
            final_probability = sum(p * w for p, w in zip(probabilities, weights))
            final_prediction = 1 if final_probability >= 0.5 else 0
            
        elif ensemble_type == "stacking":
            # Simple stacking: average probabilities
            final_probability = np.mean(probabilities)
            final_prediction = 1 if final_probability >= 0.5 else 0
            
        else:
            # Default to weighted average
            final_probability = sum(p * w for p, w in zip(probabilities, weights))
            final_prediction = 1 if final_probability >= 0.5 else 0
        
        return {
            "prediction": final_prediction,
            "probability": final_probability,
            "ensemble_type": ensemble_type,
            "individual_predictions": predictions,
            "individual_probabilities": probabilities,
            "weights": weights
        }


class ModelComparator:
    """
    Compares models in real-time or batch mode.
    """
    
    def __init__(self):
        """Initialize model comparator."""
        self.logger = get_logger(f"{__name__}.ModelComparator")
    
    def compare_models(
        self,
        model_1_name: str,
        model_1_version: str,
        model_2_name: str,
        model_2_version: str,
        test_data: List[Dict[str, Any]],
        comparison_type: str = "real_time"
    ) -> Dict[str, Any]:
        """
        Compare two models on test data.
        
        Args:
            model_1_name: First model name
            model_1_version: First model version
            model_2_name: Second model name
            model_2_version: Second model version
            test_data: Test data (list of feature vectors)
            comparison_type: Type of comparison ('real_time', 'batch', 'historical')
            
        Returns:
            Comparison results
        """
        try:
            # Load both models
            model_1 = load_model_from_mlflow(model_1_name, model_1_version)
            model_2 = load_model_from_mlflow(model_2_name, model_2_version)
            
            if not model_1 or not model_2:
                raise ValueError("Could not load one or both models")
            
            # Get predictions from both models
            predictions_1 = []
            predictions_2 = []
            probabilities_1 = []
            probabilities_2 = []
            
            for data in test_data:
                features = np.array([data.get("features", [])])
                
                # Model 1 predictions
                if hasattr(model_1, 'predict_proba'):
                    proba_1 = model_1.predict_proba(features)[0]
                    pred_1 = int(np.argmax(proba_1))
                    prob_1 = float(proba_1[1]) if len(proba_1) > 1 else float(proba_1[0])
                else:
                    pred_1 = int(model_1.predict(features)[0])
                    prob_1 = float(pred_1)
                
                # Model 2 predictions
                if hasattr(model_2, 'predict_proba'):
                    proba_2 = model_2.predict_proba(features)[0]
                    pred_2 = int(np.argmax(proba_2))
                    prob_2 = float(proba_2[1]) if len(proba_2) > 1 else float(proba_2[0])
                else:
                    pred_2 = int(model_2.predict(features)[0])
                    prob_2 = float(pred_2)
                
                predictions_1.append(pred_1)
                predictions_2.append(pred_2)
                probabilities_1.append(prob_1)
                probabilities_2.append(prob_2)
            
            # Calculate comparison metrics
            agreement = sum(1 for p1, p2 in zip(predictions_1, predictions_2) if p1 == p2) / len(predictions_1)
            avg_prob_diff = np.mean([abs(p1 - p2) for p1, p2 in zip(probabilities_1, probabilities_2)])
            
            # Determine winner (based on average probability - higher is better for risk assessment)
            avg_prob_1 = np.mean(probabilities_1)
            avg_prob_2 = np.mean(probabilities_2)
            
            winner = None
            if abs(avg_prob_1 - avg_prob_2) > 0.01:  # Significant difference
                winner = model_1_name if avg_prob_1 > avg_prob_2 else model_2_name
            
            comparison_metrics = {
                "model_1_avg_probability": float(avg_prob_1),
                "model_2_avg_probability": float(avg_prob_2),
                "agreement_rate": float(agreement),
                "average_probability_difference": float(avg_prob_diff),
                "model_1_predictions": {
                    "high_risk_count": sum(1 for p in predictions_1 if p == 1),
                    "low_risk_count": sum(1 for p in predictions_1 if p == 0)
                },
                "model_2_predictions": {
                    "high_risk_count": sum(1 for p in predictions_2 if p == 1),
                    "low_risk_count": sum(1 for p in predictions_2 if p == 0)
                }
            }
            
            differences = {
                "probability_difference": float(avg_prob_1 - avg_prob_2),
                "prediction_agreement": float(agreement),
                "disagreement_count": sum(1 for p1, p2 in zip(predictions_1, predictions_2) if p1 != p2)
            }
            
            # Save comparison result
            try:
                with get_db_session() as session:
                    comparison = ModelComparisonResult(
                        comparison_name=f"{model_1_name}_vs_{model_2_name}",
                        comparison_type=comparison_type,
                        model_1_name=model_1_name,
                        model_1_version=model_1_version,
                        model_2_name=model_2_name,
                        model_2_version=model_2_version,
                        comparison_metrics=comparison_metrics,
                        differences=differences,
                        winner=winner,
                        test_samples=len(test_data)
                    )
                    session.add(comparison)
                    session.commit()
            except Exception as e:
                self.logger.warning(f"Error saving comparison result: {e}")
            
            return {
                "model_1": {"name": model_1_name, "version": model_1_version},
                "model_2": {"name": model_2_name, "version": model_2_version},
                "comparison_metrics": comparison_metrics,
                "differences": differences,
                "winner": winner,
                "test_samples": len(test_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {e}", exc_info=True)
            raise


class MultiModelManager:
    """
    Manages multiple models for serving.
    """
    
    def __init__(self):
        """Initialize multi-model manager."""
        self.logger = get_logger(f"{__name__}.MultiModelManager")
        self.loaded_models: Dict[str, Any] = {}
        self.model_lock = threading.Lock()
        self.router = ModelRouter()
        self.ensemble_predictor = ModelEnsemblePredictor()
        self.comparator = ModelComparator()
    
    def load_model(
        self,
        model_name: str,
        model_version: str,
        model_stage: str = "Production"
    ) -> bool:
        """
        Load a model into memory.
        
        Args:
            model_name: Name of the model
            model_version: Model version
            model_stage: Model stage
            
        Returns:
            True if loaded successfully
        """
        try:
            model_key = f"{model_name}:{model_version}:{model_stage}"
            
            with self.model_lock:
                if model_key in self.loaded_models:
                    self.logger.info(f"Model {model_key} already loaded")
                    return True
                
                model = load_model_from_mlflow(model_name, model_stage)
                if model:
                    self.loaded_models[model_key] = {
                        "model": model,
                        "model_name": model_name,
                        "model_version": model_version,
                        "model_stage": model_stage,
                        "loaded_at": datetime.now(timezone.utc),
                        "usage_count": 0
                    }
                    
                    # Update registry
                    self._update_registry_status(model_name, model_version, is_loaded=True)
                    
                    self.logger.info(f"Model {model_key} loaded successfully")
                    return True
                else:
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
            self._update_registry_status(model_name, model_version, is_loaded=False, error=str(e))
            return False
    
    def get_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        model_stage: str = "Production"
    ) -> Optional[Any]:
        """
        Get a loaded model.
        
        Args:
            model_name: Name of the model
            model_version: Model version (optional)
            model_stage: Model stage
            
        Returns:
            Model object or None
        """
        with self.model_lock:
            if model_version:
                model_key = f"{model_name}:{model_version}:{model_stage}"
                if model_key in self.loaded_models:
                    self.loaded_models[model_key]["usage_count"] += 1
                    return self.loaded_models[model_key]["model"]
            else:
                # Find latest version
                matching_models = [
                    (key, data) for key, data in self.loaded_models.items()
                    if data["model_name"] == model_name and data["model_stage"] == model_stage
                ]
                if matching_models:
                    # Sort by version and get latest
                    latest = max(matching_models, key=lambda x: x[1]["model_version"])
                    latest[1]["usage_count"] += 1
                    return latest[1]["model"]
            
            return None
    
    def predict_with_routing(
        self,
        features: np.ndarray,
        request_data: Dict[str, Any],
        default_model: Any = None
    ) -> Dict[str, Any]:
        """
        Make prediction using model routing.
        
        Args:
            features: Feature array
            request_data: Request data for routing
            default_model: Default model if no routing rule matches
            
        Returns:
            Prediction results with routing metadata
        """
        model_or_models, routing_metadata = self.router.get_model_for_request(
            request_data,
            default_model
        )
        
        if model_or_models is None:
            raise ValueError("No model available for prediction")
        
        # Check if it's an ensemble
        if isinstance(model_or_models, list):
            # Ensemble prediction
            ensemble_type = routing_metadata.get("routing_type", "weighted_ensemble")
            result = self.ensemble_predictor.predict_ensemble(
                model_or_models,
                features,
                ensemble_type
            )
            result["routing_metadata"] = routing_metadata
            return result
        else:
            # Single model prediction
            model = model_or_models
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                probability = float(probabilities[1])
                prediction = int(np.argmax(probabilities))
            else:
                prediction = int(model.predict(features)[0])
                probability = float(prediction)
            
            return {
                "prediction": prediction,
                "probability": probability,
                "routing_metadata": routing_metadata
            }
    
    def _update_registry_status(
        self,
        model_name: str,
        model_version: str,
        is_loaded: bool = True,
        error: Optional[str] = None
    ):
        """Update model registry status."""
        try:
            with get_db_session() as session:
                registry = session.query(ModelRegistry).filter(
                    ModelRegistry.model_name == model_name,
                    ModelRegistry.model_version == model_version
                ).first()
                
                if registry:
                    registry.is_loaded = is_loaded
                    registry.status = "available" if is_loaded else "error"
                    if error:
                        registry.error_message = error
                    registry.last_used_at = datetime.now(timezone.utc)
                    session.commit()
        except Exception as e:
            self.logger.warning(f"Error updating registry status: {e}")


def get_multi_model_manager() -> MultiModelManager:
    """Get a singleton MultiModelManager instance."""
    global _multi_model_manager
    if '_multi_model_manager' not in globals():
        _multi_model_manager = MultiModelManager()
    return _multi_model_manager
