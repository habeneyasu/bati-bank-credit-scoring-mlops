"""
Diagnostic script to investigate why all customers are getting similar low probabilities.

This script:
1. Checks feature diversity across customers
2. Analyzes prediction distribution
3. Tests model with different feature patterns
4. Identifies potential issues
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available, some features will be limited")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, using basic analysis")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import settings
from src.utils.logging import setup_logging, get_logger
from src.database.connection import get_db_session
from src.database.models import Prediction, CustomerFeature
from sqlalchemy import func, desc
import mlflow

# Setup logging
setup_logging()
logger = get_logger(__name__)


def check_feature_diversity(limit: int = 100) -> Dict[str, Any]:
    """
    Check if features are diverse across customers.
    
    Args:
        limit: Number of recent predictions to analyze
        
    Returns:
        Dictionary with feature diversity statistics
    """
    logger.info("Checking feature diversity...")
    
    try:
        with get_db_session() as session:
            # Get recent predictions with features
            predictions = session.query(Prediction).order_by(
                desc(Prediction.created_at)
            ).limit(limit).all()
            
            if not predictions:
                logger.warning("No predictions found in database")
                return {"error": "No predictions found"}
            
            # Extract features
            feature_vectors = []
            probabilities = []
            risk_levels = []
            customer_ids = []
            
            for pred in predictions:
                if pred.features and isinstance(pred.features, list):
                    feature_vectors.append(pred.features)
                    probabilities.append(float(pred.probability))
                    risk_levels.append(pred.risk_level)
                    customer_ids.append(pred.customer_id)
            
            if not feature_vectors:
                logger.warning("No features found in predictions")
                return {"error": "No features found in predictions"}
            
            # Calculate statistics
            if HAS_NUMPY:
                probabilities_array = np.array(probabilities)
                prob_mean = float(np.mean(probabilities_array))
                prob_std = float(np.std(probabilities_array))
                prob_min = float(np.min(probabilities_array))
                prob_max = float(np.max(probabilities_array))
                prob_median = float(np.median(probabilities_array))
                prob_q25 = float(np.percentile(probabilities_array, 25))
                prob_q75 = float(np.percentile(probabilities_array, 75))
            else:
                # Basic statistics without numpy
                prob_mean = sum(probabilities) / len(probabilities)
                prob_min = min(probabilities)
                prob_max = max(probabilities)
                sorted_probs = sorted(probabilities)
                prob_median = sorted_probs[len(sorted_probs) // 2]
                prob_q25 = sorted_probs[len(sorted_probs) // 4]
                prob_q75 = sorted_probs[3 * len(sorted_probs) // 4]
                prob_std = (sum((p - prob_mean) ** 2 for p in probabilities) / len(probabilities)) ** 0.5
            
            feature_stats = {
                "total_customers": len(feature_vectors),
                "feature_count": len(feature_vectors[0]) if feature_vectors else 0,
                "probability_stats": {
                    "mean": prob_mean,
                    "std": prob_std,
                    "min": prob_min,
                    "max": prob_max,
                    "median": prob_median,
                    "q25": prob_q25,
                    "q75": prob_q75
                },
                "risk_level_distribution": {
                    "low": risk_levels.count("low"),
                    "medium": risk_levels.count("medium"),
                    "high": risk_levels.count("high")
                },
                "feature_diversity": {}
            }
            
            # Check feature variance (low variance = similar features)
            if HAS_PANDAS:
                features_df = pd.DataFrame(feature_vectors)
                for col_idx in range(features_df.shape[1]):
                    col_data = features_df.iloc[:, col_idx]
                    feature_stats["feature_diversity"][f"feature_{col_idx}"] = {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "variance": float(col_data.var()),
                        "is_constant": col_data.nunique() == 1,
                        "unique_values": int(col_data.nunique())
                    }
            else:
                # Basic feature analysis without pandas
                num_features = len(feature_vectors[0]) if feature_vectors else 0
                for col_idx in range(num_features):
                    col_values = [fv[col_idx] for fv in feature_vectors if len(fv) > col_idx]
                    if col_values:
                        col_mean = sum(col_values) / len(col_values)
                        col_min = min(col_values)
                        col_max = max(col_values)
                        col_variance = sum((v - col_mean) ** 2 for v in col_values) / len(col_values)
                        col_std = col_variance ** 0.5
                        unique_vals = len(set(col_values))
                        
                        feature_stats["feature_diversity"][f"feature_{col_idx}"] = {
                            "mean": col_mean,
                            "std": col_std,
                            "min": col_min,
                            "max": col_max,
                            "variance": col_variance,
                            "is_constant": unique_vals == 1,
                            "unique_values": unique_vals
                        }
            
            # Identify constant features (same value for all customers)
            constant_features = [
                idx for idx, stats in feature_stats["feature_diversity"].items()
                if stats.get("is_constant", False)
            ]
            
            # Identify low-variance features
            low_variance_features = [
                idx for idx, stats in feature_stats["feature_diversity"].items()
                if stats.get("variance", 1.0) < 0.01 and not stats.get("is_constant", False)
            ]
            
            feature_stats["issues"] = {
                "constant_features": constant_features,
                "low_variance_features": low_variance_features,
                "total_constant": len(constant_features),
                "total_low_variance": len(low_variance_features)
            }
            
            logger.info(f"Analyzed {len(feature_vectors)} customers")
            logger.info(f"Probability range: {feature_stats['probability_stats']['min']:.4f} - {feature_stats['probability_stats']['max']:.4f}")
            logger.info(f"Probability std: {feature_stats['probability_stats']['std']:.4f}")
            logger.info(f"Constant features: {len(constant_features)}")
            logger.info(f"Low variance features: {len(low_variance_features)}")
            
            return feature_stats
            
    except Exception as e:
        logger.error(f"Error checking feature diversity: {e}", exc_info=True)
        return {"error": str(e)}


def analyze_prediction_distribution(limit: int = 1000) -> Dict[str, Any]:
    """
    Analyze the distribution of predictions.
    
    Args:
        limit: Number of predictions to analyze
        
    Returns:
        Dictionary with prediction distribution statistics
    """
    logger.info("Analyzing prediction distribution...")
    
    try:
        with get_db_session() as session:
            # Get predictions
            predictions = session.query(
                Prediction.probability,
                Prediction.risk_level,
                Prediction.customer_score,
                Prediction.created_at
            ).order_by(desc(Prediction.created_at)).limit(limit).all()
            
            if not predictions:
                return {"error": "No predictions found"}
            
            probabilities = [float(p.probability) for p in predictions]
            risk_levels = [p.risk_level for p in predictions]
            scores = [p.customer_score for p in predictions if p.customer_score]
            
            # Calculate distribution
            if HAS_NUMPY:
                prob_array = np.array(probabilities)
                prob_mean = float(np.mean(prob_array))
                prob_std = float(np.std(prob_array))
                prob_min = float(np.min(prob_array))
                prob_max = float(np.max(prob_array))
                prob_median = float(np.median(prob_array))
                # Histogram
                bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                hist_counts = [
                    int(np.sum((prob_array >= bins[i]) & (prob_array < bins[i+1])))
                    for i in range(len(bins) - 1)
                ]
            else:
                prob_mean = sum(probabilities) / len(probabilities)
                prob_min = min(probabilities)
                prob_max = max(probabilities)
                sorted_probs = sorted(probabilities)
                prob_median = sorted_probs[len(sorted_probs) // 2]
                prob_std = (sum((p - prob_mean) ** 2 for p in probabilities) / len(probabilities)) ** 0.5
                # Basic histogram
                bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                hist_counts = []
                for i in range(len(bins) - 1):
                    count = sum(1 for p in probabilities if bins[i] <= p < bins[i+1])
                    hist_counts.append(count)
            
            distribution = {
                "total_predictions": len(predictions),
                "probability_distribution": {
                    "mean": prob_mean,
                    "std": prob_std,
                    "min": prob_min,
                    "max": prob_max,
                    "median": prob_median,
                    "histogram": {
                        "bins": bins,
                        "counts": hist_counts
                    }
                },
                "risk_level_distribution": {
                    "low": risk_levels.count("low"),
                    "medium": risk_levels.count("medium"),
                    "high": risk_levels.count("high")
                },
                "score_distribution": {
                    "mean": (sum(scores) / len(scores)) if scores else None,
                    "std": ((sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5) if scores else None,
                    "min": int(min(scores)) if scores else None,
                    "max": int(max(scores)) if scores else None
                }
            }
            
            # Check if all predictions are in low risk range
            low_risk_count = sum(1 for p in probabilities if p < settings.risk_threshold_low)
            medium_risk_count = sum(1 for p in probabilities if settings.risk_threshold_low <= p <= settings.risk_threshold_high)
            high_risk_count = sum(1 for p in probabilities if p > settings.risk_threshold_high)
            
            distribution["risk_threshold_analysis"] = {
                "low_risk_range": f"< {settings.risk_threshold_low}",
                "medium_risk_range": f"{settings.risk_threshold_low} - {settings.risk_threshold_high}",
                "high_risk_range": f"> {settings.risk_threshold_high}",
                "low_risk_count": low_risk_count,
                "medium_risk_count": medium_risk_count,
                "high_risk_count": high_risk_count,
                "low_risk_percentage": (low_risk_count / len(probabilities)) * 100 if probabilities else 0,
                "all_low_risk": low_risk_count == len(probabilities)
            }
            
            logger.info(f"Analyzed {len(predictions)} predictions")
            logger.info(f"Low risk: {low_risk_count} ({distribution['risk_threshold_analysis']['low_risk_percentage']:.1f}%)")
            logger.info(f"Medium risk: {medium_risk_count}")
            logger.info(f"High risk: {high_risk_count}")
            
            return distribution
            
    except Exception as e:
        logger.error(f"Error analyzing prediction distribution: {e}", exc_info=True)
        return {"error": str(e)}


def test_model_with_patterns() -> Dict[str, Any]:
    """
    Test model with different feature patterns to see if it responds differently.
    
    Returns:
        Dictionary with test results
    """
    logger.info("Testing model with different feature patterns...")
    
    try:
        # Load model
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
        model = mlflow.sklearn.load_model(model_uri)
        
        logger.info(f"Loaded model: {model_uri}")
        logger.info(f"Model type: {type(model).__name__}")
        
        # Create test patterns
        num_features = settings.expected_features
        
        test_patterns = {
            "all_zeros": [0.0] * num_features,
            "all_ones": [1.0] * num_features,
            "low_values": [0.1] * num_features,
            "high_values": [0.9] * num_features,
            "mixed_low_risk": [0.2] * num_features,  # Simulate low-risk customer
            "mixed_high_risk": [0.8] * num_features,  # Simulate high-risk customer
        }
        
        # Add random patterns if numpy is available
        if HAS_NUMPY:
            import random
            random.seed(42)  # For reproducibility
            test_patterns["random"] = [random.random() for _ in range(num_features)]
            test_patterns["normalized_random"] = [(random.random() * 2 - 1) for _ in range(num_features)]
        
        results = {}
        
        for pattern_name, features in test_patterns.items():
            try:
                features_array = np.array(features, dtype=np.float64).reshape(1, -1)
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_array)[0]
                    probability = float(probabilities[1])  # High-risk class probability
                    if HAS_NUMPY:
                        prediction = int(np.argmax(probabilities))
                    else:
                        prediction = int(probabilities.index(max(probabilities)))
                else:
                    pred_result = model.predict(features_array)[0]
                    prediction = int(pred_result)
                    probability = float(prediction)
                
                # Determine risk level
                if probability < settings.risk_threshold_low:
                    risk_level = "low"
                elif probability > settings.risk_threshold_high:
                    risk_level = "high"
                else:
                    risk_level = "medium"
                
                customer_score = int((1 - probability) * 100)
                
                results[pattern_name] = {
                    "probability": probability,
                    "prediction": prediction,
                    "risk_level": risk_level,
                    "customer_score": customer_score,
                    "probabilities": probabilities.tolist() if hasattr(model, 'predict_proba') else None
                }
                
                logger.info(f"{pattern_name}: probability={probability:.4f}, risk={risk_level}, score={customer_score}")
                
            except Exception as e:
                logger.error(f"Error testing pattern {pattern_name}: {e}", exc_info=True)
                results[pattern_name] = {"error": str(e)}
        
        # Analyze results
        probabilities = [r["probability"] for r in results.values() if "probability" in r]
        if probabilities:
            prob_min = min(probabilities)
            prob_max = max(probabilities)
            prob_mean = sum(probabilities) / len(probabilities)
            prob_std = (sum((p - prob_mean) ** 2 for p in probabilities) / len(probabilities)) ** 0.5
            
            results["analysis"] = {
                "probability_range": {
                    "min": prob_min,
                    "max": prob_max,
                    "mean": prob_mean,
                    "std": prob_std
                },
                "model_responsive": prob_max - prob_min > 0.1,
                "all_similar": prob_max - prob_min < 0.05
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error testing model patterns: {e}", exc_info=True)
        return {"error": str(e)}


def check_model_metrics() -> Dict[str, Any]:
    """
    Check model training metrics from MLflow.
    
    Returns:
        Dictionary with model metrics
    """
    logger.info("Checking model metrics from MLflow...")
    
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        # Get model version info
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        model_name = settings.model_name
        model_stage = settings.model_stage
        
        # Get latest model version
        if model_stage in ["Production", "Staging", "Archived", "None"]:
            latest_versions = client.get_latest_versions(model_name, stages=[model_stage])
            if not latest_versions:
                return {"error": f"No model found in {model_stage} stage"}
            model_version = latest_versions[0].version
        else:
            model_version = model_stage
        
        # Get model version details
        mv = client.get_model_version(model_name, model_version)
        run_id = mv.run_id
        
        # Get run metrics
        run = client.get_run(run_id)
        
        metrics = {
            "model_name": model_name,
            "model_version": model_version,
            "model_stage": model_stage,
            "run_id": run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time
        }
        
        # Extract key metrics
        key_metrics = {}
        for metric_name in ["roc_auc", "roc-auc", "auc", "accuracy", "precision", "recall", "f1"]:
            if metric_name in metrics["metrics"]:
                key_metrics[metric_name] = metrics["metrics"][metric_name]
        
        metrics["key_metrics"] = key_metrics
        
        logger.info(f"Model version: {model_version}")
        logger.info(f"Key metrics: {key_metrics}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error checking model metrics: {e}", exc_info=True)
        return {"error": str(e)}


def run_full_diagnostic() -> Dict[str, Any]:
    """
    Run all diagnostic checks.
    
    Returns:
        Complete diagnostic report
    """
    logger.info("=" * 80)
    logger.info("RUNNING FULL SCORING DIAGNOSTIC")
    logger.info("=" * 80)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "risk_threshold_low": settings.risk_threshold_low,
            "risk_threshold_high": settings.risk_threshold_high,
            "expected_features": settings.expected_features,
            "model_name": settings.model_name,
            "model_stage": settings.model_stage
        },
        "feature_diversity": check_feature_diversity(limit=100),
        "prediction_distribution": analyze_prediction_distribution(limit=1000),
        "model_testing": test_model_with_patterns(),
        "model_metrics": check_model_metrics()
    }
    
    # Generate summary
    summary = {
        "issues_found": [],
        "recommendations": []
    }
    
    # Check feature diversity issues
    if "feature_diversity" in report and "issues" in report["feature_diversity"]:
        issues = report["feature_diversity"]["issues"]
        if issues.get("total_constant", 0) > 0:
            summary["issues_found"].append(
                f"{issues['total_constant']} constant features (same value for all customers)"
            )
        if issues.get("total_low_variance", 0) > 0:
            summary["issues_found"].append(
                f"{issues['total_low_variance']} low-variance features (very similar values)"
            )
    
    # Check prediction distribution issues
    if "prediction_distribution" in report and "risk_threshold_analysis" in report["prediction_distribution"]:
        analysis = report["prediction_distribution"]["risk_threshold_analysis"]
        if analysis.get("all_low_risk", False):
            summary["issues_found"].append(
                "All predictions are in LOW risk range - model may be biased or not calibrated"
            )
        if analysis.get("low_risk_percentage", 0) > 90:
            summary["issues_found"].append(
                f"{analysis['low_risk_percentage']:.1f}% of predictions are LOW risk - very imbalanced"
            )
    
    # Check model responsiveness
    if "model_testing" in report and "analysis" in report["model_testing"]:
        analysis = report["model_testing"]["analysis"]
        if analysis.get("all_similar", False):
            summary["issues_found"].append(
                "Model produces very similar probabilities for different input patterns - model may not be learning properly"
            )
        if not analysis.get("model_responsive", True):
            summary["issues_found"].append(
                "Model does not respond differently to different input patterns"
            )
    
    # Generate recommendations
    if summary["issues_found"]:
        summary["recommendations"].extend([
            "Review model training data - check for class imbalance",
            "Verify feature engineering pipeline - ensure features capture risk differences",
            "Consider model recalibration using Platt scaling or isotonic regression",
            "Test with known high-risk customer patterns",
            "Review model training metrics and validation performance"
        ])
    else:
        summary["recommendations"].append("No major issues detected - review individual metrics for details")
    
    report["summary"] = summary
    
    # Save report
    report_file = project_root / "diagnostic_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Report saved to: {report_file}")
    logger.info(f"Issues found: {len(summary['issues_found'])}")
    for issue in summary["issues_found"]:
        logger.warning(f"  - {issue}")
    
    return report


if __name__ == "__main__":
    report = run_full_diagnostic()
    
    # Print summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"\nIssues Found: {len(report['summary']['issues_found'])}")
    for issue in report["summary"]["issues_found"]:
        print(f"  ⚠️  {issue}")
    
    print(f"\nRecommendations: {len(report['summary']['recommendations'])}")
    for rec in report["summary"]["recommendations"]:
        print(f"  💡 {rec}")
    
    print(f"\nFull report saved to: diagnostic_report.json")
