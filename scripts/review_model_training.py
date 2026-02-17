#!/usr/bin/env python3
"""
Review model training data and metrics.
Implements Step 2: Short-term recommendation from SCORING_DIAGNOSTIC.md
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional

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
    from src.utils.config import settings
    from src.utils.logging import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)
except ImportError:
    logger = None
    print("Warning: Could not import project modules")


def get_model_training_metrics() -> Dict[str, Any]:
    """
    Retrieve model training metrics from MLflow.
    
    Returns:
        Dictionary with model metrics and training information
    """
    if not HAS_MLFLOW:
        return {"error": "MLflow not available"}
    
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
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
        
        # Get run details
        run = client.get_run(run_id)
        
        # Extract all metrics
        all_metrics = run.data.metrics
        all_params = run.data.params
        all_tags = run.data.tags
        
        # Identify key metrics
        key_metrics = {}
        metric_categories = {
            "performance": ["roc_auc", "roc-auc", "auc", "accuracy", "precision", "recall", "f1", "f1_score"],
            "training": ["train_loss", "val_loss", "train_accuracy", "val_accuracy"],
            "data": ["train_samples", "val_samples", "test_samples", "class_distribution"]
        }
        
        for category, metric_names in metric_categories.items():
            key_metrics[category] = {}
            for metric_name in metric_names:
                # Try different variations
                for variant in [metric_name, metric_name.replace("_", "-"), metric_name.upper(), metric_name.lower()]:
                    if variant in all_metrics:
                        key_metrics[category][metric_name] = all_metrics[variant]
                        break
        
        # Check for class imbalance indicators
        class_imbalance = {}
        for key, value in all_metrics.items():
            if "class" in key.lower() or "distribution" in key.lower() or "balance" in key.lower():
                class_imbalance[key] = value
        
        # Get data info from tags or params
        data_info = {}
        for key, value in {**all_params, **all_tags}.items():
            if any(term in key.lower() for term in ["data", "sample", "class", "train", "test"]):
                data_info[key] = value
        
        return {
            "model_name": model_name,
            "model_version": model_version,
            "model_stage": model_stage,
            "run_id": run_id,
            "run_name": run.info.run_name,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "duration_seconds": (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None,
            "key_metrics": key_metrics,
            "all_metrics": all_metrics,
            "parameters": all_params,
            "tags": all_tags,
            "class_imbalance_indicators": class_imbalance,
            "data_info": data_info,
            "analysis": {
                "has_performance_metrics": len(key_metrics.get("performance", {})) > 0,
                "has_class_imbalance_info": len(class_imbalance) > 0,
                "roc_auc": key_metrics.get("performance", {}).get("roc_auc") or key_metrics.get("performance", {}).get("auc"),
                "accuracy": key_metrics.get("performance", {}).get("accuracy")
            }
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error getting model metrics: {e}", exc_info=True)
        return {"error": str(e)}


def analyze_class_distribution() -> Dict[str, Any]:
    """
    Analyze class distribution in training data.
    
    Returns:
        Dictionary with class distribution analysis
    """
    if not HAS_MLFLOW:
        return {"error": "MLflow not available"}
    
    try:
        metrics = get_model_training_metrics()
        if "error" in metrics:
            return metrics
        
        # Try to extract class distribution from metrics/tags
        class_info = {}
        
        # Look for class distribution in metrics
        for key, value in metrics.get("all_metrics", {}).items():
            if "class" in key.lower() or "distribution" in key.lower():
                class_info[key] = value
        
        # Look in tags/params
        for key, value in {**metrics.get("parameters", {}), **metrics.get("tags", {})}.items():
            if "class" in key.lower() or "distribution" in key.lower() or "balance" in key.lower():
                class_info[key] = value
        
        # Calculate imbalance if we have class counts
        imbalance_analysis = {}
        if class_info:
            # Try to find positive/negative class counts
            pos_count = None
            neg_count = None
            
            for key, value in class_info.items():
                key_lower = key.lower()
                if "positive" in key_lower or "1" in key_lower or "high_risk" in key_lower:
                    try:
                        pos_count = float(value)
                    except:
                        pass
                if "negative" in key_lower or "0" in key_lower or "low_risk" in key_lower:
                    try:
                        neg_count = float(value)
                    except:
                        pass
            
            if pos_count is not None and neg_count is not None:
                total = pos_count + neg_count
                pos_ratio = pos_count / total if total > 0 else 0
                neg_ratio = neg_count / total if total > 0 else 0
                imbalance_ratio = max(pos_ratio, neg_ratio) / min(pos_ratio, neg_ratio) if min(pos_ratio, neg_ratio) > 0 else float('inf')
                
                imbalance_analysis = {
                    "positive_class_count": pos_count,
                    "negative_class_count": neg_count,
                    "total_samples": total,
                    "positive_ratio": pos_ratio,
                    "negative_ratio": neg_ratio,
                    "imbalance_ratio": imbalance_ratio,
                    "is_imbalanced": imbalance_ratio > 2.0,  # More than 2:1 ratio
                    "is_severely_imbalanced": imbalance_ratio > 10.0  # More than 10:1 ratio
                }
        
        return {
            "class_info": class_info,
            "imbalance_analysis": imbalance_analysis,
            "has_class_info": len(class_info) > 0
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error analyzing class distribution: {e}", exc_info=True)
        return {"error": str(e)}


def generate_training_review_report() -> Dict[str, Any]:
    """
    Generate comprehensive training review report.
    
    Returns:
        Complete training review report
    """
    print("=" * 80)
    print("MODEL TRAINING REVIEW")
    print("=" * 80)
    print()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_metrics": {},
        "class_distribution": {},
        "recommendations": []
    }
    
    # Get model metrics
    print("1. Retrieving model training metrics...")
    metrics = get_model_training_metrics()
    report["model_metrics"] = metrics
    
    if "error" not in metrics:
        print(f"   ✓ Model: {metrics['model_name']} v{metrics['model_version']}")
        print(f"   ✓ Run ID: {metrics['run_id']}")
        
        if metrics.get("analysis", {}).get("roc_auc"):
            print(f"   ✓ ROC-AUC: {metrics['analysis']['roc_auc']:.4f}")
        if metrics.get("analysis", {}).get("accuracy"):
            print(f"   ✓ Accuracy: {metrics['analysis']['accuracy']:.4f}")
    else:
        print(f"   ✗ Error: {metrics['error']}")
    
    print()
    
    # Analyze class distribution
    print("2. Analyzing class distribution...")
    class_dist = analyze_class_distribution()
    report["class_distribution"] = class_dist
    
    if "error" not in class_dist:
        if class_dist.get("has_class_info"):
            print("   ✓ Class information found")
            if class_dist.get("imbalance_analysis"):
                ia = class_dist["imbalance_analysis"]
                print(f"   Positive class: {ia.get('positive_class_count', 'N/A')}")
                print(f"   Negative class: {ia.get('negative_class_count', 'N/A')}")
                print(f"   Imbalance ratio: {ia.get('imbalance_ratio', 'N/A'):.2f}" if ia.get('imbalance_ratio') != float('inf') else "   Imbalance ratio: N/A")
                if ia.get("is_severely_imbalanced"):
                    print("   ⚠️  WARNING: Severely imbalanced data detected!")
                    report["recommendations"].append("Training data is severely imbalanced - consider using class weights or resampling")
                elif ia.get("is_imbalanced"):
                    print("   ⚠️  WARNING: Imbalanced data detected")
                    report["recommendations"].append("Training data is imbalanced - consider using class weights")
        else:
            print("   ⚠️  No class distribution information found in metrics")
            report["recommendations"].append("Class distribution not logged - add this to training pipeline")
    else:
        print(f"   ✗ Error: {class_dist['error']}")
    
    print()
    
    # Generate recommendations
    print("3. Generating recommendations...")
    
    if metrics.get("analysis", {}).get("roc_auc"):
        roc_auc = metrics["analysis"]["roc_auc"]
        if roc_auc < 0.7:
            report["recommendations"].append(f"Low ROC-AUC ({roc_auc:.4f}) - model may need improvement")
        elif roc_auc < 0.85:
            report["recommendations"].append(f"Moderate ROC-AUC ({roc_auc:.4f}) - consider model improvements")
    
    if not metrics.get("analysis", {}).get("has_class_imbalance_info"):
        report["recommendations"].append("Add class distribution logging to training pipeline for better diagnostics")
    
    if report["recommendations"]:
        print("   Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")
    else:
        print("   ✓ No specific recommendations")
    
    print()
    print("=" * 80)
    print("REVIEW COMPLETE")
    print("=" * 80)
    
    # Save report
    report_file = project_root / "training_review_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_file}")
    
    return report


if __name__ == "__main__":
    report = generate_training_review_report()
