#!/usr/bin/env python3
"""
Standalone diagnostic script to investigate scoring issues.
Can run without full project dependencies.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Try to import database connection
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.database.connection import get_db_session
    from src.database.models import Prediction
    from sqlalchemy import func, desc
    HAS_DB = True
except ImportError as e:
    print(f"Warning: Database imports failed: {e}")
    HAS_DB = False

# Try to import MLflow
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    print("Warning: MLflow not available")
    HAS_MLFLOW = False


def analyze_predictions_from_db(limit: int = 100) -> Dict[str, Any]:
    """Analyze predictions from database."""
    if not HAS_DB:
        return {"error": "Database connection not available"}
    
    try:
        with get_db_session() as session:
            predictions = session.query(
                Prediction.probability,
                Prediction.risk_level,
                Prediction.customer_score,
                Prediction.features,
                Prediction.customer_id
            ).order_by(desc(Prediction.created_at)).limit(limit).all()
            
            if not predictions:
                return {"error": "No predictions found"}
            
            probabilities = [float(p.probability) for p in predictions]
            risk_levels = [p.risk_level for p in predictions]
            scores = [p.customer_score for p in predictions if p.customer_score]
            
            # Basic statistics
            prob_mean = sum(probabilities) / len(probabilities)
            prob_min = min(probabilities)
            prob_max = max(probabilities)
            prob_std = (sum((p - prob_mean) ** 2 for p in probabilities) / len(probabilities)) ** 0.5
            
            # Check feature diversity
            feature_vectors = []
            for p in predictions:
                if p.features and isinstance(p.features, list):
                    feature_vectors.append(p.features)
            
            feature_analysis = {}
            if feature_vectors:
                num_features = len(feature_vectors[0])
                for col_idx in range(num_features):
                    col_values = [fv[col_idx] for fv in feature_vectors if len(fv) > col_idx]
                    if col_values:
                        col_mean = sum(col_values) / len(col_values)
                        col_min = min(col_values)
                        col_max = max(col_values)
                        col_variance = sum((v - col_mean) ** 2 for v in col_values) / len(col_values)
                        unique_vals = len(set(col_values))
                        
                        feature_analysis[f"feature_{col_idx}"] = {
                            "mean": col_mean,
                            "min": col_min,
                            "max": col_max,
                            "variance": col_variance,
                            "is_constant": unique_vals == 1,
                            "unique_values": unique_vals
                        }
            
            constant_features = [k for k, v in feature_analysis.items() if v.get("is_constant", False)]
            low_variance_features = [k for k, v in feature_analysis.items() if v.get("variance", 1.0) < 0.01 and not v.get("is_constant", False)]
            
            return {
                "total_predictions": len(predictions),
                "probability_stats": {
                    "mean": prob_mean,
                    "std": prob_std,
                    "min": prob_min,
                    "max": prob_max,
                    "range": prob_max - prob_min
                },
                "risk_level_distribution": {
                    "low": risk_levels.count("low"),
                    "medium": risk_levels.count("medium"),
                    "high": risk_levels.count("high")
                },
                "feature_analysis": {
                    "total_features": len(feature_analysis),
                    "constant_features": len(constant_features),
                    "low_variance_features": len(low_variance_features),
                    "constant_feature_names": constant_features,
                    "low_variance_feature_names": low_variance_features
                },
                "issues": {
                    "all_low_risk": all(r == "low" for r in risk_levels),
                    "low_probability_range": prob_max - prob_min < 0.1,
                    "has_constant_features": len(constant_features) > 0,
                    "has_low_variance_features": len(low_variance_features) > 0
                }
            }
    except Exception as e:
        return {"error": str(e)}


def test_model_patterns() -> Dict[str, Any]:
    """Test model with different patterns."""
    if not HAS_MLFLOW:
        return {"error": "MLflow not available"}
    
    try:
        # Try to load settings
        try:
            from src.utils.config import settings
            model_name = settings.model_name
            model_stage = settings.model_stage
            mlflow_uri = settings.mlflow_tracking_uri
            num_features = settings.expected_features
        except:
            # Use defaults
            model_name = "credit_scoring_model"
            model_stage = "Production"
            mlflow_uri = "file:./mlruns"
            num_features = 26
        
        mlflow.set_tracking_uri(mlflow_uri)
        model_uri = f"models:/{model_name}/{model_stage}"
        model = mlflow.sklearn.load_model(model_uri)
        
        test_patterns = {
            "all_zeros": [0.0] * num_features,
            "all_ones": [1.0] * num_features,
            "low_values": [0.1] * num_features,
            "high_values": [0.9] * num_features,
            "mixed_low": [0.2] * num_features,
            "mixed_high": [0.8] * num_features,
        }
        
        results = {}
        for pattern_name, features in test_patterns.items():
            try:
                import numpy as np
                features_array = np.array(features, dtype=np.float64).reshape(1, -1)
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_array)[0]
                    probability = float(probabilities[1])
                    prediction = int(np.argmax(probabilities))
                else:
                    prediction = int(model.predict(features_array)[0])
                    probability = float(prediction)
                
                if probability < 0.30:
                    risk_level = "low"
                elif probability > 0.60:
                    risk_level = "high"
                else:
                    risk_level = "medium"
                
                results[pattern_name] = {
                    "probability": probability,
                    "risk_level": risk_level,
                    "score": int((1 - probability) * 100)
                }
            except Exception as e:
                results[pattern_name] = {"error": str(e)}
        
        # Analyze
        probs = [r["probability"] for r in results.values() if "probability" in r]
        if probs:
            results["analysis"] = {
                "probability_range": max(probs) - min(probs),
                "all_similar": max(probs) - min(probs) < 0.05,
                "model_responsive": max(probs) - min(probs) > 0.1
            }
        
        return results
    except Exception as e:
        return {"error": str(e)}


def main():
    """Run diagnostic."""
    print("=" * 80)
    print("SCORING DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    print()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "database_analysis": {},
        "model_testing": {}
    }
    
    # Database analysis
    if HAS_DB:
        print("1. Analyzing predictions from database...")
        db_analysis = analyze_predictions_from_db(limit=100)
        report["database_analysis"] = db_analysis
        
        if "error" not in db_analysis:
            print(f"   ✓ Analyzed {db_analysis['total_predictions']} predictions")
            print(f"   Probability range: {db_analysis['probability_stats']['min']:.4f} - {db_analysis['probability_stats']['max']:.4f}")
            print(f"   Probability std: {db_analysis['probability_stats']['std']:.4f}")
            print(f"   Risk distribution: Low={db_analysis['risk_level_distribution']['low']}, "
                  f"Medium={db_analysis['risk_level_distribution']['medium']}, "
                  f"High={db_analysis['risk_level_distribution']['high']}")
            
            if db_analysis.get("issues", {}).get("all_low_risk"):
                print("   ⚠️  WARNING: All predictions are LOW risk!")
            if db_analysis.get("issues", {}).get("has_constant_features"):
                print(f"   ⚠️  WARNING: {db_analysis['feature_analysis']['constant_features']} constant features found!")
        else:
            print(f"   ✗ Error: {db_analysis['error']}")
    else:
        print("1. Skipping database analysis (database not available)")
    
    print()
    
    # Model testing
    if HAS_MLFLOW:
        print("2. Testing model with different patterns...")
        model_test = test_model_patterns()
        report["model_testing"] = model_test
        
        if "error" not in model_test:
            print("   Test results:")
            for pattern, result in model_test.items():
                if "probability" in result:
                    print(f"     {pattern}: prob={result['probability']:.4f}, risk={result['risk_level']}, score={result['score']}")
            
            if "analysis" in model_test:
                analysis = model_test["analysis"]
                print(f"   Probability range across patterns: {analysis['probability_range']:.4f}")
                if analysis.get("all_similar"):
                    print("   ⚠️  WARNING: Model produces very similar probabilities for different inputs!")
                if analysis.get("model_responsive"):
                    print("   ✓ Model responds differently to different inputs")
        else:
            print(f"   ✗ Error: {model_test['error']}")
    else:
        print("2. Skipping model testing (MLflow not available)")
    
    print()
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    
    # Save report
    report_file = Path(__file__).parent.parent / "diagnostic_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_file}")
    
    # Summary
    issues = []
    if HAS_DB and "issues" in report.get("database_analysis", {}):
        issues_data = report["database_analysis"]["issues"]
        if issues_data.get("all_low_risk"):
            issues.append("All predictions are LOW risk")
        if issues_data.get("has_constant_features"):
            issues.append("Constant features detected (same value for all customers)")
        if issues_data.get("low_probability_range"):
            issues.append("Very narrow probability range (model may not be differentiating customers)")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✓ No major issues detected")


if __name__ == "__main__":
    main()
