"""
Example: Train Models with MLflow Experiment Tracking

This script demonstrates how to train models and track experiments using MLflow,
including model parameters, evaluation metrics, and artifacts.

This is part of Task 5: Model Training and Tracking.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mlflow_tracking import MLflowTracker
from src.model_training import train_models, ModelTrainer
from src.data_splitting import load_splits


def main():
    """Main function to train models with MLflow tracking."""
    
    print("=" * 100)
    print("Train Models with MLflow Experiment Tracking")
    print("=" * 100)
    print()
    
    # Initialize MLflow tracker
    tracker = MLflowTracker(
        experiment_name="credit_scoring",
        tracking_uri="file:./mlruns"  # Local file store
    )
    print()
    
    # Load data splits
    splits_dir = project_root / "data" / "processed" / "splits"
    
    if not splits_dir.exists():
        print(f"Splits directory not found: {splits_dir}")
        print("Please run data preparation first:")
        print("  python examples/prepare_data_splits.py")
        return
    
    print(f"Loading data splits from: {splits_dir}")
    X_train, X_test, y_train, y_test = load_splits(str(splits_dir))
    print()
    
    # Define models to train
    model_names = ['logistic_regression', 'decision_tree', 'random_forest']
    
    # Optional: Define custom parameters
    model_params = {
        'logistic_regression': {
            'max_iter': 1000,
            'C': 1.0
        },
        'decision_tree': {
            'max_depth': 10,
            'min_samples_split': 20
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 20
        }
    }
    
    print("=" * 100)
    print("Training Models with MLflow Tracking")
    print("=" * 100)
    print()
    
    # Train models
    trainer = train_models(
        X_train, y_train,
        X_test, y_test,
        model_names=model_names,
        random_state=42,
        model_params=model_params
    )
    
    print()
    print("=" * 100)
    print("Logging Experiments to MLflow")
    print("=" * 100)
    print()
    
    # Log each model to MLflow
    run_ids = {}
    
    for model_name in model_names:
        if model_name in trainer.models_ and model_name in trainer.metrics_:
            model = trainer.models_[model_name]
            metrics = trainer.metrics_[model_name]
            
            # Extract model parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
            else:
                params = model_params.get(model_name, {})
            
            # Log to MLflow
            run_id = tracker.log_model_training(
                model=model,
                model_name=model_name,
                model_params=params,
                metrics=metrics,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                tags={
                    'model_type': model_name,
                    'training_method': 'baseline'
                }
            )
            
            run_ids[model_name] = run_id
            print()
    
    # Get best model
    print("=" * 100)
    print("Best Model Identification")
    print("=" * 100)
    print()
    
    try:
        best_run = tracker.get_best_run(metric='test_roc_auc')
        
        print(f"Best Model: {best_run['run_name']}")
        print(f"Run ID: {best_run['run_id']}")
        print(f"Test ROC-AUC: {best_run['metric_value']:.4f}")
        print()
        
        # Register best model
        print("Registering best model in MLflow Model Registry...")
        print("-" * 100)
        
        model_version = tracker.register_model(
            run_id=best_run['run_id'],
            model_name="credit_scoring_model",
            artifact_path="model",
            stage="Staging"  # Start in Staging, can promote to Production later
        )
        
        print()
        print(f"✓ Best model registered as 'credit_scoring_model' (Version: {model_version})")
        print()
        
    except Exception as e:
        print(f"Could not identify/register best model: {e}")
        print()
    
    # Compare all runs
    print("=" * 100)
    print("Experiment Comparison")
    print("=" * 100)
    print()
    
    comparison = tracker.compare_runs(metric='test_roc_auc')
    
    if not comparison.empty:
        # Display key metrics
        display_cols = ['tags.mlflow.runName', 'metrics.test_roc_auc', 
                       'metrics.test_accuracy', 'metrics.test_f1_score']
        available_cols = [col for col in display_cols if col in comparison.columns]
        
        print("Model Performance Comparison:")
        print(comparison[available_cols].round(4))
        print()
    
    print("=" * 100)
    print("MLflow Experiment Tracking Complete!")
    print("=" * 100)
    print()
    print("Next Steps:")
    print("  1. View experiments in MLflow UI:")
    print("     Run: mlflow ui")
    print("     Then open: http://localhost:5000")
    print()
    print("  2. Compare model runs in the MLflow UI")
    print("  3. Promote best model to Production stage if ready")
    print("  4. Use registered model for deployment")
    print()


if __name__ == "__main__":
    main()

