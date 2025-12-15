"""
Example: Hyperparameter Tuning with MLflow Tracking

This script demonstrates how to perform hyperparameter tuning and track
all experiments using MLflow.

This is part of Task 5: Model Training and Tracking.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mlflow_tracking import MLflowTracker
from src.hyperparameter_tuning import HyperparameterTuner
from src.data_splitting import load_splits
from src.model_training import ModelTrainer


def main():
    """Main function to perform hyperparameter tuning with MLflow tracking."""
    
    print("=" * 100)
    print("Hyperparameter Tuning with MLflow Experiment Tracking")
    print("=" * 100)
    print()
    
    # Initialize MLflow tracker
    tracker = MLflowTracker(
        experiment_name="credit_scoring_tuning",
        tracking_uri="file:./mlruns"
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
    
    # Create hyperparameter tuner
    tuner = HyperparameterTuner(
        random_state=42,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Define models to tune
    models_to_tune = ['logistic_regression', 'random_forest']
    
    # Define parameter grids
    param_grids = {
        'logistic_regression': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'lbfgs']
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    }
    
    print("=" * 100)
    print("Hyperparameter Tuning with MLflow Tracking")
    print("=" * 100)
    print()
    
    run_ids = {}
    
    # Perform Grid Search for each model
    for model_name in models_to_tune:
        try:
            print(f"Tuning {model_name} with Grid Search...")
            print("-" * 100)
            
            param_grid = param_grids.get(model_name, None)
            
            # Perform grid search
            grid_search = tuner.grid_search(
                model_name,
                X_train, y_train,
                param_grid=param_grid
            )
            
            # Evaluate best model on test set
            best_model = tuner.best_models_[model_name]
            temp_trainer = ModelTrainer()
            test_metrics = temp_trainer.evaluate_model(
                best_model, X_test, y_test, set_name='test'
            )
            
            # Log to MLflow
            run_id = tracker.log_hyperparameter_tuning(
                search_result=grid_search,
                model_name=model_name,
                tuning_method='grid_search',
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                test_metrics=test_metrics
            )
            
            run_ids[f"{model_name}_grid_search"] = run_id
            
            print(f"✓ {model_name} tuning logged to MLflow (Run ID: {run_id})")
            print()
            
        except Exception as e:
            print(f"✗ Failed to tune {model_name}: {e}")
            print()
    
    # Perform Random Search for comparison
    print("=" * 100)
    print("Random Search (for comparison)")
    print("=" * 100)
    print()
    
    try:
        print("Performing Random Search for Random Forest...")
        
        random_search = tuner.random_search(
            'random_forest',
            X_train, y_train,
            param_distributions=param_grids.get('random_forest', None),
            n_iter=20
        )
        
        # Evaluate on test set
        best_model_rs = tuner.best_models_['random_forest']
        temp_trainer = ModelTrainer()
        test_metrics_rs = temp_trainer.evaluate_model(
            best_model_rs, X_test, y_test, set_name='test'
        )
        
        # Log to MLflow
        run_id_rs = tracker.log_hyperparameter_tuning(
            search_result=random_search,
            model_name='random_forest',
            tuning_method='random_search',
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            test_metrics=test_metrics_rs
        )
        
        run_ids['random_forest_random_search'] = run_id_rs
        
        print(f"✓ Random Search logged to MLflow (Run ID: {run_id_rs})")
        print()
        
    except Exception as e:
        print(f"✗ Random Search failed: {e}")
        print()
    
    # Get best model and register
    print("=" * 100)
    print("Best Model Registration")
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
            model_name="credit_scoring_tuned_model",
            artifact_path="model",
            stage="Staging"
        )
        
        print()
        print(f"✓ Best tuned model registered as 'credit_scoring_tuned_model' (Version: {model_version})")
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
                       'metrics.test_accuracy', 'metrics.test_f1_score',
                       'metrics.cv_best_score']
        available_cols = [col for col in display_cols if col in comparison.columns]
        
        print("Tuning Results Comparison:")
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
    print("  2. Compare all tuning runs in the MLflow UI")
    print("  3. Review best model parameters and metrics")
    print("  4. Promote best model to Production stage if ready")
    print()


if __name__ == "__main__":
    main()

