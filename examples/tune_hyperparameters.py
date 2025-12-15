"""
Example: Hyperparameter Tuning for Credit Scoring Models

This script demonstrates how to perform hyperparameter tuning using
Grid Search and Random Search to improve model performance.

This is part of Task 5: Model Training and Tracking.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hyperparameter_tuning import HyperparameterTuner, tune_hyperparameters
from src.data_splitting import load_splits
from src.model_training import ModelTrainer


def main():
    """Main function to perform hyperparameter tuning."""
    
    print("=" * 100)
    print("Hyperparameter Tuning: Grid Search and Random Search")
    print("=" * 100)
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
    
    # Display data summary
    print("Data Summary:")
    print(f"  Training set: {len(X_train):,} samples, {len(X_train.columns)} features")
    print(f"  Test set:     {len(X_test):,} samples")
    print()
    
    # Create hyperparameter tuner
    tuner = HyperparameterTuner(
        random_state=42,
        cv=5,  # 5-fold cross-validation
        scoring='roc_auc',  # Use ROC-AUC as scoring metric
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    # Define which models to tune
    models_to_tune = ['logistic_regression', 'random_forest']
    
    # Optional: Define custom parameter grids
    custom_param_grids = {
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
    print("Hyperparameter Tuning Results")
    print("=" * 100)
    print()
    
    # Perform Grid Search for each model
    print("Method: Grid Search")
    print("-" * 100)
    print()
    
    for model_name in models_to_tune:
        try:
            param_grid = custom_param_grids.get(model_name, None)
            
            grid_search = tuner.grid_search(
                model_name,
                X_train, y_train,
                param_grid=param_grid
            )
            
            # Get best model and evaluate on test set
            best_model = tuner.best_models_[model_name]
            best_params = tuner.best_params_[model_name]
            best_cv_score = tuner.best_scores_[model_name]
            
            # Evaluate on test set
            from src.model_training import ModelTrainer
            temp_trainer = ModelTrainer()
            test_metrics = temp_trainer.evaluate_model(best_model, X_test, y_test, set_name='test')
            
            print(f"\n{model_name.upper()} - Grid Search Results:")
            print(f"  Best CV Score (ROC-AUC): {best_cv_score:.4f}")
            print(f"  Test ROC-AUC: {test_metrics.get('test_roc_auc', 'N/A'):.4f}")
            print(f"  Test Accuracy: {test_metrics.get('test_accuracy', 'N/A'):.4f}")
            print(f"  Test F1-Score: {test_metrics.get('test_f1_score', 'N/A'):.4f}")
            print(f"  Best Parameters: {best_params}")
            print()
            
        except Exception as e:
            print(f"✗ Failed to tune {model_name}: {e}")
            print()
    
    # Perform Random Search for comparison (optional)
    print("=" * 100)
    print("Method: Random Search (for comparison)")
    print("=" * 100)
    print()
    
    # Use Random Search for one model as example
    try:
        print("Performing Random Search for Random Forest...")
        random_search = tuner.random_search(
            'random_forest',
            X_train, y_train,
            param_distributions=custom_param_grids.get('random_forest', None),
            n_iter=20  # Sample 20 parameter combinations
        )
        
        # Get best model from random search
        best_model_rs = tuner.best_models_['random_forest']
        best_params_rs = tuner.best_params_['random_forest']
        best_cv_score_rs = tuner.best_scores_['random_forest']
        
        # Evaluate on test set
        temp_trainer = ModelTrainer()
        test_metrics_rs = temp_trainer.evaluate_model(best_model_rs, X_test, y_test, set_name='test')
        
        print(f"\nRandom Forest - Random Search Results:")
        print(f"  Best CV Score (ROC-AUC): {best_cv_score_rs:.4f}")
        print(f"  Test ROC-AUC: {test_metrics_rs.get('test_roc_auc', 'N/A'):.4f}")
        print(f"  Test Accuracy: {test_metrics_rs.get('test_accuracy', 'N/A'):.4f}")
        print(f"  Test F1-Score: {test_metrics_rs.get('test_f1_score', 'N/A'):.4f}")
        print(f"  Best Parameters: {best_params_rs}")
        print()
        
    except Exception as e:
        print(f"✗ Random Search failed: {e}")
        print()
    
    # Get summary of all tuning results
    print("=" * 100)
    print("Tuning Results Summary")
    print("=" * 100)
    print()
    
    summary = tuner.get_results_summary()
    if not summary.empty:
        print(summary)
        print()
    
    # Save best models
    models_dir = project_root / "models" / "tuned"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving tuned models...")
    print("-" * 100)
    for model_name in tuner.best_models_.keys():
        model_path = models_dir / f"{model_name}_tuned.pkl"
        tuner.save_best_model(model_name, str(model_path))
    print()
    
    print("=" * 100)
    print("Hyperparameter Tuning Complete!")
    print("=" * 100)
    print()
    print("Next Steps:")
    print("  1. Compare tuned models with baseline models")
    print("  2. Integrate with MLflow for experiment tracking")
    print("  3. Register best tuned model in MLflow model registry")
    print()


if __name__ == "__main__":
    main()

