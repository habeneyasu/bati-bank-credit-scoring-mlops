"""
Example: Train Multiple Models for Credit Scoring

This script demonstrates how to train multiple machine learning models
for credit risk prediction.

This is part of Task 5: Model Training and Tracking.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_training import train_models, ModelTrainer
from src.data_splitting import load_splits


def main():
    """Main function to train multiple models."""
    
    print("=" * 100)
    print("Model Selection and Training")
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
    print(f"  Test set:     {len(X_test):,} samples, {len(X_test.columns)} features")
    print()
    
    print("Target Distribution:")
    print(f"  Training - Low-Risk (0): {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"  Training - High-Risk (1): {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
    print(f"  Test - Low-Risk (0):     {sum(y_test == 0):,} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
    print(f"  Test - High-Risk (1):     {sum(y_test == 1):,} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")
    print()
    
    # Define models to train
    # At least 2 models as required, but we'll train 4 for comparison
    model_names = [
        'logistic_regression',
        'decision_tree',
        'random_forest',
        # 'xgboost',  # Uncomment if XGBoost is available
        # 'lightgbm'  # Uncomment if LightGBM is available
    ]
    
    # Optional: Define custom parameters for models
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
    print("Training Models")
    print("=" * 100)
    print()
    print(f"Models to train: {', '.join(model_names)}")
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
    print("Model Evaluation Results")
    print("=" * 100)
    print()
    
    # Get metrics summary
    metrics_summary = trainer.get_metrics_summary()
    
    # Display metrics
    print("Model Performance Metrics:")
    print("-" * 100)
    
    # Format metrics for display
    display_metrics = ['model', 'train_accuracy', 'test_accuracy', 
                      'train_precision', 'test_precision',
                      'train_recall', 'test_recall',
                      'train_f1_score', 'test_f1_score',
                      'train_roc_auc', 'test_roc_auc']
    
    available_cols = [col for col in display_metrics if col in metrics_summary.columns]
    print(metrics_summary[available_cols].round(4))
    print()
    
    # Find best model
    try:
        best_name, best_model = trainer.get_best_model(metric='test_roc_auc')
        best_metrics = trainer.metrics_[best_name]
        
        print("=" * 100)
        print("Best Model (by Test ROC-AUC)")
        print("=" * 100)
        print()
        print(f"Model: {best_name}")
        print(f"Test ROC-AUC: {best_metrics.get('test_roc_auc', 'N/A'):.4f}")
        print(f"Test Accuracy: {best_metrics.get('test_accuracy', 'N/A'):.4f}")
        print(f"Test F1-Score: {best_metrics.get('test_f1_score', 'N/A'):.4f}")
        print()
    except Exception as e:
        print(f"Could not determine best model: {e}")
        print()
    
    # Save models
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving trained models...")
    print("-" * 100)
    for model_name in trainer.models_.keys():
        model_path = models_dir / f"{model_name}.pkl"
        trainer.save_model(model_name, str(model_path))
    print()
    
    print("=" * 100)
    print("Model Training Complete!")
    print("=" * 100)
    print()
    print("Next Steps:")
    print("  1. Review model performance metrics")
    print("  2. Integrate with MLflow for experiment tracking")
    print("  3. Perform hyperparameter tuning")
    print("  4. Register best model in MLflow model registry")
    print()


if __name__ == "__main__":
    main()

