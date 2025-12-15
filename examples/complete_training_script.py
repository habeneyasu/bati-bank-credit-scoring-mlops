"""
Complete Training Script for Task 5

This script performs a complete model training workflow:
1. Reproducible train/test split (random_state=42)
2. Train at least two models
3. Run Grid Search and Random Search hyperparameter tuning
4. Compute all metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
5. Log everything to MLflow (parameters, metrics, artifacts)

This directly addresses Task 5 requirements.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_splitting import DataSplitter
from src.model_training import ModelTrainer
from src.hyperparameter_tuning import HyperparameterTuner
from src.mlflow_tracking import MLflowTracker
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def main():
    """Complete training script addressing all Task 5 requirements."""
    
    print("=" * 100)
    print("Complete Model Training Script - Task 5")
    print("=" * 100)
    print()
    
    # ============================================================================
    # STEP 1: Reproducible Train/Test Split
    # ============================================================================
    print("Step 1: Reproducible Train/Test Split")
    print("-" * 100)
    print()
    
    # Load processed data with target
    data_path = project_root / "data" / "processed" / "processed_data_with_target.csv"
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please run Task 4 first to create processed_data_with_target.csv")
        return
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    print()
    
    # Separate features and target
    target_col = 'is_high_risk'
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in data")
        return
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Features: {len(X.columns)}")
    print(f"Target distribution:")
    print(f"  Low-risk (0): {sum(y == 0):,} ({sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  High-risk (1): {sum(y == 1):,} ({sum(y == 1)/len(y)*100:.1f}%)")
    print()
    
    # Perform reproducible train/test split with random_state=42
    print("Performing train/test split with random_state=42...")
    splitter = DataSplitter(
        test_size=0.2,
        random_state=42,  # Reproducibility
        stratify=True      # Maintain class distribution
    )
    
    X_train, X_test, y_train, y_test = splitter.split_data(df, target_col=target_col)
    
    print(f"✓ Train set: {len(X_train):,} samples")
    print(f"✓ Test set:  {len(X_test):,} samples")
    print(f"✓ Random state: 42 (reproducible)")
    print()
    
    # ============================================================================
    # STEP 2: Initialize MLflow Tracker
    # ============================================================================
    print("Step 2: Initialize MLflow Tracking")
    print("-" * 100)
    print()
    
    tracker = MLflowTracker(
        experiment_name="credit_scoring",
        tracking_uri="file:./mlruns"
    )
    print()
    
    # ============================================================================
    # STEP 3: Train At Least Two Models
    # ============================================================================
    print("Step 3: Train At Least Two Models")
    print("-" * 100)
    print()
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Train at least 2 models (actually training 3 for better comparison)
    model_names = ['logistic_regression', 'random_forest']
    
    print(f"Training models: {', '.join(model_names)}")
    print()
    
    for model_name in model_names:
        print(f"Training {model_name}...")
        model, metrics = trainer.train_and_evaluate(
            model_name,
            X_train, y_train,
            X_test, y_test
        )
        
        print(f"✓ {model_name} trained")
        print(f"  Test Accuracy:  {metrics.get('test_accuracy', 0):.4f}")
        print(f"  Test ROC-AUC:   {metrics.get('test_roc_auc', 0):.4f}")
        print()
    
    # ============================================================================
    # STEP 4: Compute All Required Metrics
    # ============================================================================
    print("Step 4: Compute All Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)")
    print("-" * 100)
    print()
    
    all_metrics = {}
    
    for model_name in model_names:
        model = trainer.models_[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Compute all required metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
        }
        
        all_metrics[model_name] = metrics
        
        print(f"{model_name} Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print()
    
    # ============================================================================
    # STEP 5: Log Everything to MLflow (Parameters, Metrics, Artifacts)
    # ============================================================================
    print("Step 5: Log to MLflow (Parameters, Metrics, Artifacts)")
    print("-" * 100)
    print()
    
    run_ids = {}
    
    for model_name in model_names:
        model = trainer.models_[model_name]
        metrics = all_metrics[model_name]
        
        # Get model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
        else:
            params = {}
        
        # Prepare metrics with train/test prefixes
        train_metrics = trainer.metrics_[model_name]
        full_metrics = {
            **train_metrics,  # Includes train_* and test_* metrics
            **{f'test_{k}': v for k, v in metrics.items()}  # Ensure all test metrics
        }
        
        # Log to MLflow
        print(f"Logging {model_name} to MLflow...")
        run_id = tracker.log_model_training(
            model=model,
            model_name=model_name,
            model_params=params,
            metrics=full_metrics,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            tags={
                'model_type': model_name,
                'training_method': 'baseline',
                'random_state': '42'
            }
        )
        
        run_ids[model_name] = run_id
        print(f"✓ Logged to MLflow (Run ID: {run_id})")
        print(f"  Parameters logged: {len(params)}")
        print(f"  Metrics logged: {len(full_metrics)}")
        print(f"  Model artifact saved")
        print()
    
    # ============================================================================
    # STEP 6: Hyperparameter Tuning - Grid Search and Random Search
    # ============================================================================
    print("Step 6: Hyperparameter Tuning (Grid Search and Random Search)")
    print("-" * 100)
    print()
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        random_state=42,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Define parameter grids
    param_grids = {
        'logistic_regression': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
    }
    
    # Grid Search
    print("Performing Grid Search...")
    print()
    
    for model_name in model_names:
        print(f"Grid Search for {model_name}...")
        param_grid = param_grids.get(model_name, {})
        
        # Perform grid search
        grid_search = tuner.grid_search(
            model_name,
            X_train, y_train,
            param_grid=param_grid
        )
        
        # Get best model
        best_model = tuner.best_models_[model_name]
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Compute all metrics
        test_metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_pred, zero_division=0),
            'test_f1_score': f1_score(y_test, y_pred, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
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
        
        print(f"✓ Grid Search complete for {model_name}")
        print(f"  Best CV Score: {grid_search.best_score_:.4f}")
        print(f"  Test ROC-AUC:  {test_metrics['test_roc_auc']:.4f}")
        print(f"  Logged to MLflow (Run ID: {run_id})")
        print()
    
    # Random Search
    print("Performing Random Search...")
    print()
    
    for model_name in model_names:
        print(f"Random Search for {model_name}...")
        param_grid = param_grids.get(model_name, {})
        
        # Perform random search (n_iter=10 for faster execution)
        random_search = tuner.random_search(
            model_name,
            X_train, y_train,
            param_distributions=param_grid,
            n_iter=10
        )
        
        # Get best model
        best_model = tuner.best_models_[f"{model_name}_random"] if f"{model_name}_random" in tuner.best_models_ else tuner.best_models_[model_name]
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Compute all metrics
        test_metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_pred, zero_division=0),
            'test_f1_score': f1_score(y_test, y_pred, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log to MLflow
        run_id = tracker.log_hyperparameter_tuning(
            search_result=random_search,
            model_name=model_name,
            tuning_method='random_search',
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            test_metrics=test_metrics
        )
        
        print(f"✓ Random Search complete for {model_name}")
        print(f"  Best CV Score: {random_search.best_score_:.4f}")
        print(f"  Test ROC-AUC:  {test_metrics['test_roc_auc']:.4f}")
        print(f"  Logged to MLflow (Run ID: {run_id})")
        print()
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("=" * 100)
    print("Training Complete - Summary")
    print("=" * 100)
    print()
    print("✓ Reproducible train/test split (random_state=42)")
    print(f"✓ Trained {len(model_names)} models: {', '.join(model_names)}")
    print("✓ Computed all metrics: Accuracy, Precision, Recall, F1, ROC-AUC")
    print("✓ Performed Grid Search hyperparameter tuning")
    print("✓ Performed Random Search hyperparameter tuning")
    print("✓ Logged all experiments to MLflow:")
    print(f"  - Parameters logged")
    print(f"  - Metrics logged")
    print(f"  - Model artifacts saved")
    print()
    print("View experiments:")
    print("  mlflow ui --backend-store-uri file:./mlruns")
    print("  Then open: http://localhost:5000")
    print()


if __name__ == "__main__":
    main()

