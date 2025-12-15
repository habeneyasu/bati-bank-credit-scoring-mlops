# MLflow Experiment Tracking Guide

## Overview

This guide explains how to use MLflow for experiment tracking, model versioning, and model registry in the credit scoring project.

## Quick Start

### 1. Train Models with MLflow Tracking

```bash
python examples/train_with_mlflow.py
```

This will:
- Train multiple models (Logistic Regression, Decision Tree, Random Forest)
- Log all experiments to MLflow
- Track parameters, metrics, and artifacts
- Register the best model

### 2. Hyperparameter Tuning with MLflow

```bash
python examples/tune_with_mlflow.py
```

This will:
- Perform Grid Search and Random Search
- Track all tuning experiments
- Compare different parameter combinations
- Register the best tuned model

### 3. View Experiments in MLflow UI

```bash
# Option 1: Use the helper script
./scripts/start_mlflow_ui.sh

# Option 2: Run directly
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Then open: http://localhost:5000

## Features

### Experiment Tracking

All experiments automatically log:
- **Parameters**: Model hyperparameters, training configuration
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Artifacts**: Trained models, datasets info
- **Tags**: Model type, training method, tuning method

### Model Registry

Best models are registered in MLflow Model Registry with:
- **Versioning**: Automatic version tracking
- **Stages**: None → Staging → Production → Archived
- **Metadata**: Parameters, metrics, and artifacts for each version

### Experiment Comparison

Compare all runs in MLflow UI:
- Sort by any metric (e.g., test_roc_auc)
- Filter by parameters or tags
- Visualize metrics over time
- Download model artifacts

## Usage Examples

### Basic Training with Tracking

```python
from src.mlflow_tracking import MLflowTracker
from src.model_training import train_models

# Create tracker
tracker = MLflowTracker(experiment_name="credit_scoring")

# Train models
trainer = train_models(X_train, y_train, X_test, y_test)

# Log each model
for model_name, (model, metrics) in trainer.models_.items():
    tracker.log_model_training(
        model=model,
        model_name=model_name,
        model_params=model.get_params(),
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
```

### Hyperparameter Tuning with Tracking

```python
from src.mlflow_tracking import MLflowTracker
from src.hyperparameter_tuning import HyperparameterTuner

# Create tracker and tuner
tracker = MLflowTracker(experiment_name="credit_scoring_tuning")
tuner = HyperparameterTuner(random_state=42, cv=5)

# Perform grid search
grid_search = tuner.grid_search('random_forest', X_train, y_train)

# Log to MLflow
tracker.log_hyperparameter_tuning(
    search_result=grid_search,
    model_name='random_forest',
    tuning_method='grid_search',
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)
```

### Register Best Model

```python
# Get best model
best_run = tracker.get_best_run(metric='test_roc_auc')

# Register in model registry
tracker.register_model(
    run_id=best_run['run_id'],
    model_name="credit_scoring_model",
    stage="Production"
)
```

## MLflow UI Features

### Viewing Experiments

1. **Experiments List**: See all experiments and their metrics
2. **Run Details**: View parameters, metrics, and artifacts for each run
3. **Compare Runs**: Side-by-side comparison of multiple runs
4. **Model Registry**: View registered models and their versions

### Comparing Models

1. Select multiple runs in the UI
2. Click "Compare" to see:
   - Parameter differences
   - Metric comparisons
   - Model artifacts

### Model Registry

1. Navigate to "Models" tab
2. View registered models:
   - All versions
   - Current stage
   - Metrics and parameters
3. Transition stages:
   - Staging → Production
   - Production → Archived

## File Structure

```
mlruns/                    # MLflow tracking data
├── 0/                    # Experiment ID
│   └── runs/             # Individual runs
│       └── <run_id>/     # Run artifacts and metadata
└── models/                # Registered models
    └── credit_scoring_model/
        └── versions/     # Model versions
```

## Best Practices

1. **Use Descriptive Experiment Names**: 
   - `credit_scoring_baseline`
   - `credit_scoring_tuning`
   - `credit_scoring_production`

2. **Tag Your Runs**:
   - Model type
   - Training method
   - Dataset version

3. **Register Production Models**:
   - Always register models going to production
   - Use appropriate stages (Staging → Production)
   - Document model versions

4. **Compare Before Promoting**:
   - Compare new models with existing production models
   - Ensure performance improvements
   - Document changes

## Troubleshooting

### MLflow UI Not Starting

```bash
# Check if port 5000 is available
lsof -i :5000

# Use different port
mlflow ui --port 5001
```

### Experiments Not Showing

- Check tracking URI: `file:./mlruns`
- Verify experiment name matches
- Check `mlruns/` directory exists

### Model Registry Issues

- Ensure MLflow version >= 2.0
- Check model registry URI is set correctly
- Verify model was logged successfully

---

**Status**: ✅ MLflow Integration Complete

