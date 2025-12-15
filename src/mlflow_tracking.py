"""
MLflow Experiment Tracking Module for Credit Scoring

This module provides MLflow integration for experiment tracking, model versioning,
and model registry.

Features:
- Automatic experiment tracking (parameters, metrics, artifacts)
- Model logging and versioning
- Model registry integration
- Experiment comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Tuple
import warnings
from pathlib import Path
import joblib

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

warnings.filterwarnings('ignore')


class MLflowTracker:
    """
    MLflow experiment tracking for credit scoring models.
    
    Provides comprehensive tracking of:
    - Model parameters
    - Evaluation metrics
    - Model artifacts
    - Model versions
    """
    
    def __init__(
        self,
        experiment_name: str = "credit_scoring",
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None
    ):
        """
        Initialize MLflow Tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (default: local file store)
            registry_uri: MLflow model registry URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.registry_uri = registry_uri
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            print(f"Warning: Could not set up experiment: {e}")
            experiment_id = "0"
        
        mlflow.set_experiment(experiment_name)
        self.experiment_id = experiment_id
        
        # Initialize MLflow client
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
    
    def log_model_training(
        self,
        model: Any,
        model_name: str,
        model_params: Dict,
        metrics: Dict[str, float],
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        tags: Optional[Dict[str, str]] = None,
        artifact_path: str = "model"
    ) -> str:
        """
        Log a trained model to MLflow.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            model_params: Model hyperparameters
            metrics: Dictionary of evaluation metrics
            X_train: Training features (optional, for logging dataset info)
            y_train: Training target (optional)
            X_test: Test features (optional)
            y_test: Test target (optional)
            tags: Additional tags for the run
            artifact_path: Path to save model artifact
        
        Returns:
            Run ID of the logged experiment
        
        Example:
            >>> tracker = MLflowTracker(experiment_name="credit_scoring")
            >>> run_id = tracker.log_model_training(
            ...     model, "random_forest",
            ...     model_params={'n_estimators': 100},
            ...     metrics={'test_roc_auc': 0.95, 'test_accuracy': 0.90}
            ... )
        """
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            
            # Log parameters
            print(f"Logging experiment for {model_name}...")
            mlflow.log_params(model_params)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log dataset info
            if X_train is not None:
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("train_features", len(X_train.columns))
            if X_test is not None:
                mlflow.log_param("test_samples", len(X_test))
            if y_train is not None:
                mlflow.log_param("train_class_0", int((y_train == 0).sum()))
                mlflow.log_param("train_class_1", int((y_train == 1).sum()))
            if y_test is not None:
                mlflow.log_param("test_class_0", int((y_test == 0).sum()))
                mlflow.log_param("test_class_1", int((y_test == 1).sum()))
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=None  # Will register separately if needed
            )
            
            print(f"✓ Experiment logged (Run ID: {run_id})")
            
            return run_id
    
    def log_hyperparameter_tuning(
        self,
        search_result: Any,  # GridSearchCV or RandomizedSearchCV
        model_name: str,
        tuning_method: str,  # 'grid_search' or 'random_search'
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        test_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Log hyperparameter tuning results to MLflow.
        
        Args:
            search_result: GridSearchCV or RandomizedSearchCV result
            model_name: Name of the model
            tuning_method: 'grid_search' or 'random_search'
            X_train: Training features (optional)
            y_train: Training target (optional)
            X_test: Test features (optional)
            y_test: Test target (optional)
            test_metrics: Test set metrics (optional)
        
        Returns:
            Run ID of the logged experiment
        """
        best_model = search_result.best_estimator_
        best_params = search_result.best_params_
        best_score = search_result.best_score_
        
        # Prepare metrics
        metrics = {
            'cv_best_score': best_score,
            'cv_mean_score': search_result.cv_results_['mean_test_score'].max(),
            'cv_std_score': search_result.cv_results_['std_test_score'][
                search_result.cv_results_['mean_test_score'].argmax()
            ]
        }
        
        # Add test metrics if provided
        if test_metrics:
            metrics.update(test_metrics)
        
        # Log tuning method and parameters
        params = best_params.copy()
        params['tuning_method'] = tuning_method
        params['cv_folds'] = search_result.cv
        
        # Log number of parameter combinations tried
        if tuning_method == 'grid_search':
            params['n_combinations'] = len(search_result.cv_results_['params'])
        else:  # random_search
            params['n_iter'] = search_result.n_iter
        
        # Log experiment
        run_id = self.log_model_training(
            model=best_model,
            model_name=f"{model_name}_{tuning_method}",
            model_params=params,
            metrics=metrics,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            tags={'tuning_method': tuning_method, 'model_type': model_name}
        )
        
        return run_id
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        stage: str = "None"
    ) -> str:
        """
        Register a model in MLflow Model Registry.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            artifact_path: Path to model artifact in the run
            stage: Model stage ('None', 'Staging', 'Production', 'Archived')
        
        Returns:
            Registered model version
        
        Example:
            >>> tracker = MLflowTracker()
            >>> run_id = tracker.log_model_training(...)
            >>> version = tracker.register_model(run_id, "best_credit_model", stage="Production")
        """
        try:
            # Get model URI
            model_uri = f"runs:/{run_id}/{artifact_path}"
            
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            print(f"✓ Model registered: {model_name} (Version: {model_version.version})")
            
            # Transition to stage if specified
            if stage != "None":
                self.transition_model_stage(model_name, model_version.version, stage)
            
            return model_version.version
            
        except Exception as e:
            print(f"Error registering model: {e}")
            raise
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ):
        """
        Transition a model to a specific stage.
        
        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            print(f"✓ Model {model_name} v{version} transitioned to {stage}")
        except Exception as e:
            print(f"Error transitioning model stage: {e}")
            raise
    
    def get_best_run(
        self,
        metric: str = "test_roc_auc",
        ascending: bool = False
    ) -> Dict:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric name to compare
            ascending: If True, lower is better (default: False, higher is better)
        
        Returns:
            Dictionary with best run information
        """
        # Get all runs in experiment
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )
        
        if runs.empty:
            raise ValueError("No runs found in experiment")
        
        best_run = runs.iloc[0]
        
        return {
            'run_id': best_run['run_id'],
            'run_name': best_run.get('tags.mlflow.runName', 'Unknown'),
            'metric_name': metric,
            'metric_value': best_run[f'metrics.{metric}'],
            'params': {k.replace('params.', ''): v for k, v in best_run.items() if k.startswith('params.')}
        }
    
    def compare_runs(
        self,
        run_ids: Optional[list] = None,
        metric: str = "test_roc_auc"
    ) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare (if None, compares all runs)
            metric: Primary metric for comparison
        
        Returns:
            DataFrame with run comparisons
        """
        if run_ids:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"run_id IN ({','.join(run_ids)})"
            )
        else:
            runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        
        if runs.empty:
            return pd.DataFrame()
        
        # Select relevant columns
        comparison_cols = ['run_id', 'tags.mlflow.runName']
        metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
        param_cols = [col for col in runs.columns if col.startswith('params.')]
        
        comparison = runs[comparison_cols + metric_cols + param_cols].copy()
        comparison = comparison.sort_values(f'metrics.{metric}', ascending=False)
        
        return comparison


def track_experiment(
    experiment_name: str = "credit_scoring",
    tracking_uri: Optional[str] = None,
    model: Optional[Any] = None,
    model_name: Optional[str] = None,
    model_params: Optional[Dict] = None,
    metrics: Optional[Dict[str, float]] = None,
    **kwargs
) -> str:
    """
    Convenience function to track an experiment.
    
    Args:
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking URI (optional)
        model: Trained model
        model_name: Model name
        model_params: Model parameters
        metrics: Evaluation metrics
        **kwargs: Additional arguments for log_model_training
    
    Returns:
        Run ID
    """
    tracker = MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri
    )
    return tracker.log_model_training(
        model=model,
        model_name=model_name,
        model_params=model_params or {},
        metrics=metrics or {},
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    print("=" * 100)
    print("MLflow Experiment Tracking Module")
    print("=" * 100)
    print()
    print("This module provides MLflow integration for experiment tracking.")
    print()
    print("Example usage:")
    print("""
    from src.mlflow_tracking import MLflowTracker
    from src.model_training import train_models
    
    # Create tracker
    tracker = MLflowTracker(experiment_name="credit_scoring")
    
    # Train models and track
    trainer = train_models(X_train, y_train, X_test, y_test)
    
    for model_name, (model, metrics) in trainer.models_.items():
        tracker.log_model_training(
            model=model,
            model_name=model_name,
            model_params={},
            metrics=metrics,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
    
    # Get best model and register
    best_run = tracker.get_best_run(metric='test_roc_auc')
    tracker.register_model(best_run['run_id'], "best_credit_model", stage="Production")
    """)

