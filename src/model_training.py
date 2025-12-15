"""
Model Training Module for Credit Scoring

This module provides functions to train multiple machine learning models
for credit risk prediction, with MLflow integration for experiment tracking.

Supported Models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import warnings
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Train and evaluate multiple machine learning models for credit scoring.
    
    Supports:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost (if available)
    - LightGBM (if available)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Model Trainer.
        
        Args:
            random_state: Random state for reproducibility (default: 42)
        """
        self.random_state = random_state
        self.models_ = {}
        self.metrics_ = {}
    
    def _create_model(self, model_name: str, **kwargs) -> Any:
        """
        Create a model instance based on model name.
        
        Args:
            model_name: Name of the model ('logistic_regression', 'decision_tree',
                       'random_forest', 'xgboost', 'lightgbm')
            **kwargs: Additional parameters for the model
        
        Returns:
            Model instance
        """
        model_name_lower = model_name.lower()
        
        if model_name_lower == 'logistic_regression':
            # Set default max_iter if not provided
            if 'max_iter' not in kwargs:
                kwargs['max_iter'] = 1000
            return LogisticRegression(
                random_state=self.random_state,
                **kwargs
            )
        
        elif model_name_lower == 'decision_tree':
            return DecisionTreeClassifier(
                random_state=self.random_state,
                **kwargs
            )
        
        elif model_name_lower == 'random_forest':
            return RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                **kwargs
            )
        
        elif model_name_lower == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not available. Install with: pip install xgboost")
            return xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                **kwargs
            )
        
        elif model_name_lower == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
            return lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1,
                **kwargs
            )
        
        else:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Supported models: logistic_regression, decision_tree, random_forest, xgboost, lightgbm"
            )
    
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **model_params
    ) -> Any:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            **model_params: Additional parameters for the model
        
        Returns:
            Trained model
        """
        print(f"Training {model_name}...")
        
        # Create model
        model = self._create_model(model_name, **model_params)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Store model
        self.models_[model_name] = model
        
        print(f"✓ {model_name} training complete")
        
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        set_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate a model and return metrics.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            set_name: Name of the dataset (for logging)
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
        }
        
        # ROC-AUC (requires probability predictions)
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        # Add set name prefix
        metrics = {f'{set_name}_{k}': v for k, v in metrics.items()}
        
        return metrics
    
    def train_and_evaluate(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        **model_params
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train a model and evaluate it on test set.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            **model_params: Additional parameters for the model
        
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        # Train model
        model = self.train_model(model_name, X_train, y_train, **model_params)
        
        # Evaluate on training set
        train_metrics = self.evaluate_model(model, X_train, y_train, set_name='train')
        
        # Evaluate on test set
        test_metrics = self.evaluate_model(model, X_test, y_test, set_name='test')
        
        # Combine metrics
        all_metrics = {**train_metrics, **test_metrics}
        
        # Store metrics
        self.metrics_[model_name] = all_metrics
        
        return model, all_metrics
    
    def train_multiple_models(
        self,
        model_names: list,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_params: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Tuple[Any, Dict[str, float]]]:
        """
        Train multiple models and evaluate them.
        
        Args:
            model_names: List of model names to train
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            model_params: Dictionary mapping model names to their parameters
        
        Returns:
            Dictionary mapping model names to (model, metrics) tuples
        
        Example:
            >>> trainer = ModelTrainer(random_state=42)
            >>> results = trainer.train_multiple_models(
            ...     ['logistic_regression', 'random_forest'],
            ...     X_train, y_train, X_test, y_test
            ... )
        """
        if model_params is None:
            model_params = {}
        
        results = {}
        
        print("=" * 100)
        print("Training Multiple Models")
        print("=" * 100)
        print()
        
        for model_name in model_names:
            try:
                params = model_params.get(model_name, {})
                model, metrics = self.train_and_evaluate(
                    model_name,
                    X_train, y_train,
                    X_test, y_test,
                    **params
                )
                results[model_name] = (model, metrics)
                print()
            except Exception as e:
                print(f"✗ Failed to train {model_name}: {e}")
                print()
        
        return results
    
    def get_best_model(self, metric: str = 'test_roc_auc') -> Tuple[str, Any]:
        """
        Get the best model based on a metric.
        
        Args:
            metric: Metric to use for comparison (default: 'test_roc_auc')
        
        Returns:
            Tuple of (model_name, model)
        """
        if not self.metrics_:
            raise ValueError("No models have been trained yet.")
        
        best_model_name = None
        best_score = -np.inf
        
        for model_name, metrics in self.metrics_.items():
            if metric in metrics:
                score = metrics[metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"Metric '{metric}' not found in any model metrics.")
        
        return best_model_name, self.models_[best_model_name]
    
    def save_model(self, model_name: str, file_path: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            file_path: Path to save the model
        """
        if model_name not in self.models_:
            raise ValueError(f"Model '{model_name}' has not been trained yet.")
        
        joblib.dump(self.models_[model_name], file_path)
        print(f"Model saved to: {file_path}")
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Get summary of all model metrics.
        
        Returns:
            DataFrame with metrics for all models
        """
        if not self.metrics_:
            return pd.DataFrame()
        
        metrics_list = []
        for model_name, metrics in self.metrics_.items():
            row = {'model': model_name}
            row.update(metrics)
            metrics_list.append(row)
        
        return pd.DataFrame(metrics_list)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_names: Optional[list] = None,
    random_state: int = 42,
    model_params: Optional[Dict[str, Dict]] = None
) -> ModelTrainer:
    """
    Convenience function to train multiple models.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_names: List of models to train (default: ['logistic_regression', 'random_forest'])
        random_state: Random state for reproducibility
        model_params: Dictionary mapping model names to their parameters
    
    Returns:
        ModelTrainer instance with trained models
    
    Example:
        >>> trainer = train_models(
        ...     X_train, y_train, X_test, y_test,
        ...     model_names=['logistic_regression', 'random_forest', 'xgboost']
        ... )
        >>> summary = trainer.get_metrics_summary()
        >>> print(summary)
    """
    if model_names is None:
        model_names = ['logistic_regression', 'random_forest']
    
    trainer = ModelTrainer(random_state=random_state)
    trainer.train_multiple_models(
        model_names,
        X_train, y_train,
        X_test, y_test,
        model_params=model_params
    )
    
    return trainer


if __name__ == "__main__":
    # Example usage
    print("=" * 100)
    print("Model Training Module - Example Usage")
    print("=" * 100)
    print()
    
    print("This module provides functions to train multiple ML models for credit scoring.")
    print()
    print("Example usage:")
    print("""
    from src.model_training import train_models
    from src.data_splitting import load_splits
    
    # Load data splits
    X_train, X_test, y_train, y_test = load_splits('data/processed/splits')
    
    # Train multiple models
    trainer = train_models(
        X_train, y_train, X_test, y_test,
        model_names=['logistic_regression', 'decision_tree', 'random_forest', 'xgboost'],
        random_state=42
    )
    
    # Get metrics summary
    summary = trainer.get_metrics_summary()
    print(summary)
    
    # Get best model
    best_name, best_model = trainer.get_best_model(metric='test_roc_auc')
    print(f"Best model: {best_name}")
    """)

