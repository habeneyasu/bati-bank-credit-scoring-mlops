"""
Hyperparameter Tuning Module for Credit Scoring Models

This module provides Grid Search and Random Search capabilities for
optimizing model hyperparameters.

Features:
- Grid Search: Exhaustive search over specified parameter grid
- Random Search: Random sampling from parameter distribution
- Cross-validation for robust evaluation
- Integration with existing model training infrastructure
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
import joblib
from pathlib import Path

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, f1_score

# Import model classes
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    Hyperparameter tuning using Grid Search or Random Search.
    
    Supports tuning for:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost (if available)
    - LightGBM (if available)
    """
    
    def __init__(
        self,
        random_state: int = 42,
        cv: int = 5,
        scoring: str = 'roc_auc',
        n_jobs: int = -1,
        verbose: int = 1
    ):
        """
        Initialize Hyperparameter Tuner.
        
        Args:
            random_state: Random state for reproducibility
            cv: Number of cross-validation folds (default: 5)
            scoring: Scoring metric (default: 'roc_auc')
            n_jobs: Number of parallel jobs (default: -1 for all cores)
            verbose: Verbosity level (default: 1)
        """
        self.random_state = random_state
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.best_models_ = {}
        self.best_params_ = {}
        self.best_scores_ = {}
        self.search_results_ = {}
    
    def _get_default_param_grids(self) -> Dict[str, Dict]:
        """
        Get default parameter grids for each model type.
        
        Returns:
            Dictionary mapping model names to parameter grids
        """
        param_grids = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'decision_tree': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        }
        
        if XGBOOST_AVAILABLE:
            param_grids['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        if LIGHTGBM_AVAILABLE:
            param_grids['lightgbm'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        return param_grids
    
    def _get_model_class(self, model_name: str):
        """Get the model class for a given model name."""
        model_name_lower = model_name.lower()
        
        if model_name_lower == 'logistic_regression':
            return LogisticRegression
        elif model_name_lower == 'decision_tree':
            return DecisionTreeClassifier
        elif model_name_lower == 'random_forest':
            return RandomForestClassifier
        elif model_name_lower == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not available")
            return xgb.XGBClassifier
        elif model_name_lower == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not available")
            return lgb.LGBMClassifier
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def grid_search(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """
        Perform Grid Search for hyperparameter tuning.
        
        Args:
            model_name: Name of the model to tune
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid to search (if None, uses default)
            **kwargs: Additional arguments for GridSearchCV
        
        Returns:
            Fitted GridSearchCV object with best model
        
        Example:
            >>> tuner = HyperparameterTuner(random_state=42)
            >>> grid_search = tuner.grid_search(
            ...     'random_forest',
            ...     X_train, y_train,
            ...     param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10]}
            ... )
        """
        print(f"Performing Grid Search for {model_name}...")
        print(f"  CV folds: {self.cv}")
        print(f"  Scoring: {self.scoring}")
        
        # Get model class
        model_class = self._get_model_class(model_name)
        
        # Get parameter grid
        if param_grid is None:
            default_grids = self._get_default_param_grids()
            param_grid = default_grids.get(model_name, {})
            if not param_grid:
                raise ValueError(f"No default parameter grid for {model_name}")
        
        # Create base model with fixed parameters
        base_params = {'random_state': self.random_state}
        
        # Handle model-specific parameters
        if model_name.lower() == 'logistic_regression':
            base_params['max_iter'] = 1000
        elif model_name.lower() == 'xgboost':
            base_params['eval_metric'] = 'logloss'
        elif model_name.lower() == 'lightgbm':
            base_params['verbose'] = -1
        
        base_model = model_class(**base_params)
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=True,
            **kwargs
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Store results
        self.best_models_[model_name] = grid_search.best_estimator_
        self.best_params_[model_name] = grid_search.best_params_
        self.best_scores_[model_name] = grid_search.best_score_
        self.search_results_[model_name] = grid_search
        
        print(f"✓ Grid Search complete for {model_name}")
        print(f"  Best score: {grid_search.best_score_:.4f}")
        print(f"  Best parameters: {grid_search.best_params_}")
        
        return grid_search
    
    def random_search(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_distributions: Optional[Dict] = None,
        n_iter: int = 20,
        **kwargs
    ) -> Any:
        """
        Perform Random Search for hyperparameter tuning.
        
        Args:
            model_name: Name of the model to tune
            X_train: Training features
            y_train: Training target
            param_distributions: Parameter distributions to sample from
            n_iter: Number of parameter settings sampled (default: 20)
            **kwargs: Additional arguments for RandomizedSearchCV
        
        Returns:
            Fitted RandomizedSearchCV object with best model
        
        Example:
            >>> tuner = HyperparameterTuner(random_state=42)
            >>> random_search = tuner.random_search(
            ...     'random_forest',
            ...     X_train, y_train,
            ...     n_iter=50
            ... )
        """
        print(f"Performing Random Search for {model_name}...")
        print(f"  CV folds: {self.cv}")
        print(f"  Scoring: {self.scoring}")
        print(f"  Iterations: {n_iter}")
        
        # Get model class
        model_class = self._get_model_class(model_name)
        
        # Get parameter distributions
        if param_distributions is None:
            # Convert default grids to distributions for random search
            default_grids = self._get_default_param_grids()
            param_distributions = default_grids.get(model_name, {})
            if not param_distributions:
                raise ValueError(f"No default parameter grid for {model_name}")
        
        # Create base model with fixed parameters
        base_params = {'random_state': self.random_state}
        
        # Handle model-specific parameters
        if model_name.lower() == 'logistic_regression':
            base_params['max_iter'] = 1000
        elif model_name.lower() == 'xgboost':
            base_params['eval_metric'] = 'logloss'
        elif model_name.lower() == 'lightgbm':
            base_params['verbose'] = -1
        
        base_model = model_class(**base_params)
        
        # Create RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            return_train_score=True,
            **kwargs
        )
        
        # Fit random search
        random_search.fit(X_train, y_train)
        
        # Store results
        self.best_models_[model_name] = random_search.best_estimator_
        self.best_params_[model_name] = random_search.best_params_
        self.best_scores_[model_name] = random_search.best_score_
        self.search_results_[model_name] = random_search
        
        print(f"✓ Random Search complete for {model_name}")
        print(f"  Best score: {random_search.best_score_:.4f}")
        print(f"  Best parameters: {random_search.best_params_}")
        
        return random_search
    
    def get_best_model(self, model_name: str) -> Tuple[Any, Dict, float]:
        """
        Get the best model, parameters, and score for a given model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        if model_name not in self.best_models_:
            raise ValueError(f"Model {model_name} has not been tuned yet.")
        
        return (
            self.best_models_[model_name],
            self.best_params_[model_name],
            self.best_scores_[model_name]
        )
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary of all tuning results.
        
        Returns:
            DataFrame with tuning results for all models
        """
        if not self.best_models_:
            return pd.DataFrame()
        
        results = []
        for model_name in self.best_models_.keys():
            results.append({
                'model': model_name,
                'best_score': self.best_scores_[model_name],
                'best_params': str(self.best_params_[model_name])
            })
        
        return pd.DataFrame(results)
    
    def save_best_model(self, model_name: str, file_path: str):
        """Save the best tuned model to disk."""
        if model_name not in self.best_models_:
            raise ValueError(f"Model {model_name} has not been tuned yet.")
        
        joblib.dump(self.best_models_[model_name], file_path)
        print(f"Best model saved to: {file_path}")


def tune_hyperparameters(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'grid_search',  # 'grid_search' or 'random_search'
    param_grid: Optional[Dict] = None,
    n_iter: int = 20,  # For random search
    cv: int = 5,
    scoring: str = 'roc_auc',
    random_state: int = 42
) -> Tuple[Any, Dict, float]:
    """
    Convenience function to tune hyperparameters for a model.
    
    Args:
        model_name: Name of the model to tune
        X_train: Training features
        y_train: Training target
        method: Tuning method ('grid_search' or 'random_search')
        param_grid: Parameter grid/distributions
        n_iter: Number of iterations for random search
        cv: Number of CV folds
        scoring: Scoring metric
        random_state: Random state
    
    Returns:
        Tuple of (best_model, best_params, best_score)
    
    Example:
        >>> best_model, best_params, best_score = tune_hyperparameters(
        ...     'random_forest',
        ...     X_train, y_train,
        ...     method='grid_search'
        ... )
    """
    tuner = HyperparameterTuner(
        random_state=random_state,
        cv=cv,
        scoring=scoring
    )
    
    if method == 'grid_search':
        search_result = tuner.grid_search(
            model_name,
            X_train, y_train,
            param_grid=param_grid
        )
    elif method == 'random_search':
        search_result = tuner.random_search(
            model_name,
            X_train, y_train,
            param_distributions=param_grid,
            n_iter=n_iter
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'grid_search' or 'random_search'")
    
    return tuner.get_best_model(model_name)


if __name__ == "__main__":
    # Example usage
    print("=" * 100)
    print("Hyperparameter Tuning Module - Example Usage")
    print("=" * 100)
    print()
    print("This module provides Grid Search and Random Search for hyperparameter tuning.")
    print()
    print("Example usage:")
    print("""
    from src.hyperparameter_tuning import HyperparameterTuner
    from src.data_splitting import load_splits
    
    # Load data splits
    X_train, X_test, y_train, y_test = load_splits('data/processed/splits')
    
    # Create tuner
    tuner = HyperparameterTuner(random_state=42, cv=5, scoring='roc_auc')
    
    # Grid Search
    grid_search = tuner.grid_search('random_forest', X_train, y_train)
    
    # Random Search
    random_search = tuner.random_search('random_forest', X_train, y_train, n_iter=50)
    
    # Get best model
    best_model, best_params, best_score = tuner.get_best_model('random_forest')
    print(f"Best score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    """)

