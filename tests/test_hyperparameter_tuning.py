"""
Unit tests for Hyperparameter Tuning Module.

Run with: pytest tests/test_hyperparameter_tuning.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hyperparameter_tuning import HyperparameterTuner, tune_hyperparameters


class TestHyperparameterTuner:
    """Tests for HyperparameterTuner class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        # Create target with some relationship to features
        y = pd.Series(
            ((X['feature1'] + X['feature2']) > 0).astype(int)
        )
        
        return X, y
    
    def test_grid_search_logistic_regression(self, sample_data):
        """Test Grid Search for logistic regression."""
        X, y = sample_data
        
        tuner = HyperparameterTuner(random_state=42, cv=3, verbose=0)
        
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        }
        
        grid_search = tuner.grid_search(
            'logistic_regression',
            X, y,
            param_grid=param_grid
        )
        
        assert grid_search is not None
        assert 'logistic_regression' in tuner.best_models_
        assert 'logistic_regression' in tuner.best_params_
        assert 'logistic_regression' in tuner.best_scores_
    
    def test_grid_search_random_forest(self, sample_data):
        """Test Grid Search for random forest."""
        X, y = sample_data
        
        tuner = HyperparameterTuner(random_state=42, cv=3, verbose=0)
        
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [5, 10]
        }
        
        grid_search = tuner.grid_search(
            'random_forest',
            X, y,
            param_grid=param_grid
        )
        
        assert grid_search is not None
        assert 'random_forest' in tuner.best_models_
        assert tuner.best_params_['random_forest']['n_estimators'] in [10, 20]
    
    def test_random_search(self, sample_data):
        """Test Random Search."""
        X, y = sample_data
        
        tuner = HyperparameterTuner(random_state=42, cv=3, verbose=0)
        
        param_distributions = {
            'n_estimators': [10, 20, 50],
            'max_depth': [5, 10, 15]
        }
        
        random_search = tuner.random_search(
            'random_forest',
            X, y,
            param_distributions=param_distributions,
            n_iter=5
        )
        
        assert random_search is not None
        assert 'random_forest' in tuner.best_models_
    
    def test_get_best_model(self, sample_data):
        """Test getting best model."""
        X, y = sample_data
        
        tuner = HyperparameterTuner(random_state=42, cv=3, verbose=0)
        
        param_grid = {'C': [0.1, 1.0]}
        tuner.grid_search('logistic_regression', X, y, param_grid=param_grid)
        
        best_model, best_params, best_score = tuner.get_best_model('logistic_regression')
        
        assert best_model is not None
        assert isinstance(best_params, dict)
        assert isinstance(best_score, (int, float))
        assert best_score > 0
    
    def test_get_results_summary(self, sample_data):
        """Test getting results summary."""
        X, y = sample_data
        
        tuner = HyperparameterTuner(random_state=42, cv=3, verbose=0)
        
        # Tune multiple models
        param_grid_lr = {'C': [0.1, 1.0]}
        tuner.grid_search('logistic_regression', X, y, param_grid=param_grid_lr)
        
        param_grid_rf = {'n_estimators': [10, 20]}
        tuner.grid_search('random_forest', X, y, param_grid=param_grid_rf)
        
        summary = tuner.get_results_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert 'model' in summary.columns
        assert 'best_score' in summary.columns
    
    def test_save_best_model(self, sample_data, tmp_path):
        """Test saving best model."""
        X, y = sample_data
        
        tuner = HyperparameterTuner(random_state=42, cv=3, verbose=0)
        
        param_grid = {'C': [0.1, 1.0]}
        tuner.grid_search('logistic_regression', X, y, param_grid=param_grid)
        
        model_path = tmp_path / "best_model.pkl"
        tuner.save_best_model('logistic_regression', str(model_path))
        
        assert model_path.exists()
    
    def test_invalid_model_name(self, sample_data):
        """Test error handling for invalid model name."""
        X, y = sample_data
        
        tuner = HyperparameterTuner(random_state=42, cv=3, verbose=0)
        
        with pytest.raises(ValueError, match="Unknown model"):
            tuner.grid_search('invalid_model', X, y)
    
    def test_get_best_model_not_tuned(self):
        """Test error when model not tuned."""
        tuner = HyperparameterTuner(random_state=42)
        
        with pytest.raises(ValueError, match="not been tuned"):
            tuner.get_best_model('logistic_regression')
    
    def test_default_param_grids(self):
        """Test default parameter grids."""
        tuner = HyperparameterTuner(random_state=42)
        param_grids = tuner._get_default_param_grids()
        
        assert 'logistic_regression' in param_grids
        assert 'decision_tree' in param_grids
        assert 'random_forest' in param_grids
        assert isinstance(param_grids['logistic_regression'], dict)


class TestTuneHyperparametersFunction:
    """Tests for convenience function tune_hyperparameters."""
    
    def test_tune_hyperparameters_grid_search(self):
        """Test convenience function with grid search."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        y = pd.Series(np.random.choice([0, 1], n_samples))
        
        param_grid = {'C': [0.1, 1.0]}
        
        best_model, best_params, best_score = tune_hyperparameters(
            'logistic_regression',
            X, y,
            method='grid_search',
            param_grid=param_grid,
            cv=3,
            random_state=42
        )
        
        assert best_model is not None
        assert isinstance(best_params, dict)
        assert isinstance(best_score, (int, float))
    
    def test_tune_hyperparameters_random_search(self):
        """Test convenience function with random search."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        y = pd.Series(np.random.choice([0, 1], n_samples))
        
        param_distributions = {'n_estimators': [10, 20, 50]}
        
        best_model, best_params, best_score = tune_hyperparameters(
            'random_forest',
            X, y,
            method='random_search',
            param_grid=param_distributions,
            n_iter=5,
            cv=3,
            random_state=42
        )
        
        assert best_model is not None
        assert isinstance(best_params, dict)
        assert isinstance(best_score, (int, float))
    
    def test_invalid_method(self):
        """Test error for invalid method."""
        np.random.seed(42)
        X = pd.DataFrame({'feature1': np.random.randn(50)})
        y = pd.Series(np.random.choice([0, 1], 50))
        
        with pytest.raises(ValueError, match="Unknown method"):
            tune_hyperparameters(
                'logistic_regression',
                X, y,
                method='invalid_method'
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

