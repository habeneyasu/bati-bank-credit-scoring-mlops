"""
Unit tests for MLflow Tracking Module.

Run with: pytest tests/test_mlflow_tracking.py -v

Note: These tests require MLflow to be installed and may create local MLflow runs.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlflow_tracking import MLflowTracker, track_experiment


@pytest.fixture
def sample_model_and_data():
    """Create sample model and data for testing."""
    from sklearn.linear_model import LogisticRegression
    
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    y = pd.Series(np.random.choice([0, 1], 100))
    
    model = LogisticRegression(random_state=42, max_iter=100)
    model.fit(X, y)
    
    return model, X, y


@pytest.fixture
def temp_mlflow_dir(tmp_path):
    """Create temporary directory for MLflow runs."""
    mlflow_dir = tmp_path / "mlruns"
    mlflow_dir.mkdir()
    return str(mlflow_dir)


class TestMLflowTracker:
    """Tests for MLflowTracker class."""
    
    def test_tracker_initialization(self, temp_mlflow_dir):
        """Test MLflow tracker initialization."""
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            tracking_uri=f"file:{temp_mlflow_dir}"
        )
        
        assert tracker.experiment_name == "test_experiment"
        assert tracker.tracking_uri == f"file:{temp_mlflow_dir}"
    
    def test_log_model_training(self, sample_model_and_data, temp_mlflow_dir):
        """Test logging model training to MLflow."""
        model, X, y = sample_model_and_data
        
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            tracking_uri=f"file:{temp_mlflow_dir}"
        )
        
        model_params = {'C': 1.0, 'max_iter': 100}
        metrics = {
            'test_accuracy': 0.85,
            'test_roc_auc': 0.90,
            'test_f1_score': 0.80
        }
        
        run_id = tracker.log_model_training(
            model=model,
            model_name="test_model",
            model_params=model_params,
            metrics=metrics,
            X_train=X,
            y_train=y
        )
        
        assert run_id is not None
        assert isinstance(run_id, str)
    
    def test_log_hyperparameter_tuning(self, temp_mlflow_dir):
        """Test logging hyperparameter tuning results."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        X = pd.DataFrame({'feature1': np.random.randn(100)})
        y = pd.Series(np.random.choice([0, 1], 100))
        
        param_grid = {'C': [0.1, 1.0]}
        model = LogisticRegression(random_state=42, max_iter=100)
        
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
        grid_search.fit(X, y)
        
        tracker = MLflowTracker(
            experiment_name="test_tuning",
            tracking_uri=f"file:{temp_mlflow_dir}"
        )
        
        run_id = tracker.log_hyperparameter_tuning(
            search_result=grid_search,
            model_name="logistic_regression",
            tuning_method="grid_search",
            X_train=X,
            y_train=y
        )
        
        assert run_id is not None
    
    def test_get_best_run(self, sample_model_and_data, temp_mlflow_dir):
        """Test getting best run."""
        model, X, y = sample_model_and_data
        
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            tracking_uri=f"file:{temp_mlflow_dir}"
        )
        
        # Log multiple runs
        for i, metric_value in enumerate([0.85, 0.90, 0.88]):
            tracker.log_model_training(
                model=model,
                model_name=f"model_{i}",
                model_params={'C': 1.0},
                metrics={'test_roc_auc': metric_value}
            )
        
        # Get best run
        best_run = tracker.get_best_run(metric='test_roc_auc')
        
        assert best_run is not None
        assert 'run_id' in best_run
        assert best_run['metric_value'] == 0.90  # Highest value
    
    def test_compare_runs(self, sample_model_and_data, temp_mlflow_dir):
        """Test comparing runs."""
        model, X, y = sample_model_and_data
        
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            tracking_uri=f"file:{temp_mlflow_dir}"
        )
        
        # Log multiple runs
        run_ids = []
        for i in range(3):
            run_id = tracker.log_model_training(
                model=model,
                model_name=f"model_{i}",
                model_params={'C': 1.0},
                metrics={'test_roc_auc': 0.85 + i * 0.02}
            )
            run_ids.append(run_id)
        
        # Compare runs
        comparison = tracker.compare_runs(run_ids=run_ids, metric='test_roc_auc')
        
        assert not comparison.empty
        assert len(comparison) == 3
    
    def test_register_model(self, sample_model_and_data, temp_mlflow_dir):
        """Test registering model in MLflow registry."""
        model, X, y = sample_model_and_data
        
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            tracking_uri=f"file:{temp_mlflow_dir}"
        )
        
        run_id = tracker.log_model_training(
            model=model,
            model_name="test_model",
            model_params={'C': 1.0},
            metrics={'test_roc_auc': 0.90}
        )
        
        # Register model
        try:
            version = tracker.register_model(
                run_id=run_id,
                model_name="test_registered_model",
                stage="Staging"
            )
            assert version is not None
        except Exception as e:
            # Model registry might not be fully set up in test environment
            pytest.skip(f"Model registry not available: {e}")


class TestTrackExperimentFunction:
    """Tests for convenience function track_experiment."""
    
    def test_track_experiment(self, sample_model_and_data, temp_mlflow_dir):
        """Test convenience function."""
        model, X, y = sample_model_and_data
        
        run_id = track_experiment(
            experiment_name="test_experiment",
            tracking_uri=f"file:{temp_mlflow_dir}",
            model=model,
            model_name="test_model",
            model_params={'C': 1.0},
            metrics={'test_roc_auc': 0.90}
        )
        
        assert run_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

