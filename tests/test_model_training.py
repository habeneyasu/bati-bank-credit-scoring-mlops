"""
Unit tests for Model Training Module.

Run with: pytest tests/test_model_training.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_training import ModelTrainer, train_models


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
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
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def test_train_logistic_regression(self, sample_data):
        """Test training logistic regression model."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_state=42)
        model = trainer.train_model('logistic_regression', X_train, y_train)
        
        assert model is not None
        assert 'logistic_regression' in trainer.models_
        assert hasattr(model, 'predict')
    
    def test_train_decision_tree(self, sample_data):
        """Test training decision tree model."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_state=42)
        model = trainer.train_model('decision_tree', X_train, y_train)
        
        assert model is not None
        assert 'decision_tree' in trainer.models_
    
    def test_train_random_forest(self, sample_data):
        """Test training random forest model."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_state=42)
        model = trainer.train_model('random_forest', X_train, y_train)
        
        assert model is not None
        assert 'random_forest' in trainer.models_
    
    def test_evaluate_model(self, sample_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_state=42)
        model = trainer.train_model('logistic_regression', X_train, y_train)
        
        metrics = trainer.evaluate_model(model, X_test, y_test, set_name='test')
        
        assert 'test_accuracy' in metrics
        assert 'test_precision' in metrics
        assert 'test_recall' in metrics
        assert 'test_f1_score' in metrics
        assert 0 <= metrics['test_accuracy'] <= 1
    
    def test_train_and_evaluate(self, sample_data):
        """Test train and evaluate workflow."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_state=42)
        model, metrics = trainer.train_and_evaluate(
            'logistic_regression',
            X_train, y_train,
            X_test, y_test
        )
        
        assert model is not None
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' in metrics
        assert 'logistic_regression' in trainer.metrics_
    
    def test_train_multiple_models(self, sample_data):
        """Test training multiple models."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_state=42)
        results = trainer.train_multiple_models(
            ['logistic_regression', 'decision_tree', 'random_forest'],
            X_train, y_train,
            X_test, y_test
        )
        
        assert len(results) == 3
        assert 'logistic_regression' in results
        assert 'decision_tree' in results
        assert 'random_forest' in results
        
        # Check that all models are stored
        assert len(trainer.models_) == 3
        assert len(trainer.metrics_) == 3
    
    def test_get_best_model(self, sample_data):
        """Test getting best model."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_state=42)
        trainer.train_multiple_models(
            ['logistic_regression', 'random_forest'],
            X_train, y_train,
            X_test, y_test
        )
        
        best_name, best_model = trainer.get_best_model(metric='test_roc_auc')
        
        assert best_name in ['logistic_regression', 'random_forest']
        assert best_model is not None
    
    def test_get_metrics_summary(self, sample_data):
        """Test getting metrics summary."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_state=42)
        trainer.train_multiple_models(
            ['logistic_regression', 'random_forest'],
            X_train, y_train,
            X_test, y_test
        )
        
        summary = trainer.get_metrics_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert 'model' in summary.columns
        assert 'test_accuracy' in summary.columns
    
    def test_save_model(self, sample_data, tmp_path):
        """Test saving model."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_state=42)
        trainer.train_model('logistic_regression', X_train, y_train)
        
        model_path = tmp_path / "model.pkl"
        trainer.save_model('logistic_regression', str(model_path))
        
        assert model_path.exists()
    
    def test_invalid_model_name(self, sample_data):
        """Test error handling for invalid model name."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_state=42)
        
        with pytest.raises(ValueError, match="Unknown model"):
            trainer.train_model('invalid_model', X_train, y_train)
    
    def test_get_best_model_no_models(self):
        """Test error when no models trained."""
        trainer = ModelTrainer(random_state=42)
        
        with pytest.raises(ValueError, match="No models"):
            trainer.get_best_model()
    
    def test_custom_model_parameters(self, sample_data):
        """Test training with custom parameters."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = ModelTrainer(random_state=42)
        
        # Train with custom parameters
        model = trainer.train_model(
            'decision_tree',
            X_train, y_train,
            max_depth=5,
            min_samples_split=10
        )
        
        assert model.max_depth == 5
        assert model.min_samples_split == 10


class TestTrainModelsFunction:
    """Tests for convenience function train_models."""
    
    def test_train_models_function(self):
        """Test the convenience function."""
        np.random.seed(42)
        n_samples = 100
        
        X_train = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        y_train = pd.Series(np.random.choice([0, 1], n_samples))
        
        X_test = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20)
        })
        y_test = pd.Series(np.random.choice([0, 1], 20))
        
        trainer = train_models(
            X_train, y_train,
            X_test, y_test,
            model_names=['logistic_regression', 'decision_tree'],
            random_state=42
        )
        
        assert len(trainer.models_) == 2
        assert 'logistic_regression' in trainer.models_
        assert 'decision_tree' in trainer.models_


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

