"""
Unit tests for model explainability (src/models/explainability.py).

Tests verify SHAP explanation generation, feature importance, and visualization.

Run with: pytest tests/test_models_explainability.py -v
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    pytest.skip("SHAP not available", allow_module_level=True)

from src.models.explainability import ModelExplainer, create_explainer


class TestModelExplainer:
    """Tests for ModelExplainer class."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a mock tree-based model."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create simple training data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def sample_background_data(self):
        """Create sample background data for SHAP."""
        np.random.seed(42)
        return np.random.randn(50, 5)
    
    @pytest.fixture
    def feature_names(self):
        """Create feature names."""
        return [f"feature_{i}" for i in range(5)]
    
    @pytest.fixture
    def explainer(self, sample_model, sample_background_data, feature_names):
        """Create ModelExplainer instance."""
        return ModelExplainer(
            model=sample_model,
            background_data=sample_background_data,
            feature_names=feature_names,
            explainer_type="tree"
        )
    
    def test_explainer_initialization(self, sample_model, sample_background_data, feature_names):
        """Test that explainer initializes correctly."""
        explainer = ModelExplainer(
            model=sample_model,
            background_data=sample_background_data,
            feature_names=feature_names
        )
        
        assert explainer.model is sample_model
        assert explainer.explainer is not None
        assert len(explainer.feature_names) == 5
    
    def test_explainer_initialization_without_background(self, sample_model, feature_names):
        """Test explainer initialization without background data (tree models)."""
        explainer = ModelExplainer(
            model=sample_model,
            feature_names=feature_names,
            explainer_type="tree"
        )
        
        assert explainer.explainer is not None
    
    def test_explainer_auto_detection(self, sample_model, sample_background_data, feature_names):
        """Test automatic explainer type detection."""
        explainer = ModelExplainer(
            model=sample_model,
            background_data=sample_background_data,
            feature_names=feature_names,
            explainer_type="auto"
        )
        
        # Should detect tree-based model
        assert explainer.explainer is not None
    
    def test_explain_instance(self, explainer):
        """Test explaining a single instance."""
        instance = np.array([[0.5, -0.3, 0.1, 0.2, -0.1]])
        
        explanation = explainer.explain_instance(instance)
        
        assert "shap_values" in explanation
        assert "base_value" in explanation
        assert "feature_names" in explanation
        assert "feature_importance" in explanation
        assert "prediction" in explanation
        assert "probability" in explanation
        assert "explanation_summary" in explanation
        
        assert len(explanation["shap_values"]) == 5
        assert len(explanation["feature_importance"]) == 5
        assert explanation["prediction"] in [0, 1]
        assert 0 <= explanation["probability"] <= 1
    
    def test_explain_instance_feature_importance_ordering(self, explainer):
        """Test that feature importance is sorted by absolute SHAP value."""
        instance = np.array([[0.5, -0.3, 0.1, 0.2, -0.1]])
        
        explanation = explainer.explain_instance(instance)
        
        # Check that feature importance is sorted by absolute value
        abs_values = [abs(feat["shap_value"]) for feat in explanation["feature_importance"]]
        assert abs_values == sorted(abs_values, reverse=True)
    
    def test_explain_batch(self, explainer):
        """Test explaining multiple instances."""
        instances = np.array([
            [0.5, -0.3, 0.1, 0.2, -0.1],
            [-0.2, 0.4, -0.1, 0.3, 0.2]
        ])
        
        explanations = explainer.explain_batch(instances)
        
        assert len(explanations) == 2
        assert all("shap_values" in exp for exp in explanations)
    
    def test_get_feature_importance_global(self, explainer, sample_background_data):
        """Test global feature importance calculation."""
        importance = explainer.get_feature_importance_global(
            sample_data=sample_background_data,
            n_samples=30
        )
        
        assert "feature_importance" in importance
        assert "n_samples" in importance
        assert len(importance["feature_importance"]) == 5
        assert importance["n_samples"] <= 30
    
    def test_explanation_summary_format(self, explainer):
        """Test that explanation summary is human-readable."""
        instance = np.array([[0.5, -0.3, 0.1, 0.2, -0.1]])
        
        explanation = explainer.explain_instance(instance)
        
        summary = explanation["explanation_summary"]
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Prediction:" in summary
        assert "Key factors:" in summary
    
    def test_plot_waterfall(self, explainer):
        """Test waterfall plot generation."""
        instance = np.array([[0.5, -0.3, 0.1, 0.2, -0.1]])
        
        plot_base64 = explainer.plot_waterfall(instance)
        
        # Plot may be None if matplotlib not available, but shouldn't raise error
        if plot_base64 is not None:
            assert isinstance(plot_base64, str)
            assert len(plot_base64) > 0
    
    def test_plot_summary(self, explainer, sample_background_data):
        """Test summary plot generation."""
        plot_base64 = explainer.plot_summary(
            sample_data=sample_background_data,
            n_samples=20
        )
        
        # Plot may be None if matplotlib not available
        if plot_base64 is not None:
            assert isinstance(plot_base64, str)
            assert len(plot_base64) > 0
    
    def test_explainer_with_linear_model(self):
        """Test explainer with linear model."""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        explainer = ModelExplainer(
            model=model,
            background_data=X[:50],
            feature_names=[f"feature_{i}" for i in range(5)],
            explainer_type="linear"
        )
        
        instance = np.array([[0.5, -0.3, 0.1, 0.2, -0.1]])
        explanation = explainer.explain_instance(instance)
        
        assert "shap_values" in explanation
        assert len(explanation["shap_values"]) == 5
    
    def test_explainer_with_kernel_explainer(self):
        """Test explainer with kernel explainer (fallback)."""
        from sklearn.svm import SVC
        
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = SVC(probability=True, random_state=42)
        model.fit(X, y)
        
        explainer = ModelExplainer(
            model=model,
            background_data=X[:20],
            feature_names=[f"feature_{i}" for i in range(5)],
            explainer_type="kernel"
        )
        
        instance = np.array([[0.5, -0.3, 0.1, 0.2, -0.1]])
        explanation = explainer.explain_instance(instance, max_evals=10)
        
        assert "shap_values" in explanation
    
    def test_explainer_missing_shap_raises_error(self):
        """Test that missing SHAP library raises appropriate error."""
        with patch('src.models.explainability.SHAP_AVAILABLE', False):
            with pytest.raises(ImportError, match="SHAP library is required"):
                ModelExplainer(
                    model=Mock(),
                    feature_names=["feature_1"]
                )


class TestCreateExplainer:
    """Tests for create_explainer factory function."""
    
    def test_create_explainer(self):
        """Test factory function."""
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = (X[:, 0] > 0).astype(int)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        explainer = create_explainer(
            model=model,
            background_data=X[:20],
            feature_names=[f"feature_{i}" for i in range(5)]
        )
        
        assert isinstance(explainer, ModelExplainer)
        assert explainer.model is model
