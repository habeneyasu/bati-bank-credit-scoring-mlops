"""
Pytest configuration and shared fixtures for all tests.

This module provides common fixtures used across test suites.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import tempfile
import shutil
from typing import Generator
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable MLflow telemetry to prevent logging errors in tests
# This must be set before MLflow is imported
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"


@pytest.fixture(scope="session", autouse=True)
def disable_mlflow_telemetry():
    """Disable MLflow telemetry for all tests to prevent logging errors."""
    # Ensure environment variable is set
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    yield


def pytest_configure(config):
    """Configure pytest to suppress MLflow telemetry logging errors."""
    import logging
    import sys
    
    # Patch the logging module's Handler.emit to catch ValueError from closed files
    original_emit = logging.Handler.emit
    
    def safe_emit(self, record):
        """Safe emit that catches ValueError from closed file handles."""
        try:
            return original_emit(self, record)
        except ValueError as e:
            # Suppress errors about closed file handles (common with MLflow telemetry)
            if "I/O operation on closed file" in str(e):
                # Check if it's from MLflow telemetry by looking at the call stack
                import traceback
                tb = traceback.format_exc()
                if "mlflow.telemetry" in tb or "mlflow/telemetry" in tb:
                    return  # Suppress MLflow telemetry errors
            # Re-raise other ValueErrors
            raise
        except OSError as e:
            # Suppress OSError from closed files
            if "closed file" in str(e).lower():
                return
            raise
    
    # Only patch if not already patched
    if logging.Handler.emit == original_emit:
        logging.Handler.emit = safe_emit


def pytest_runtest_setup(item):
    """Setup hook to ensure MLflow telemetry is disabled."""
    # Ensure environment variable is set before each test
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"

# Lazy imports to avoid import errors when dependencies aren't installed
# These will be imported only when fixtures are used
try:
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.utils.config import Settings
    _API_AVAILABLE = True
except ImportError as e:
    _API_AVAILABLE = False
    # Create dummy objects for type hints
    TestClient = None
    app = None
    Settings = None


@pytest.fixture
def sample_transaction_data() -> pd.DataFrame:
    """Create sample transaction data for testing."""
    np.random.seed(42)
    base_date = pd.Timestamp('2019-02-13')
    
    n_transactions = 100
    data = []
    for i in range(n_transactions):
        data.append({
            'CustomerId': f'C{i % 10}',
            'TransactionStartTime': base_date - pd.Timedelta(days=np.random.randint(0, 90)),
            'Amount': np.random.normal(1000, 500),
            'Category': np.random.choice(['A', 'B', 'C']),
            'Value': np.random.normal(1200, 600)
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_rfm_data() -> pd.DataFrame:
    """Create sample RFM data for testing."""
    np.random.seed(42)
    
    data = []
    for i in range(20):
        data.append({
            'CustomerId': f'C{i}',
            'recency': np.random.uniform(0, 90),
            'frequency': np.random.randint(1, 50),
            'monetary': np.random.uniform(100, 100000)
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample training data (X, y) for model testing."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.randn(n_samples),
        'feature5': np.random.randn(n_samples)
    })
    
    # Create target with some relationship to features
    y = pd.Series(
        ((X['feature1'] + X['feature2']) > 0).astype(int)
    )
    
    return X, y


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_mlflow_dir(temp_dir: Path) -> str:
    """Create a temporary MLflow tracking directory."""
    mlflow_dir = temp_dir / "mlruns"
    mlflow_dir.mkdir()
    return str(mlflow_dir)


@pytest.fixture
def api_client():
    """Create a test client for the API."""
    if not _API_AVAILABLE:
        pytest.skip("API dependencies not installed. Run: pip install -r requirements.txt")
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict = Mock(return_value=np.array([0]))
    model.predict_proba = Mock(return_value=np.array([[0.85, 0.15]]))
    return model


@pytest.fixture
def sample_prediction_request() -> dict:
    """Create a sample prediction request payload."""
    return {
        "features": [0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 
                    0.849, -0.994, -0.006, 0.853, 0.170, -0.068, -0.312, 
                    -0.167, 0.164, -0.193, -0.025, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0]
    }


@pytest.fixture
def test_settings():
    """Create test settings with overrides."""
    if not _API_AVAILABLE:
        pytest.skip("API dependencies not installed. Run: pip install -r requirements.txt")
    from src.utils.config import Settings
    return Settings(
        environment="testing",
        mlflow_tracking_uri="file:./test_mlruns",
        enable_rate_limiting=False,
        log_level="DEBUG"
    )
