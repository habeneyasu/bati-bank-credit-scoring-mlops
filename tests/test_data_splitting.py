"""
Unit tests for Data Splitting Module.

Run with: pytest tests/test_data_splitting.py -v
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

from src.data_splitting import (
    split_data,
    split_data_from_file,
    load_splits,
    get_split_summary
)


class TestSplitData:
    """Tests for split_data function."""
    
    def test_basic_split(self):
        """Test basic train/test split."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        X_train, X_test, y_train, y_test = split_data(
            X, y,
            test_size=0.2,
            random_state=42
        )
        
        # Check shapes
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        
        # Check that all data is used
        assert len(X_train) + len(X_test) == len(X)
    
    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        np.random.seed(42)
        X = pd.DataFrame({'feature1': np.random.randn(100)})
        y = pd.Series(np.random.choice([0, 1], 100))
        
        # Split twice with same random_state
        splits1 = split_data(X, y, test_size=0.2, random_state=42)
        splits2 = split_data(X, y, test_size=0.2, random_state=42)
        
        X_train1, X_test1, y_train1, y_test1 = splits1
        X_train2, X_test2, y_train2, y_test2 = splits2
        
        # Should produce identical splits
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)
    
    def test_stratified_split(self):
        """Test stratified splitting maintains class distribution."""
        # Create imbalanced dataset
        X = pd.DataFrame({'feature1': range(100)})
        y = pd.Series([0] * 80 + [1] * 20)  # 80% class 0, 20% class 1
        
        X_train, X_test, y_train, y_test = split_data(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=True
        )
        
        # Check that class distribution is maintained
        train_pct = y_train.sum() / len(y_train)
        test_pct = y_test.sum() / len(y_test)
        original_pct = y.sum() / len(y)
        
        # Should be approximately the same
        assert abs(train_pct - original_pct) < 0.05
        assert abs(test_pct - original_pct) < 0.05
    
    def test_train_val_test_split(self):
        """Test train/validation/test split."""
        np.random.seed(42)
        X = pd.DataFrame({'feature1': np.random.randn(100)})
        y = pd.Series(np.random.choice([0, 1], 100))
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y,
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        
        # Check shapes (approximately)
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(X_test) == 20  # 20% of 100
        assert len(X_val) == 10   # 10% of 100
        assert len(X_train) == 70  # Remaining
    
    def test_no_stratify(self):
        """Test split without stratification."""
        X = pd.DataFrame({'feature1': range(100)})
        y = pd.Series([0] * 80 + [1] * 20)
        
        X_train, X_test, y_train, y_test = split_data(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=False
        )
        
        # Should still work
        assert len(X_train) + len(X_test) == len(X)
    
    def test_validation_errors(self):
        """Test that validation errors are raised appropriately."""
        X = pd.DataFrame({'feature1': range(100)})
        y = pd.Series(range(100))
        
        # Different lengths
        with pytest.raises(ValueError, match="same length"):
            split_data(X, y[:50], test_size=0.2)
        
        # Wrong types
        with pytest.raises(TypeError, match="DataFrame"):
            split_data([1, 2, 3], y, test_size=0.2)
        
        with pytest.raises(TypeError, match="Series"):
            split_data(X, [1, 2, 3], test_size=0.2)


class TestSplitDataFromFile:
    """Tests for split_data_from_file function."""
    
    @pytest.fixture
    def sample_data_file(self, tmp_path):
        """Create a sample data file for testing."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'is_high_risk': np.random.choice([0, 1], 100)
        })
        
        file_path = tmp_path / "test_data.csv"
        df.to_csv(file_path, index=False)
        return str(file_path)
    
    def test_split_from_file(self, sample_data_file, tmp_path):
        """Test splitting data from file."""
        output_dir = tmp_path / "splits"
        
        splits = split_data_from_file(
            sample_data_file,
            target_col='is_high_risk',
            test_size=0.2,
            random_state=42,
            save_splits=True,
            output_dir=str(output_dir)
        )
        
        X_train, X_test, y_train, y_test = splits
        
        # Check splits
        assert len(X_train) + len(X_test) == 100
        assert 'is_high_risk' not in X_train.columns
        assert 'is_high_risk' not in X_test.columns
        
        # Check files were saved
        assert (output_dir / "X_train.csv").exists()
        assert (output_dir / "X_test.csv").exists()
        assert (output_dir / "y_train.csv").exists()
        assert (output_dir / "y_test.csv").exists()
    
    def test_missing_target_column(self, sample_data_file):
        """Test error when target column is missing."""
        with pytest.raises(ValueError, match="not found"):
            split_data_from_file(
                sample_data_file,
                target_col='nonexistent',
                test_size=0.2
            )


class TestLoadSplits:
    """Tests for load_splits function."""
    
    @pytest.fixture
    def sample_splits(self, tmp_path):
        """Create sample split files."""
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        
        # Create sample data
        np.random.seed(42)
        X_train = pd.DataFrame({'feature1': np.random.randn(80)})
        y_train = pd.Series(np.random.choice([0, 1], 80))
        X_test = pd.DataFrame({'feature1': np.random.randn(20)})
        y_test = pd.Series(np.random.choice([0, 1], 20))
        
        # Save
        X_train.to_csv(splits_dir / "X_train.csv", index=False)
        y_train.to_csv(splits_dir / "y_train.csv", index=False)
        X_test.to_csv(splits_dir / "X_test.csv", index=False)
        y_test.to_csv(splits_dir / "y_test.csv", index=False)
        
        return str(splits_dir)
    
    def test_load_splits(self, sample_splits):
        """Test loading splits from disk."""
        X_train, X_test, y_train, y_test = load_splits(sample_splits)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20


class TestGetSplitSummary:
    """Tests for get_split_summary function."""
    
    def test_split_summary(self):
        """Test split summary generation."""
        y_train = pd.Series([0] * 70 + [1] * 10)
        y_test = pd.Series([0] * 20 + [1] * 0)
        
        summary = get_split_summary(y_train, y_test)
        
        assert 'split' in summary.columns
        assert 'n_samples' in summary.columns
        assert 'class_0' in summary.columns
        assert 'class_1' in summary.columns
        assert len(summary) == 2  # train and test
    
    def test_split_summary_with_val(self):
        """Test split summary with validation set."""
        y_train = pd.Series([0] * 60 + [1] * 10)
        y_val = pd.Series([0] * 10 + [1] * 0)
        y_test = pd.Series([0] * 20 + [1] * 0)
        
        summary = get_split_summary(y_train, y_test, y_val)
        
        assert len(summary) == 3  # train, val, and test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

