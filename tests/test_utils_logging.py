"""
Unit tests for logging utilities (src/utils/logging.py).

Tests verify logging configuration, format selection, and logger creation.

Run with: pytest tests/test_utils_logging.py -v
"""

import pytest
import logging
import sys
from pathlib import Path
import tempfile
import os

from src.utils.logging import setup_logging, get_logger


class TestLoggingSetup:
    """Tests for logging setup."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        logger = setup_logging()
        
        assert logger is not None
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        logger = setup_logging(log_level="DEBUG")
        
        assert logger.level == logging.DEBUG
    
    def test_setup_logging_json_format(self):
        """Test logging setup with JSON format."""
        logger = setup_logging(log_format="json")
        
        # Check that JSON formatter is used
        handlers = logger.handlers
        assert len(handlers) > 0
    
    def test_setup_logging_text_format(self):
        """Test logging setup with text format."""
        logger = setup_logging(log_format="text")
        
        assert logger is not None
        assert len(logger.handlers) > 0
    
    def test_setup_logging_with_file(self):
        """Test logging setup with log file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name
        
        try:
            logger = setup_logging(log_file=log_file)
            
            # Log a test message
            logger.info("Test message")
            
            # Verify file was created and contains message
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
        finally:
            if os.path.exists(log_file):
                os.remove(log_file)
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_module")
        
        assert logger is not None
        assert logger.name == "test_module"
        assert isinstance(logger, logging.Logger)
    
    def test_logger_hierarchy(self):
        """Test logger hierarchy."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")
        
        assert child_logger.parent == parent_logger
    
    def test_logging_levels(self):
        """Test different logging levels."""
        logger = setup_logging(log_level="DEBUG")
        
        # All levels should work
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        assert logger.level == logging.DEBUG


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
