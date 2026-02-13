"""
Example usage of the database layer.

This file demonstrates how to use the database models, repositories, and services
with proper exception handling and logging.
"""

from typing import List
from decimal import Decimal

from src.database.connection import get_db_session
from src.database.models import Prediction, User, RawTransaction
from src.database.repositories import (
    PredictionRepository,
    UserRepository,
    RawTransactionRepository
)
from src.database.services import (
    PredictionService,
    RawTransactionService,
    BusinessKPIService
)
from src.database.exceptions import (
    DatabaseError,
    RecordNotFoundError,
    DuplicateRecordError,
    DatabaseIntegrityError
)
from datetime import datetime
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Example 1: Save a Prediction
# ============================================================================

def example_save_prediction():
    """Example: Save a prediction using the service layer."""
    try:
        with get_db_session() as session:
            service = PredictionService(session)
            
            prediction = service.save_prediction(
                prediction_id="pred_abc123xyz",
                customer_id="CUST-12345",
                prediction=0,
                probability=Decimal("0.15"),
                customer_score=850,
                risk_level="low",
                features=[0.0, -0.046, -0.072, -0.349, -0.045, -2.156, -0.101, 0.849, -0.994,
                         -0.006, 0.853, 0.170, -0.068, -0.312, -0.167, 0.164, -0.193, -0.025,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                model_name="credit_scoring_model",
                model_version="v1.0",
                model_stage="Production",
                latency_ms=Decimal("45.2")
            )
            
            logger.info(f"Prediction saved: {prediction.prediction_id}")
            return prediction
            
    except DuplicateRecordError as e:
        logger.error(f"Prediction already exists: {e}")
        raise
    except DatabaseError as e:
        logger.error(f"Failed to save prediction: {e}")
        raise


# ============================================================================
# Example 2: Get Customer Predictions
# ============================================================================

def example_get_customer_predictions(customer_id: str) -> List[Prediction]:
    """Example: Get all predictions for a customer."""
    try:
        with get_db_session() as session:
            service = PredictionService(session)
            
            predictions = service.get_customer_predictions(
                customer_id=customer_id,
                limit=100
            )
            
            logger.info(f"Found {len(predictions)} predictions for customer {customer_id}")
            return predictions
            
    except DatabaseError as e:
        logger.error(f"Failed to get customer predictions: {e}")
        raise


# ============================================================================
# Example 3: Using Repository Directly
# ============================================================================

def example_repository_usage():
    """Example: Using repository directly for more control."""
    try:
        with get_db_session() as session:
            repo = PredictionRepository(session)
            
            # Get by ID
            prediction = repo.get_by_id("pred_abc123xyz")
            if prediction:
                logger.info(f"Found prediction: {prediction.prediction_id}")
            
            # Get by customer ID
            customer_predictions = repo.get_by_customer_id("CUST-12345", limit=10)
            logger.info(f"Found {len(customer_predictions)} predictions")
            
            # Get by risk level
            high_risk = repo.get_by_risk_level("high", limit=50)
            logger.info(f"Found {len(high_risk)} high-risk predictions")
            
    except DatabaseError as e:
        logger.error(f"Repository operation failed: {e}")
        raise


# ============================================================================
# Example 4: Bulk Upload Transactions
# ============================================================================

def example_bulk_upload_transactions(transactions: List[dict]):
    """Example: Bulk upload transactions."""
    try:
        with get_db_session() as session:
            service = RawTransactionService(session)
            
            count = service.upload_transactions(
                transactions=transactions,
                uploaded_by="admin",
                data_source="ecommerce_platform",
                file_name="transactions_2026_02_12.csv",
                data_version="v1"
            )
            
            logger.info(f"Successfully uploaded {count} transactions")
            return count
            
    except DatabaseError as e:
        logger.error(f"Failed to upload transactions: {e}")
        raise


# ============================================================================
# Example 5: Error Handling Pattern
# ============================================================================

def example_error_handling():
    """Example: Professional error handling pattern."""
    try:
        with get_db_session() as session:
            repo = UserRepository(session)
            
            # Try to get user
            user = repo.get_by_username("admin")
            
            if user is None:
                logger.warning("User not found")
                # Handle not found case
                return None
            
            logger.info(f"User found: {user.username}")
            return user
            
    except RecordNotFoundError as e:
        logger.warning(f"Record not found: {e}")
        # Handle not found
        return None
    except DuplicateRecordError as e:
        logger.error(f"Duplicate record: {e}")
        # Handle duplicate
        raise
    except DatabaseIntegrityError as e:
        logger.error(f"Integrity error: {e}")
        # Handle integrity violation
        raise
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        # Handle general database error
        raise
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        # Handle unexpected errors
        raise


# ============================================================================
# Example 6: Transaction Management
# ============================================================================

def example_transaction_management():
    """Example: Manual transaction management."""
    try:
        with get_db_session() as session:
            # All operations in this block are part of one transaction
            pred_repo = PredictionRepository(session)
            user_repo = UserRepository(session)
            
            # Create prediction
            prediction = pred_repo.create(
                prediction_id="pred_test123",
                customer_id="CUST-12345",
                prediction=0,
                probability=Decimal("0.15"),
                risk_level="low",
                features=[],
                model_name="test_model",
                model_version="v1.0",
                model_stage="Production"
            )
            
            # Update user
            user = user_repo.get_by_username("admin")
            if user:
                user_repo.update(user.user_id, last_login_at=datetime.utcnow())
            
            # If any operation fails, entire transaction is rolled back
            # If all succeed, transaction is committed automatically
            
    except DatabaseError as e:
        logger.error(f"Transaction failed: {e}")
        # Transaction is automatically rolled back
        raise
