"""
Repository pattern implementation for database operations.

Each repository provides a clean interface for database operations with:
- Professional exception handling
- Comprehensive logging
- Type hints
- Transaction management
"""

from typing import Optional, List, Dict, Any, TypeVar, Generic
from datetime import datetime, date
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy import and_, or_, func, desc, asc

from src.utils.logging import get_logger
from src.database.connection import get_db_session
from src.database.models import (
    User, Role, UserRole, Permission, RolePermission, AuditLog,
    RawTransaction, RFMMetric, ProcessedFeature, DataSplit,
    Prediction, CustomerFeature, DataVersion, ModelMetadata,
    BusinessKPI, PerformanceMetric, DriftMetric
)
from src.database.exceptions import (
    DatabaseError,
    DatabaseQueryError,
    DatabaseIntegrityError,
    RecordNotFoundError,
    DuplicateRecordError
)

logger = get_logger(__name__)

T = TypeVar('T')


class BaseRepository(Generic[T]):
    """
    Base repository class with common database operations.
    
    Provides CRUD operations with professional exception handling and logging.
    """
    
    def __init__(self, model_class: type, session: Optional[Session] = None):
        """
        Initialize repository.
        
        Args:
            model_class: SQLAlchemy model class
            session: Optional database session (creates new if not provided)
        """
        self.model_class = model_class
        self.model_name = model_class.__name__
        self._session = session
        self.logger = get_logger(f"{__name__}.{self.model_name}Repository")
    
    @property
    def session(self) -> Session:
        """Get database session."""
        if self._session is None:
            # This should be used with context manager in practice
            raise DatabaseError("Session not provided. Use with get_db_session() context manager.")
        return self._session
    
    def create(self, **kwargs) -> T:
        """
        Create a new record.
        
        Args:
            **kwargs: Model attributes
            
        Returns:
            Created model instance
            
        Raises:
            DuplicateRecordError: If record already exists
            DatabaseIntegrityError: If integrity constraints violated
            DatabaseError: For other database errors
        """
        try:
            self.logger.info(
                f"Creating {self.model_name}",
                extra={"attributes": list(kwargs.keys())}
            )
            
            instance = self.model_class(**kwargs)
            self.session.add(instance)
            self.session.flush()  # Flush to get ID if needed
            
            self.logger.info(
                f"{self.model_name} created successfully",
                extra={"id": getattr(instance, f"{self.model_name.lower()}_id", None) or getattr(instance, "id", None)}
            )
            
            return instance
            
        except IntegrityError as e:
            self.session.rollback()
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            
            # Check for duplicate key violation
            if "duplicate key" in error_msg.lower() or "unique constraint" in error_msg.lower():
                self.logger.warning(
                    f"Duplicate {self.model_name} detected",
                    extra={"error": error_msg}
                )
                raise DuplicateRecordError(
                    self.model_name,
                    "key",
                    str(kwargs),
                    original_error=e
                )
            
            self.logger.error(
                f"Integrity error creating {self.model_name}",
                extra={"error": error_msg},
                exc_info=True
            )
            raise DatabaseIntegrityError(
                f"Integrity error creating {self.model_name}: {error_msg}",
                original_error=e
            )
        except SQLAlchemyError as e:
            self.session.rollback()
            self.logger.error(
                f"Database error creating {self.model_name}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseQueryError(
                f"Database error creating {self.model_name}: {str(e)}",
                original_error=e
            )
        except Exception as e:
            self.session.rollback()
            self.logger.error(
                f"Unexpected error creating {self.model_name}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseError(
                f"Unexpected error creating {self.model_name}: {str(e)}",
                original_error=e
            )
    
    def get_by_id(self, record_id: Any) -> Optional[T]:
        """
        Get record by primary key.
        
        Args:
            record_id: Primary key value
            
        Returns:
            Model instance or None if not found
        """
        try:
            self.logger.debug(f"Getting {self.model_name} by ID: {record_id}")
            instance = self.session.query(self.model_class).get(record_id)
            
            if instance:
                self.logger.debug(f"{self.model_name} found: {record_id}")
            else:
                self.logger.debug(f"{self.model_name} not found: {record_id}")
            
            return instance
            
        except SQLAlchemyError as e:
            self.logger.error(
                f"Database error getting {self.model_name} by ID",
                extra={"id": record_id, "error": str(e)},
                exc_info=True
            )
            raise DatabaseQueryError(
                f"Database error getting {self.model_name}: {str(e)}",
                original_error=e
            )
    
    def get_by_id_or_raise(self, record_id: Any) -> T:
        """
        Get record by primary key or raise exception if not found.
        
        Args:
            record_id: Primary key value
            
        Returns:
            Model instance
            
        Raises:
            RecordNotFoundError: If record not found
        """
        instance = self.get_by_id(record_id)
        if instance is None:
            raise RecordNotFoundError(self.model_name, str(record_id))
        return instance
    
    def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """
        Get all records.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of model instances
        """
        try:
            self.logger.debug(f"Getting all {self.model_name} records (limit={limit}, offset={offset})")
            
            query = self.session.query(self.model_class)
            if limit:
                query = query.limit(limit).offset(offset)
            
            instances = query.all()
            self.logger.debug(f"Found {len(instances)} {self.model_name} records")
            
            return instances
            
        except SQLAlchemyError as e:
            self.logger.error(
                f"Database error getting all {self.model_name}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseQueryError(
                f"Database error getting all {self.model_name}: {str(e)}",
                original_error=e
            )
    
    def update(self, record_id: Any, **kwargs) -> T:
        """
        Update a record.
        
        Args:
            record_id: Primary key value
            **kwargs: Attributes to update
            
        Returns:
            Updated model instance
            
        Raises:
            RecordNotFoundError: If record not found
        """
        try:
            instance = self.get_by_id_or_raise(record_id)
            
            self.logger.info(
                f"Updating {self.model_name}",
                extra={"id": record_id, "attributes": list(kwargs.keys())}
            )
            
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            self.session.flush()
            
            self.logger.info(f"{self.model_name} updated successfully: {record_id}")
            
            return instance
            
        except RecordNotFoundError:
            raise
        except IntegrityError as e:
            self.session.rollback()
            self.logger.error(
                f"Integrity error updating {self.model_name}",
                extra={"id": record_id, "error": str(e)},
                exc_info=True
            )
            raise DatabaseIntegrityError(
                f"Integrity error updating {self.model_name}: {str(e)}",
                original_error=e
            )
        except SQLAlchemyError as e:
            self.session.rollback()
            self.logger.error(
                f"Database error updating {self.model_name}",
                extra={"id": record_id, "error": str(e)},
                exc_info=True
            )
            raise DatabaseQueryError(
                f"Database error updating {self.model_name}: {str(e)}",
                original_error=e
            )
    
    def delete(self, record_id: Any) -> bool:
        """
        Delete a record.
        
        Args:
            record_id: Primary key value
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            DatabaseError: If deletion fails
        """
        try:
            instance = self.get_by_id(record_id)
            if instance is None:
                self.logger.warning(f"{self.model_name} not found for deletion: {record_id}")
                return False
            
            self.logger.info(f"Deleting {self.model_name}: {record_id}")
            self.session.delete(instance)
            self.session.flush()
            
            self.logger.info(f"{self.model_name} deleted successfully: {record_id}")
            return True
            
        except SQLAlchemyError as e:
            self.session.rollback()
            self.logger.error(
                f"Database error deleting {self.model_name}",
                extra={"id": record_id, "error": str(e)},
                exc_info=True
            )
            raise DatabaseQueryError(
                f"Database error deleting {self.model_name}: {str(e)}",
                original_error=e
            )
    
    def count(self) -> int:
        """
        Count total records.
        
        Returns:
            Total count
        """
        try:
            count = self.session.query(func.count(self.model_class.id)).scalar()
            return count or 0
        except SQLAlchemyError as e:
            self.logger.error(
                f"Database error counting {self.model_name}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseQueryError(
                f"Database error counting {self.model_name}: {str(e)}",
                original_error=e
            )


# ============================================================================
# SPECIFIC REPOSITORIES
# ============================================================================

class UserRepository(BaseRepository[User]):
    """Repository for User model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(User, session)
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            return self.session.query(User).filter(User.username == username).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting user by username: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting user: {str(e)}", original_error=e)
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            return self.session.query(User).filter(User.email == email).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting user by email: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting user: {str(e)}", original_error=e)
    
    def get_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        try:
            return self.session.query(User).filter(
                and_(
                    User.api_key == api_key,
                    User.is_active == True,
                    or_(
                        User.api_key_expires_at.is_(None),
                        User.api_key_expires_at > datetime.utcnow()
                    )
                )
            ).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting user by API key: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting user: {str(e)}", original_error=e)


class RoleRepository(BaseRepository[Role]):
    """Repository for Role model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(Role, session)
    
    def get_by_code(self, role_code: str) -> Optional[Role]:
        """Get role by code."""
        try:
            return self.session.query(Role).filter(Role.role_code == role_code).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting role by code: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting role: {str(e)}", original_error=e)


class PredictionRepository(BaseRepository[Prediction]):
    """Repository for Prediction model with specialized queries."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(Prediction, session)
    
    def get_by_customer_id(
        self,
        customer_id: str,
        limit: Optional[int] = None,
        order_by: str = "created_at",
        descending: bool = True
    ) -> List[Prediction]:
        """
        Get predictions by customer ID.
        
        Args:
            customer_id: Customer identifier
            limit: Maximum number of records
            order_by: Field to order by
            descending: Order descending if True
            
        Returns:
            List of predictions
        """
        try:
            self.logger.debug(f"Getting predictions for customer: {customer_id}")
            
            query = self.session.query(Prediction).filter(
                Prediction.customer_id_indexed == customer_id
            )
            
            # Order by
            order_field = getattr(Prediction, order_by, Prediction.created_at)
            if descending:
                query = query.order_by(desc(order_field))
            else:
                query = query.order_by(asc(order_field))
            
            # Limit
            if limit:
                query = query.limit(limit)
            
            predictions = query.all()
            self.logger.debug(f"Found {len(predictions)} predictions for customer {customer_id}")
            
            return predictions
            
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error getting predictions by customer ID: {e}",
                extra={"customer_id": customer_id},
                exc_info=True
            )
            raise DatabaseQueryError(
                f"Error getting predictions: {str(e)}",
                original_error=e
            )
    
    def get_by_date_range(
        self,
        start_date: date,
        end_date: date,
        limit: Optional[int] = None
    ) -> List[Prediction]:
        """Get predictions within date range."""
        try:
            query = self.session.query(Prediction).filter(
                and_(
                    Prediction.created_at_date >= start_date,
                    Prediction.created_at_date <= end_date
                )
            ).order_by(desc(Prediction.created_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting predictions by date range: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting predictions: {str(e)}", original_error=e)
    
    def get_by_risk_level(
        self,
        risk_level: str,
        limit: Optional[int] = None
    ) -> List[Prediction]:
        """Get predictions by risk level."""
        try:
            query = self.session.query(Prediction).filter(
                Prediction.risk_level == risk_level
            ).order_by(desc(Prediction.created_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting predictions by risk level: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting predictions: {str(e)}", original_error=e)


class RawTransactionRepository(BaseRepository[RawTransaction]):
    """Repository for RawTransaction model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(RawTransaction, session)
    
    def get_by_customer_id(
        self,
        customer_id: str,
        limit: Optional[int] = None
    ) -> List[RawTransaction]:
        """Get transactions by customer ID."""
        try:
            query = self.session.query(RawTransaction).filter(
                RawTransaction.customer_id == customer_id
            ).order_by(desc(RawTransaction.transaction_start_time))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting transactions by customer ID: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting transactions: {str(e)}", original_error=e)
    
    def bulk_create(self, transactions: List[Dict[str, Any]]) -> int:
        """
        Bulk create transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Number of transactions created
        """
        try:
            self.logger.info(f"Bulk creating {len(transactions)} transactions")
            
            instances = [RawTransaction(**txn) for txn in transactions]
            self.session.bulk_save_objects(instances)
            self.session.flush()
            
            self.logger.info(f"Successfully created {len(instances)} transactions")
            return len(instances)
            
        except IntegrityError as e:
            self.session.rollback()
            self.logger.error(f"Integrity error in bulk create: {e}", exc_info=True)
            raise DatabaseIntegrityError(f"Bulk create failed: {str(e)}", original_error=e)
        except SQLAlchemyError as e:
            self.session.rollback()
            self.logger.error(f"Database error in bulk create: {e}", exc_info=True)
            raise DatabaseQueryError(f"Bulk create failed: {str(e)}", original_error=e)


class RFMMetricRepository(BaseRepository[RFMMetric]):
    """Repository for RFMMetric model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(RFMMetric, session)
    
    def get_by_cluster(self, cluster: int) -> List[RFMMetric]:
        """Get RFM metrics by cluster."""
        try:
            return self.session.query(RFMMetric).filter(
                RFMMetric.cluster == cluster
            ).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting RFM metrics by cluster: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting RFM metrics: {str(e)}", original_error=e)


class ProcessedFeatureRepository(BaseRepository[ProcessedFeature]):
    """Repository for ProcessedFeature model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(ProcessedFeature, session)
    
    def get_feature_vector(self, customer_id: str) -> Optional[List[float]]:
        """Get feature vector for a customer."""
        try:
            feature = self.get_by_id(customer_id)
            if feature and feature.feature_vector:
                return [float(f) for f in feature.feature_vector]
            return None
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting feature vector: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting feature vector: {str(e)}", original_error=e)


class BusinessKPIRepository(BaseRepository[BusinessKPI]):
    """Repository for BusinessKPI model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(BusinessKPI, session)
    
    def get_by_period(
        self,
        period_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[BusinessKPI]:
        """Get KPI by period."""
        try:
            return self.session.query(BusinessKPI).filter(
                and_(
                    BusinessKPI.period_type == period_type,
                    BusinessKPI.period_start == start_date,
                    BusinessKPI.period_end == end_date
                )
            ).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting KPI by period: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting KPI: {str(e)}", original_error=e)
    
    def get_latest(self, period_type: str) -> Optional[BusinessKPI]:
        """Get latest KPI for period type."""
        try:
            return self.session.query(BusinessKPI).filter(
                BusinessKPI.period_type == period_type
            ).order_by(desc(BusinessKPI.period_start)).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting latest KPI: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting KPI: {str(e)}", original_error=e)
