"""
Repository pattern implementation for database operations.

Each repository provides a clean interface for database operations with:
- Professional exception handling
- Comprehensive logging
- Type hints
- Transaction management
"""

from typing import Optional, List, Dict, Any, TypeVar, Generic
from datetime import datetime, date, timezone
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy import and_, or_, func, desc, asc

from src.utils.logging import get_logger
from src.database.connection import get_db_session
from src.database.models import (
    User, Role, UserRole, Permission, RolePermission, AuditLog,
    RawTransaction, RFMMetric, ProcessedFeature, DataSplit,
    Prediction, CustomerFeature, DataVersion, DataLineage, ModelMetadata,
    BusinessKPI, PerformanceMetric, DriftMetric,
    Experiment, ExperimentAssignment, ExperimentMetric,
    RetrainingJob, RetrainingSchedule, ModelValidationRule,
    BatchPredictionJob, BatchPredictionSchedule, BatchPredictionResult, BatchPredictionLog,
    ModelRoutingRule, ModelRegistry, ModelComparisonResult, ModelEnsemble
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


class CustomerFeatureRepository(BaseRepository[CustomerFeature]):
    """Repository for CustomerFeature model (Feature Store)."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(CustomerFeature, session)
    
    def get_feature_vector(self, customer_id: str) -> Optional[List[float]]:
        """Get feature vector for a customer from feature store."""
        try:
            feature = self.get_by_id(customer_id)
            if feature and feature.feature_vector:
                return [float(f) for f in feature.feature_vector]
            return None
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting feature vector: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting feature vector: {str(e)}", original_error=e)
    
    def batch_get_feature_vectors(self, customer_ids: List[str]) -> Dict[str, List[float]]:
        """Get feature vectors for multiple customers."""
        try:
            features = self.session.query(CustomerFeature).filter(
                CustomerFeature.customer_id.in_(customer_ids)
            ).all()
            
            result = {}
            for feature in features:
                if feature and feature.feature_vector:
                    result[feature.customer_id] = [float(f) for f in feature.feature_vector]
            
            return result
        except SQLAlchemyError as e:
            self.logger.error(f"Error batch getting feature vectors: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error batch getting feature vectors: {str(e)}", original_error=e)
    
    def upsert_feature(
        self,
        customer_id: str,
        feature_vector: List[float],
        recency_normalized: Optional[float] = None,
        frequency_normalized: Optional[float] = None,
        monetary_normalized: Optional[float] = None,
        transaction_hour: Optional[float] = None,
        transaction_day: Optional[float] = None,
        transaction_month: Optional[float] = None,
        transaction_year: Optional[float] = None,
        transaction_dayofweek: Optional[float] = None,
        aggregate_features: Optional[Dict[str, Any]] = None,
        categorical_features: Optional[Dict[str, Any]] = None,
        feature_version: Optional[str] = None,
        data_version: Optional[str] = None
    ) -> CustomerFeature:
        """Insert or update customer features in feature store."""
        try:
            # Check if feature exists
            existing = self.get_by_id(customer_id)
            
            if existing:
                # Update existing
                existing.feature_vector = feature_vector
                if recency_normalized is not None:
                    existing.recency_normalized = recency_normalized
                if frequency_normalized is not None:
                    existing.frequency_normalized = frequency_normalized
                if monetary_normalized is not None:
                    existing.monetary_normalized = monetary_normalized
                if transaction_hour is not None:
                    existing.transaction_hour = transaction_hour
                if transaction_day is not None:
                    existing.transaction_day = transaction_day
                if transaction_month is not None:
                    existing.transaction_month = transaction_month
                if transaction_year is not None:
                    existing.transaction_year = transaction_year
                if transaction_dayofweek is not None:
                    existing.transaction_dayofweek = transaction_dayofweek
                if aggregate_features is not None:
                    existing.aggregate_features = aggregate_features
                if categorical_features is not None:
                    existing.categorical_features = categorical_features
                if feature_version is not None:
                    existing.feature_version = feature_version
                if data_version is not None:
                    existing.data_version = data_version
                
                self.session.commit()
                return existing
            else:
                # Create new
                new_feature = CustomerFeature(
                    customer_id=customer_id,
                    feature_vector=feature_vector,
                    recency_normalized=recency_normalized,
                    frequency_normalized=frequency_normalized,
                    monetary_normalized=monetary_normalized,
                    transaction_hour=transaction_hour,
                    transaction_day=transaction_day,
                    transaction_month=transaction_month,
                    transaction_year=transaction_year,
                    transaction_dayofweek=transaction_dayofweek,
                    aggregate_features=aggregate_features,
                    categorical_features=categorical_features,
                    feature_version=feature_version,
                    data_version=data_version
                )
                self.session.add(new_feature)
                self.session.commit()
                return new_feature
                
        except SQLAlchemyError as e:
            self.session.rollback()
            self.logger.error(f"Error upserting feature: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error upserting feature: {str(e)}", original_error=e)


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


class DataVersionRepository(BaseRepository[DataVersion]):
    """Repository for DataVersion model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(DataVersion, session)
    
    def get_by_type_and_version(
        self,
        data_type: str,
        version: str
    ) -> Optional[DataVersion]:
        """Get data version by type and version."""
        try:
            return self.session.query(DataVersion).filter(
                and_(
                    DataVersion.data_type == data_type,
                    DataVersion.version == version
                )
            ).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting data version: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting data version: {str(e)}", original_error=e)
    
    def get_latest_by_type(self, data_type: str) -> Optional[DataVersion]:
        """Get latest version for a data type."""
        try:
            return self.session.query(DataVersion).filter(
                DataVersion.data_type == data_type
            ).order_by(desc(DataVersion.created_at)).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting latest data version: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting latest data version: {str(e)}", original_error=e)
    
    def list_by_type(self, data_type: str) -> List[DataVersion]:
        """List all versions for a data type."""
        try:
            return self.session.query(DataVersion).filter(
                DataVersion.data_type == data_type
            ).order_by(desc(DataVersion.created_at)).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error listing data versions: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error listing data versions: {str(e)}", original_error=e)


class ExperimentRepository(BaseRepository[Experiment]):
    """Repository for Experiment model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(Experiment, session)
    
    def get_by_name(self, experiment_name: str) -> Optional[Experiment]:
        """Get experiment by name."""
        try:
            return self.session.query(Experiment).filter(
                Experiment.experiment_name == experiment_name
            ).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting experiment by name: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting experiment: {str(e)}", original_error=e)
    
    def get_running_experiments(self) -> List[Experiment]:
        """Get all running experiments."""
        try:
            return self.session.query(Experiment).filter(
                Experiment.status == "running"
            ).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting running experiments: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting running experiments: {str(e)}", original_error=e)
    
    def create_experiment(
        self,
        experiment_name: str,
        variants: List[Dict[str, Any]],
        description: Optional[str] = None,
        traffic_percentage: int = 100,
        assignment_method: str = "hash",
        primary_metric: str = "accuracy",
        minimum_sample_size: int = 1000,
        significance_level: float = 0.05,
        minimum_improvement: float = 0.01,
        created_by: Optional[str] = None
    ) -> Experiment:
        """Create a new experiment."""
        try:
            experiment = Experiment(
                experiment_name=experiment_name,
                description=description,
                status="draft",
                variants=variants,
                traffic_percentage=traffic_percentage,
                assignment_method=assignment_method,
                primary_metric=primary_metric,
                minimum_sample_size=minimum_sample_size,
                significance_level=significance_level,
                minimum_improvement=minimum_improvement,
                created_by=created_by
            )
            self.session.add(experiment)
            self.session.commit()
            return experiment
        except SQLAlchemyError as e:
            self.session.rollback()
            self.logger.error(f"Error creating experiment: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error creating experiment: {str(e)}", original_error=e)


class ExperimentAssignmentRepository(BaseRepository[ExperimentAssignment]):
    """Repository for ExperimentAssignment model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(ExperimentAssignment, session)
    
    def get_by_experiment_and_entity(
        self,
        experiment_id: int,
        entity_id: str,
        entity_type: str = "customer"
    ) -> Optional[ExperimentAssignment]:
        """Get assignment for a specific entity in an experiment."""
        try:
            return self.session.query(ExperimentAssignment).filter(
                and_(
                    ExperimentAssignment.experiment_id == experiment_id,
                    ExperimentAssignment.entity_id == entity_id,
                    ExperimentAssignment.entity_type == entity_type
                )
            ).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting assignment: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting assignment: {str(e)}", original_error=e)
    
    def get_by_experiment(self, experiment_id: int) -> List[ExperimentAssignment]:
        """Get all assignments for an experiment."""
        try:
            return self.session.query(ExperimentAssignment).filter(
                ExperimentAssignment.experiment_id == experiment_id
            ).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting assignments: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting assignments: {str(e)}", original_error=e)
    
    def create(
        self,
        experiment_id: int,
        entity_id: str,
        entity_type: str,
        variant_name: str,
        assignment_hash: Optional[str] = None
    ) -> ExperimentAssignment:
        """Create a new assignment."""
        try:
            assignment = ExperimentAssignment(
                experiment_id=experiment_id,
                entity_id=entity_id,
                entity_type=entity_type,
                variant_name=variant_name,
                assignment_hash=assignment_hash
            )
            self.session.add(assignment)
            return assignment
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating assignment: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error creating assignment: {str(e)}", original_error=e)


class ExperimentMetricRepository(BaseRepository[ExperimentMetric]):
    """Repository for ExperimentMetric model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(ExperimentMetric, session)
    
    def get_latest_by_experiment_and_variant(
        self,
        experiment_id: int,
        variant_name: str
    ) -> Optional[ExperimentMetric]:
        """Get latest metrics for a variant in an experiment."""
        try:
            return self.session.query(ExperimentMetric).filter(
                and_(
                    ExperimentMetric.experiment_id == experiment_id,
                    ExperimentMetric.variant_name == variant_name
                )
            ).order_by(desc(ExperimentMetric.calculated_at)).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting latest metrics: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting latest metrics: {str(e)}", original_error=e)
    
    def get_by_experiment(
        self,
        experiment_id: int,
        variant_name: Optional[str] = None
    ) -> List[ExperimentMetric]:
        """Get all metrics for an experiment, optionally filtered by variant."""
        try:
            query = self.session.query(ExperimentMetric).filter(
                ExperimentMetric.experiment_id == experiment_id
            )
            
            if variant_name:
                query = query.filter(ExperimentMetric.variant_name == variant_name)
            
            return query.order_by(desc(ExperimentMetric.calculated_at)).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting metrics: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting metrics: {str(e)}", original_error=e)


class DataLineageRepository(BaseRepository[DataLineage]):
    """Repository for DataLineage model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(DataLineage, session)
    
    def get_by_source_version(
        self,
        source_data_version_id: int
    ) -> List[DataLineage]:
        """Get all lineage records for a source data version."""
        try:
            return self.session.query(DataLineage).filter(
                DataLineage.source_data_version_id == source_data_version_id
            ).order_by(desc(DataLineage.created_at)).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting lineage by source: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting lineage: {str(e)}", original_error=e)
    
    def get_by_target(
        self,
        target_type: str,
        target_id: str
    ) -> List[DataLineage]:
        """Get all lineage records for a target (e.g., model, prediction)."""
        try:
            return self.session.query(DataLineage).filter(
                and_(
                    DataLineage.target_type == target_type,
                    DataLineage.target_id == target_id
                )
            ).order_by(desc(DataLineage.created_at)).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting lineage by target: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting lineage: {str(e)}", original_error=e)
    
    def get_lineage_graph(
        self,
        data_version_id: Optional[int] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None
    ) -> List[DataLineage]:
        """Get lineage graph with optional filters."""
        try:
            query = self.session.query(DataLineage)
            
            if data_version_id:
                query = query.filter(DataLineage.source_data_version_id == data_version_id)
            if target_type:
                query = query.filter(DataLineage.target_type == target_type)
            if target_id:
                query = query.filter(DataLineage.target_id == target_id)
            
            return query.order_by(desc(DataLineage.created_at)).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting lineage graph: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting lineage graph: {str(e)}", original_error=e)


class RetrainingJobRepository(BaseRepository[RetrainingJob]):
    """Repository for RetrainingJob model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(RetrainingJob, session)
    
    def get_by_status(self, status: str) -> List[RetrainingJob]:
        """Get jobs by status."""
        try:
            return self.session.query(RetrainingJob).filter(
                RetrainingJob.status == status
            ).order_by(desc(RetrainingJob.created_at)).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting jobs by status: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting jobs: {str(e)}", original_error=e)
    
    def get_recent_jobs(self, limit: int = 10) -> List[RetrainingJob]:
        """Get recent retraining jobs."""
        try:
            return self.session.query(RetrainingJob).order_by(
                desc(RetrainingJob.created_at)
            ).limit(limit).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting recent jobs: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting recent jobs: {str(e)}", original_error=e)
    
    def create_job(
        self,
        job_name: str,
        model_name: str,
        model_type: str = "random_forest",
        trigger_type: str = "manual",
        trigger_metadata: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> RetrainingJob:
        """Create a new retraining job."""
        try:
            job = RetrainingJob(
                job_name=job_name,
                model_name=model_name,
                model_type=model_type,
                trigger_type=trigger_type,
                trigger_metadata=trigger_metadata,
                hyperparameters=hyperparameters,
                status="pending",
                created_by=created_by,
                job_metadata=training_config
            )
            self.session.add(job)
            self.session.flush()
            return job
        except SQLAlchemyError as e:
            self.session.rollback()
            self.logger.error(f"Error creating retraining job: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error creating job: {str(e)}", original_error=e)


class RetrainingScheduleRepository(BaseRepository[RetrainingSchedule]):
    """Repository for RetrainingSchedule model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(RetrainingSchedule, session)
    
    def get_active_schedules(self) -> List[RetrainingSchedule]:
        """Get all active schedules."""
        try:
            return self.session.query(RetrainingSchedule).filter(
                RetrainingSchedule.is_active == True
            ).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting active schedules: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting schedules: {str(e)}", original_error=e)
    
    def get_due_schedules(self) -> List[RetrainingSchedule]:
        """Get schedules that are due to run."""
        try:
            now = datetime.now(timezone.utc)
            return self.session.query(RetrainingSchedule).filter(
                and_(
                    RetrainingSchedule.is_active == True,
                    or_(
                        RetrainingSchedule.next_run_at <= now,
                        RetrainingSchedule.next_run_at.is_(None)
                    )
                )
            ).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting due schedules: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting due schedules: {str(e)}", original_error=e)
    
    def get_by_name(self, schedule_name: str) -> Optional[RetrainingSchedule]:
        """Get schedule by name."""
        try:
            return self.session.query(RetrainingSchedule).filter(
                RetrainingSchedule.schedule_name == schedule_name
            ).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting schedule by name: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting schedule: {str(e)}", original_error=e)


class ModelValidationRuleRepository(BaseRepository[ModelValidationRule]):
    """Repository for ModelValidationRule model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(ModelValidationRule, session)
    
    def get_active_rules_for_model(self, model_name: str) -> List[ModelValidationRule]:
        """Get active validation rules for a model."""
        try:
            return self.session.query(ModelValidationRule).filter(
                and_(
                    ModelValidationRule.model_name == model_name,
                    ModelValidationRule.is_active == True
                )
            ).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting validation rules: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting rules: {str(e)}", original_error=e)
    
    def get_by_name(self, rule_name: str) -> Optional[ModelValidationRule]:
        """Get rule by name."""
        try:
            return self.session.query(ModelValidationRule).filter(
                ModelValidationRule.rule_name == rule_name
            ).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting rule by name: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting rule: {str(e)}", original_error=e)


class BatchPredictionJobRepository(BaseRepository[BatchPredictionJob]):
    """Repository for BatchPredictionJob model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(BatchPredictionJob, session)
    
    def get_by_status(self, status: str) -> List[BatchPredictionJob]:
        """Get jobs by status."""
        try:
            return self.session.query(BatchPredictionJob).filter(
                BatchPredictionJob.status == status
            ).order_by(desc(BatchPredictionJob.created_at)).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting jobs by status: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting jobs: {str(e)}", original_error=e)
    
    def get_recent_jobs(self, limit: int = 10) -> List[BatchPredictionJob]:
        """Get recent batch prediction jobs."""
        try:
            return self.session.query(BatchPredictionJob).order_by(
                desc(BatchPredictionJob.created_at)
            ).limit(limit).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting recent jobs: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting recent jobs: {str(e)}", original_error=e)
    
    def create_job(
        self,
        job_name: str,
        input_source: str,
        input_config: Dict[str, Any],
        output_format: str,
        output_config: Dict[str, Any],
        model_name: str,
        trigger_type: str = "manual",
        batch_size: int = 1000,
        max_workers: int = 4,
        use_feature_store: bool = True,
        model_version: Optional[str] = None,
        model_stage: str = "Production",
        created_by: Optional[str] = None
    ) -> BatchPredictionJob:
        """Create a new batch prediction job."""
        try:
            job = BatchPredictionJob(
                job_name=job_name,
                trigger_type=trigger_type,
                input_source=input_source,
                input_config=input_config,
                output_format=output_format,
                output_config=output_config,
                model_name=model_name,
                model_version=model_version,
                model_stage=model_stage,
                batch_size=batch_size,
                max_workers=max_workers,
                use_feature_store=use_feature_store,
                status="pending",
                created_by=created_by
            )
            self.session.add(job)
            self.session.flush()
            return job
        except SQLAlchemyError as e:
            self.session.rollback()
            self.logger.error(f"Error creating batch job: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error creating job: {str(e)}", original_error=e)


class BatchPredictionScheduleRepository(BaseRepository[BatchPredictionSchedule]):
    """Repository for BatchPredictionSchedule model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(BatchPredictionSchedule, session)
    
    def get_active_schedules(self) -> List[BatchPredictionSchedule]:
        """Get all active schedules."""
        try:
            return self.session.query(BatchPredictionSchedule).filter(
                BatchPredictionSchedule.is_active == True
            ).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting active schedules: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting schedules: {str(e)}", original_error=e)
    
    def get_due_schedules(self) -> List[BatchPredictionSchedule]:
        """Get schedules that are due to run."""
        try:
            now = datetime.now(timezone.utc)
            return self.session.query(BatchPredictionSchedule).filter(
                and_(
                    BatchPredictionSchedule.is_active == True,
                    or_(
                        BatchPredictionSchedule.next_run_at <= now,
                        BatchPredictionSchedule.next_run_at.is_(None)
                    )
                )
            ).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting due schedules: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting due schedules: {str(e)}", original_error=e)
"""
Multi-Model Serving Repositories

These repositories are added to repositories.py but kept separate for clarity.
"""

from typing import Optional, List, Dict
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, desc

from src.utils.logging import get_logger
from src.database.models import (
    ModelRoutingRule, ModelRegistry, ModelEnsemble
)
from src.database.exceptions import DatabaseQueryError
from src.database.repositories import BaseRepository

logger = get_logger(__name__)


class ModelRoutingRuleRepository(BaseRepository[ModelRoutingRule]):
    """Repository for ModelRoutingRule model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(ModelRoutingRule, session)
    
    def get_active_rules(self) -> List[ModelRoutingRule]:
        """Get all active routing rules sorted by priority."""
        try:
            return self.session.query(ModelRoutingRule).filter(
                ModelRoutingRule.is_active == True
            ).order_by(desc(ModelRoutingRule.priority)).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting active routing rules: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting routing rules: {str(e)}", original_error=e)
    
    def get_by_name(self, rule_name: str) -> Optional[ModelRoutingRule]:
        """Get rule by name."""
        try:
            return self.session.query(ModelRoutingRule).filter(
                ModelRoutingRule.rule_name == rule_name
            ).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting rule by name: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting rule: {str(e)}", original_error=e)


class ModelRegistryRepository(BaseRepository[ModelRegistry]):
    """Repository for ModelRegistry model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(ModelRegistry, session)
    
    def get_loaded_models(self) -> List[ModelRegistry]:
        """Get all loaded models."""
        try:
            return self.session.query(ModelRegistry).filter(
                ModelRegistry.is_loaded == True
            ).order_by(desc(ModelRegistry.load_priority)).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting loaded models: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting loaded models: {str(e)}", original_error=e)
    
    def get_by_name_and_version(
        self,
        model_name: str,
        model_version: str
    ) -> Optional[ModelRegistry]:
        """Get model by name and version."""
        try:
            return self.session.query(ModelRegistry).filter(
                and_(
                    ModelRegistry.model_name == model_name,
                    ModelRegistry.model_version == model_version
                )
            ).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting model by name and version: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting model: {str(e)}", original_error=e)
    
    def register_model(
        self,
        model_name: str,
        model_version: str,
        model_stage: str = "Production",
        model_type: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_model_uri: Optional[str] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> ModelRegistry:
        """Register a model in the registry."""
        try:
            # Check if already exists
            existing = self.get_by_name_and_version(model_name, model_version)
            if existing:
                return existing
            
            registry = ModelRegistry(
                model_name=model_name,
                model_version=model_version,
                model_stage=model_stage,
                model_type=model_type,
                mlflow_run_id=mlflow_run_id,
                mlflow_model_uri=mlflow_model_uri,
                accuracy=performance_metrics.get("accuracy") if performance_metrics else None,
                roc_auc=performance_metrics.get("roc_auc") if performance_metrics else None,
                precision=performance_metrics.get("precision") if performance_metrics else None,
                recall=performance_metrics.get("recall") if performance_metrics else None,
                f1_score=performance_metrics.get("f1_score") if performance_metrics else None,
                status="available"
            )
            
            self.session.add(registry)
            self.session.flush()
            return registry
            
        except SQLAlchemyError as e:
            self.session.rollback()
            self.logger.error(f"Error registering model: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error registering model: {str(e)}", original_error=e)


class ModelEnsembleRepository(BaseRepository[ModelEnsemble]):
    """Repository for ModelEnsemble model."""
    
    def __init__(self, session: Optional[Session] = None):
        super().__init__(ModelEnsemble, session)
    
    def get_active_ensembles(self) -> List[ModelEnsemble]:
        """Get all active ensembles."""
        try:
            return self.session.query(ModelEnsemble).filter(
                ModelEnsemble.is_active == True
            ).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting active ensembles: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting ensembles: {str(e)}", original_error=e)
    
    def get_by_name(self, ensemble_name: str) -> Optional[ModelEnsemble]:
        """Get ensemble by name."""
        try:
            return self.session.query(ModelEnsemble).filter(
                ModelEnsemble.ensemble_name == ensemble_name
            ).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting ensemble by name: {e}", exc_info=True)
            raise DatabaseQueryError(f"Error getting ensemble: {str(e)}", original_error=e)
