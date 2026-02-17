"""
Service layer for database operations.

Provides high-level business logic for database operations with proper
exception handling and logging.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, case, Integer

from src.utils.logging import get_logger
from src.database.connection import get_db_session
from src.database.models import (
    Prediction, RawTransaction, BusinessKPI, DataVersion, DataLineage
)
from src.database.repositories import (
    PredictionRepository,
    RawTransactionRepository,
    BusinessKPIRepository,
    UserRepository,
    DataVersionRepository,
    DataLineageRepository
)
from src.database.exceptions import (
    DatabaseError,
    RecordNotFoundError,
    DuplicateRecordError
)

logger = get_logger(__name__)


class PredictionService:
    """Service for prediction operations."""
    
    def __init__(self, session: Session):
        """
        Initialize prediction service.
        
        Args:
            session: Database session
        """
        self.session = session
        self.repository = PredictionRepository(session)
        self.logger = get_logger(f"{__name__}.PredictionService")
    
    def save_prediction(
        self,
        prediction_id: str,
        customer_id: Optional[str],
        prediction: int,
        probability: Decimal,
        customer_score: Optional[int],
        risk_level: str,
        features: List[float],
        model_name: str,
        model_version: str,
        model_stage: str,
        latency_ms: Optional[float] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
        response_metadata: Optional[Dict[str, Any]] = None
    ) -> Prediction:
        """
        Save a prediction to the database.
        
        Args:
            prediction_id: Unique prediction identifier
            customer_id: Customer identifier
            prediction: Binary prediction (0 or 1)
            probability: Probability of high-risk
            customer_score: Credit score (0-100)
            risk_level: Risk level ('low', 'medium', 'high')
            features: List of 26 feature values
            model_name: Model name
            model_version: Model version
            model_stage: Model stage
            latency_ms: Prediction latency in milliseconds
            request_metadata: Additional request metadata
            response_metadata: Additional response metadata
            
        Returns:
            Saved Prediction instance
            
        Raises:
            DatabaseError: If save fails
        """
        try:
            self.logger.info(
                "Saving prediction",
                extra={
                    "prediction_id": prediction_id,
                    "customer_id": customer_id,
                    "risk_level": risk_level,
                    "model_version": model_version
                }
            )
            
            # Calculate created_at_date from current timestamp
            now = datetime.utcnow()
            created_at_date = now.date()
            
            prediction_obj = self.repository.create(
                prediction_id=prediction_id,
                customer_id=customer_id,
                customer_id_indexed=customer_id,  # For indexed lookups
                prediction=prediction,
                probability=Decimal(str(probability)),
                customer_score=customer_score,
                risk_level=risk_level,
                features=features,  # Will be stored as JSONB
                model_name=model_name,
                model_version=model_version,
                model_stage=model_stage,
                latency_ms=Decimal(str(latency_ms)) if latency_ms else None,
                request_metadata=request_metadata,
                response_metadata=response_metadata,
                created_at_date=created_at_date
            )
            
            self.logger.info(
                "Prediction saved successfully",
                extra={"prediction_id": prediction_id}
            )
            
            return prediction_obj
            
        except DuplicateRecordError:
            self.logger.warning(
                "Prediction already exists",
                extra={"prediction_id": prediction_id}
            )
            raise
        except Exception as e:
            self.logger.error(
                "Failed to save prediction",
                extra={
                    "prediction_id": prediction_id,
                    "error": str(e)
                },
                exc_info=True
            )
            raise DatabaseError(
                f"Failed to save prediction: {str(e)}",
                original_error=e
            )
    
    def get_customer_predictions(
        self,
        customer_id: str,
        limit: Optional[int] = 100,
        offset: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        risk_level: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> List[Prediction]:
        """
        Get all predictions for a customer with filtering.
        
        Args:
            customer_id: Customer identifier
            limit: Maximum number of predictions to return
            offset: Number of predictions to skip
            start_date: Filter predictions from this date (inclusive)
            end_date: Filter predictions to this date (inclusive)
            risk_level: Filter by risk level ('low', 'medium', 'high')
            model_version: Filter by model version
            
        Returns:
            List of predictions ordered by most recent first
        """
        try:
            self.logger.debug(f"Getting predictions for customer: {customer_id}")
            predictions = self.repository.get_by_customer_id(
                customer_id=customer_id,
                limit=limit,
                offset=offset,
                start_date=start_date,
                end_date=end_date,
                risk_level=risk_level,
                model_version=model_version
            )
            self.logger.debug(f"Found {len(predictions)} predictions for customer {customer_id}")
            return predictions
        except Exception as e:
            self.logger.error(
                f"Failed to get customer predictions: {e}",
                extra={"customer_id": customer_id},
                exc_info=True
            )
            raise DatabaseError(
                f"Failed to get customer predictions: {str(e)}",
                original_error=e
            )
    
    def get_recent_predictions(
        self,
        limit: int = 100
    ) -> List[Prediction]:
        """
        Get recent predictions ordered by creation time.
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List of recent predictions
        """
        try:
            from sqlalchemy import desc
            
            predictions = self.session.query(Prediction).order_by(
                desc(Prediction.created_at)
            ).limit(limit).all()
            
            return predictions
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting recent predictions: {e}", exc_info=True)
            raise DatabaseQueryError(
                f"Error getting recent predictions: {str(e)}",
                original_error=e
            )
    
    def get_customer_prediction_analytics(
        self,
        customer_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Get analytics/aggregations for customer predictions.
        
        Args:
            customer_id: Customer identifier
            start_date: Filter predictions from this date (inclusive)
            end_date: Filter predictions to this date (inclusive)
            
        Returns:
            Dictionary with analytics data
        """
        try:
            from sqlalchemy import func, case
            
            query = self.session.query(
                func.count(Prediction.prediction_id).label('total_count'),
                func.avg(Prediction.probability).label('avg_probability'),
                func.avg(Prediction.customer_score).label('avg_score'),
                func.avg(Prediction.latency_ms).label('avg_latency_ms'),
                func.count(case((Prediction.risk_level == 'low', 1))).label('low_risk_count'),
                func.count(case((Prediction.risk_level == 'medium', 1))).label('medium_risk_count'),
                func.count(case((Prediction.risk_level == 'high', 1))).label('high_risk_count'),
                func.min(Prediction.created_at).label('first_prediction'),
                func.max(Prediction.created_at).label('last_prediction')
            ).filter(
                Prediction.customer_id_indexed == customer_id
            )
            
            # Apply date filters
            if start_date:
                query = query.filter(Prediction.created_at_date >= start_date)
            if end_date:
                query = query.filter(Prediction.created_at_date <= end_date)
            
            result = query.first()
            
            if result and result.total_count > 0:
                return {
                    "total_count": result.total_count or 0,
                    "average_probability": float(result.avg_probability) if result.avg_probability else None,
                    "average_score": float(result.avg_score) if result.avg_score else None,
                    "average_latency_ms": float(result.avg_latency_ms) if result.avg_latency_ms else None,
                    "risk_level_distribution": {
                        "low": result.low_risk_count or 0,
                        "medium": result.medium_risk_count or 0,
                        "high": result.high_risk_count or 0
                    },
                    "first_prediction": result.first_prediction.isoformat() if result.first_prediction else None,
                    "last_prediction": result.last_prediction.isoformat() if result.last_prediction else None
                }
            else:
                return {
                    "total_count": 0,
                    "average_probability": None,
                    "average_score": None,
                    "average_latency_ms": None,
                    "risk_level_distribution": {
                        "low": 0,
                        "medium": 0,
                        "high": 0
                    },
                    "first_prediction": None,
                    "last_prediction": None
                }
                
        except Exception as e:
            self.logger.error(
                f"Failed to get customer prediction analytics: {e}",
                extra={"customer_id": customer_id},
                exc_info=True
            )
            raise DatabaseError(
                f"Failed to get customer prediction analytics: {str(e)}",
                original_error=e
            )
    
    def count_customer_predictions(
        self,
        customer_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        risk_level: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> int:
        """
        Count predictions for a customer with filtering.
        
        Args:
            customer_id: Customer identifier
            start_date: Filter predictions from this date (inclusive)
            end_date: Filter predictions to this date (inclusive)
            risk_level: Filter by risk level
            model_version: Filter by model version
            
        Returns:
            Total count of matching predictions
        """
        try:
            return self.repository.count_by_customer_id(
                customer_id=customer_id,
                start_date=start_date,
                end_date=end_date,
                risk_level=risk_level,
                model_version=model_version
            )
        except Exception as e:
            self.logger.error(
                f"Failed to count customer predictions: {e}",
                extra={"customer_id": customer_id},
                exc_info=True
            )
            raise DatabaseError(
                f"Failed to count customer predictions: {str(e)}",
                original_error=e
            )
    
    def get_prediction_by_id(self, prediction_id: str) -> Prediction:
        """
        Get prediction by ID.
        
        Args:
            prediction_id: Prediction identifier
            
        Returns:
            Prediction instance
            
        Raises:
            RecordNotFoundError: If prediction not found
        """
        try:
            return self.repository.get_by_id_or_raise(prediction_id)
        except RecordNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                f"Failed to get prediction: {e}",
                extra={"prediction_id": prediction_id},
                exc_info=True
            )
            raise DatabaseError(
                f"Failed to get prediction: {str(e)}",
                original_error=e
            )


class RawTransactionService:
    """Service for raw transaction operations."""
    
    def __init__(self, session: Session):
        """
        Initialize raw transaction service.
        
        Args:
            session: Database session
        """
        self.session = session
        self.repository = RawTransactionRepository(session)
        self.logger = get_logger(f"{__name__}.RawTransactionService")
    
    def upload_transactions(
        self,
        transactions: List[Dict[str, Any]],
        uploaded_by: str,
        data_source: str,
        file_name: str,
        data_version: Optional[str] = None
    ) -> int:
        """
        Upload multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
            uploaded_by: Username who uploaded the data
            data_source: Source system name
            file_name: Original file name
            data_version: Data version identifier
            
        Returns:
            Number of transactions uploaded
        """
        try:
            self.logger.info(
                "Uploading transactions",
                extra={
                    "count": len(transactions),
                    "uploaded_by": uploaded_by,
                    "data_source": data_source,
                    "file_name": file_name
                }
            )
            
            # Add upload metadata to each transaction
            for txn in transactions:
                txn["uploaded_by"] = uploaded_by
                txn["data_source"] = data_source
                txn["file_name"] = file_name
                if data_version:
                    txn["data_version"] = data_version
            
            count = self.repository.bulk_create(transactions)
            
            self.logger.info(
                f"Successfully uploaded {count} transactions",
                extra={"uploaded_by": uploaded_by}
            )
            
            return count
            
        except Exception as e:
            self.logger.error(
                "Failed to upload transactions",
                extra={
                    "count": len(transactions),
                    "error": str(e)
                },
                exc_info=True
            )
            raise DatabaseError(
                f"Failed to upload transactions: {str(e)}",
                original_error=e
            )


class BusinessKPIService:
    """Service for business KPI operations."""
    
    def __init__(self, session: Session):
        """
        Initialize business KPI service.
        
        Args:
            session: Database session
        """
        self.session = session
        self.repository = BusinessKPIRepository(session)
        self.logger = get_logger(f"{__name__}.BusinessKPIService")
    
    def calculate_and_save_kpis(
        self,
        period_start: datetime,
        period_end: datetime,
        period_type: str
    ) -> BusinessKPI:
        """
        Calculate and save business KPIs for a time period.
        
        Args:
            period_start: Period start timestamp
            period_end: Period end timestamp
            period_type: Period type ('hourly', 'daily', 'weekly', 'monthly')
            
        Returns:
            Saved BusinessKPI instance
        """
        try:
            self.logger.info(
                "Calculating business KPIs",
                extra={
                    "period_type": period_type,
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat()
                }
            )
            
            # Check if KPI already exists
            existing = self.repository.get_by_period(period_type, period_start, period_end)
            if existing:
                self.logger.warning("KPI already exists for this period, updating")
                repository = self.repository
            else:
                repository = self.repository
            
            # Calculate KPIs from predictions
            from sqlalchemy import func, and_, case, Integer
            
            stats = self.session.query(
                func.count(Prediction.prediction_id).label("total"),
                func.sum(case((Prediction.risk_level == "low", 1), else_=0)).label("approvals"),
                func.sum(case((Prediction.risk_level == "high", 1), else_=0)).label("rejections"),
                func.sum(case((Prediction.risk_level == "medium", 1), else_=0)).label("reviews"),
                func.avg(Prediction.probability).label("avg_risk_score"),
                func.percentile_cont(0.5).within_group(Prediction.probability).label("median_risk_score"),
                func.avg(Prediction.latency_ms).label("avg_latency"),
                func.percentile_cont(0.95).within_group(Prediction.latency_ms).label("p95_latency"),
                func.percentile_cont(0.99).within_group(Prediction.latency_ms).label("p99_latency"),
                func.count(func.distinct(Prediction.customer_id)).label("unique_customers")
            ).filter(
                and_(
                    Prediction.created_at >= period_start,
                    Prediction.created_at <= period_end
                )
            ).first()
            
            # Calculate rates
            total = stats.total or 0
            approvals = stats.approvals or 0
            rejections = stats.rejections or 0
            reviews = stats.reviews or 0
            
            approval_rate = Decimal(str(approvals / total)) if total > 0 else Decimal("0")
            rejection_rate = Decimal(str(rejections / total)) if total > 0 else Decimal("0")
            review_rate = Decimal(str(reviews / total)) if total > 0 else Decimal("0")
            
            # Create or update KPI
            kpi_data = {
                "period_start": period_start,
                "period_end": period_end,
                "period_type": period_type,
                "total_predictions": total,
                "approval_count": approvals,
                "rejection_count": rejections,
                "review_count": reviews,
                "approval_rate": approval_rate,
                "rejection_rate": rejection_rate,
                "review_rate": review_rate,
                "avg_risk_score": Decimal(str(stats.avg_risk_score)) if stats.avg_risk_score else None,
                "median_risk_score": Decimal(str(stats.median_risk_score)) if stats.median_risk_score else None,
                "unique_customers": stats.unique_customers,
                "avg_latency_ms": Decimal(str(stats.avg_latency)) if stats.avg_latency else None,
                "p95_latency_ms": Decimal(str(stats.p95_latency)) if stats.p95_latency else None,
                "p99_latency_ms": Decimal(str(stats.p99_latency)) if stats.p99_latency else None
            }
            
            if existing:
                kpi = self.repository.update(existing.id, **kpi_data)
            else:
                kpi = self.repository.create(**kpi_data)
            
            self.logger.info(
                "Business KPIs calculated and saved",
                extra={
                    "period_type": period_type,
                    "total_predictions": total,
                    "approval_rate": float(approval_rate)
                }
            )
            
            return kpi
            
        except Exception as e:
            self.logger.error(
                "Failed to calculate and save KPIs",
                extra={
                    "period_type": period_type,
                    "error": str(e)
                },
                exc_info=True
            )
            raise DatabaseError(
                f"Failed to calculate KPIs: {str(e)}",
                original_error=e
            )


class DataVersionService:
    """Service for data versioning operations."""
    
    def __init__(self, session: Session):
        """
        Initialize data version service.
        
        Args:
            session: Database session
        """
        self.session = session
        self.repository = DataVersionRepository(session)
        self.logger = get_logger(f"{__name__}.DataVersionService")
    
    def create_version(
        self,
        data_type: str,
        version: str,
        file_path: str,
        file_size: int,
        checksum_sha256: str,
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ) -> DataVersion:
        """
        Create a new data version record.
        
        Args:
            data_type: Type of data ('raw_transactions', 'processed', 'features', etc.)
            version: Version string (e.g., 'v1', 'v2')
            file_path: Path to the data file
            file_size: Size in bytes
            checksum_sha256: SHA256 checksum
            metadata: Additional metadata (JSON)
            dependencies: List of dependency versions
            
        Returns:
            Created DataVersion instance
        """
        try:
            # Check if version already exists
            existing = self.repository.get_by_type_and_version(data_type, version)
            if existing:
                self.logger.warning(
                    f"Data version {data_type}:{version} already exists",
                    extra={"data_type": data_type, "version": version}
                )
                return existing
            
            version_data = {
                "data_type": data_type,
                "version": version,
                "file_path": file_path,
                "file_size": file_size,
                "checksum_sha256": checksum_sha256,
                "data_metadata": metadata or {},
                "dependencies": dependencies or []
            }
            
            data_version = self.repository.create(**version_data)
            
            self.logger.info(
                f"Created data version {data_type}:{version}",
                extra={"data_type": data_type, "version": version}
            )
            
            return data_version
            
        except Exception as e:
            self.logger.error(
                f"Failed to create data version: {e}",
                extra={"data_type": data_type, "version": version},
                exc_info=True
            )
            raise DatabaseError(
                f"Failed to create data version: {str(e)}",
                original_error=e
            )
    
    def get_latest_version(self, data_type: str) -> Optional[DataVersion]:
        """Get latest version for a data type."""
        return self.repository.get_latest_by_type(data_type)
    
    def get_version(self, data_type: str, version: str) -> Optional[DataVersion]:
        """Get specific version."""
        return self.repository.get_by_type_and_version(data_type, version)


class DataLineageService:
    """Service for data lineage tracking operations."""
    
    def __init__(self, session: Session):
        """
        Initialize data lineage service.
        
        Args:
            session: Database session
        """
        self.session = session
        self.repository = DataLineageRepository(session)
        self.logger = get_logger(f"{__name__}.DataLineageService")
    
    def create_lineage(
        self,
        source_data_version_id: int,
        source_data_type: str,
        source_version: str,
        target_type: str,
        target_id: str,
        target_name: Optional[str] = None,
        relationship_type: str = "used_for",
        operation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataLineage:
        """
        Create a lineage record linking data version to a target.
        
        Args:
            source_data_version_id: ID of the source data version
            source_data_type: Type of source data
            source_version: Version of source data
            target_type: Type of target ('model', 'prediction', 'feature_set', etc.)
            target_id: ID of target (model_version, prediction_id, etc.)
            target_name: Human-readable name of target
            relationship_type: Type of relationship ('trained_on', 'used_for', 'derived_from', etc.)
            operation: Operation performed ('training', 'prediction', 'feature_engineering', etc.)
            metadata: Additional context
            
        Returns:
            Created DataLineage instance
        """
        try:
            lineage_data = {
                "source_data_version_id": source_data_version_id,
                "source_data_type": source_data_type,
                "source_version": source_version,
                "target_type": target_type,
                "target_id": target_id,
                "target_name": target_name,
                "relationship_type": relationship_type,
                "operation": operation,
                "lineage_metadata": metadata or {}
            }
            
            lineage = self.repository.create(**lineage_data)
            
            self.logger.info(
                f"Created lineage: {source_data_type}:{source_version} -> {target_type}:{target_id}",
                extra={
                    "source_data_type": source_data_type,
                    "source_version": source_version,
                    "target_type": target_type,
                    "target_id": target_id
                }
            )
            
            return lineage
            
        except Exception as e:
            self.logger.error(
                f"Failed to create lineage: {e}",
                extra={
                    "source_data_version_id": source_data_version_id,
                    "target_type": target_type,
                    "target_id": target_id
                },
                exc_info=True
            )
            raise DatabaseError(
                f"Failed to create lineage: {str(e)}",
                original_error=e
            )
    
    def get_lineage_by_source(self, source_data_version_id: int) -> List[DataLineage]:
        """Get all lineage records for a source data version."""
        return self.repository.get_by_source_version(source_data_version_id)
    
    def get_lineage_by_target(
        self,
        target_type: str,
        target_id: str
    ) -> List[DataLineage]:
        """Get all lineage records for a target."""
        return self.repository.get_by_target(target_type, target_id)
    
    def get_lineage_graph(
        self,
        data_version_id: Optional[int] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None
    ) -> List[DataLineage]:
        """Get lineage graph with optional filters."""
        return self.repository.get_lineage_graph(data_version_id, target_type, target_id)
