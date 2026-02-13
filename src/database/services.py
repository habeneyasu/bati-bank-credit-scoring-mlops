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
from src.database.models import Prediction, RawTransaction, BusinessKPI
from src.database.repositories import (
    PredictionRepository,
    RawTransactionRepository,
    BusinessKPIRepository,
    UserRepository
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
            customer_score: Credit score (0-1000)
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
        limit: Optional[int] = 100
    ) -> List[Prediction]:
        """
        Get all predictions for a customer.
        
        Args:
            customer_id: Customer identifier
            limit: Maximum number of predictions to return
            
        Returns:
            List of predictions ordered by most recent first
        """
        try:
            self.logger.debug(f"Getting predictions for customer: {customer_id}")
            predictions = self.repository.get_by_customer_id(
                customer_id=customer_id,
                limit=limit
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
