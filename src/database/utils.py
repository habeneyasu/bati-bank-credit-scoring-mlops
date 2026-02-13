"""
Database utility functions for common operations.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from src.utils.logging import get_logger
from src.database.connection import get_db_session
from src.database.models import Prediction, BusinessKPI
from src.database.exceptions import DatabaseError

logger = get_logger(__name__)


def get_prediction_statistics(
    session: Session,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> Dict[str, Any]:
    """
    Get prediction statistics for a date range.
    
    Args:
        session: Database session
        start_date: Start date (default: 30 days ago)
        end_date: End date (default: today)
        
    Returns:
        Dictionary with statistics
    """
    try:
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        logger.debug(
            "Getting prediction statistics",
            extra={"start_date": str(start_date), "end_date": str(end_date)}
        )
        
        from sqlalchemy import case, Integer
        
        stats = session.query(
            func.count(Prediction.prediction_id).label("total"),
            func.sum(case((Prediction.risk_level == "low", 1), else_=0)).label("low_risk"),
            func.sum(case((Prediction.risk_level == "medium", 1), else_=0)).label("medium_risk"),
            func.sum(case((Prediction.risk_level == "high", 1), else_=0)).label("high_risk"),
            func.avg(Prediction.probability).label("avg_probability"),
            func.avg(Prediction.customer_score).label("avg_score"),
            func.avg(Prediction.latency_ms).label("avg_latency"),
            func.count(func.distinct(Prediction.customer_id)).label("unique_customers")
        ).filter(
            and_(
                Prediction.created_at_date >= start_date,
                Prediction.created_at_date <= end_date
            )
        ).first()
        
        result = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_predictions": stats.total or 0,
            "low_risk_count": stats.low_risk or 0,
            "medium_risk_count": stats.medium_risk or 0,
            "high_risk_count": stats.high_risk or 0,
            "avg_probability": float(stats.avg_probability) if stats.avg_probability else None,
            "avg_customer_score": float(stats.avg_score) if stats.avg_score else None,
            "avg_latency_ms": float(stats.avg_latency) if stats.avg_latency else None,
            "unique_customers": stats.unique_customers or 0
        }
        
        logger.debug(f"Statistics retrieved: {result['total_predictions']} predictions")
        return result
        
    except Exception as e:
        logger.error(f"Error getting prediction statistics: {e}", exc_info=True)
        raise DatabaseError(f"Failed to get statistics: {str(e)}", original_error=e)


def check_database_health(session: Session) -> Dict[str, Any]:
    """
    Check database health and connectivity.
    
    Args:
        session: Database session
        
    Returns:
        Dictionary with health status
    """
    try:
        # Test basic query
        result = session.execute("SELECT 1").scalar()
        
        # Get table counts
        from src.database.models import (
            User, Role, Prediction, RawTransaction, BusinessKPI
        )
        
        counts = {
            "users": session.query(func.count(User.user_id)).scalar() or 0,
            "roles": session.query(func.count(Role.role_id)).scalar() or 0,
            "predictions": session.query(func.count(Prediction.prediction_id)).scalar() or 0,
            "raw_transactions": session.query(func.count(RawTransaction.transaction_id)).scalar() or 0,
            "business_kpis": session.query(func.count(BusinessKPI.id)).scalar() or 0
        }
        
        return {
            "status": "healthy",
            "connected": True,
            "table_counts": counts,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
