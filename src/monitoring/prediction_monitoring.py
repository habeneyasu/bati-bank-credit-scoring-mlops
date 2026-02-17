"""
Prediction Monitoring Module

Implements Step 4: Long-term recommendation from SCORING_DIAGNOSTIC.md

Monitors prediction distributions to detect:
- Prediction clustering (all probabilities in narrow range)
- Model bias issues
- Calibration problems
- Risk level distribution anomalies
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from src.utils.logging import get_logger
from src.utils.config import settings
from src.database.connection import get_db_session
from src.database.models import Prediction
from src.monitoring.drift_detection import DriftMonitor
from sqlalchemy import func, and_

logger = get_logger(__name__)


class PredictionMonitor:
    """
    Monitor prediction distributions for clustering and bias issues.
    """
    
    def __init__(self):
        """Initialize prediction monitor."""
        self.logger = get_logger(f"{__name__}.PredictionMonitor")
        self.drift_monitor = DriftMonitor()
    
    def monitor_recent_predictions(
        self,
        hours: int = 24,
        min_samples: int = 50,
        probability_range_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Monitor recent predictions for clustering issues.
        
        Args:
            hours: Number of hours to look back
            min_samples: Minimum samples required
            probability_range_threshold: Alert if probability range < threshold
            
        Returns:
            Monitoring results
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            with get_db_session() as session:
                # Get recent predictions
                predictions = session.query(
                    Prediction.probability,
                    Prediction.risk_level,
                    Prediction.customer_id
                ).filter(
                    Prediction.created_at >= cutoff_time
                ).all()
                
                if len(predictions) < min_samples:
                    return {
                        "monitoring_status": "insufficient_data",
                        "sample_count": len(predictions),
                        "required_samples": min_samples,
                        "message": f"Only {len(predictions)} predictions found (need {min_samples})"
                    }
                
                # Extract probabilities
                probabilities = np.array([float(p.probability) for p in predictions])
                risk_levels = [p.risk_level for p in predictions]
                
                # Use drift detector's clustering detection
                from src.monitoring.drift_detection import DriftDetector
                detector = DriftDetector()
                clustering_result = detector.monitor_prediction_clustering(
                    probabilities,
                    probability_range_threshold=probability_range_threshold,
                    min_samples=min_samples
                )
                
                # Additional analysis
                risk_distribution = {
                    "low": risk_levels.count("low"),
                    "medium": risk_levels.count("medium"),
                    "high": risk_levels.count("high"),
                    "total": len(risk_levels)
                }
                
                risk_percentages = {
                    "low": (risk_distribution["low"] / risk_distribution["total"]) * 100,
                    "medium": (risk_distribution["medium"] / risk_distribution["total"]) * 100,
                    "high": (risk_distribution["high"] / risk_distribution["total"]) * 100
                }
                
                # Check for imbalanced risk distribution
                all_same_risk = (
                    risk_distribution["low"] == risk_distribution["total"] or
                    risk_distribution["medium"] == risk_distribution["total"] or
                    risk_distribution["high"] == risk_distribution["total"]
                )
                
                # Check if one risk level dominates (>90%)
                dominant_risk = None
                for risk, count in risk_distribution.items():
                    if risk != "total" and count / risk_distribution["total"] > 0.9:
                        dominant_risk = risk
                        break
                
                return {
                    "monitoring_status": "complete",
                    "time_window_hours": hours,
                    "sample_count": len(predictions),
                    "cutoff_time": cutoff_time.isoformat(),
                    "clustering_analysis": clustering_result,
                    "risk_distribution": risk_distribution,
                    "risk_percentages": risk_percentages,
                    "issues": {
                        "all_same_risk": all_same_risk,
                        "dominant_risk_level": dominant_risk,
                        "clustering_detected": clustering_result.get("clustering_detected", False),
                        "severity": clustering_result.get("severity", "none")
                    },
                    "alerts": self._generate_alerts(
                        clustering_result,
                        all_same_risk,
                        dominant_risk,
                        risk_percentages
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error monitoring predictions: {e}", exc_info=True)
            return {
                "monitoring_status": "error",
                "error": str(e)
            }
    
    def _generate_alerts(
        self,
        clustering_result: Dict[str, Any],
        all_same_risk: bool,
        dominant_risk: Optional[str],
        risk_percentages: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate alerts based on monitoring results."""
        alerts = []
        
        if clustering_result.get("clustering_detected", False):
            severity = clustering_result.get("severity", "none")
            alerts.append({
                "type": "prediction_clustering",
                "severity": severity,
                "message": f"Prediction clustering detected ({severity} severity)",
                "details": clustering_result.get("recommendations", [])
            })
        
        if all_same_risk:
            alerts.append({
                "type": "risk_distribution_anomaly",
                "severity": "critical",
                "message": "All predictions have the same risk level",
                "details": [
                    "This indicates a serious model issue",
                    "Model may not be differentiating between customers",
                    "Review model training and calibration"
                ]
            })
        
        if dominant_risk:
            alerts.append({
                "type": "imbalanced_risk_distribution",
                "severity": "major",
                "message": f"{dominant_risk.upper()} risk dominates ({risk_percentages[dominant_risk]:.1f}%)",
                "details": [
                    f"Over 90% of predictions are {dominant_risk} risk",
                    "Model may be biased or not calibrated properly"
                ]
            })
        
        return alerts
    
    def get_prediction_statistics(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get prediction statistics over time period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Statistics dictionary
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            
            with get_db_session() as session:
                # Get statistics
                stats_query = session.query(
                    func.count(Prediction.prediction_id).label('total'),
                    func.avg(Prediction.probability).label('avg_prob'),
                    func.min(Prediction.probability).label('min_prob'),
                    func.max(Prediction.probability).label('max_prob'),
                    func.stddev(Prediction.probability).label('std_prob')
                ).filter(
                    Prediction.created_at >= cutoff_time
                )
                
                result = stats_query.first()
                
                if not result or result.total == 0:
                    return {"error": "No predictions found"}
                
                # Get risk level distribution
                risk_dist = session.query(
                    Prediction.risk_level,
                    func.count(Prediction.prediction_id).label('count')
                ).filter(
                    Prediction.created_at >= cutoff_time
                ).group_by(Prediction.risk_level).all()
                
                risk_distribution = {risk: count for risk, count in risk_dist}
                
                return {
                    "period_days": days,
                    "cutoff_time": cutoff_time.isoformat(),
                    "total_predictions": result.total,
                    "probability_stats": {
                        "mean": float(result.avg_prob) if result.avg_prob else None,
                        "min": float(result.min_prob) if result.min_prob else None,
                        "max": float(result.max_prob) if result.max_prob else None,
                        "std": float(result.std_prob) if result.std_prob else None,
                        "range": float(result.max_prob - result.min_prob) if result.max_prob and result.min_prob else None
                    },
                    "risk_distribution": risk_distribution,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting prediction statistics: {e}", exc_info=True)
            return {"error": str(e)}


def get_prediction_monitor() -> PredictionMonitor:
    """Get singleton PredictionMonitor instance."""
    if not hasattr(get_prediction_monitor, "_instance"):
        get_prediction_monitor._instance = PredictionMonitor()
    return get_prediction_monitor._instance
