"""
A/B Testing Framework

Provides comprehensive A/B testing capabilities:
- Traffic splitting with consistent hashing
- Multi-model management
- Statistical significance testing
- Automated winner selection
"""

import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from decimal import Decimal

from src.utils.logging import get_logger
from src.database.connection import get_db_session
from src.database.models import Experiment, ExperimentAssignment, ExperimentMetric, Prediction
from src.database.repositories import ExperimentRepository, ExperimentAssignmentRepository, ExperimentMetricRepository

logger = get_logger(__name__)

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - statistical tests will be limited")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - statistical tests will be limited")


class TrafficSplitter:
    """
    Traffic splitter for A/B testing.
    
    Uses consistent hashing to ensure the same customer/request
    always gets the same variant assignment.
    """
    
    def __init__(self, assignment_method: str = "hash"):
        """
        Initialize traffic splitter.
        
        Args:
            assignment_method: Method for assignment ('hash', 'random')
        """
        self.assignment_method = assignment_method
        self.logger = get_logger(f"{__name__}.TrafficSplitter")
    
    def assign_variant(
        self,
        entity_id: str,
        variants: List[Dict[str, Any]],
        traffic_percentage: int = 100
    ) -> Optional[str]:
        """
        Assign a variant to an entity (customer/request).
        
        Args:
            entity_id: Customer ID or request ID
            variants: List of variant configurations with traffic_percentage
            traffic_percentage: Overall traffic percentage for experiment (0-100)
            
        Returns:
            Variant name or None if entity not in experiment
        """
        if not variants:
            return None
        
        # Check if entity should be in experiment (based on traffic_percentage)
        if traffic_percentage < 100:
            # Use hash to determine if entity is in experiment
            hash_value = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
            if (hash_value % 100) >= traffic_percentage:
                return None  # Entity not in experiment
        
        if self.assignment_method == "hash":
            return self._hash_based_assignment(entity_id, variants)
        elif self.assignment_method == "random":
            return self._random_assignment(variants)
        else:
            self.logger.warning(f"Unknown assignment method: {self.assignment_method}, using hash")
            return self._hash_based_assignment(entity_id, variants)
    
    def _hash_based_assignment(self, entity_id: str, variants: List[Dict[str, Any]]) -> str:
        """Assign variant using consistent hashing."""
        # Create hash from entity_id
        hash_value = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
        
        # Calculate cumulative percentages
        cumulative = 0
        for variant in variants:
            cumulative += variant.get("traffic_percentage", 0)
            if (hash_value % 100) < cumulative:
                return variant["name"]
        
        # Fallback to first variant
        return variants[0]["name"]
    
    def _random_assignment(self, variants: List[Dict[str, Any]]) -> str:
        """Assign variant randomly based on traffic percentages."""
        # Calculate weights
        weights = [v.get("traffic_percentage", 0) for v in variants]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return variants[0]["name"]
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Random selection based on weights
        return np.random.choice([v["name"] for v in variants], p=weights)


class StatisticalAnalyzer:
    """
    Statistical analysis for A/B testing.
    
    Provides statistical significance testing, confidence intervals,
    and winner determination.
    """
    
    def __init__(self):
        """Initialize statistical analyzer."""
        self.logger = get_logger(f"{__name__}.StatisticalAnalyzer")
    
    def t_test(
        self,
        control_values: List[float],
        treatment_values: List[float],
        alternative: str = "two-sided"
    ) -> Dict[str, Any]:
        """
        Perform t-test for comparing two groups.
        
        Args:
            control_values: Values from control group
            treatment_values: Values from treatment group
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Dictionary with t-statistic, p-value, and conclusion
        """
        if len(control_values) < 2 or len(treatment_values) < 2:
            return {
                "error": "Insufficient data for t-test",
                "p_value": 1.0,
                "significant": False
            }
        
        if not SCIPY_AVAILABLE:
            return {
                "error": "scipy not available for statistical tests",
                "p_value": 1.0,
                "significant": False
            }
        
        try:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            
            # Adjust for one-sided test if needed
            if alternative == "greater":
                p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
            elif alternative == "less":
                p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2
            
            return {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "control_mean": float(np.mean(control_values)),
                "treatment_mean": float(np.mean(treatment_values)),
                "control_std": float(np.std(control_values)),
                "treatment_std": float(np.std(treatment_values)),
                "control_size": len(control_values),
                "treatment_size": len(treatment_values)
            }
        except Exception as e:
            self.logger.error(f"Error performing t-test: {e}", exc_info=True)
            return {
                "error": str(e),
                "p_value": 1.0,
                "significant": False
            }
    
    def chi_square_test(
        self,
        control_success: int,
        control_total: int,
        treatment_success: int,
        treatment_total: int
    ) -> Dict[str, Any]:
        """
        Perform chi-square test for comparing proportions.
        
        Args:
            control_success: Number of successes in control
            control_total: Total in control
            treatment_success: Number of successes in treatment
            treatment_total: Total in treatment
            
        Returns:
            Dictionary with chi-square statistic, p-value, and conclusion
        """
        if not SCIPY_AVAILABLE:
            return {
                "error": "scipy not available for statistical tests",
                "p_value": 1.0,
                "significant": False
            }
        
        try:
            # Create contingency table
            contingency = np.array([
                [control_success, control_total - control_success],
                [treatment_success, treatment_total - treatment_success]
            ])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            control_rate = control_success / control_total if control_total > 0 else 0
            treatment_rate = treatment_success / treatment_total if treatment_total > 0 else 0
            
            return {
                "chi2_statistic": float(chi2),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "significant": p_value < 0.05,
                "control_rate": float(control_rate),
                "treatment_rate": float(treatment_rate),
                "control_success": control_success,
                "control_total": control_total,
                "treatment_success": treatment_success,
                "treatment_total": treatment_total
            }
        except Exception as e:
            self.logger.error(f"Error performing chi-square test: {e}", exc_info=True)
            return {
                "error": str(e),
                "p_value": 1.0,
                "significant": False
            }
    
    def confidence_interval(
        self,
        values: List[float],
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate confidence interval for a sample.
        
        Args:
            values: Sample values
            confidence_level: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Dictionary with lower and upper bounds
        """
        if len(values) < 2:
            return {"lower": 0.0, "upper": 0.0}
        
        try:
            mean = np.mean(values)
            std = np.std(values, ddof=1)  # Sample standard deviation
            n = len(values)
            
            # Calculate standard error
            se = std / np.sqrt(n)
            
            # Get critical value (t-distribution for small samples, normal for large)
            if SCIPY_AVAILABLE:
                if n < 30:
                    critical_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
                else:
                    critical_value = stats.norm.ppf((1 + confidence_level) / 2)
            else:
                # Fallback: use normal approximation
                import math
                # Z-score for confidence level (approximate)
                if confidence_level == 0.95:
                    critical_value = 1.96
                elif confidence_level == 0.99:
                    critical_value = 2.58
                else:
                    critical_value = 1.96  # Default to 95%
            
            margin = critical_value * se
            
            return {
                "lower": float(mean - margin),
                "upper": float(mean + margin),
                "mean": float(mean),
                "std": float(std),
                "n": n
            }
        except Exception as e:
            self.logger.error(f"Error calculating confidence interval: {e}", exc_info=True)
            return {"lower": 0.0, "upper": 0.0}
    
    def compare_variants(
        self,
        control_metrics: Dict[str, Any],
        treatment_metrics: Dict[str, Any],
        primary_metric: str = "accuracy",
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compare two variants and determine winner.
        
        Args:
            control_metrics: Metrics for control variant
            treatment_metrics: Metrics for treatment variant
            primary_metric: Primary metric to compare
            significance_level: Significance level (default 0.05)
            
        Returns:
            Comparison results with winner determination
        """
        control_value = control_metrics.get(primary_metric)
        treatment_value = treatment_metrics.get(primary_metric)
        
        if control_value is None or treatment_value is None:
            return {
                "error": f"Primary metric '{primary_metric}' not found in metrics",
                "winner": None
            }
        
        # For continuous metrics, use t-test
        # For now, assume we have sample data - in production, would get from predictions
        # This is a simplified version - would need actual prediction values
        
        improvement = treatment_value - control_value
        improvement_pct = (improvement / control_value * 100) if control_value > 0 else 0
        
        # Determine winner (simplified - would use actual statistical test)
        winner = None
        if improvement_pct > 0:
            winner = "treatment"
        elif improvement_pct < 0:
            winner = "control"
        else:
            winner = "tie"
        
        return {
            "control_value": float(control_value),
            "treatment_value": float(treatment_value),
            "improvement": float(improvement),
            "improvement_pct": float(improvement_pct),
            "winner": winner,
            "significant": False  # Would be determined by actual statistical test
        }


class ABTestingFramework:
    """
    Main A/B testing framework.
    
    Manages experiments, traffic splitting, model routing,
    and statistical analysis.
    """
    
    def __init__(self):
        """Initialize A/B testing framework."""
        self.traffic_splitter = TrafficSplitter()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.logger = get_logger(f"{__name__}.ABTestingFramework")
        self.loaded_models: Dict[str, Any] = {}  # variant_name -> model
    
    def get_assignment(
        self,
        experiment_id: int,
        entity_id: str,
        entity_type: str = "customer"
    ) -> Optional[str]:
        """
        Get variant assignment for an entity.
        
        Checks database first, then creates new assignment if needed.
        
        Args:
            experiment_id: Experiment ID
            entity_id: Customer ID or request ID
            entity_type: Type of entity ('customer', 'request')
            
        Returns:
            Variant name or None if not in experiment
        """
        try:
            with get_db_session() as session:
                assignment_repo = ExperimentAssignmentRepository(session)
                experiment_repo = ExperimentRepository(session)
                
                # Check for existing assignment
                assignment = assignment_repo.get_by_experiment_and_entity(
                    experiment_id, entity_id, entity_type
                )
                
                if assignment:
                    return assignment.variant_name
                
                # Get experiment
                experiment = experiment_repo.get_by_id(experiment_id)
                if not experiment or experiment.status != "running":
                    return None
                
                # Create new assignment
                variants = experiment.variants if isinstance(experiment.variants, list) else []
                variant_name = self.traffic_splitter.assign_variant(
                    entity_id,
                    variants,
                    experiment.traffic_percentage
                )
                
                if variant_name:
                    # Save assignment
                    assignment_repo.create(
                        experiment_id=experiment_id,
                        entity_id=entity_id,
                        entity_type=entity_type,
                        variant_name=variant_name,
                        assignment_hash=hashlib.md5(entity_id.encode()).hexdigest()
                    )
                    session.commit()
                
                return variant_name
                
        except Exception as e:
            self.logger.error(f"Error getting assignment: {e}", exc_info=True)
            return None
    
    def load_model_for_variant(
        self,
        variant_name: str,
        model_name: str,
        model_version: str
    ) -> bool:
        """
        Load a model for a variant.
        
        Args:
            variant_name: Variant name
            model_name: Model name
            model_version: Model version
            
        Returns:
            True if loaded successfully
        """
        try:
            import mlflow
            import mlflow.sklearn
            
            # Set MLflow tracking URI
            from src.utils.config import settings
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            
            # Load model from registry
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.sklearn.load_model(model_uri)
            if model:
                self.loaded_models[variant_name] = model
                self.logger.info(f"Loaded model for variant {variant_name}: {model_name} v{model_version}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error loading model for variant {variant_name}: {e}", exc_info=True)
            return False
    
    def get_model_for_variant(self, variant_name: str) -> Optional[Any]:
        """
        Get loaded model for a variant.
        
        Args:
            variant_name: Variant name
            
        Returns:
            Model object or None
        """
        return self.loaded_models.get(variant_name)
    
    def calculate_experiment_metrics(
        self,
        experiment_id: int,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics for all variants in an experiment.
        
        Args:
            experiment_id: Experiment ID
            period_start: Start of period (default: experiment start)
            period_end: End of period (default: now)
            
        Returns:
            Dictionary mapping variant_name to metrics
        """
        try:
            with get_db_session() as session:
                experiment_repo = ExperimentRepository(session)
                prediction_repo = session.query(Prediction)
                
                experiment = experiment_repo.get_by_id(experiment_id)
                if not experiment:
                    return {}
                
                variants = experiment.variants if isinstance(experiment.variants, list) else []
                variant_names = [v["name"] for v in variants]
                
                # Get period
                if not period_start:
                    period_start = experiment.actual_started_at or experiment.start_date
                if not period_end:
                    period_end = datetime.now(timezone.utc)
                
                # Get predictions for this experiment
                # Note: In production, would track experiment_id in Prediction model
                # For now, we'll use model_version to identify variants
                variant_metrics = {}
                
                for variant in variants:
                    variant_name = variant["name"]
                    model_version = variant.get("model_version")
                    
                    if not model_version:
                        continue
                    
                    # Get predictions for this model version
                    predictions = session.query(Prediction).filter(
                        Prediction.model_version == model_version,
                        Prediction.created_at >= period_start,
                        Prediction.created_at <= period_end
                    ).all()
                    
                    if not predictions:
                        continue
                    
                    # Calculate metrics
                    total = len(predictions)
                    probabilities = [float(p.probability) for p in predictions]
                    latencies = [float(p.latency_ms) for p in predictions if p.latency_ms]
                    
                    # Risk level counts
                    high_risk = sum(1 for p in predictions if p.risk_level == "high")
                    low_risk = sum(1 for p in predictions if p.risk_level == "low")
                    
                    # Accuracy (would need ground truth - simplified)
                    # In production, would compare predictions to actual outcomes
                    accuracy = None
                    
                    # Calculate statistics
                    mean_prob = np.mean(probabilities) if probabilities else 0
                    std_prob = np.std(probabilities) if len(probabilities) > 1 else 0
                    
                    avg_latency = np.mean(latencies) if latencies else None
                    p95_latency = np.percentile(latencies, 95) if len(latencies) > 0 else None
                    
                    variant_metrics[variant_name] = {
                        "sample_size": total,
                        "total_predictions": total,
                        "high_risk_predictions": high_risk,
                        "low_risk_predictions": low_risk,
                        "mean_probability": float(mean_prob),
                        "std_probability": float(std_prob),
                        "avg_latency_ms": float(avg_latency) if avg_latency else None,
                        "p95_latency_ms": float(p95_latency) if p95_latency else None,
                        "accuracy": accuracy
                    }
                
                return variant_metrics
                
        except Exception as e:
            self.logger.error(f"Error calculating experiment metrics: {e}", exc_info=True)
            return {}
    
    def determine_winner(
        self,
        experiment_id: int,
        primary_metric: str = "accuracy",
        significance_level: float = 0.05,
        minimum_improvement: float = 0.01
    ) -> Optional[Dict[str, Any]]:
        """
        Determine winner of experiment based on statistical analysis.
        
        Args:
            experiment_id: Experiment ID
            primary_metric: Primary metric to compare
            significance_level: Significance level
            minimum_improvement: Minimum improvement to declare winner
            
        Returns:
            Winner information or None
        """
        try:
            with get_db_session() as session:
                experiment_repo = ExperimentRepository(session)
                experiment = experiment_repo.get_by_id(experiment_id)
                
                if not experiment:
                    return None
                
                # Get metrics for all variants
                variant_metrics = self.calculate_experiment_metrics(experiment_id)
                
                if len(variant_metrics) < 2:
                    return {"error": "Need at least 2 variants to determine winner"}
                
                # Find control variant (first variant or explicitly marked)
                variants = experiment.variants if isinstance(experiment.variants, list) else []
                control_variant = None
                for v in variants:
                    if v.get("is_control", False) or v["name"] == "control":
                        control_variant = v["name"]
                        break
                
                if not control_variant:
                    control_variant = variants[0]["name"] if variants else None
                
                if not control_variant or control_variant not in variant_metrics:
                    return {"error": "Control variant not found"}
                
                control_metrics = variant_metrics[control_variant]
                
                # Compare each treatment variant to control
                best_variant = control_variant
                best_improvement = 0
                comparisons = {}
                
                for variant_name, metrics in variant_metrics.items():
                    if variant_name == control_variant:
                        continue
                    
                    comparison = self.statistical_analyzer.compare_variants(
                        control_metrics,
                        metrics,
                        primary_metric,
                        significance_level
                    )
                    
                    comparisons[variant_name] = comparison
                    
                    if comparison.get("improvement_pct", 0) > best_improvement:
                        best_improvement = comparison["improvement_pct"]
                        best_variant = variant_name
                
                # Check if improvement meets minimum threshold
                if best_improvement < (minimum_improvement * 100):
                    best_variant = control_variant  # No significant improvement
                
                return {
                    "winner": best_variant,
                    "control_variant": control_variant,
                    "improvement_pct": best_improvement,
                    "comparisons": comparisons,
                    "meets_threshold": best_improvement >= (minimum_improvement * 100)
                }
                
        except Exception as e:
            self.logger.error(f"Error determining winner: {e}", exc_info=True)
            return None


def get_ab_testing_framework() -> ABTestingFramework:
    """Get a singleton ABTestingFramework instance."""
    global _ab_testing_framework
    if '_ab_testing_framework' not in globals():
        _ab_testing_framework = ABTestingFramework()
    return _ab_testing_framework
