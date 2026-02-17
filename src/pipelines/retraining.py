"""
Automated Model Retraining Pipeline

Provides comprehensive retraining capabilities:
- Scheduled retraining jobs
- Trigger-based retraining (drift, new data, performance degradation)
- Automated model validation
- Automated model promotion
- Rollback on performance degradation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import uuid
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.config import settings
from src.database.connection import get_db_session
from src.database.models import (
    RetrainingJob, RetrainingSchedule, ModelValidationRule,
    ModelMetadata
)
from src.database.repositories import (
    RetrainingJobRepository, RetrainingScheduleRepository,
    ModelValidationRuleRepository
)
from sqlalchemy import and_, desc, or_
from src.models.training import ModelTrainer
from src.models.tracking import MLflowTracker
from src.features.splitting import load_splits
from src.monitoring.drift_detection import DriftDetector

logger = get_logger(__name__)


class ModelValidator:
    """
    Validates models against performance thresholds and baseline comparisons.
    """
    
    def __init__(self):
        """Initialize model validator."""
        self.logger = get_logger(f"{__name__}.ModelValidator")
    
    def validate_model(
        self,
        metrics: Dict[str, float],
        model_name: str,
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Validate model against validation rules.
        
        Args:
            metrics: Model metrics (accuracy, roc_auc, etc.)
            model_name: Name of the model
            baseline_metrics: Metrics from baseline model for comparison
            
        Returns:
            Validation results with pass/fail status
        """
        try:
            with get_db_session() as session:
                rule_repo = ModelValidationRuleRepository(session)
                rules = rule_repo.get_active_rules_for_model(model_name)
                
                if not rules:
                    self.logger.warning(f"No validation rules found for model {model_name}, validation passed by default")
                    return {
                        "passed": True,
                        "errors": [],
                        "warnings": [],
                        "rule_results": []
                    }
                
                errors = []
                warnings = []
                rule_results = []
                
                for rule in rules:
                    metric_value = metrics.get(rule.metric_name)
                    
                    if metric_value is None:
                        warning = f"Metric '{rule.metric_name}' not found in model metrics"
                        if rule.is_required:
                            errors.append(warning)
                        else:
                            warnings.append(warning)
                        rule_results.append({
                            "rule_name": rule.rule_name,
                            "metric": rule.metric_name,
                            "status": "error" if rule.is_required else "warning",
                            "message": warning
                        })
                        continue
                    
                    # Evaluate rule based on comparison type
                    if rule.comparison_type == "absolute":
                        passed = self._evaluate_absolute_rule(
                            metric_value, rule.comparison_operator, float(rule.threshold_value)
                        )
                    elif rule.comparison_type == "relative_to_baseline":
                        if baseline_metrics is None:
                            warnings.append(f"Baseline metrics not provided for relative comparison in rule {rule.rule_name}")
                            passed = True  # Skip if no baseline
                        else:
                            baseline_value = baseline_metrics.get(rule.metric_name)
                            if baseline_value is None:
                                warnings.append(f"Baseline metric '{rule.metric_name}' not found for rule {rule.rule_name}")
                                passed = True
                            else:
                                passed = self._evaluate_relative_rule(
                                    metric_value, baseline_value, rule.comparison_operator, float(rule.threshold_value)
                                )
                    elif rule.comparison_type == "relative_improvement":
                        if baseline_metrics is None:
                            warnings.append(f"Baseline metrics not provided for improvement check in rule {rule.rule_name}")
                            passed = True
                        else:
                            baseline_value = baseline_metrics.get(rule.metric_name)
                            if baseline_value is None:
                                warnings.append(f"Baseline metric '{rule.metric_name}' not found for rule {rule.rule_name}")
                                passed = True
                            else:
                                improvement = (metric_value - baseline_value) / baseline_value if baseline_value > 0 else 0
                                min_improvement = float(rule.minimum_improvement) if rule.minimum_improvement else 0.0
                                passed = improvement >= min_improvement
                    else:
                        passed = True  # Unknown comparison type, pass by default
                    
                    rule_results.append({
                        "rule_name": rule.rule_name,
                        "metric": rule.metric_name,
                        "metric_value": float(metric_value),
                        "threshold": float(rule.threshold_value),
                        "comparison_type": rule.comparison_type,
                        "passed": passed,
                        "status": "passed" if passed else ("error" if rule.is_required else "warning")
                    })
                    
                    if not passed:
                        message = (
                            f"Rule '{rule.rule_name}': {rule.metric_name} = {metric_value:.4f} "
                            f"{rule.comparison_operator} {rule.threshold_value} failed"
                        )
                        if rule.is_required:
                            errors.append(message)
                        else:
                            warnings.append(message)
                
                validation_passed = len(errors) == 0
                
                return {
                    "passed": validation_passed,
                    "errors": errors,
                    "warnings": warnings,
                    "rule_results": rule_results
                }
                
        except Exception as e:
            self.logger.error(f"Error validating model: {e}", exc_info=True)
            return {
                "passed": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "rule_results": []
            }
    
    def _evaluate_absolute_rule(
        self,
        value: float,
        operator: str,
        threshold: float
    ) -> bool:
        """Evaluate absolute rule."""
        if operator == ">":
            return value > threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 1e-6
        elif operator == "!=":
            return abs(value - threshold) >= 1e-6
        else:
            return True
    
    def _evaluate_relative_rule(
        self,
        value: float,
        baseline: float,
        operator: str,
        threshold: float
    ) -> bool:
        """Evaluate relative rule (value relative to baseline)."""
        if baseline == 0:
            return True  # Skip if baseline is zero
        
        relative_value = value / baseline
        return self._evaluate_absolute_rule(relative_value, operator, threshold)


class ModelPromoter:
    """
    Handles model promotion to staging/production and rollback.
    """
    
    def __init__(self):
        """Initialize model promoter."""
        self.logger = get_logger(f"{__name__}.ModelPromoter")
    
    def promote_model(
        self,
        model_name: str,
        model_version: str,
        target_stage: str = "Staging",
        source_stage: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Promote model to target stage in MLflow.
        
        Args:
            model_name: Name of the model
            model_version: Model version to promote
            target_stage: Target stage ('Staging', 'Production')
            source_stage: Source stage (optional)
            
        Returns:
            Promotion result
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
            
            # Transition model to target stage
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=target_stage
            )
            
            self.logger.info(
                f"Promoted model {model_name} v{model_version} to {target_stage}",
                extra={
                    "model_name": model_name,
                    "model_version": model_version,
                    "target_stage": target_stage
                }
            )
            
            return {
                "success": True,
                "model_name": model_name,
                "model_version": model_version,
                "target_stage": target_stage,
                "promoted_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error promoting model: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name,
                "model_version": model_version,
                "target_stage": target_stage
            }
    
    def rollback_model(
        self,
        model_name: str,
        previous_version: str,
        current_version: str
    ) -> Dict[str, Any]:
        """
        Rollback model to previous version.
        
        Args:
            model_name: Name of the model
            previous_version: Version to rollback to
            current_version: Current version to replace
            
        Returns:
            Rollback result
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
            
            # Get current production model stage
            current_model = client.get_model_version(model_name, current_version)
            current_stage = current_model.current_stage
            
            # Transition current version to Archived
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Archived"
            )
            
            # Promote previous version back to production
            client.transition_model_version_stage(
                name=model_name,
                version=previous_version,
                stage=current_stage
            )
            
            self.logger.warning(
                f"Rolled back model {model_name} from v{current_version} to v{previous_version}",
                extra={
                    "model_name": model_name,
                    "previous_version": previous_version,
                    "current_version": current_version,
                    "stage": current_stage
                }
            )
            
            return {
                "success": True,
                "model_name": model_name,
                "previous_version": previous_version,
                "current_version": current_version,
                "rolled_back_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error rolling back model: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name,
                "previous_version": previous_version,
                "current_version": current_version
            }


class RetrainingPipeline:
    """
    Main retraining pipeline orchestrator.
    
    Handles:
    - Training new models
    - Model validation
    - Model promotion
    - Rollback on failure
    """
    
    def __init__(self):
        """Initialize retraining pipeline."""
        self.validator = ModelValidator()
        self.promoter = ModelPromoter()
        self.logger = get_logger(f"{__name__}.RetrainingPipeline")
    
    def run_retraining_job(
        self,
        job_id: int,
        trigger_type: str = "manual",
        trigger_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a retraining job.
        
        Args:
            job_id: Retraining job ID
            trigger_type: Type of trigger ('scheduled', 'drift', 'new_data', 'manual', 'performance_degradation')
            trigger_metadata: Additional trigger metadata
            
        Returns:
            Job execution result
        """
        try:
            with get_db_session() as session:
                job_repo = RetrainingJobRepository(session)
                job = job_repo.get_by_id(job_id)
                
                if not job:
                    raise ValueError(f"Retraining job {job_id} not found")
                
                # Update job status
                job.status = "running"
                job.started_at = datetime.now(timezone.utc)
                job.trigger_type = trigger_type
                if trigger_metadata:
                    job.trigger_metadata = trigger_metadata
                session.commit()
                
                self.logger.info(f"Starting retraining job {job_id}: {job.job_name}")
                
                # Load training data
                # Use project root to construct data directory path
                project_root = Path(__file__).parent.parent.parent
                splits_dir = project_root / "data" / "processed" / "splits"
                if not splits_dir.exists():
                    raise ValueError(f"Data splits directory not found: {splits_dir}")
                
                X_train, X_test, y_train, y_test = load_splits(str(splits_dir))
                
                # Train model
                trainer = ModelTrainer(random_state=42)
                model_type = job.model_type or "random_forest"
                
                model, metrics = trainer.train_and_evaluate(
                    model_type,
                    X_train, y_train,
                    X_test, y_test
                )
                
                # Log to MLflow
                tracker = MLflowTracker(
                    experiment_name=job.mlflow_experiment_name or "credit_scoring",
                    tracking_uri=settings.mlflow_tracking_uri
                )
                
                mlflow_run_id = tracker.log_model_training(
                    model=model,
                    model_name=job.model_name,
                    model_params=job.hyperparameters or {},
                    metrics=metrics,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    data_version=job.training_data_version
                )
                
                # Register model and get version
                import mlflow
                from mlflow.tracking import MlflowClient
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
                
                # Register model from the run
                model_uri = f"runs:/{mlflow_run_id}/model"
                try:
                    model_version = client.create_model_version(
                        name=job.model_name,
                        source=model_uri,
                        run_id=mlflow_run_id
                    )
                    latest_version = model_version.version
                except Exception as e:
                    self.logger.warning(f"Could not register model version: {e}")
                    # Try to get latest version
                    model_versions = client.search_model_versions(f"name='{job.model_name}'")
                    latest_version = model_versions[0].version if model_versions else None
                
                # Get baseline metrics for comparison
                baseline_metrics = self._get_baseline_metrics(job.model_name, session)
                
                # Validate model
                validation_result = self.validator.validate_model(
                    metrics=metrics,
                    model_name=job.model_name,
                    baseline_metrics=baseline_metrics
                )
                
                # Update job with results
                job.training_metrics = metrics
                job.test_metrics = metrics
                job.validation_passed = validation_result["passed"]
                job.validation_errors = validation_result["errors"]
                job.baseline_comparison = {
                    "baseline_metrics": baseline_metrics,
                    "new_model_metrics": metrics,
                    "improvement": self._calculate_improvement(metrics, baseline_metrics) if baseline_metrics else None
                }
                job.mlflow_run_id = mlflow_run_id
                job.model_version = latest_version
                
                # Auto-promote if validation passed
                if validation_result["passed"] and latest_version:
                    # Promote to Staging first
                    promotion_result = self.promoter.promote_model(
                        model_name=job.model_name,
                        model_version=latest_version,
                        target_stage="Staging"
                    )
                    
                    if promotion_result["success"]:
                        job.promotion_status = "promoted"
                        job.promoted_to_stage = "Staging"
                        job.promotion_timestamp = datetime.now(timezone.utc)
                        self.logger.info(f"Model {job.model_name} v{latest_version} promoted to Staging")
                    else:
                        job.promotion_status = "rejected"
                        job.error_message = f"Promotion failed: {promotion_result.get('error')}"
                else:
                    job.promotion_status = "rejected"
                    if not validation_result["passed"]:
                        job.error_message = f"Validation failed: {', '.join(validation_result['errors'])}"
                    elif not latest_version:
                        job.error_message = "Model version not found in MLflow"
                    else:
                        job.error_message = "Unknown error"
                
                job.status = "completed"
                job.completed_at = datetime.now(timezone.utc)
                session.commit()
                
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "validation_passed": validation_result["passed"],
                    "promotion_status": job.promotion_status,
                    "model_version": latest_version,
                    "metrics": metrics,
                    "baseline_comparison": job.baseline_comparison
                }
                
        except Exception as e:
            self.logger.error(f"Error running retraining job {job_id}: {e}", exc_info=True)
            
            # Update job status
            try:
                with get_db_session() as session:
                    job_repo = RetrainingJobRepository(session)
                    job = job_repo.get_by_id(job_id)
                    if job:
                        job.status = "failed"
                        job.error_message = str(e)
                        job.completed_at = datetime.now(timezone.utc)
                        session.commit()
            except:
                pass
            
            raise
    
    def _get_baseline_metrics(
        self,
        model_name: str,
        session: Any
    ) -> Optional[Dict[str, float]]:
        """Get baseline model metrics for comparison."""
        try:
            # Get production model from ModelMetadata
            baseline_model = session.query(ModelMetadata).filter(
                and_(
                    ModelMetadata.model_name == model_name,
                    ModelMetadata.model_stage == "Production",
                    ModelMetadata.is_active == True
                )
            ).order_by(desc(ModelMetadata.deployed_at)).first()
            
            if baseline_model:
                return {
                    "accuracy": float(baseline_model.accuracy) if baseline_model.accuracy else None,
                    "roc_auc": float(baseline_model.roc_auc) if baseline_model.roc_auc else None,
                    "precision": float(baseline_model.precision) if baseline_model.precision else None,
                    "recall": float(baseline_model.recall) if baseline_model.recall else None,
                    "f1_score": float(baseline_model.f1_score) if baseline_model.f1_score else None,
                }
        except Exception as e:
            self.logger.warning(f"Could not get baseline metrics: {e}")
        
        return None
    
    def _calculate_improvement(
        self,
        new_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate improvement percentage for each metric."""
        improvement = {}
        for metric_name, new_value in new_metrics.items():
            baseline_value = baseline_metrics.get(metric_name)
            if baseline_value is not None and baseline_value > 0:
                improvement[metric_name] = ((new_value - baseline_value) / baseline_value) * 100
            else:
                improvement[metric_name] = 0.0
        return improvement


class RetrainingScheduler:
    """
    Manages scheduled retraining jobs and trigger-based retraining.
    """
    
    def __init__(self):
        """Initialize retraining scheduler."""
        self.pipeline = RetrainingPipeline()
        self.logger = get_logger(f"{__name__}.RetrainingScheduler")
    
    def check_scheduled_jobs(self) -> List[int]:
        """
        Check for scheduled jobs that need to run.
        
        Returns:
            List of job IDs that should be executed
        """
        try:
            with get_db_session() as session:
                schedule_repo = RetrainingScheduleRepository(session)
                schedules = schedule_repo.get_due_schedules()
                
                job_ids = []
                for schedule in schedules:
                    # Create retraining job from schedule
                    job_id = self._create_job_from_schedule(schedule, session)
                    if job_id:
                        job_ids.append(job_id)
                
                return job_ids
                
        except Exception as e:
            self.logger.error(f"Error checking scheduled jobs: {e}", exc_info=True)
            return []
    
    def _create_job_from_schedule(
        self,
        schedule: RetrainingSchedule,
        session: Any
    ) -> Optional[int]:
        """Create a retraining job from a schedule."""
        try:
            job_repo = RetrainingJobRepository(session)
            
            job_name = f"{schedule.schedule_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            training_config = schedule.training_config or {}
            
            job = job_repo.create_job(
                job_name=job_name,
                model_name=schedule.model_name,
                model_type=training_config.get("model_type", "random_forest"),
                trigger_type="scheduled",
                trigger_metadata={"schedule_id": schedule.schedule_id},
                hyperparameters=training_config.get("hyperparameters", {}),
                training_config=training_config
            )
            
            # Update schedule
            schedule.last_run_at = datetime.now(timezone.utc)
            # Calculate next run time (simplified - would use cron parser in production)
            schedule.next_run_at = self._calculate_next_run(schedule)
            
            session.commit()
            
            return job.job_id
            
        except Exception as e:
            self.logger.error(f"Error creating job from schedule: {e}", exc_info=True)
            return None
    
    def _calculate_next_run(self, schedule: RetrainingSchedule) -> datetime:
        """Calculate next run time based on schedule type."""
        now = datetime.now(timezone.utc)
        
        if schedule.schedule_type == "daily":
            return now + timedelta(days=1)
        elif schedule.schedule_type == "weekly":
            return now + timedelta(weeks=1)
        elif schedule.schedule_type == "monthly":
            return now + timedelta(days=30)
        else:
            # For cron, would parse cron expression
            return now + timedelta(days=1)
    
    def trigger_on_drift(
        self,
        model_name: str,
        drift_metadata: Dict[str, Any]
    ) -> Optional[int]:
        """
        Trigger retraining based on drift detection.
        
        Args:
            model_name: Name of the model
            drift_metadata: Drift detection metadata
            
        Returns:
            Created job ID or None
        """
        try:
            with get_db_session() as session:
                job_repo = RetrainingJobRepository(session)
                
                job_name = f"drift_retraining_{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                
                job = job_repo.create_job(
                    job_name=job_name,
                    model_name=model_name,
                    model_type="random_forest",  # Default, could be configurable
                    trigger_type="drift",
                    trigger_metadata=drift_metadata
                )
                
                session.commit()
                
                self.logger.info(f"Created drift-triggered retraining job {job.job_id} for model {model_name}")
                
                return job.job_id
                
        except Exception as e:
            self.logger.error(f"Error triggering drift retraining: {e}", exc_info=True)
            return None


def get_retraining_pipeline() -> RetrainingPipeline:
    """Get a singleton RetrainingPipeline instance."""
    global _retraining_pipeline
    if '_retraining_pipeline' not in globals():
        _retraining_pipeline = RetrainingPipeline()
    return _retraining_pipeline
