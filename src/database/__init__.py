"""
Database package for credit scoring MLOps system.

This package provides:
- SQLAlchemy ORM models for all database tables
- Database connection management
- Repository pattern for data access
- Professional exception handling
- Comprehensive logging
"""

from src.database.connection import DatabaseManager, get_db_session
from src.database.models import (
    Base,
    User,
    Role,
    UserRole,
    Permission,
    RolePermission,
    AuditLog,
    RawTransaction,
    RFMMetric,
    ProcessedFeature,
    DataSplit,
    Prediction,
    CustomerFeature,
    DataVersion,
    DataLineage,
    ModelMetadata,
    BusinessKPI,
    PerformanceMetric,
    DriftMetric,
    Experiment,
    ExperimentAssignment,
    ExperimentMetric,
    RetrainingJob,
    RetrainingSchedule,
    ModelValidationRule,
    BatchPredictionJob,
    BatchPredictionSchedule,
    BatchPredictionResult,
    BatchPredictionLog,
    ModelRoutingRule,
    ModelRegistry,
    ModelComparisonResult,
    ModelEnsemble
)
from src.database.repositories import (
    BaseRepository,
    UserRepository,
    RoleRepository,
    PredictionRepository,
    RawTransactionRepository,
    RFMMetricRepository,
    ProcessedFeatureRepository,
    CustomerFeatureRepository,
    BusinessKPIRepository,
    DataVersionRepository,
    DataLineageRepository,
    ModelRoutingRuleRepository,
    ModelRegistryRepository,
    ModelEnsembleRepository
)
from src.database.services import (
    PredictionService,
    RawTransactionService,
    BusinessKPIService,
    DataVersionService,
    DataLineageService
)
from src.database.exceptions import (
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseIntegrityError,
    RecordNotFoundError,
    DuplicateRecordError
)

__all__ = [
    # Connection
    "DatabaseManager",
    "get_db_session",
    # Models
    "Base",
    "User",
    "Role",
    "UserRole",
    "Permission",
    "RolePermission",
    "AuditLog",
    "RawTransaction",
    "RFMMetric",
    "ProcessedFeature",
    "DataSplit",
    "Prediction",
    "CustomerFeature",
    "DataVersion",
    "DataLineage",
    "ModelMetadata",
    "BusinessKPI",
    "PerformanceMetric",
    "DriftMetric",
    "Experiment",
    "ExperimentAssignment",
    "ExperimentMetric",
    "RetrainingJob",
    "RetrainingSchedule",
    "ModelValidationRule",
    "BatchPredictionJob",
    "BatchPredictionSchedule",
    "BatchPredictionResult",
    "BatchPredictionLog",
    "ModelRoutingRule",
    "ModelRegistry",
    "ModelComparisonResult",
    "ModelEnsemble",
    # Repositories
    "BaseRepository",
    "UserRepository",
    "RoleRepository",
    "PredictionRepository",
    "RawTransactionRepository",
    "RFMMetricRepository",
    "ProcessedFeatureRepository",
    "CustomerFeatureRepository",
    "BusinessKPIRepository",
    "DataVersionRepository",
    "DataLineageRepository",
    "ExperimentRepository",
    "ExperimentAssignmentRepository",
    "ExperimentMetricRepository",
    "RetrainingJobRepository",
    "RetrainingScheduleRepository",
    "ModelValidationRuleRepository",
    "BatchPredictionJobRepository",
    "BatchPredictionScheduleRepository",
    "ModelRoutingRuleRepository",
    "ModelRegistryRepository",
    "ModelEnsembleRepository",
    # Services
    "PredictionService",
    "RawTransactionService",
    "BusinessKPIService",
    "DataVersionService",
    "DataLineageService",
    # Exceptions
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "DatabaseIntegrityError",
    "RecordNotFoundError",
    "DuplicateRecordError",
]
