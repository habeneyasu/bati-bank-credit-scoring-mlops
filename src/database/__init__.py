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
    ModelMetadata,
    BusinessKPI,
    PerformanceMetric,
    DriftMetric
)
from src.database.repositories import (
    BaseRepository,
    UserRepository,
    RoleRepository,
    PredictionRepository,
    RawTransactionRepository,
    RFMMetricRepository,
    ProcessedFeatureRepository,
    BusinessKPIRepository
)
from src.database.services import (
    PredictionService,
    RawTransactionService,
    BusinessKPIService
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
    "ModelMetadata",
    "BusinessKPI",
    "PerformanceMetric",
    "DriftMetric",
    # Repositories
    "BaseRepository",
    "UserRepository",
    "RoleRepository",
    "PredictionRepository",
    "RawTransactionRepository",
    "RFMMetricRepository",
    "ProcessedFeatureRepository",
    "BusinessKPIRepository",
    # Services
    "PredictionService",
    "RawTransactionService",
    "BusinessKPIService",
    # Exceptions
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "DatabaseIntegrityError",
    "RecordNotFoundError",
    "DuplicateRecordError",
]
