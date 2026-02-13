# Database Layer Documentation

## Overview

This package provides a professional database layer with:
- **SQLAlchemy ORM models** for all database tables
- **Repository pattern** for data access
- **Service layer** for business logic
- **Professional exception handling**
- **Comprehensive logging**
- **Connection pooling** and transaction management

## Architecture

```
┌─────────────────────────────────────────┐
│         Service Layer                    │
│  (Business Logic)                        │
│  - PredictionService                     │
│  - RawTransactionService                 │
│  - BusinessKPIService                    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         Repository Layer                │
│  (Data Access)                          │
│  - BaseRepository                       │
│  - PredictionRepository                 │
│  - UserRepository                       │
│  - etc.                                  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         Model Layer                     │
│  (SQLAlchemy ORM)                       │
│  - User, Role, Prediction              │
│  - RawTransaction, RFMMetric            │
│  - etc.                                  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      Database Connection                │
│  - DatabaseManager                      │
│  - Connection Pooling                   │
│  - Transaction Management               │
└─────────────────────────────────────────┘
```

## Quick Start

### 1. Save a Prediction

```python
from src.database.connection import get_db_session
from src.database.services import PredictionService
from decimal import Decimal

with get_db_session() as session:
    service = PredictionService(session)
    
    prediction = service.save_prediction(
        prediction_id="pred_abc123",
        customer_id="CUST-12345",
        prediction=0,
        probability=Decimal("0.15"),
        customer_score=850,
        risk_level="low",
        features=[0.0, -0.046, ...],  # 26 features
        model_name="credit_scoring_model",
        model_version="v1.0",
        model_stage="Production",
        latency_ms=Decimal("45.2")
    )
```

### 2. Get Customer Predictions

```python
from src.database.connection import get_db_session
from src.database.services import PredictionService

with get_db_session() as session:
    service = PredictionService(session)
    
    predictions = service.get_customer_predictions(
        customer_id="CUST-12345",
        limit=100
    )
```

### 3. Using Repository Directly

```python
from src.database.connection import get_db_session
from src.database.repositories import PredictionRepository

with get_db_session() as session:
    repo = PredictionRepository(session)
    
    # Get by ID
    prediction = repo.get_by_id("pred_abc123")
    
    # Get by customer
    predictions = repo.get_by_customer_id("CUST-12345")
    
    # Get by risk level
    high_risk = repo.get_by_risk_level("high")
```

## Exception Handling

All database operations raise specific exceptions:

```python
from src.database.exceptions import (
    DatabaseError,
    RecordNotFoundError,
    DuplicateRecordError,
    DatabaseIntegrityError
)

try:
    prediction = service.get_prediction_by_id("pred_abc123")
except RecordNotFoundError:
    # Handle not found
    pass
except DuplicateRecordError:
    # Handle duplicate
    pass
except DatabaseError as e:
    # Handle other database errors
    logger.error(f"Database error: {e}")
```

## Logging

All operations are automatically logged:

- **INFO**: Successful operations (create, update, delete)
- **DEBUG**: Query operations
- **WARNING**: Non-critical issues (duplicates, not found)
- **ERROR**: Database errors with full stack traces

## Models

All database tables have corresponding SQLAlchemy models:

- `User`, `Role`, `UserRole`, `Permission`, `RolePermission`, `AuditLog`
- `RawTransaction`
- `RFMMetric`, `ProcessedFeature`, `DataSplit`
- `Prediction`, `CustomerFeature`
- `DataVersion`, `ModelMetadata`
- `BusinessKPI`, `PerformanceMetric`, `DriftMetric`

## Repositories

Each model has a repository with CRUD operations:

- `create()` - Create new record
- `get_by_id()` - Get by primary key
- `get_by_id_or_raise()` - Get or raise exception
- `get_all()` - Get all records (with pagination)
- `update()` - Update record
- `delete()` - Delete record
- `count()` - Count records

Plus model-specific methods (e.g., `get_by_customer_id()`).

## Services

High-level business logic services:

- `PredictionService` - Prediction operations
- `RawTransactionService` - Transaction upload operations
- `BusinessKPIService` - KPI calculation and storage

## Connection Management

Always use the context manager:

```python
from src.database.connection import get_db_session

with get_db_session() as session:
    # Use session here
    # Transaction is automatically committed on success
    # Automatically rolled back on error
```

## Configuration

Database connection is configured via environment variables:

```bash
DATABASE_URL=postgresql://user:pass@host:port/dbname
# OR
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=mlops_db
DATABASE_USER=postgres
DATABASE_PASSWORD=password
```

## Best Practices

1. **Always use context managers** for sessions
2. **Catch specific exceptions** (RecordNotFoundError, DuplicateRecordError)
3. **Use services for business logic**, repositories for data access
4. **Let exceptions propagate** - they're already logged
5. **Use type hints** for better IDE support
6. **Check logs** for debugging database issues

## See Also

- `example_usage.py` - Complete usage examples
- `models.py` - All database models
- `repositories.py` - Repository implementations
- `services.py` - Service layer implementations
