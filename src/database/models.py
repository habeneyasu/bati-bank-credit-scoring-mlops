"""
SQLAlchemy ORM models for all database tables.

This module defines all database models using SQLAlchemy ORM with:
- Type hints for all fields
- Proper relationships
- Indexes for performance
- Constraints for data integrity
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from decimal import Decimal
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Date, Numeric, Text,
    ForeignKey, CheckConstraint, Index, JSON, ARRAY, BigInteger
)
from sqlalchemy.dialects.postgresql import JSONB, UUID, INET
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


# ============================================================================
# SECURITY & ACCESS CONTROL MODELS
# ============================================================================

class User(Base):
    """User model for authentication and access control."""
    
    __tablename__ = "users"
    
    # Primary Key
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    
    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # User Information
    full_name = Column(String(255))
    department = Column(String(100))
    position = Column(String(100))
    
    # Account Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Security
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True))
    password_changed_at = Column(DateTime(timezone=True))
    last_login_at = Column(DateTime(timezone=True))
    
    # API Access
    api_key = Column(String(255), unique=True, index=True)
    api_key_created_at = Column(DateTime(timezone=True))
    api_key_expires_at = Column(DateTime(timezone=True))
    
    # Metadata
    created_by = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at = Column(DateTime(timezone=True))  # Soft delete
    
    # Relationships
    user_roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'", name="chk_email_format"),
    )
    
    def __repr__(self) -> str:
        """String representation of User."""
        return f"<User(user_id={self.user_id}, username='{self.username}', email='{self.email}')>"


class Role(Base):
    """Role model for access control."""
    
    __tablename__ = "roles"
    
    # Primary Key
    role_id = Column(Integer, primary_key=True, autoincrement=True)
    role_name = Column(String(100), unique=True, nullable=False)
    role_code = Column(String(50), unique=True, nullable=False, index=True)
    
    # Role Description
    description = Column(Text)
    
    # Role Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user_roles = relationship("UserRole", back_populates="role", cascade="all, delete-orphan")
    role_permissions = relationship("RolePermission", back_populates="role", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        """String representation of Role."""
        return f"<Role(role_id={self.role_id}, role_name='{self.role_name}', role_code='{self.role_code}')>"


class UserRole(Base):
    """User-Role mapping table (many-to-many)."""
    
    __tablename__ = "user_roles"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Keys
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    role_id = Column(Integer, ForeignKey("roles.role_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Assignment Metadata
    assigned_by = Column(String(100))
    assigned_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), index=True)
    
    # Relationships
    user = relationship("User", back_populates="user_roles")
    role = relationship("Role", back_populates="user_roles")
    
    # Constraints
    __table_args__ = (
        Index("uq_user_role", "user_id", "role_id", unique=True),
    )
    
    def __repr__(self) -> str:
        """String representation of UserRole."""
        return f"<UserRole(user_id={self.user_id}, role_id={self.role_id})>"


class Permission(Base):
    """Permission model for granular access control."""
    
    __tablename__ = "permissions"
    
    # Primary Key
    permission_id = Column(Integer, primary_key=True, autoincrement=True)
    permission_name = Column(String(100), unique=True, nullable=False)
    permission_code = Column(String(50), unique=True, nullable=False, index=True)
    
    # Permission Details
    resource_type = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False)
    description = Column(Text)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    role_permissions = relationship("RolePermission", back_populates="permission", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        """String representation of Permission."""
        return f"<Permission(permission_id={self.permission_id}, permission_code='{self.permission_code}')>"


class RolePermission(Base):
    """Role-Permission mapping table (many-to-many)."""
    
    __tablename__ = "role_permissions"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Keys
    role_id = Column(Integer, ForeignKey("roles.role_id", ondelete="CASCADE"), nullable=False, index=True)
    permission_id = Column(Integer, ForeignKey("permissions.permission_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Relationships
    role = relationship("Role", back_populates="role_permissions")
    permission = relationship("Permission", back_populates="role_permissions")
    
    # Constraints
    __table_args__ = (
        Index("uq_role_permission", "role_id", "permission_id", unique=True),
    )
    
    def __repr__(self) -> str:
        """String representation of RolePermission."""
        return f"<RolePermission(role_id={self.role_id}, permission_id={self.permission_id})>"


class AuditLog(Base):
    """Audit log model for tracking user actions."""
    
    __tablename__ = "audit_logs"
    
    # Primary Key
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # User Information
    user_id = Column(Integer, ForeignKey("users.user_id"), index=True)
    username = Column(String(100))
    
    # Action Details
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50))
    resource_id = Column(String(100))
    
    # Request Details
    ip_address = Column(INET)
    user_agent = Column(Text)
    request_method = Column(String(10))
    request_path = Column(Text)
    
    # Result
    status_code = Column(Integer)
    success = Column(Boolean)
    error_message = Column(Text)
    
    # Metadata
    log_metadata = Column(JSONB)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index("idx_audit_logs_resource", "resource_type", "resource_id"),
    )
    
    def __repr__(self) -> str:
        """String representation of AuditLog."""
        return f"<AuditLog(log_id={self.log_id}, action='{self.action}', username='{self.username}')>"


# ============================================================================
# RAW DATA MODELS
# ============================================================================

class RawTransaction(Base):
    """Raw transaction data model."""
    
    __tablename__ = "raw_transactions"
    
    # Primary Key
    transaction_id = Column(String(100), primary_key=True)
    
    # Batch Information
    batch_id = Column(String(100), index=True)
    
    # Account Information
    account_id = Column(String(100))
    subscription_id = Column(String(100))
    customer_id = Column(String(100), nullable=False, index=True)
    
    # Transaction Details
    currency_code = Column(String(10))
    country_code = Column(String(10))
    provider_id = Column(String(100))
    product_id = Column(String(100))
    product_category = Column(String(100), index=True)
    channel_id = Column(String(100), index=True)
    
    # Financial Details
    amount = Column(Numeric(15, 2), nullable=False)
    value = Column(Numeric(15, 2))
    
    # Transaction Metadata
    transaction_start_time = Column(DateTime(timezone=True), nullable=False, index=True)
    pricing_strategy = Column(Integer)
    fraud_result = Column(Integer, default=0, nullable=False)
    
    # Data Quality Flags
    is_valid = Column(Boolean, default=True, nullable=False)
    validation_errors = Column(JSONB)
    
    # Upload Information
    uploaded_by = Column(String(100))
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    data_source = Column(String(100))
    file_name = Column(String(255))
    
    # Data Versioning
    data_version = Column(String(50))
    checksum_sha256 = Column(String(64))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_raw_transactions_customer_time", "customer_id", "transaction_start_time"),
    )
    
    def __repr__(self) -> str:
        """String representation of RawTransaction."""
        return f"<RawTransaction(transaction_id='{self.transaction_id}', customer_id='{self.customer_id}')>"


# ============================================================================
# PROCESSED DATA MODELS
# ============================================================================

class RFMMetric(Base):
    """RFM metrics model (customer-level aggregations)."""
    
    __tablename__ = "rfm_metrics"
    
    # Primary Key
    customer_id = Column(String(100), primary_key=True)
    
    # RFM Metrics
    recency = Column(Integer, nullable=False)
    frequency = Column(Integer, nullable=False)
    monetary = Column(Numeric(15, 2), nullable=False)
    
    # Normalized RFM
    recency_normalized = Column(Numeric(10, 6))
    frequency_normalized = Column(Numeric(10, 6))
    monetary_normalized = Column(Numeric(10, 6))
    
    # Clustering Information
    cluster = Column(Integer, index=True)
    cluster_label = Column(String(50))
    
    # Target Variable
    is_high_risk = Column(Integer, index=True)
    
    # Processing Metadata
    processed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    processing_version = Column(String(50))
    data_version = Column(String(50))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    processed_features = relationship("ProcessedFeature", back_populates="rfm_metrics", uselist=False)
    
    def __repr__(self) -> str:
        """String representation of RFMMetric."""
        return f"<RFMMetric(customer_id='{self.customer_id}', cluster={self.cluster}, is_high_risk={self.is_high_risk})>"


class ProcessedFeature(Base):
    """Processed features model (26 engineered features per customer)."""
    
    __tablename__ = "processed_features"
    
    # Primary Key
    customer_id = Column(String(100), ForeignKey("rfm_metrics.customer_id", ondelete="CASCADE"), primary_key=True)
    
    # Temporal Features
    transaction_hour = Column(Numeric(10, 6))
    transaction_day = Column(Numeric(10, 6))
    transaction_month = Column(Numeric(10, 6))
    transaction_year = Column(Numeric(10, 6))
    transaction_dayofweek = Column(Numeric(10, 6))
    
    # RFM Features (normalized)
    recency_normalized = Column(Numeric(10, 6))
    frequency_normalized = Column(Numeric(10, 6))
    monetary_normalized = Column(Numeric(10, 6))
    
    # Aggregate Features
    aggregate_features = Column(JSONB)
    
    # Categorical Encodings
    categorical_features = Column(JSONB)
    
    # Complete Feature Vector (26 features)
    feature_vector = Column(ARRAY(Numeric(10, 6)), nullable=False)
    
    # Processing Metadata
    processed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    feature_engineering_version = Column(String(50), index=True)
    data_version = Column(String(50))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    rfm_metrics = relationship("RFMMetric", back_populates="processed_features")
    
    def __repr__(self) -> str:
        """String representation of ProcessedFeature."""
        return f"<ProcessedFeature(customer_id='{self.customer_id}', features_count={len(self.feature_vector) if self.feature_vector else 0})>"


class DataSplit(Base):
    """Data splits model (train/validation/test splits)."""
    
    __tablename__ = "data_splits"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Customer Information
    customer_id = Column(String(100), nullable=False, index=True)
    
    # Split Information
    split_type = Column(String(20), nullable=False, index=True)  # 'train', 'validation', 'test'
    split_version = Column(String(50), index=True)
    
    # Target Variable
    target_value = Column(Integer)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Constraints
    __table_args__ = (
        Index("uq_data_splits", "customer_id", "split_type", "split_version", unique=True),
    )
    
    def __repr__(self) -> str:
        """String representation of DataSplit."""
        return f"<DataSplit(customer_id='{self.customer_id}', split_type='{self.split_type}')>"


# ============================================================================
# PREDICTION MODELS
# ============================================================================

class Prediction(Base):
    """Prediction model for storing all predictions."""
    
    __tablename__ = "predictions"
    
    # Primary Key
    prediction_id = Column(String(50), primary_key=True)
    
    # Customer Information
    customer_id = Column(String(100))
    customer_id_indexed = Column(String(100), index=True)
    
    # Prediction Details
    prediction = Column(Integer, nullable=False)  # 0 or 1
    probability = Column(Numeric(5, 4), nullable=False)  # 0.0000 to 1.0000
    customer_score = Column(Integer)  # 0-100 scale
    risk_level = Column(String(10), nullable=False, index=True)  # 'low', 'medium', 'high'
    
    # Features
    features = Column(JSONB, nullable=False)
    
    # Model Information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False, index=True)
    model_stage = Column(String(20), nullable=False)
    
    # Performance Metrics
    latency_ms = Column(Numeric(10, 2))
    request_size_bytes = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    created_at_date = Column(Date, index=True)
    
    # Metadata
    request_metadata = Column(JSONB)
    response_metadata = Column(JSONB)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("prediction IN (0, 1)", name="chk_prediction"),
        CheckConstraint("probability >= 0 AND probability <= 1", name="chk_probability"),
        CheckConstraint("customer_score IS NULL OR (customer_score >= 0 AND customer_score <= 1000)", name="chk_customer_score"),
        CheckConstraint("risk_level IN ('low', 'medium', 'high')", name="chk_risk_level"),
        Index("idx_predictions_customer_date", "customer_id_indexed", "created_at_date"),
    )
    
    def __repr__(self) -> str:
        """String representation of Prediction."""
        return f"<Prediction(prediction_id='{self.prediction_id}', customer_id='{self.customer_id}', risk_level='{self.risk_level}')>"


class CustomerFeature(Base):
    """Customer features model for feature store (online serving)."""
    
    __tablename__ = "customer_features"
    
    # Primary Key
    customer_id = Column(String(100), primary_key=True)
    
    # RFM Features
    recency_normalized = Column(Numeric(10, 6))
    frequency_normalized = Column(Numeric(10, 6))
    monetary_normalized = Column(Numeric(10, 6))
    
    # Temporal Features
    transaction_hour = Column(Numeric(10, 6))
    transaction_day = Column(Numeric(10, 6))
    transaction_month = Column(Numeric(10, 6))
    transaction_year = Column(Numeric(10, 6))
    transaction_dayofweek = Column(Numeric(10, 6))
    
    # Aggregate Features
    aggregate_features = Column(JSONB)
    
    # Categorical Encodings
    categorical_features = Column(JSONB)
    
    # All 26 Features
    feature_vector = Column(ARRAY(Numeric(10, 6)), nullable=False)
    
    # Metadata
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, index=True)
    feature_version = Column(String(50), index=True)
    data_version = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self) -> str:
        """String representation of CustomerFeature."""
        return f"<CustomerFeature(customer_id='{self.customer_id}', last_updated='{self.last_updated}')>"


# ============================================================================
# METADATA MODELS
# ============================================================================

class DataVersion(Base):
    """Data versioning model."""
    
    __tablename__ = "data_versions"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Version Information
    data_type = Column(String(50), nullable=False, index=True)  # 'dataset', 'features', 'splits', 'artifacts'
    version = Column(String(50), nullable=False)  # 'v1', 'v2', etc.
    
    # File Information
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer)  # Size in bytes
    checksum_sha256 = Column(String(64), nullable=False)
    
    # Metadata
    data_metadata = Column(JSONB)
    dependencies = Column(ARRAY(Text))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    
    # Constraints
    __table_args__ = (
        Index("uq_data_versions", "data_type", "version", unique=True),
    )
    
    def __repr__(self) -> str:
        """String representation of DataVersion."""
        return f"<DataVersion(data_type='{self.data_type}', version='{self.version}')>"


class DataLineage(Base):
    """Data lineage tracking model to track relationships between data versions, models, and predictions."""
    
    __tablename__ = "data_lineage"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Source (upstream) - what data version was used
    source_data_version_id = Column(Integer, ForeignKey("data_versions.id"), nullable=False, index=True)
    source_data_type = Column(String(50), nullable=False)  # 'raw_transactions', 'processed', 'features', etc.
    source_version = Column(String(50), nullable=False)
    
    # Target (downstream) - what was created from this data
    target_type = Column(String(50), nullable=False, index=True)  # 'model', 'prediction', 'feature_set', 'processed_data'
    target_id = Column(String(100), nullable=False, index=True)  # model_version, prediction_id, etc.
    target_name = Column(String(200))  # Human-readable name
    
    # Relationship metadata
    relationship_type = Column(String(50), nullable=False)  # 'trained_on', 'used_for', 'derived_from', 'generated_from'
    operation = Column(String(100))  # 'training', 'prediction', 'feature_engineering', 'processing'
    
    # Additional context
    lineage_metadata = Column(JSONB)  # Additional context (timestamp, user, etc.)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    
    # Relationships
    source_data_version = relationship("DataVersion", foreign_keys=[source_data_version_id])
    
    # Constraints
    __table_args__ = (
        Index("idx_lineage_source", "source_data_version_id", "target_type", "target_id"),
        Index("idx_lineage_target", "target_type", "target_id"),
    )
    
    def __repr__(self) -> str:
        """String representation of DataLineage."""
        return f"<DataLineage(source={self.source_data_type}:{self.source_version} -> {self.target_type}:{self.target_id})>"


class ModelMetadata(Base):
    """Model metadata model for tracking model versions."""
    
    __tablename__ = "model_metadata"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Model Information
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False, index=True)
    model_stage = Column(String(20), nullable=False, index=True)  # 'Production', 'Staging', 'Archived'
    
    # MLflow Integration
    mlflow_run_id = Column(String(50))
    mlflow_experiment_name = Column(String(100))
    
    # Performance Metrics
    roc_auc = Column(Numeric(5, 4))
    accuracy = Column(Numeric(5, 4))
    precision = Column(Numeric(5, 4))
    recall = Column(Numeric(5, 4))
    f1_score = Column(Numeric(5, 4))
    
    # Training Information
    training_data_version = Column(String(50))
    feature_version = Column(String(50))
    hyperparameters = Column(JSONB)
    
    # Deployment Information
    deployed_at = Column(DateTime(timezone=True))
    deployed_by = Column(String(100))
    deployment_environment = Column(String(50))
    
    # Status
    is_active = Column(Boolean, default=False, nullable=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Constraints
    __table_args__ = (
        Index("uq_model_metadata", "model_name", "model_version", unique=True),
    )
    
    def __repr__(self) -> str:
        """String representation of ModelMetadata."""
        return f"<ModelMetadata(model_name='{self.model_name}', model_version='{self.model_version}', stage='{self.model_stage}')>"


# ============================================================================
# BUSINESS & MONITORING MODELS
# ============================================================================

class BusinessKPI(Base):
    """Business KPI tracking model."""
    
    __tablename__ = "business_kpis"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Time Period
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False)
    period_type = Column(String(20), nullable=False, index=True)  # 'hourly', 'daily', 'weekly', 'monthly'
    
    # KPI Metrics
    total_predictions = Column(Integer, default=0, nullable=False)
    approval_count = Column(Integer, default=0, nullable=False)  # Low risk
    rejection_count = Column(Integer, default=0, nullable=False)  # High risk
    review_count = Column(Integer, default=0, nullable=False)  # Medium risk
    
    approval_rate = Column(Numeric(5, 4))
    rejection_rate = Column(Numeric(5, 4))
    review_rate = Column(Numeric(5, 4))
    
    avg_risk_score = Column(Numeric(5, 4))
    median_risk_score = Column(Numeric(5, 4))
    
    # Customer Metrics
    unique_customers = Column(Integer)
    new_customers = Column(Integer)
    
    # Performance Metrics
    avg_latency_ms = Column(Numeric(10, 2))
    p95_latency_ms = Column(Numeric(10, 2))
    p99_latency_ms = Column(Numeric(10, 2))
    error_rate = Column(Numeric(5, 4))
    
    # Timestamps
    calculated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Constraints
    __table_args__ = (
        Index("uq_business_kpis", "period_start", "period_end", "period_type", unique=True),
    )
    
    def __repr__(self) -> str:
        """String representation of BusinessKPI."""
        return f"<BusinessKPI(period_type='{self.period_type}', total_predictions={self.total_predictions})>"


class PerformanceMetric(Base):
    """Performance metrics model (time-series)."""
    
    __tablename__ = "performance_metrics"
    
    # Primary Key (composite with time)
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Time
    time = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Metrics
    endpoint = Column(String(100), nullable=False, index=True)
    latency_ms = Column(Numeric(10, 2), nullable=False)
    status_code = Column(Integer)
    error = Column(Boolean, default=False)
    
    # Request Information
    customer_id = Column(String(100), index=True)
    model_version = Column(String(50))
    
    # Additional Metadata
    performance_metadata = Column(JSONB)
    
    # Indexes
    __table_args__ = (
        Index("idx_performance_metrics_endpoint", "endpoint", "time"),
        Index("idx_performance_metrics_customer", "customer_id", "time"),
    )
    
    def __repr__(self) -> str:
        """String representation of PerformanceMetric."""
        return f"<PerformanceMetric(endpoint='{self.endpoint}', latency_ms={self.latency_ms}, time='{self.time}')>"


class DriftMetric(Base):
    """Drift detection metrics model (time-series)."""
    
    __tablename__ = "drift_metrics"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Time
    time = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Feature Information
    feature_name = Column(String(100), nullable=False, index=True)
    
    # Drift Metrics
    psi = Column(Numeric(10, 6))  # Population Stability Index
    ks_statistic = Column(Numeric(10, 6))  # Kolmogorov-Smirnov statistic
    chi_square = Column(Numeric(10, 6))  # Chi-square statistic
    
    # Drift Status
    is_drifted = Column(Boolean, default=False, nullable=False, index=True)
    drift_severity = Column(String(20))  # 'none', 'minor', 'major'
    
    # Reference Distribution
    reference_distribution = Column(JSONB)
    current_distribution = Column(JSONB)
    
    # Metadata
    model_version = Column(String(50))
    drift_metadata = Column(JSONB)
    
    # Indexes
    __table_args__ = (
        Index("idx_drift_metrics_feature", "feature_name", "time"),
        Index("idx_drift_metrics_drifted", "is_drifted", "time"),
    )
    
    def __repr__(self) -> str:
        """String representation of DriftMetric."""
        return f"<DriftMetric(feature_name='{self.feature_name}', is_drifted={self.is_drifted}, psi={self.psi})>"


# ============================================================================
# A/B TESTING MODELS
# ============================================================================

class Experiment(Base):
    """A/B testing experiment model."""
    
    __tablename__ = "experiments"
    
    # Primary Key
    experiment_id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_name = Column(String(100), unique=True, nullable=False, index=True)
    
    # Experiment Configuration
    description = Column(Text)
    status = Column(String(20), nullable=False, default="draft", index=True)  # 'draft', 'running', 'paused', 'completed', 'cancelled'
    
    # Variants Configuration
    variants = Column(JSONB, nullable=False)  # List of variant configs
    
    # Traffic Splitting
    traffic_percentage = Column(Integer, default=100, nullable=False)  # 0-100
    assignment_method = Column(String(50), default="hash", nullable=False)  # 'hash', 'random', 'customer_segment'
    
    # Experiment Dates
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))
    actual_started_at = Column(DateTime(timezone=True))
    actual_ended_at = Column(DateTime(timezone=True))
    
    # Success Criteria
    primary_metric = Column(String(50), default="accuracy", nullable=False)
    minimum_sample_size = Column(Integer, default=1000, nullable=False)
    significance_level = Column(Numeric(5, 4), default=0.05, nullable=False)
    minimum_improvement = Column(Numeric(5, 4), default=0.01, nullable=False)  # 1%
    
    # Results
    winner_variant = Column(String(100))
    statistical_significance = Column(Numeric(5, 4))  # p-value
    confidence_interval = Column(JSONB)
    conclusion = Column(Text)
    
    # Metadata
    created_by = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    assignments = relationship("ExperimentAssignment", back_populates="experiment", cascade="all, delete-orphan")
    metrics = relationship("ExperimentMetric", back_populates="experiment", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('draft', 'running', 'paused', 'completed', 'cancelled')", name="chk_status"),
        CheckConstraint("traffic_percentage >= 0 AND traffic_percentage <= 100", name="chk_traffic_percentage"),
        CheckConstraint("significance_level > 0 AND significance_level < 1", name="chk_significance_level"),
    )
    
    def __repr__(self) -> str:
        """String representation of Experiment."""
        return f"<Experiment(experiment_id={self.experiment_id}, name='{self.experiment_name}', status='{self.status}')>"


class ExperimentAssignment(Base):
    """Experiment assignment model (tracks which variant each customer/request gets)."""
    
    __tablename__ = "experiment_assignments"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Key
    experiment_id = Column(Integer, ForeignKey("experiments.experiment_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Assignment Details
    entity_id = Column(String(100), nullable=False)  # Customer ID or request ID
    entity_type = Column(String(50), default="customer", nullable=False)  # 'customer', 'request'
    variant_name = Column(String(100), nullable=False, index=True)
    
    # Assignment Metadata
    assignment_hash = Column(String(64))
    assigned_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="assignments")
    
    # Constraints
    __table_args__ = (
        Index("uq_experiment_entity", "experiment_id", "entity_id", "entity_type", unique=True),
    )
    
    def __repr__(self) -> str:
        """String representation of ExperimentAssignment."""
        return f"<ExperimentAssignment(experiment_id={self.experiment_id}, entity_id='{self.entity_id}', variant='{self.variant_name}')>"


class ExperimentMetric(Base):
    """Experiment metrics model (aggregated metrics per variant)."""
    
    __tablename__ = "experiment_metrics"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Key
    experiment_id = Column(Integer, ForeignKey("experiments.experiment_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Variant Information
    variant_name = Column(String(100), nullable=False, index=True)
    
    # Metrics
    sample_size = Column(Integer, default=0, nullable=False)
    accuracy = Column(Numeric(10, 6))
    roc_auc = Column(Numeric(10, 6))
    precision = Column(Numeric(10, 6))
    recall = Column(Numeric(10, 6))
    f1_score = Column(Numeric(10, 6))
    avg_latency_ms = Column(Numeric(10, 2))
    p95_latency_ms = Column(Numeric(10, 2))
    error_rate = Column(Numeric(10, 6))
    
    # Business Metrics
    total_predictions = Column(Integer, default=0, nullable=False)
    high_risk_predictions = Column(Integer, default=0, nullable=False)
    low_risk_predictions = Column(Integer, default=0, nullable=False)
    
    # Statistical Metrics
    mean_value = Column(Numeric(10, 6))  # Mean of primary metric
    std_value = Column(Numeric(10, 6))  # Standard deviation
    confidence_interval_lower = Column(Numeric(10, 6))
    confidence_interval_upper = Column(Numeric(10, 6))
    
    # Timestamps
    calculated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    period_start = Column(DateTime(timezone=True))
    period_end = Column(DateTime(timezone=True))
    
    # Relationships
    experiment = relationship("Experiment", back_populates="metrics")
    
    def __repr__(self) -> str:
        """String representation of ExperimentMetric."""
        return f"<ExperimentMetric(experiment_id={self.experiment_id}, variant='{self.variant_name}', sample_size={self.sample_size})>"


# ============================================================================
# MODEL RETRAINING MODELS
# ============================================================================

class RetrainingJob(Base):
    """Model retraining job model."""
    
    __tablename__ = "retraining_jobs"
    
    # Primary Key
    job_id = Column(Integer, primary_key=True, autoincrement=True)
    job_name = Column(String(100), nullable=False)
    
    # Job Configuration
    trigger_type = Column(String(50), nullable=False, index=True)  # 'scheduled', 'drift', 'new_data', 'manual', 'performance_degradation'
    trigger_metadata = Column(JSONB)
    
    # Training Configuration
    model_name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50))
    training_data_version = Column(String(50))
    feature_version = Column(String(50))
    hyperparameters = Column(JSONB)
    
    # Job Status
    status = Column(String(20), nullable=False, default="pending", index=True)  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    
    # Training Results
    training_metrics = Column(JSONB)
    validation_metrics = Column(JSONB)
    test_metrics = Column(JSONB)
    
    # Model Validation
    validation_passed = Column(Boolean)
    validation_errors = Column(ARRAY(Text))
    baseline_comparison = Column(JSONB)
    
    # Model Promotion
    promotion_status = Column(String(20))  # 'pending', 'promoted', 'rejected', 'rolled_back'
    promoted_to_stage = Column(String(20))  # 'Staging', 'Production'
    promotion_timestamp = Column(DateTime(timezone=True))
    
    # MLflow Integration
    mlflow_run_id = Column(String(50), index=True)
    mlflow_experiment_name = Column(String(100))
    model_version = Column(String(50))
    
    # Timestamps
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Metadata
    created_by = Column(String(100))
    error_message = Column(Text)
    job_metadata = Column(JSONB)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'cancelled')", name="chk_retraining_status"),
        CheckConstraint("trigger_type IN ('scheduled', 'drift', 'new_data', 'manual', 'performance_degradation')", name="chk_trigger_type"),
    )
    
    def __repr__(self) -> str:
        """String representation of RetrainingJob."""
        return f"<RetrainingJob(job_id={self.job_id}, job_name='{self.job_name}', status='{self.status}')>"


class RetrainingSchedule(Base):
    """Retraining schedule model."""
    
    __tablename__ = "retraining_schedules"
    
    # Primary Key
    schedule_id = Column(Integer, primary_key=True, autoincrement=True)
    schedule_name = Column(String(100), unique=True, nullable=False)
    
    # Schedule Configuration
    model_name = Column(String(100), nullable=False, index=True)
    schedule_type = Column(String(20), nullable=False)  # 'daily', 'weekly', 'monthly', 'cron'
    schedule_config = Column(JSONB, nullable=False)
    
    # Schedule Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    last_run_at = Column(DateTime(timezone=True))
    next_run_at = Column(DateTime(timezone=True), index=True)
    
    # Training Configuration
    training_config = Column(JSONB)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Metadata
    created_by = Column(String(100))
    description = Column(Text)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("schedule_type IN ('daily', 'weekly', 'monthly', 'cron')", name="chk_schedule_type"),
    )
    
    def __repr__(self) -> str:
        """String representation of RetrainingSchedule."""
        return f"<RetrainingSchedule(schedule_id={self.schedule_id}, schedule_name='{self.schedule_name}', is_active={self.is_active})>"


class ModelValidationRule(Base):
    """Model validation rule model."""
    
    __tablename__ = "model_validation_rules"
    
    # Primary Key
    rule_id = Column(Integer, primary_key=True, autoincrement=True)
    rule_name = Column(String(100), unique=True, nullable=False)
    
    # Rule Configuration
    model_name = Column(String(100), nullable=False, index=True)
    metric_name = Column(String(50), nullable=False)  # 'accuracy', 'roc_auc', etc.
    comparison_operator = Column(String(10), nullable=False)  # '>', '>=', '<', '<=', '=='
    threshold_value = Column(Numeric(10, 6), nullable=False)
    comparison_type = Column(String(20), default="absolute", nullable=False)  # 'absolute', 'relative_to_baseline', 'relative_improvement'
    
    # Baseline Configuration
    baseline_model_version = Column(String(50))
    minimum_improvement = Column(Numeric(5, 4))  # Minimum improvement percentage
    
    # Rule Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_required = Column(Boolean, default=True, nullable=False)  # If False, violation is warning
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Metadata
    description = Column(Text)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("comparison_operator IN ('>', '>=', '<', '<=', '==', '!=')", name="chk_comparison_operator"),
        CheckConstraint("comparison_type IN ('absolute', 'relative_to_baseline', 'relative_improvement')", name="chk_comparison_type"),
    )
    
    def __repr__(self) -> str:
        """String representation of ModelValidationRule."""
        return f"<ModelValidationRule(rule_id={self.rule_id}, rule_name='{self.rule_name}', metric='{self.metric_name}')>"


# ============================================================================
# BATCH PREDICTION MODELS
# ============================================================================

class BatchPredictionJob(Base):
    """Batch prediction job model."""
    
    __tablename__ = "batch_prediction_jobs"
    
    # Primary Key
    job_id = Column(Integer, primary_key=True, autoincrement=True)
    job_name = Column(String(100), nullable=False)
    
    # Job Configuration
    trigger_type = Column(String(50), nullable=False, index=True)  # 'manual', 'scheduled', 'event'
    schedule_id = Column(Integer, index=True)
    
    # Input Configuration
    input_source = Column(String(50), nullable=False)  # 'database', 'file', 'api'
    input_config = Column(JSONB, nullable=False)
    
    # Processing Configuration
    batch_size = Column(Integer, default=1000, nullable=False)
    max_workers = Column(Integer, default=4, nullable=False)
    use_feature_store = Column(Boolean, default=True, nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    model_stage = Column(String(20), default="Production", nullable=False)
    
    # Output Configuration
    output_format = Column(String(20), nullable=False)  # 'database', 'file', 's3', 'parquet', 'csv'
    output_config = Column(JSONB, nullable=False)
    
    # Job Status
    status = Column(String(20), nullable=False, default="pending", index=True)  # 'pending', 'running', 'completed', 'failed', 'cancelled', 'paused'
    
    # Progress Tracking
    total_records = Column(Integer)
    processed_records = Column(Integer, default=0, nullable=False)
    failed_records = Column(Integer, default=0, nullable=False)
    progress_percentage = Column(Numeric(5, 2), default=0.0, nullable=False)
    
    # Results
    output_path = Column(String(500))
    output_file_size_bytes = Column(BigInteger)
    records_per_second = Column(Numeric(10, 2))
    
    # Error Handling
    error_message = Column(Text)
    error_count = Column(Integer, default=0, nullable=False)
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=3, nullable=False)
    
    # Timestamps
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Metadata
    created_by = Column(String(100))
    job_metadata = Column(JSONB)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'paused')", name="chk_batch_job_status"),
        CheckConstraint("trigger_type IN ('manual', 'scheduled', 'event')", name="chk_trigger_type"),
        CheckConstraint("input_source IN ('database', 'file', 'api')", name="chk_input_source"),
        CheckConstraint("output_format IN ('database', 'file', 's3', 'parquet', 'csv')", name="chk_output_format"),
    )
    
    def __repr__(self) -> str:
        """String representation of BatchPredictionJob."""
        return f"<BatchPredictionJob(job_id={self.job_id}, job_name='{self.job_name}', status='{self.status}')>"


class BatchPredictionSchedule(Base):
    """Batch prediction schedule model."""
    
    __tablename__ = "batch_prediction_schedules"
    
    # Primary Key
    schedule_id = Column(Integer, primary_key=True, autoincrement=True)
    schedule_name = Column(String(100), unique=True, nullable=False)
    
    # Schedule Configuration
    schedule_type = Column(String(20), nullable=False)  # 'daily', 'weekly', 'monthly', 'cron'
    schedule_config = Column(JSONB, nullable=False)
    
    # Schedule Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    last_run_at = Column(DateTime(timezone=True))
    next_run_at = Column(DateTime(timezone=True), index=True)
    
    # Job Configuration (template)
    job_config = Column(JSONB, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Metadata
    created_by = Column(String(100))
    description = Column(Text)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("schedule_type IN ('daily', 'weekly', 'monthly', 'cron')", name="chk_batch_schedule_type"),
    )
    
    def __repr__(self) -> str:
        """String representation of BatchPredictionSchedule."""
        return f"<BatchPredictionSchedule(schedule_id={self.schedule_id}, schedule_name='{self.schedule_name}', is_active={self.is_active})>"


class BatchPredictionResult(Base):
    """Batch prediction result model."""
    
    __tablename__ = "batch_prediction_results"
    
    # Primary Key
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("batch_prediction_jobs.job_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Customer Information
    customer_id = Column(String(100), nullable=False, index=True)
    
    # Prediction Results
    prediction = Column(Integer, nullable=False)  # 0 or 1
    probability = Column(Numeric(5, 4), nullable=False)
    customer_score = Column(Integer)
    risk_level = Column(String(10), nullable=False)
    
    # Features (optional)
    features = Column(JSONB)
    
    # Model Information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Processing Metadata
    processing_time_ms = Column(Numeric(10, 2))
    row_number = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("prediction IN (0, 1)", name="chk_batch_prediction"),
        CheckConstraint("probability >= 0 AND probability <= 1", name="chk_batch_probability"),
        CheckConstraint("risk_level IN ('low', 'medium', 'high')", name="chk_batch_risk_level"),
    )
    
    def __repr__(self) -> str:
        """String representation of BatchPredictionResult."""
        return f"<BatchPredictionResult(result_id={self.result_id}, job_id={self.job_id}, customer_id='{self.customer_id}')>"


class BatchPredictionLog(Base):
    """Batch prediction log model."""
    
    __tablename__ = "batch_prediction_logs"
    
    # Primary Key
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("batch_prediction_jobs.job_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Log Details
    log_level = Column(String(20), nullable=False, index=True)  # 'INFO', 'WARNING', 'ERROR', 'DEBUG'
    message = Column(Text, nullable=False)
    error_details = Column(JSONB)
    
    # Context
    record_index = Column(Integer)
    customer_id = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("log_level IN ('INFO', 'WARNING', 'ERROR', 'DEBUG')", name="chk_batch_log_level"),
    )
    
    def __repr__(self) -> str:
        """String representation of BatchPredictionLog."""
        return f"<BatchPredictionLog(log_id={self.log_id}, job_id={self.job_id}, log_level='{self.log_level}')>"


# ============================================================================
# MULTI-MODEL SERVING MODELS
# ============================================================================

class ModelRoutingRule(Base):
    """Model routing rule model."""
    
    __tablename__ = "model_routing_rules"
    
    # Primary Key
    rule_id = Column(Integer, primary_key=True, autoincrement=True)
    rule_name = Column(String(100), unique=True, nullable=False)
    
    # Rule Configuration
    priority = Column(Integer, nullable=False, default=0, index=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Routing Criteria
    routing_criteria = Column(JSONB, nullable=False)
    routing_type = Column(String(50), nullable=False)  # 'single', 'ensemble', 'weighted_ensemble', 'comparison'
    
    # Target Models
    target_models = Column(JSONB, nullable=False)
    model_weights = Column(JSONB)
    
    # Fallback Configuration
    fallback_model_name = Column(String(100))
    fallback_model_stage = Column(String(20), default="Production", nullable=False)
    
    # Metadata
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(100))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("routing_type IN ('single', 'ensemble', 'weighted_ensemble', 'comparison')", name="chk_routing_type"),
    )
    
    def __repr__(self) -> str:
        """String representation of ModelRoutingRule."""
        return f"<ModelRoutingRule(rule_id={self.rule_id}, rule_name='{self.rule_name}', routing_type='{self.routing_type}')>"


class ModelRegistry(Base):
    """Extended model registry for multi-model serving."""
    
    __tablename__ = "model_registry"
    
    # Primary Key
    registry_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Model Information
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    model_stage = Column(String(20), nullable=False, index=True)  # 'Production', 'Staging', 'Archived'
    
    # Model Metadata
    model_type = Column(String(50))
    mlflow_run_id = Column(String(50))
    mlflow_model_uri = Column(String(500))
    
    # Performance Metrics
    accuracy = Column(Numeric(5, 4))
    roc_auc = Column(Numeric(5, 4))
    precision = Column(Numeric(5, 4))
    recall = Column(Numeric(5, 4))
    f1_score = Column(Numeric(5, 4))
    
    # Serving Configuration
    is_loaded = Column(Boolean, default=False, nullable=False, index=True)
    load_priority = Column(Integer, default=0, nullable=False, index=True)
    max_concurrent_requests = Column(Integer, default=100, nullable=False)
    
    # Resource Requirements
    memory_usage_mb = Column(Integer)
    cpu_usage_percent = Column(Numeric(5, 2))
    
    # Status
    status = Column(String(20), default="available", nullable=False, index=True)  # 'available', 'loading', 'unavailable', 'error'
    error_message = Column(Text)
    
    # Timestamps
    registered_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_used_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Metadata
    model_metadata = Column("metadata", JSONB)  # Column name is 'metadata' in DB, but Python attribute is 'model_metadata'
    
    # Constraints
    __table_args__ = (
        CheckConstraint("model_stage IN ('Production', 'Staging', 'Archived')", name="chk_model_stage"),
        CheckConstraint("status IN ('available', 'loading', 'unavailable', 'error')", name="chk_model_status"),
    )
    # Note: Unique constraint on (model_name, model_version) is defined in SQL schema
    
    def __repr__(self) -> str:
        """String representation of ModelRegistry."""
        return f"<ModelRegistry(registry_id={self.registry_id}, model_name='{self.model_name}', version='{self.model_version}', stage='{self.model_stage}')>"


class ModelComparisonResult(Base):
    """Model comparison result model."""
    
    __tablename__ = "model_comparison_results"
    
    # Primary Key
    comparison_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Comparison Configuration
    comparison_name = Column(String(100))
    comparison_type = Column(String(50), nullable=False, index=True)  # 'real_time', 'batch', 'historical'
    
    # Models Compared
    model_1_name = Column(String(100), nullable=False, index=True)
    model_1_version = Column(String(50), nullable=False)
    model_2_name = Column(String(100), nullable=False, index=True)
    model_2_version = Column(String(50), nullable=False)
    
    # Comparison Metrics
    comparison_metrics = Column(JSONB, nullable=False)
    differences = Column(JSONB)
    winner = Column(String(100))
    
    # Test Data
    test_samples = Column(Integer)
    test_customer_ids = Column(ARRAY(Text))
    
    # Timestamps
    compared_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    created_by = Column(String(100))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("comparison_type IN ('real_time', 'batch', 'historical')", name="chk_comparison_type"),
    )
    
    def __repr__(self) -> str:
        """String representation of ModelComparisonResult."""
        return f"<ModelComparisonResult(comparison_id={self.comparison_id}, model_1='{self.model_1_name}', model_2='{self.model_2_name}')>"


class ModelEnsemble(Base):
    """Model ensemble configuration model."""
    
    __tablename__ = "model_ensembles"
    
    # Primary Key
    ensemble_id = Column(Integer, primary_key=True, autoincrement=True)
    ensemble_name = Column(String(100), unique=True, nullable=False)
    
    # Ensemble Configuration
    ensemble_type = Column(String(50), nullable=False)  # 'voting', 'weighted_average', 'stacking'
    model_configs = Column(JSONB, nullable=False)
    
    # Ensemble Metadata
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    description = Column(Text)
    
    # Performance
    ensemble_metrics = Column(JSONB)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(100))
    
    # Constraints
    __table_args__ = (
        CheckConstraint("ensemble_type IN ('voting', 'weighted_average', 'stacking')", name="chk_ensemble_type"),
    )
    
    def __repr__(self) -> str:
        """String representation of ModelEnsemble."""
        return f"<ModelEnsemble(ensemble_id={self.ensemble_id}, ensemble_name='{self.ensemble_name}', ensemble_type='{self.ensemble_type}')>"
