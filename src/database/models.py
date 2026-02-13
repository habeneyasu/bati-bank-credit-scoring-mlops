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
    ForeignKey, CheckConstraint, Index, JSON, ARRAY, INET
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
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
    metadata = Column(JSONB)
    
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
    customer_score = Column(Integer)  # 0-1000 scale
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
    metadata = Column(JSONB)
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
    metadata = Column(JSONB)
    
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
    metadata = Column(JSONB)
    
    # Indexes
    __table_args__ = (
        Index("idx_drift_metrics_feature", "feature_name", "time"),
        Index("idx_drift_metrics_drifted", "is_drifted", "time"),
    )
    
    def __repr__(self) -> str:
        """String representation of DriftMetric."""
        return f"<DriftMetric(feature_name='{self.feature_name}', is_drifted={self.is_drifted}, psi={self.psi})>"
