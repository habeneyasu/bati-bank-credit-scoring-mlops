"""
Configuration management for the credit scoring application.

Uses pydantic-settings for environment variable management with validation.
"""

from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default="file:./mlruns",
        description="MLflow tracking URI"
    )
    model_name: str = Field(
        default="credit_scoring_model",
        description="Name of the registered model"
    )
    model_stage: str = Field(
        default="Production",
        description="Model stage (Production, Staging, None)"
    )
    
    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    api_port: int = Field(
        default=8000,
        description="API port number",
        ge=1,
        le=65535
    )
    api_workers: int = Field(
        default=1,
        description="Number of API workers",
        ge=1
    )
    api_reload: bool = Field(
        default=False,
        description="Enable auto-reload for development"
    )
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow CORS credentials"
    )
    cors_allow_methods: List[str] = Field(
        default=["*"],
        description="Allowed CORS methods"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"],
        description="Allowed CORS headers"
    )
    
    # Risk Thresholds
    # NOTE: These thresholds are used as fallback when adaptive thresholds cannot be calculated.
    # The system uses adaptive percentile-based thresholds (33rd and 67th percentiles) from recent
    # predictions to ensure a balanced distribution across low, medium, and high risk categories.
    # These fixed thresholds are only used when there are fewer than 10 recent predictions.
    risk_threshold_low: float = Field(
        default=0.15,  # Lower threshold for low risk (probability < threshold = low risk)
        description="Low risk threshold (probability < threshold). Used as fallback when adaptive thresholds unavailable.",
        ge=0.0,
        le=1.0
    )
    risk_threshold_high: float = Field(
        default=0.20,  # Higher threshold for high risk (probability > threshold = high risk)
        description="High risk threshold (probability > threshold). Used as fallback when adaptive thresholds unavailable.",
        ge=0.0,
        le=1.0
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json, text)"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path (optional)"
    )
    
    # Model Configuration
    expected_features: int = Field(
        default=26,
        description="Expected number of input features",
        ge=1
    )
    
    # Security Configuration
    rate_limit_per_minute: int = Field(
        default=60,
        description="Rate limit per minute per IP",
        ge=1
    )
    enable_rate_limiting: bool = Field(
        default=False,
        description="Enable rate limiting"
    )
    
    # PII Redaction Configuration (GDPR/CCPA Compliance)
    enable_pii_redaction: bool = Field(
        default=True,
        description="Enable PII redaction in logs and responses"
    )
    pii_redaction_strategy: str = Field(
        default="mask",
        description="PII redaction strategy: mask, hash, remove, partial_mask"
    )
    enable_pii_redaction_logging: bool = Field(
        default=True,
        description="Log PII redaction actions"
    )
    redact_in_logs: bool = Field(
        default=True,
        description="Redact PII in application logs"
    )
    redact_in_responses: bool = Field(
        default=False,
        description="Redact PII in API responses (use with caution)"
    )
    
    # Distributed Tracing Configuration (OpenTelemetry)
    enable_distributed_tracing: bool = Field(
        default=False,
        description="Enable OpenTelemetry distributed tracing"
    )
    tracing_service_name: str = Field(
        default="credit-scoring-api",
        description="Service name for tracing"
    )
    tracing_service_version: str = Field(
        default="1.0.0",
        description="Service version for tracing"
    )
    tracing_exporter: str = Field(
        default="console",
        description="Tracing exporter type: console, otlp, jaeger, zipkin"
    )
    tracing_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP exporter endpoint"
    )
    tracing_jaeger_endpoint: str = Field(
        default="http://localhost:14268/api/traces",
        description="Jaeger collector endpoint"
    )
    tracing_zipkin_endpoint: str = Field(
        default="http://localhost:9411/api/v2/spans",
        description="Zipkin collector endpoint"
    )
    tracing_sample_rate: float = Field(
        default=1.0,
        description="Trace sampling rate (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Monitoring Configuration
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus-style metrics"
    )
    metrics_path: str = Field(
        default="/metrics",
        description="Metrics endpoint path"
    )
    
    # Caching Configuration
    use_redis_cache: bool = Field(
        default=False,
        description="Use Redis for caching (requires redis package)"
    )
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL (default: redis://localhost:6379/0)"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        description="Cache time-to-live in seconds",
        ge=1
    )
    enable_prediction_cache: bool = Field(
        default=True,
        description="Enable caching for predictions"
    )
    
    # Performance Configuration
    target_p95_latency_ms: float = Field(
        default=200.0,
        description="Target 95th percentile latency in milliseconds",
        ge=1.0
    )
    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )
    
    # Environment
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # Database Configuration
    database_url: Optional[str] = Field(
        default=None,
        description="Database connection URL (postgresql://user:pass@host:port/dbname)"
    )
    database_host: str = Field(
        default="localhost",
        description="Database host"
    )
    database_port: int = Field(
        default=5432,
        description="Database port",
        ge=1,
        le=65535
    )
    database_name: str = Field(
        default="mlops_db",
        description="Database name"
    )
    database_user: str = Field(
        default="postgres",
        description="Database user"
    )
    database_password: Optional[str] = Field(
        default=None,
        description="Database password"
    )
    database_pool_size: int = Field(
        default=5,
        description="Database connection pool size",
        ge=1
    )
    database_max_overflow: int = Field(
        default=10,
        description="Database connection pool max overflow",
        ge=0
    )
    database_echo: bool = Field(
        default=False,
        description="Echo SQL queries (for debugging)"
    )
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("cors_allow_methods", pre=True)
    def parse_cors_methods(cls, v):
        """Parse CORS methods from string or list."""
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    @validator("cors_allow_headers", pre=True)
    def parse_cors_headers(cls, v):
        """Parse CORS headers from string or list."""
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator("risk_threshold_high")
    def validate_risk_thresholds(cls, v, values):
        """Validate risk thresholds."""
        if "risk_threshold_low" in values and v <= values["risk_threshold_low"]:
            raise ValueError("risk_threshold_high must be greater than risk_threshold_low")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()
