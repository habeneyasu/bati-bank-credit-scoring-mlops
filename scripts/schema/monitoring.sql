-- Monitoring Tables Schema (TimescaleDB)
-- Purpose: Store time-series data for monitoring, drift detection, and performance metrics

-- Enable TimescaleDB extension (if using TimescaleDB)
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Performance Metrics Table
CREATE TABLE IF NOT EXISTS performance_metrics (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Metrics
    endpoint VARCHAR(100) NOT NULL,  -- '/predict', '/explain', etc.
    latency_ms DECIMAL(10,2) NOT NULL,
    status_code INTEGER,
    error BOOLEAN DEFAULT FALSE,
    
    -- Request Information
    customer_id VARCHAR(100),
    model_version VARCHAR(50),
    
    -- Additional Metadata
    metadata JSONB
);

-- Convert to hypertable (TimescaleDB) - uncomment if using TimescaleDB
-- SELECT create_hypertable('performance_metrics', 'time');

-- Indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_endpoint ON performance_metrics(endpoint, time DESC);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_customer ON performance_metrics(customer_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_time ON performance_metrics(time DESC);

-- Drift Detection Metrics Table
CREATE TABLE IF NOT EXISTS drift_metrics (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Feature Information
    feature_name VARCHAR(100) NOT NULL,
    
    -- Drift Metrics
    psi DECIMAL(10,6),  -- Population Stability Index
    ks_statistic DECIMAL(10,6),  -- Kolmogorov-Smirnov statistic
    chi_square DECIMAL(10,6),  -- Chi-square statistic
    
    -- Drift Status
    is_drifted BOOLEAN DEFAULT FALSE,
    drift_severity VARCHAR(20),  -- 'none', 'minor', 'major'
    
    -- Reference Distribution
    reference_distribution JSONB,
    current_distribution JSONB,
    
    -- Metadata
    model_version VARCHAR(50),
    metadata JSONB
);

-- Convert to hypertable (TimescaleDB) - uncomment if using TimescaleDB
-- SELECT create_hypertable('drift_metrics', 'time');

-- Indexes
CREATE INDEX IF NOT EXISTS idx_drift_metrics_feature ON drift_metrics(feature_name, time DESC);
CREATE INDEX IF NOT EXISTS idx_drift_metrics_drifted ON drift_metrics(is_drifted, time DESC);
CREATE INDEX IF NOT EXISTS idx_drift_metrics_time ON drift_metrics(time DESC);

-- Comments
COMMENT ON TABLE performance_metrics IS 'Time-series performance metrics for API endpoints';
COMMENT ON TABLE drift_metrics IS 'Time-series drift detection metrics for features and predictions';
