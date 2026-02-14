-- Metadata Tables Schema
-- Purpose: Track data versions, model metadata, and business KPIs

-- Data Versioning Table
CREATE TABLE IF NOT EXISTS data_versions (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- Version Information
    data_type VARCHAR(50) NOT NULL,  -- 'dataset', 'features', 'splits', 'artifacts'
    version VARCHAR(50) NOT NULL,  -- 'v1', 'v2', etc.
    
    -- File Information
    file_path TEXT NOT NULL,
    file_size BIGINT,  -- Size in bytes
    checksum_sha256 VARCHAR(64) NOT NULL,  -- SHA256 checksum
    
    -- Metadata (using data_metadata to avoid SQLAlchemy reserved word conflict)
    data_metadata JSONB,  -- Additional metadata (shape, columns, etc.)
    dependencies TEXT[],  -- Array of dependency versions
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Unique Constraint
    UNIQUE(data_type, version)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_data_versions_type ON data_versions(data_type);
CREATE INDEX IF NOT EXISTS idx_data_versions_created ON data_versions(created_at);

-- Model Metadata Table
CREATE TABLE IF NOT EXISTS model_metadata (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- Model Information
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_stage VARCHAR(20) NOT NULL,  -- 'Production', 'Staging', 'Archived'
    
    -- MLflow Integration
    mlflow_run_id VARCHAR(50),
    mlflow_experiment_name VARCHAR(100),
    
    -- Performance Metrics
    roc_auc DECIMAL(5,4),
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    
    -- Training Information
    training_data_version VARCHAR(50),
    feature_version VARCHAR(50),
    hyperparameters JSONB,
    
    -- Deployment Information
    deployed_at TIMESTAMP WITH TIME ZONE,
    deployed_by VARCHAR(100),
    deployment_environment VARCHAR(50),  -- 'production', 'staging'
    
    -- Status
    is_active BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Unique Constraint
    UNIQUE(model_name, model_version)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_model_metadata_name ON model_metadata(model_name);
CREATE INDEX IF NOT EXISTS idx_model_metadata_stage ON model_metadata(model_stage);
CREATE INDEX IF NOT EXISTS idx_model_metadata_active ON model_metadata(is_active);

-- Data Lineage Table
CREATE TABLE IF NOT EXISTS data_lineage (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- Source (upstream) - what data version was used
    source_data_version_id INTEGER NOT NULL REFERENCES data_versions(id),
    source_data_type VARCHAR(50) NOT NULL,  -- 'raw_transactions', 'processed', 'features', etc.
    source_version VARCHAR(50) NOT NULL,
    
    -- Target (downstream) - what was created from this data
    target_type VARCHAR(50) NOT NULL,  -- 'model', 'prediction', 'feature_set', 'processed_data'
    target_id VARCHAR(100) NOT NULL,  -- model_version, prediction_id, etc.
    target_name VARCHAR(200),  -- Human-readable name
    
    -- Relationship metadata
    relationship_type VARCHAR(50) NOT NULL,  -- 'trained_on', 'used_for', 'derived_from', 'generated_from'
    operation VARCHAR(100),  -- 'training', 'prediction', 'feature_engineering', 'processing'
    
    -- Additional context (note: using lineage_metadata instead of metadata to avoid SQLAlchemy conflict)
    lineage_metadata JSONB,  -- Additional context (timestamp, user, etc.)
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_lineage_source ON data_lineage(source_data_version_id, target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_lineage_target ON data_lineage(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_lineage_source_version ON data_lineage(source_data_version_id);
CREATE INDEX IF NOT EXISTS idx_lineage_created ON data_lineage(created_at);

-- Business KPIs Table
CREATE TABLE IF NOT EXISTS business_kpis (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- Time Period
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    period_type VARCHAR(20) NOT NULL,  -- 'hourly', 'daily', 'weekly', 'monthly'
    
    -- KPI Metrics
    total_predictions INTEGER NOT NULL DEFAULT 0,
    approval_count INTEGER NOT NULL DEFAULT 0,  -- Low risk
    rejection_count INTEGER NOT NULL DEFAULT 0,  -- High risk
    review_count INTEGER NOT NULL DEFAULT 0,  -- Medium risk
    
    approval_rate DECIMAL(5,4),  -- approval_count / total_predictions
    rejection_rate DECIMAL(5,4),  -- rejection_count / total_predictions
    review_rate DECIMAL(5,4),  -- review_count / total_predictions
    
    avg_risk_score DECIMAL(5,4),  -- Average probability
    median_risk_score DECIMAL(5,4),
    
    -- Customer Metrics
    unique_customers INTEGER,
    new_customers INTEGER,
    
    -- Performance Metrics
    avg_latency_ms DECIMAL(10,2),
    p95_latency_ms DECIMAL(10,2),
    p99_latency_ms DECIMAL(10,2),
    error_rate DECIMAL(5,4),
    
    -- Timestamps
    calculated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Unique Constraint
    UNIQUE(period_start, period_end, period_type)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_business_kpis_period ON business_kpis(period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_business_kpis_type ON business_kpis(period_type);

-- Comments
COMMENT ON TABLE data_versions IS 'Tracks versions of datasets, features, and artifacts with checksums';
COMMENT ON TABLE model_metadata IS 'Tracks model versions, performance metrics, and deployment information';
COMMENT ON TABLE data_lineage IS 'Tracks relationships between data versions, models, and predictions';
COMMENT ON TABLE business_kpis IS 'Stores aggregated business metrics for reporting and analytics';
