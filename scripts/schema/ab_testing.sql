-- A/B Testing Framework Schema
-- Purpose: Support model A/B testing with traffic splitting and statistical analysis

-- Experiments Table
CREATE TABLE IF NOT EXISTS experiments (
    -- Primary Key
    experiment_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL UNIQUE,
    
    -- Experiment Configuration
    description TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',  -- 'draft', 'running', 'paused', 'completed', 'cancelled'
    
    -- Variants Configuration (stored as JSONB for flexibility)
    variants JSONB NOT NULL,  -- [{"name": "control", "model_version": "v1", "traffic_percentage": 50}, ...]
    
    -- Traffic Splitting
    traffic_percentage INTEGER DEFAULT 100,  -- Percentage of traffic to include in experiment (0-100)
    assignment_method VARCHAR(50) DEFAULT 'hash',  -- 'hash', 'random', 'customer_segment'
    
    -- Experiment Dates
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    actual_started_at TIMESTAMP WITH TIME ZONE,
    actual_ended_at TIMESTAMP WITH TIME ZONE,
    
    -- Success Criteria
    primary_metric VARCHAR(50) DEFAULT 'accuracy',  -- 'accuracy', 'roc_auc', 'precision', 'recall', 'f1', 'latency'
    minimum_sample_size INTEGER DEFAULT 1000,  -- Minimum samples per variant
    significance_level DECIMAL(5, 4) DEFAULT 0.05,  -- p-value threshold (default 0.05)
    minimum_improvement DECIMAL(5, 4) DEFAULT 0.01,  -- Minimum improvement to declare winner (1%)
    
    -- Results
    winner_variant VARCHAR(100),  -- Name of winning variant
    statistical_significance DECIMAL(5, 4),  -- Calculated p-value
    confidence_interval JSONB,  -- Confidence interval for difference
    conclusion TEXT,  -- Automated conclusion text
    
    -- Metadata
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_status CHECK (status IN ('draft', 'running', 'paused', 'completed', 'cancelled')),
    CONSTRAINT chk_traffic_percentage CHECK (traffic_percentage >= 0 AND traffic_percentage <= 100),
    CONSTRAINT chk_significance_level CHECK (significance_level > 0 AND significance_level < 1)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(experiment_name);
CREATE INDEX IF NOT EXISTS idx_experiments_dates ON experiments(start_date, end_date);

-- Experiment Assignments Table (tracks which variant each customer/request gets)
CREATE TABLE IF NOT EXISTS experiment_assignments (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- Foreign Key
    experiment_id INTEGER NOT NULL,
    
    -- Assignment Details
    entity_id VARCHAR(100) NOT NULL,  -- Customer ID or request ID
    entity_type VARCHAR(50) DEFAULT 'customer',  -- 'customer', 'request'
    variant_name VARCHAR(100) NOT NULL,  -- Which variant was assigned
    
    -- Assignment Metadata
    assignment_hash VARCHAR(64),  -- Hash used for consistent assignment
    assigned_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT fk_experiment_assignments_experiment FOREIGN KEY (experiment_id) 
        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    CONSTRAINT uq_experiment_entity UNIQUE (experiment_id, entity_id, entity_type)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_experiment_assignments_experiment ON experiment_assignments(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_assignments_entity ON experiment_assignments(entity_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_experiment_assignments_variant ON experiment_assignments(variant_name);

-- Experiment Metrics Table (aggregated metrics per variant)
CREATE TABLE IF NOT EXISTS experiment_metrics (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- Foreign Key
    experiment_id INTEGER NOT NULL,
    
    -- Variant Information
    variant_name VARCHAR(100) NOT NULL,
    
    -- Metrics
    sample_size INTEGER DEFAULT 0,
    accuracy DECIMAL(10, 6),
    roc_auc DECIMAL(10, 6),
    precision DECIMAL(10, 6),
    recall DECIMAL(10, 6),
    f1_score DECIMAL(10, 6),
    avg_latency_ms DECIMAL(10, 2),
    p95_latency_ms DECIMAL(10, 2),
    error_rate DECIMAL(10, 6),
    
    -- Business Metrics
    total_predictions INTEGER DEFAULT 0,
    high_risk_predictions INTEGER DEFAULT 0,
    low_risk_predictions INTEGER DEFAULT 0,
    
    -- Statistical Metrics
    mean_value DECIMAL(10, 6),  -- Mean of primary metric
    std_value DECIMAL(10, 6),  -- Standard deviation
    confidence_interval_lower DECIMAL(10, 6),
    confidence_interval_upper DECIMAL(10, 6),
    
    -- Timestamps
    calculated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT fk_experiment_metrics_experiment FOREIGN KEY (experiment_id) 
        REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_experiment_metrics_experiment ON experiment_metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_metrics_variant ON experiment_metrics(variant_name);
CREATE INDEX IF NOT EXISTS idx_experiment_metrics_calculated ON experiment_metrics(calculated_at);

-- Comments
COMMENT ON TABLE experiments IS 'A/B testing experiments configuration and results';
COMMENT ON TABLE experiment_assignments IS 'Tracks which variant each customer/request is assigned to';
COMMENT ON TABLE experiment_metrics IS 'Aggregated performance metrics per variant for statistical analysis';
