-- Automated Model Retraining Pipeline Schema
-- Purpose: Track retraining jobs, triggers, validation results, and promotions

-- Retraining Jobs Table
CREATE TABLE IF NOT EXISTS retraining_jobs (
    -- Primary Key
    job_id SERIAL PRIMARY KEY,
    job_name VARCHAR(100) NOT NULL,
    
    -- Job Configuration
    trigger_type VARCHAR(50) NOT NULL,  -- 'scheduled', 'drift', 'new_data', 'manual', 'performance_degradation'
    trigger_metadata JSONB,  -- Additional trigger information
    
    -- Training Configuration
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50),  -- 'logistic_regression', 'random_forest', etc.
    training_data_version VARCHAR(50),
    feature_version VARCHAR(50),
    hyperparameters JSONB,
    
    -- Job Status
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- 'pending', 'running', 'completed', 'failed', 'cancelled'
    
    -- Training Results
    training_metrics JSONB,  -- Training set metrics
    validation_metrics JSONB,  -- Validation set metrics
    test_metrics JSONB,  -- Test set metrics
    
    -- Model Validation
    validation_passed BOOLEAN,
    validation_errors TEXT[],
    baseline_comparison JSONB,  -- Comparison with baseline model
    
    -- Model Promotion
    promotion_status VARCHAR(20),  -- 'pending', 'promoted', 'rejected', 'rolled_back'
    promoted_to_stage VARCHAR(20),  -- 'Staging', 'Production'
    promotion_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- MLflow Integration
    mlflow_run_id VARCHAR(50),
    mlflow_experiment_name VARCHAR(100),
    model_version VARCHAR(50),  -- MLflow model version
    
    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Metadata
    created_by VARCHAR(100),
    error_message TEXT,
    job_metadata JSONB,
    
    -- Constraints
    CONSTRAINT chk_retraining_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_trigger_type CHECK (trigger_type IN ('scheduled', 'drift', 'new_data', 'manual', 'performance_degradation'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_retraining_jobs_status ON retraining_jobs(status);
CREATE INDEX IF NOT EXISTS idx_retraining_jobs_trigger ON retraining_jobs(trigger_type);
CREATE INDEX IF NOT EXISTS idx_retraining_jobs_model ON retraining_jobs(model_name);
CREATE INDEX IF NOT EXISTS idx_retraining_jobs_created ON retraining_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_retraining_jobs_mlflow ON retraining_jobs(mlflow_run_id);

-- Retraining Schedule Table
CREATE TABLE IF NOT EXISTS retraining_schedules (
    -- Primary Key
    schedule_id SERIAL PRIMARY KEY,
    schedule_name VARCHAR(100) NOT NULL UNIQUE,
    
    -- Schedule Configuration
    model_name VARCHAR(100) NOT NULL,
    schedule_type VARCHAR(20) NOT NULL,  -- 'daily', 'weekly', 'monthly', 'cron'
    schedule_config JSONB NOT NULL,  -- Cron expression or schedule details
    
    -- Schedule Status
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMP WITH TIME ZONE,
    next_run_at TIMESTAMP WITH TIME ZONE,
    
    -- Training Configuration
    training_config JSONB,  -- Model type, hyperparameters, etc.
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Metadata
    created_by VARCHAR(100),
    description TEXT,
    
    -- Constraints
    CONSTRAINT chk_schedule_type CHECK (schedule_type IN ('daily', 'weekly', 'monthly', 'cron'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_retraining_schedules_active ON retraining_schedules(is_active, next_run_at);
CREATE INDEX IF NOT EXISTS idx_retraining_schedules_model ON retraining_schedules(model_name);

-- Model Validation Rules Table
CREATE TABLE IF NOT EXISTS model_validation_rules (
    -- Primary Key
    rule_id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL UNIQUE,
    
    -- Rule Configuration
    model_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,  -- 'accuracy', 'roc_auc', 'precision', etc.
    comparison_operator VARCHAR(10) NOT NULL,  -- '>', '>=', '<', '<=', '=='
    threshold_value DECIMAL(10, 6) NOT NULL,
    comparison_type VARCHAR(20) DEFAULT 'absolute',  -- 'absolute', 'relative_to_baseline', 'relative_improvement'
    
    -- Baseline Configuration
    baseline_model_version VARCHAR(50),
    minimum_improvement DECIMAL(5, 4),  -- Minimum improvement percentage (e.g., 0.01 for 1%)
    
    -- Rule Status
    is_active BOOLEAN DEFAULT TRUE,
    is_required BOOLEAN DEFAULT TRUE,  -- If False, violation is warning, not error
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Metadata
    description TEXT,
    
    -- Constraints
    CONSTRAINT chk_comparison_operator CHECK (comparison_operator IN ('>', '>=', '<', '<=', '==', '!=')),
    CONSTRAINT chk_comparison_type CHECK (comparison_type IN ('absolute', 'relative_to_baseline', 'relative_improvement'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_validation_rules_model ON model_validation_rules(model_name, is_active);
CREATE INDEX IF NOT EXISTS idx_validation_rules_metric ON model_validation_rules(metric_name);

-- Comments
COMMENT ON TABLE retraining_jobs IS 'Tracks automated model retraining jobs with validation and promotion status';
COMMENT ON TABLE retraining_schedules IS 'Stores scheduled retraining configurations';
COMMENT ON TABLE model_validation_rules IS 'Defines validation rules for model promotion';
