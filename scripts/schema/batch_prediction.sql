-- Batch Prediction Pipeline Schema
-- Purpose: Track batch prediction jobs, schedules, and results

-- Batch Prediction Jobs Table
CREATE TABLE IF NOT EXISTS batch_prediction_jobs (
    -- Primary Key
    job_id SERIAL PRIMARY KEY,
    job_name VARCHAR(100) NOT NULL,
    
    -- Job Configuration
    trigger_type VARCHAR(50) NOT NULL,  -- 'manual', 'scheduled', 'event'
    schedule_id INTEGER,  -- Reference to schedule if scheduled
    
    -- Input Configuration
    input_source VARCHAR(50) NOT NULL,  -- 'database', 'file', 'api'
    input_config JSONB NOT NULL,  -- Source-specific configuration
    
    -- Processing Configuration
    batch_size INTEGER DEFAULT 1000,
    max_workers INTEGER DEFAULT 4,
    use_feature_store BOOLEAN DEFAULT TRUE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    model_stage VARCHAR(20) DEFAULT 'Production',
    
    -- Output Configuration
    output_format VARCHAR(20) NOT NULL,  -- 'database', 'file', 's3'
    output_config JSONB NOT NULL,  -- Output-specific configuration
    
    -- Job Status
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- 'pending', 'running', 'completed', 'failed', 'cancelled', 'paused'
    
    -- Progress Tracking
    total_records INTEGER,
    processed_records INTEGER DEFAULT 0,
    failed_records INTEGER DEFAULT 0,
    progress_percentage DECIMAL(5, 2) DEFAULT 0.0,
    
    -- Results
    output_path VARCHAR(500),  -- Path to output file/directory
    output_file_size_bytes BIGINT,
    records_per_second DECIMAL(10, 2),
    
    -- Error Handling
    error_message TEXT,
    error_count INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Metadata
    created_by VARCHAR(100),
    job_metadata JSONB,
    
    -- Constraints
    CONSTRAINT chk_batch_job_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'paused')),
    CONSTRAINT chk_trigger_type CHECK (trigger_type IN ('manual', 'scheduled', 'event')),
    CONSTRAINT chk_input_source CHECK (input_source IN ('database', 'file', 'api')),
    CONSTRAINT chk_output_format CHECK (output_format IN ('database', 'file', 's3', 'parquet', 'csv'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_prediction_jobs(status);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_trigger ON batch_prediction_jobs(trigger_type);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_created ON batch_prediction_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_schedule ON batch_prediction_jobs(schedule_id);

-- Batch Prediction Schedules Table
CREATE TABLE IF NOT EXISTS batch_prediction_schedules (
    -- Primary Key
    schedule_id SERIAL PRIMARY KEY,
    schedule_name VARCHAR(100) NOT NULL UNIQUE,
    
    -- Schedule Configuration
    schedule_type VARCHAR(20) NOT NULL,  -- 'daily', 'weekly', 'monthly', 'cron'
    schedule_config JSONB NOT NULL,  -- Cron expression or schedule details
    
    -- Schedule Status
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMP WITH TIME ZONE,
    next_run_at TIMESTAMP WITH TIME ZONE,
    
    -- Job Configuration (template)
    job_config JSONB NOT NULL,  -- Template for creating batch jobs
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Metadata
    created_by VARCHAR(100),
    description TEXT,
    
    -- Constraints
    CONSTRAINT chk_batch_schedule_type CHECK (schedule_type IN ('daily', 'weekly', 'monthly', 'cron'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_batch_schedules_active ON batch_prediction_schedules(is_active, next_run_at);

-- Batch Prediction Results Table (for database output)
CREATE TABLE IF NOT EXISTS batch_prediction_results (
    -- Primary Key
    result_id SERIAL PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES batch_prediction_jobs(job_id) ON DELETE CASCADE,
    
    -- Customer Information
    customer_id VARCHAR(100) NOT NULL,
    
    -- Prediction Results
    prediction INTEGER NOT NULL,  -- 0 or 1
    probability DECIMAL(5, 4) NOT NULL,
    customer_score INTEGER,
    risk_level VARCHAR(10) NOT NULL,
    
    -- Features (optional, for reference)
    features JSONB,
    
    -- Model Information
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    
    -- Processing Metadata
    processing_time_ms DECIMAL(10, 2),
    row_number INTEGER,  -- Original row number in input
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_batch_prediction CHECK (prediction IN (0, 1)),
    CONSTRAINT chk_batch_probability CHECK (probability >= 0 AND probability <= 1),
    CONSTRAINT chk_batch_risk_level CHECK (risk_level IN ('low', 'medium', 'high'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_batch_results_job ON batch_prediction_results(job_id);
CREATE INDEX IF NOT EXISTS idx_batch_results_customer ON batch_prediction_results(customer_id);
CREATE INDEX IF NOT EXISTS idx_batch_results_created ON batch_prediction_results(created_at DESC);

-- Batch Prediction Job Logs Table
CREATE TABLE IF NOT EXISTS batch_prediction_logs (
    -- Primary Key
    log_id SERIAL PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES batch_prediction_jobs(job_id) ON DELETE CASCADE,
    
    -- Log Details
    log_level VARCHAR(20) NOT NULL,  -- 'INFO', 'WARNING', 'ERROR', 'DEBUG'
    message TEXT NOT NULL,
    error_details JSONB,
    
    -- Context
    record_index INTEGER,  -- Which record caused the log entry
    customer_id VARCHAR(100),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_batch_log_level CHECK (log_level IN ('INFO', 'WARNING', 'ERROR', 'DEBUG'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_batch_logs_job ON batch_prediction_logs(job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_batch_logs_level ON batch_prediction_logs(log_level);

-- Comments
COMMENT ON TABLE batch_prediction_jobs IS 'Tracks batch prediction jobs with progress and status';
COMMENT ON TABLE batch_prediction_schedules IS 'Stores scheduled batch prediction configurations';
COMMENT ON TABLE batch_prediction_results IS 'Stores batch prediction results when output format is database';
COMMENT ON TABLE batch_prediction_logs IS 'Stores detailed logs for batch prediction jobs';
