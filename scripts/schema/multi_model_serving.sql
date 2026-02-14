-- Multi-Model Serving Schema
-- Purpose: Support multiple models simultaneously with routing and ensemble serving

-- Model Routing Rules Table
CREATE TABLE IF NOT EXISTS model_routing_rules (
    -- Primary Key
    rule_id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL UNIQUE,
    
    -- Rule Configuration
    priority INTEGER NOT NULL DEFAULT 0,  -- Higher priority = evaluated first
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Routing Criteria
    routing_criteria JSONB NOT NULL,  -- Conditions for routing (e.g., customer segment, amount range)
    routing_type VARCHAR(50) NOT NULL,  -- 'single', 'ensemble', 'weighted_ensemble'
    
    -- Target Models
    target_models JSONB NOT NULL,  -- List of model names/versions to use
    model_weights JSONB,  -- Weights for ensemble (if routing_type is ensemble)
    
    -- Fallback Configuration
    fallback_model_name VARCHAR(100),
    fallback_model_stage VARCHAR(20) DEFAULT 'Production',
    
    -- Metadata
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    
    -- Constraints
    CONSTRAINT chk_routing_type CHECK (routing_type IN ('single', 'ensemble', 'weighted_ensemble', 'comparison'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_routing_rules_active ON model_routing_rules(is_active, priority DESC);
CREATE INDEX IF NOT EXISTS idx_routing_rules_type ON model_routing_rules(routing_type);

-- Model Registry (Extended) Table
-- Tracks all available models for multi-model serving
CREATE TABLE IF NOT EXISTS model_registry (
    -- Primary Key
    registry_id SERIAL PRIMARY KEY,
    
    -- Model Information
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_stage VARCHAR(20) NOT NULL,  -- 'Production', 'Staging', 'Archived'
    
    -- Model Metadata
    model_type VARCHAR(50),  -- 'logistic_regression', 'random_forest', etc.
    mlflow_run_id VARCHAR(50),
    mlflow_model_uri VARCHAR(500),
    
    -- Performance Metrics
    accuracy DECIMAL(5, 4),
    roc_auc DECIMAL(5, 4),
    precision DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    
    -- Serving Configuration
    is_loaded BOOLEAN DEFAULT FALSE,
    load_priority INTEGER DEFAULT 0,  -- Priority for loading (higher = load first)
    max_concurrent_requests INTEGER DEFAULT 100,
    
    -- Resource Requirements
    memory_usage_mb INTEGER,
    cpu_usage_percent DECIMAL(5, 2),
    
    -- Status
    status VARCHAR(20) DEFAULT 'available',  -- 'available', 'loading', 'unavailable', 'error'
    error_message TEXT,
    
    -- Timestamps
    registered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB,
    
    -- Constraints
    CONSTRAINT chk_model_stage CHECK (model_stage IN ('Production', 'Staging', 'Archived')),
    CONSTRAINT chk_model_status CHECK (status IN ('available', 'loading', 'unavailable', 'error')),
    UNIQUE(model_name, model_version)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_model_registry_name ON model_registry(model_name);
CREATE INDEX IF NOT EXISTS idx_model_registry_stage ON model_registry(model_stage);
CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry(status, is_loaded);
CREATE INDEX IF NOT EXISTS idx_model_registry_priority ON model_registry(load_priority DESC);

-- Model Comparison Results Table
CREATE TABLE IF NOT EXISTS model_comparison_results (
    -- Primary Key
    comparison_id SERIAL PRIMARY KEY,
    
    -- Comparison Configuration
    comparison_name VARCHAR(100),
    comparison_type VARCHAR(50) NOT NULL,  -- 'real_time', 'batch', 'historical'
    
    -- Models Compared
    model_1_name VARCHAR(100) NOT NULL,
    model_1_version VARCHAR(50) NOT NULL,
    model_2_name VARCHAR(100) NOT NULL,
    model_2_version VARCHAR(50) NOT NULL,
    
    -- Comparison Metrics
    comparison_metrics JSONB NOT NULL,  -- Side-by-side metrics comparison
    differences JSONB,  -- Key differences between models
    winner VARCHAR(100),  -- Which model performed better
    
    -- Test Data
    test_samples INTEGER,
    test_customer_ids TEXT[],  -- Customer IDs used for comparison
    
    -- Timestamps
    compared_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    
    -- Constraints
    CONSTRAINT chk_comparison_type CHECK (comparison_type IN ('real_time', 'batch', 'historical'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_comparison_models ON model_comparison_results(model_1_name, model_2_name);
CREATE INDEX IF NOT EXISTS idx_comparison_type ON model_comparison_results(comparison_type);
CREATE INDEX IF NOT EXISTS idx_comparison_date ON model_comparison_results(compared_at DESC);

-- Model Ensemble Configurations Table
CREATE TABLE IF NOT EXISTS model_ensembles (
    -- Primary Key
    ensemble_id SERIAL PRIMARY KEY,
    ensemble_name VARCHAR(100) NOT NULL UNIQUE,
    
    -- Ensemble Configuration
    ensemble_type VARCHAR(50) NOT NULL,  -- 'voting', 'weighted_average', 'stacking'
    model_configs JSONB NOT NULL,  -- List of models with weights/configurations
    
    -- Ensemble Metadata
    is_active BOOLEAN DEFAULT TRUE,
    description TEXT,
    
    -- Performance
    ensemble_metrics JSONB,  -- Performance metrics for the ensemble
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    
    -- Constraints
    CONSTRAINT chk_ensemble_type CHECK (ensemble_type IN ('voting', 'weighted_average', 'stacking'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ensembles_active ON model_ensembles(is_active);

-- Comments
COMMENT ON TABLE model_routing_rules IS 'Defines routing rules for selecting models based on criteria';
COMMENT ON TABLE model_registry IS 'Extended registry tracking all available models for multi-model serving';
COMMENT ON TABLE model_comparison_results IS 'Stores results of real-time model comparisons';
COMMENT ON TABLE model_ensembles IS 'Defines model ensemble configurations';
