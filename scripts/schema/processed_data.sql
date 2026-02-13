-- Processed Data Tables Schema
-- Purpose: Store processed and feature-engineered data

-- RFM Metrics Table (Customer-level aggregations)
CREATE TABLE IF NOT EXISTS rfm_metrics (
    -- Primary Key
    customer_id VARCHAR(100) PRIMARY KEY,
    
    -- RFM Metrics
    recency INTEGER NOT NULL,  -- Days since last transaction
    frequency INTEGER NOT NULL,  -- Total transaction count
    monetary DECIMAL(15,2) NOT NULL,  -- Total transaction amount
    
    -- Normalized RFM (for model input)
    recency_normalized DECIMAL(10,6),
    frequency_normalized DECIMAL(10,6),
    monetary_normalized DECIMAL(10,6),
    
    -- Clustering Information
    cluster INTEGER,  -- K-Means cluster assignment (0, 1, 2)
    cluster_label VARCHAR(50),  -- 'High Risk', 'Medium Risk', 'Low Risk'
    
    -- Target Variable
    is_high_risk INTEGER,  -- Binary target: 0 (low risk) or 1 (high risk)
    
    -- Processing Metadata
    processed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    processing_version VARCHAR(50),  -- Version of processing pipeline
    data_version VARCHAR(50),  -- Version of source data used
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_rfm_metrics_cluster ON rfm_metrics(cluster);
CREATE INDEX IF NOT EXISTS idx_rfm_metrics_is_high_risk ON rfm_metrics(is_high_risk);
CREATE INDEX IF NOT EXISTS idx_rfm_metrics_processed_at ON rfm_metrics(processed_at);

-- Processed Features Table (26 engineered features per customer)
CREATE TABLE IF NOT EXISTS processed_features (
    -- Primary Key
    customer_id VARCHAR(100) PRIMARY KEY,
    
    -- Temporal Features
    transaction_hour DECIMAL(10,6),
    transaction_day DECIMAL(10,6),
    transaction_month DECIMAL(10,6),
    transaction_year DECIMAL(10,6),
    transaction_dayofweek DECIMAL(10,6),
    
    -- RFM Features (normalized)
    recency_normalized DECIMAL(10,6),
    frequency_normalized DECIMAL(10,6),
    monetary_normalized DECIMAL(10,6),
    
    -- Aggregate Features (stored as JSONB for flexibility)
    aggregate_features JSONB,  -- Transaction counts, amounts by category/channel
    
    -- Categorical Encodings
    categorical_features JSONB,  -- One-hot, WOE encodings
    
    -- Complete Feature Vector (26 features)
    feature_vector DECIMAL(10,6)[] NOT NULL,  -- Array of 26 features ready for model
    
    -- Processing Metadata
    processed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    feature_engineering_version VARCHAR(50),
    data_version VARCHAR(50),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Foreign Key to RFM Metrics
    CONSTRAINT fk_processed_features_customer FOREIGN KEY (customer_id) 
        REFERENCES rfm_metrics(customer_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_processed_features_processed_at ON processed_features(processed_at);
CREATE INDEX IF NOT EXISTS idx_processed_features_version ON processed_features(feature_engineering_version);

-- Data Splits Table (Train/Validation/Test splits)
CREATE TABLE IF NOT EXISTS data_splits (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- Customer Information
    customer_id VARCHAR(100) NOT NULL,
    
    -- Split Information
    split_type VARCHAR(20) NOT NULL,  -- 'train', 'validation', 'test'
    split_version VARCHAR(50),  -- Version of split (for reproducibility)
    
    -- Target Variable
    target_value INTEGER,  -- is_high_risk value
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Unique Constraint
    UNIQUE(customer_id, split_type, split_version)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_data_splits_customer_id ON data_splits(customer_id);
CREATE INDEX IF NOT EXISTS idx_data_splits_split_type ON data_splits(split_type);
CREATE INDEX IF NOT EXISTS idx_data_splits_version ON data_splits(split_version);

-- Comments
COMMENT ON TABLE rfm_metrics IS 'Customer-level RFM metrics and clustering results';
COMMENT ON TABLE processed_features IS 'Feature-engineered data ready for model training/inference';
COMMENT ON TABLE data_splits IS 'Train/validation/test data splits for model training';
COMMENT ON COLUMN processed_features.feature_vector IS 'Array of 26 normalized features ready for model input';
