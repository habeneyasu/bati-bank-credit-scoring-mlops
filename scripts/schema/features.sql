-- Feature Store Schema
-- Purpose: Store pre-computed features for fast online serving

CREATE TABLE IF NOT EXISTS customer_features (
    -- Primary Key
    customer_id VARCHAR(100) PRIMARY KEY,
    
    -- RFM Features
    recency_normalized DECIMAL(10,6),
    frequency_normalized DECIMAL(10,6),
    monetary_normalized DECIMAL(10,6),
    
    -- Temporal Features
    transaction_hour DECIMAL(10,6),
    transaction_day DECIMAL(10,6),
    transaction_month DECIMAL(10,6),
    transaction_year DECIMAL(10,6),
    transaction_dayofweek DECIMAL(10,6),
    
    -- Aggregate Features (stored as JSONB for flexibility)
    aggregate_features JSONB,  -- Transaction counts, amounts by category/channel
    
    -- Categorical Encodings
    categorical_features JSONB,  -- One-hot, WOE encodings
    
    -- All 26 Features (for direct model input)
    feature_vector DECIMAL(10,6)[] NOT NULL,  -- Array of 26 features
    
    -- Metadata
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    feature_version VARCHAR(50),  -- Version of feature engineering pipeline
    data_version VARCHAR(50),  -- Version of source data
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_customer_features_updated ON customer_features(last_updated);
CREATE INDEX IF NOT EXISTS idx_customer_features_version ON customer_features(feature_version);

-- Comments
COMMENT ON TABLE customer_features IS 'Feature store for pre-computed customer features (online serving)';
COMMENT ON COLUMN customer_features.feature_vector IS 'Array of 26 normalized features ready for model input';
COMMENT ON COLUMN customer_features.last_updated IS 'Timestamp of last feature update';
