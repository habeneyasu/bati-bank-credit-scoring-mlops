-- Predictions Table Schema
-- Purpose: Store all predictions for audit trail, compliance, and analytics

CREATE TABLE IF NOT EXISTS predictions (
    -- Primary Key
    prediction_id VARCHAR(50) PRIMARY KEY,
    
    -- Customer Information
    customer_id VARCHAR(100),
    customer_id_indexed VARCHAR(100),  -- For faster lookups
    
    -- Prediction Details
    prediction INTEGER NOT NULL,  -- 0 or 1
    probability DECIMAL(5,4) NOT NULL,  -- 0.0000 to 1.0000
    customer_score INTEGER,  -- Credit score (0-1000 scale, higher = lower risk)
    risk_level VARCHAR(10) NOT NULL,  -- 'low', 'medium', 'high'
    
    -- Features (stored as JSONB for flexibility)
    features JSONB NOT NULL,  -- Array of 26 feature values
    
    -- Model Information
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_stage VARCHAR(20) NOT NULL,  -- 'Production', 'Staging', etc.
    
    -- Performance Metrics
    latency_ms DECIMAL(10,2),  -- Prediction latency in milliseconds
    request_size_bytes INTEGER,  -- Size of request
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at_date DATE GENERATED ALWAYS AS (created_at::DATE) STORED,
    
    -- Metadata
    request_metadata JSONB,  -- Additional request context
    response_metadata JSONB,  -- Additional response context
    
    -- Constraints
    CONSTRAINT chk_prediction CHECK (prediction IN (0, 1)),
    CONSTRAINT chk_probability CHECK (probability >= 0 AND probability <= 1),
    CONSTRAINT chk_customer_score CHECK (customer_score IS NULL OR (customer_score >= 0 AND customer_score <= 1000)),
    CONSTRAINT chk_risk_level CHECK (risk_level IN ('low', 'medium', 'high'))
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_predictions_customer_id ON predictions(customer_id_indexed);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at_date ON predictions(created_at_date);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON predictions(risk_level);
CREATE INDEX IF NOT EXISTS idx_predictions_customer_score ON predictions(customer_score);
CREATE INDEX IF NOT EXISTS idx_predictions_customer_date ON predictions(customer_id_indexed, created_at_date);

-- Comments
COMMENT ON TABLE predictions IS 'Stores all predictions for audit trail, compliance (GDPR/CCPA), and analytics';
COMMENT ON COLUMN predictions.prediction_id IS 'Unique identifier for each prediction (e.g., pred_abc123xyz)';
COMMENT ON COLUMN predictions.customer_id IS 'Customer identifier for tracking and analytics';
COMMENT ON COLUMN predictions.probability IS 'Probability of high-risk (0.0 to 1.0)';
COMMENT ON COLUMN predictions.customer_score IS 'Credit score on 0-1000 scale (higher = lower risk). Calculated as: (1 - probability) * 1000';
COMMENT ON COLUMN predictions.risk_level IS 'Human-readable risk level: low, medium, or high';
COMMENT ON COLUMN predictions.features IS 'JSONB array of 26 feature values used for prediction';
COMMENT ON COLUMN predictions.model_version IS 'Model version used for this prediction';
