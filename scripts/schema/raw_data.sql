-- Raw Data Table Schema
-- Purpose: Store original transaction data from data sources

CREATE TABLE IF NOT EXISTS raw_transactions (
    -- Primary Key
    transaction_id VARCHAR(100) PRIMARY KEY,
    
    -- Batch Information
    batch_id VARCHAR(100),
    
    -- Account Information
    account_id VARCHAR(100),
    subscription_id VARCHAR(100),
    customer_id VARCHAR(100) NOT NULL,  -- Customer identifier
    
    -- Transaction Details
    currency_code VARCHAR(10),
    country_code VARCHAR(10),
    provider_id VARCHAR(100),
    product_id VARCHAR(100),
    product_category VARCHAR(100),
    channel_id VARCHAR(100),
    
    -- Financial Details
    amount DECIMAL(15,2) NOT NULL,
    value DECIMAL(15,2),
    
    -- Transaction Metadata
    transaction_start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    pricing_strategy INTEGER,
    fraud_result INTEGER DEFAULT 0,
    
    -- Data Quality Flags
    is_valid BOOLEAN DEFAULT TRUE,
    validation_errors JSONB,  -- Store validation error details
    
    -- Upload Information
    uploaded_by VARCHAR(100),  -- User who uploaded this data
    uploaded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    data_source VARCHAR(100),  -- Source system/file name
    file_name VARCHAR(255),  -- Original file name
    
    -- Data Versioning
    data_version VARCHAR(50),  -- Version of data schema/format
    checksum_sha256 VARCHAR(64),  -- For data integrity
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_raw_transactions_customer_id ON raw_transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_raw_transactions_transaction_start_time ON raw_transactions(transaction_start_time);
CREATE INDEX IF NOT EXISTS idx_raw_transactions_uploaded_at ON raw_transactions(uploaded_at);
CREATE INDEX IF NOT EXISTS idx_raw_transactions_batch_id ON raw_transactions(batch_id);
CREATE INDEX IF NOT EXISTS idx_raw_transactions_product_category ON raw_transactions(product_category);
CREATE INDEX IF NOT EXISTS idx_raw_transactions_channel_id ON raw_transactions(channel_id);
CREATE INDEX IF NOT EXISTS idx_raw_transactions_customer_time ON raw_transactions(customer_id, transaction_start_time);

-- Partitioning by date (for large-scale deployments)
-- Example: CREATE TABLE raw_transactions_2026_02 PARTITION OF raw_transactions
--     FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

-- Comments
COMMENT ON TABLE raw_transactions IS 'Stores original raw transaction data from data sources';
COMMENT ON COLUMN raw_transactions.transaction_id IS 'Unique transaction identifier';
COMMENT ON COLUMN raw_transactions.customer_id IS 'Customer identifier for linking to predictions and features';
COMMENT ON COLUMN raw_transactions.uploaded_by IS 'User who uploaded this transaction data';
COMMENT ON COLUMN raw_transactions.data_version IS 'Version of data schema/format for tracking changes';
