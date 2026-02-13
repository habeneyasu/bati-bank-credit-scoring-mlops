-- Database Initialization Script
-- Purpose: Create database, schemas, and all tables
-- 
-- IMPORTANT: This script should be run from the project root directory
-- Usage: psql -U postgres -d mlops_db -f scripts/init_db.sql
--
-- Or if database doesn't exist:
--   1. psql -U postgres -c "CREATE DATABASE mlops_db;"
--   2. psql -U postgres -d mlops_db -f scripts/init_db.sql

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS predictions;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS metadata;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO public, predictions, features, metadata, monitoring;

-- Note: The \i commands below require running from project root directory
-- If running from a different location, use absolute paths or adjust accordingly

-- Create tables
\i scripts/schema/security.sql
\i scripts/schema/raw_data.sql
\i scripts/schema/processed_data.sql
\i scripts/schema/predictions.sql
\i scripts/schema/features.sql
\i scripts/schema/metadata.sql
\i scripts/schema/monitoring.sql

-- Create views and functions
\i scripts/schema/access_control_views.sql
\i scripts/schema/access_control_functions.sql

-- Seed initial data
\i scripts/schema/seed_roles_permissions.sql

-- Create read-only user for analytics
-- CREATE USER analytics_user WITH PASSWORD 'analytics_password';
-- GRANT CONNECT ON DATABASE credit_scoring TO analytics_user;
-- GRANT USAGE ON SCHEMA predictions, metadata TO analytics_user;
-- GRANT SELECT ON ALL TABLES IN SCHEMA predictions, metadata TO analytics_user;

-- Create application user
-- CREATE USER mlops_user WITH PASSWORD 'mlops_password';
-- GRANT ALL PRIVILEGES ON DATABASE credit_scoring TO mlops_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA predictions, features, metadata, monitoring TO mlops_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Database initialization complete!';
    RAISE NOTICE 'Schemas created: predictions, features, metadata, monitoring';
END $$;
