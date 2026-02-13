"""
Database trigger functions for automatic field population.

These functions are executed automatically by PostgreSQL triggers.
"""

# This SQL will be executed to create the trigger function
TRIGGER_FUNCTION_SQL = """
-- Trigger function to automatically set created_at_date from created_at
CREATE OR REPLACE FUNCTION set_created_at_date()
RETURNS TRIGGER AS $$
BEGIN
    -- Automatically set created_at_date from created_at timestamp
    NEW.created_at_date := DATE(NEW.created_at);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for predictions table
DROP TRIGGER IF EXISTS trigger_set_created_at_date ON predictions;

CREATE TRIGGER trigger_set_created_at_date
    BEFORE INSERT ON predictions
    FOR EACH ROW
    EXECUTE FUNCTION set_created_at_date();

-- Comment
COMMENT ON FUNCTION set_created_at_date() IS 'Automatically sets created_at_date from created_at timestamp on insert';
"""
