-- Access Control Helper Functions
-- Purpose: Functions to check user permissions

-- Function: Check if user has permission
CREATE OR REPLACE FUNCTION user_has_permission(
    p_username VARCHAR,
    p_permission_code VARCHAR
) RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM user_permissions
        WHERE username = p_username
          AND permission_code = p_permission_code
          AND user_active = TRUE
          AND role_active = TRUE
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function: Check if user has any of the specified permissions
CREATE OR REPLACE FUNCTION user_has_any_permission(
    p_username VARCHAR,
    p_permission_codes VARCHAR[]
) RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM user_permissions
        WHERE username = p_username
          AND permission_code = ANY(p_permission_codes)
          AND user_active = TRUE
          AND role_active = TRUE
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function: Check if user has role
CREATE OR REPLACE FUNCTION user_has_role(
    p_username VARCHAR,
    p_role_code VARCHAR
) RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM users u
        INNER JOIN user_roles ur ON u.user_id = ur.user_id
        INNER JOIN roles r ON ur.role_id = r.role_id
        WHERE u.username = p_username
          AND r.role_code = p_role_code
          AND u.is_active = TRUE
          AND r.is_active = TRUE
          AND (ur.expires_at IS NULL OR ur.expires_at > NOW())
          AND (u.deleted_at IS NULL)
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function: Get user permissions
CREATE OR REPLACE FUNCTION get_user_permissions(
    p_username VARCHAR
) RETURNS TABLE (
    permission_code VARCHAR,
    resource_type VARCHAR,
    action VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT
        up.permission_code,
        up.resource_type,
        up.action
    FROM user_permissions up
    WHERE up.username = p_username
    ORDER BY up.resource_type, up.action;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function: Get user roles
CREATE OR REPLACE FUNCTION get_user_roles(
    p_username VARCHAR
) RETURNS TABLE (
    role_code VARCHAR,
    role_name VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT
        r.role_code,
        r.role_name
    FROM users u
    INNER JOIN user_roles ur ON u.user_id = ur.user_id
    INNER JOIN roles r ON ur.role_id = r.role_id
    WHERE u.username = p_username
      AND u.is_active = TRUE
      AND r.is_active = TRUE
      AND (ur.expires_at IS NULL OR ur.expires_at > NOW())
      AND (u.deleted_at IS NULL)
    ORDER BY r.role_name;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Comments
COMMENT ON FUNCTION user_has_permission IS 'Check if a user has a specific permission';
COMMENT ON FUNCTION user_has_any_permission IS 'Check if a user has any of the specified permissions';
COMMENT ON FUNCTION user_has_role IS 'Check if a user has a specific role';
COMMENT ON FUNCTION get_user_permissions IS 'Get all permissions for a user';
COMMENT ON FUNCTION get_user_roles IS 'Get all roles for a user';
