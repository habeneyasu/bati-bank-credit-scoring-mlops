-- Access Control Helper Views
-- Purpose: Simplify permission checking queries

-- View: User Permissions
-- Shows all permissions for each user (through their roles)
CREATE OR REPLACE VIEW user_permissions AS
SELECT DISTINCT
    u.user_id,
    u.username,
    u.email,
    u.is_active AS user_active,
    r.role_id,
    r.role_name,
    r.role_code,
    r.is_active AS role_active,
    p.permission_id,
    p.permission_name,
    p.permission_code,
    p.resource_type,
    p.action
FROM users u
INNER JOIN user_roles ur ON u.user_id = ur.user_id
INNER JOIN roles r ON ur.role_id = r.role_id
INNER JOIN role_permissions rp ON r.role_id = rp.role_id
INNER JOIN permissions p ON rp.permission_id = p.permission_id
WHERE u.is_active = TRUE
  AND r.is_active = TRUE
  AND (ur.expires_at IS NULL OR ur.expires_at > NOW())
  AND (u.deleted_at IS NULL);

-- View: User Roles Summary
-- Shows all roles assigned to each user
CREATE OR REPLACE VIEW user_roles_summary AS
SELECT
    u.user_id,
    u.username,
    u.email,
    u.is_active,
    STRING_AGG(r.role_name, ', ' ORDER BY r.role_name) AS roles,
    STRING_AGG(r.role_code, ', ' ORDER BY r.role_code) AS role_codes
FROM users u
LEFT JOIN user_roles ur ON u.user_id = ur.user_id
LEFT JOIN roles r ON ur.role_id = r.role_id
WHERE u.deleted_at IS NULL
GROUP BY u.user_id, u.username, u.email, u.is_active;

-- View: Role Permissions Summary
-- Shows all permissions for each role
CREATE OR REPLACE VIEW role_permissions_summary AS
SELECT
    r.role_id,
    r.role_name,
    r.role_code,
    r.is_active,
    STRING_AGG(p.permission_name, ', ' ORDER BY p.permission_name) AS permissions,
    STRING_AGG(p.permission_code, ', ' ORDER BY p.permission_code) AS permission_codes
FROM roles r
LEFT JOIN role_permissions rp ON r.role_id = rp.role_id
LEFT JOIN permissions p ON rp.permission_id = p.permission_id
GROUP BY r.role_id, r.role_name, r.role_code, r.is_active;

-- Comments
COMMENT ON VIEW user_permissions IS 'Shows all permissions for each active user through their roles';
COMMENT ON VIEW user_roles_summary IS 'Summary of roles assigned to each user';
COMMENT ON VIEW role_permissions_summary IS 'Summary of permissions assigned to each role';
