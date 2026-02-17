-- Grant model:write permission to all roles
-- This script adds the model:write permission to all existing roles
-- so that all users can create A/B experiments and retraining jobs

-- Ensure the permission exists
INSERT INTO permissions (permission_name, permission_code, resource_type, action, description)
VALUES (
    'Manage Model Experiments',
    'model:write',
    'model',
    'write',
    'Create and manage A/B testing experiments and retraining jobs'
)
ON CONFLICT (permission_code) DO NOTHING;

-- Grant model:write permission to all roles
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.role_id, p.permission_id
FROM roles r, permissions p
WHERE p.permission_code = 'model:write'
  AND r.is_active = TRUE
  AND NOT EXISTS (
    SELECT 1 
    FROM role_permissions rp 
    WHERE rp.role_id = r.role_id 
      AND rp.permission_id = p.permission_id
  )
ON CONFLICT DO NOTHING;

-- Success message
DO $$
DECLARE
    roles_updated INTEGER;
BEGIN
    SELECT COUNT(*) INTO roles_updated
    FROM role_permissions rp
    JOIN permissions p ON rp.permission_id = p.permission_id
    WHERE p.permission_code = 'model:write';
    
    RAISE NOTICE 'model:write permission granted to all active roles!';
    RAISE NOTICE 'Total roles with model:write permission: %', roles_updated;
END $$;
