-- Add model:write permission for A/B Testing
-- This script adds the missing model:write permission and assigns it to model_developer role

-- Insert the permission if it doesn't exist
INSERT INTO permissions (permission_name, permission_code, resource_type, action, description)
VALUES (
    'Manage Model Experiments',
    'model:write',
    'model',
    'write',
    'Create and manage A/B testing experiments'
)
ON CONFLICT (permission_code) DO NOTHING;

-- Assign model:write permission to model_developer role
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.role_id, p.permission_id
FROM roles r, permissions p
WHERE r.role_code = 'model_developer'
  AND p.permission_code = 'model:write'
ON CONFLICT DO NOTHING;

-- Note: Super admin already has all permissions, so no need to explicitly assign it

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'model:write permission added successfully!';
    RAISE NOTICE 'Permission assigned to model_developer role';
END $$;
