-- Seed Data for Roles and Permissions
-- Purpose: Initialize default roles and permissions for the system

-- Insert Default Roles
INSERT INTO roles (role_name, role_code, description, is_active) VALUES
    ('Super Administrator', 'super_admin', 'Full system access with all permissions', TRUE),
    ('Data Administrator', 'data_admin', 'Can upload raw data, manage processed data, and view all metrics', TRUE),
    ('Data Analyst', 'data_analyst', 'Can view data metrics, predictions, and model performance', TRUE),
    ('Model Developer', 'model_developer', 'Can train models, view model performance, and manage model registry', TRUE),
    ('Business User', 'business_user', 'Can view predictions, business KPIs, and dashboards', TRUE),
    ('Viewer', 'viewer', 'Read-only access to dashboards and reports', TRUE)
ON CONFLICT (role_code) DO NOTHING;

-- Insert Permissions
INSERT INTO permissions (permission_name, permission_code, resource_type, action, description) VALUES
    -- Data Permissions
    ('Upload Raw Data', 'data:upload', 'data', 'upload', 'Upload raw transaction data via dashboard'),
    ('View Raw Data', 'data:read', 'data', 'read', 'View raw transaction data'),
    ('Delete Raw Data', 'data:delete', 'data', 'delete', 'Delete raw transaction data'),
    ('Process Data', 'data:process', 'data', 'write', 'Trigger data processing pipeline'),
    ('View Data Metrics', 'data:metrics', 'data', 'read', 'View data quality metrics and statistics'),
    
    -- Prediction Permissions
    ('View Predictions', 'prediction:read', 'prediction', 'read', 'View prediction results'),
    ('View Customer Predictions', 'prediction:customer', 'prediction', 'read', 'View predictions for specific customers'),
    ('Export Predictions', 'prediction:export', 'prediction', 'read', 'Export prediction data'),
    
    -- Model Permissions
    ('View Model Performance', 'model:performance', 'model', 'read', 'View model performance metrics'),
    ('Train Models', 'model:train', 'model', 'write', 'Train new models'),
    ('Deploy Models', 'model:deploy', 'model', 'write', 'Deploy models to production'),
    ('View Model Registry', 'model:registry', 'model', 'read', 'View model registry and versions'),
    ('Manage Model Registry', 'model:manage', 'model', 'write', 'Manage model registry and versions'),
    ('Manage Model Experiments', 'model:write', 'model', 'write', 'Create and manage A/B testing experiments'),
    
    -- Dashboard Permissions
    ('View Dashboard', 'dashboard:read', 'dashboard', 'read', 'View main dashboard'),
    ('View Business KPIs', 'dashboard:kpis', 'dashboard', 'read', 'View business KPI dashboard'),
    ('View Monitoring Dashboard', 'dashboard:monitoring', 'dashboard', 'read', 'View monitoring and performance dashboard'),
    
    -- User Management Permissions
    ('View Users', 'user:read', 'user', 'read', 'View user list and details'),
    ('Create Users', 'user:create', 'user', 'write', 'Create new users'),
    ('Update Users', 'user:update', 'user', 'write', 'Update user information'),
    ('Delete Users', 'user:delete', 'user', 'delete', 'Delete users'),
    ('Manage Roles', 'role:manage', 'role', 'write', 'Manage roles and permissions'),
    
    -- System Permissions
    ('View System Logs', 'system:logs', 'system', 'read', 'View system logs'),
    ('Manage System Settings', 'system:settings', 'system', 'write', 'Manage system configuration')
ON CONFLICT (permission_code) DO NOTHING;

-- Assign Permissions to Roles
-- Super Admin: All permissions
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.role_id, p.permission_id
FROM roles r, permissions p
WHERE r.role_code = 'super_admin'
ON CONFLICT DO NOTHING;

-- Data Admin: Data and prediction permissions + model:write
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.role_id, p.permission_id
FROM roles r, permissions p
WHERE r.role_code = 'data_admin'
  AND p.permission_code IN (
    'data:upload', 'data:read', 'data:delete', 'data:process', 'data:metrics',
    'prediction:read', 'prediction:customer', 'prediction:export',
    'model:write',
    'dashboard:read', 'dashboard:kpis', 'dashboard:monitoring'
  )
ON CONFLICT DO NOTHING;

-- Data Analyst: Read-only data and prediction permissions + model:write
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.role_id, p.permission_id
FROM roles r, permissions p
WHERE r.role_code = 'data_analyst'
  AND p.permission_code IN (
    'data:read', 'data:metrics',
    'prediction:read', 'prediction:customer', 'prediction:export',
    'model:performance', 'model:registry', 'model:write',
    'dashboard:read', 'dashboard:kpis', 'dashboard:monitoring'
  )
ON CONFLICT DO NOTHING;

-- Model Developer: Model and data read permissions
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.role_id, p.permission_id
FROM roles r, permissions p
WHERE r.role_code = 'model_developer'
  AND p.permission_code IN (
    'data:read', 'data:metrics',
    'model:performance', 'model:train', 'model:deploy', 'model:registry', 'model:manage', 'model:write',
    'dashboard:read', 'dashboard:monitoring'
  )
ON CONFLICT DO NOTHING;

-- Business User: Business-focused permissions + model:write
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.role_id, p.permission_id
FROM roles r, permissions p
WHERE r.role_code = 'business_user'
  AND p.permission_code IN (
    'prediction:read', 'prediction:customer',
    'model:write',
    'dashboard:read', 'dashboard:kpis'
  )
ON CONFLICT DO NOTHING;

-- Viewer: Read-only dashboard access + model:write
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.role_id, p.permission_id
FROM roles r, permissions p
WHERE r.role_code = 'viewer'
  AND p.permission_code IN (
    'dashboard:read', 'dashboard:kpis',
    'model:write'
  )
ON CONFLICT DO NOTHING;

-- Create default admin user (password should be changed on first login)
-- Password hash for 'admin123' (bcrypt, rounds=12)
-- In production, use a secure password hashing library
INSERT INTO users (username, email, password_hash, full_name, is_active, is_verified, is_superuser)
VALUES (
    'admin',
    'admin@batibank.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyY5Y5Y5Y5Y5',  -- Change this!
    'System Administrator',
    TRUE,
    TRUE,
    TRUE
)
ON CONFLICT (username) DO NOTHING;

-- Assign super_admin role to default admin user
INSERT INTO user_roles (user_id, role_id, assigned_by)
SELECT u.user_id, r.role_id, 'system'
FROM users u, roles r
WHERE u.username = 'admin' AND r.role_code = 'super_admin'
ON CONFLICT DO NOTHING;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Roles and permissions seeded successfully!';
    RAISE NOTICE 'Default admin user created: username=admin, password=admin123 (CHANGE THIS!)';
END $$;
