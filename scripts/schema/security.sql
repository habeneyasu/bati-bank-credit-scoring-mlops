-- Security and Access Control Tables Schema
-- Purpose: Manage users, roles, and permissions for system access

-- Users Table
CREATE TABLE IF NOT EXISTS users (
    -- Primary Key
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    
    -- Authentication
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,  -- Hashed password (bcrypt/argon2)
    
    -- User Information
    full_name VARCHAR(255),
    department VARCHAR(100),
    position VARCHAR(100),
    
    -- Account Status
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,  -- Email verification
    is_superuser BOOLEAN DEFAULT FALSE,  -- Super admin flag
    
    -- Security
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,  -- Account lockout expiration
    password_changed_at TIMESTAMP WITH TIME ZONE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    
    -- API Access
    api_key VARCHAR(255) UNIQUE,  -- For API authentication
    api_key_created_at TIMESTAMP WITH TIME ZONE,
    api_key_expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    created_by VARCHAR(100),  -- User who created this account
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,  -- Soft delete
    
    -- Constraints
    CONSTRAINT chk_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- Roles Table
CREATE TABLE IF NOT EXISTS roles (
    -- Primary Key
    role_id SERIAL PRIMARY KEY,
    role_name VARCHAR(100) NOT NULL UNIQUE,
    role_code VARCHAR(50) NOT NULL UNIQUE,  -- For programmatic access (e.g., 'admin', 'data_analyst')
    
    -- Role Description
    description TEXT,
    
    -- Role Status
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- User Roles Mapping Table (Many-to-Many)
CREATE TABLE IF NOT EXISTS user_roles (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- Foreign Keys
    user_id INTEGER NOT NULL,
    role_id INTEGER NOT NULL,
    
    -- Assignment Metadata
    assigned_by VARCHAR(100),  -- User who assigned this role
    assigned_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,  -- Optional role expiration
    
    -- Constraints
    CONSTRAINT fk_user_roles_user FOREIGN KEY (user_id) 
        REFERENCES users(user_id) ON DELETE CASCADE,
    CONSTRAINT fk_user_roles_role FOREIGN KEY (role_id) 
        REFERENCES roles(role_id) ON DELETE CASCADE,
    CONSTRAINT uq_user_role UNIQUE (user_id, role_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_user_roles_user_id ON user_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_role_id ON user_roles(role_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_expires_at ON user_roles(expires_at);

-- Permissions Table (Granular permissions)
CREATE TABLE IF NOT EXISTS permissions (
    -- Primary Key
    permission_id SERIAL PRIMARY KEY,
    permission_name VARCHAR(100) NOT NULL UNIQUE,
    permission_code VARCHAR(50) NOT NULL UNIQUE,  -- For programmatic access
    
    -- Permission Details
    resource_type VARCHAR(50) NOT NULL,  -- 'data', 'model', 'prediction', 'dashboard', etc.
    action VARCHAR(50) NOT NULL,  -- 'read', 'write', 'delete', 'upload', 'view', etc.
    description TEXT,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Role Permissions Mapping Table (Many-to-Many)
CREATE TABLE IF NOT EXISTS role_permissions (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- Foreign Keys
    role_id INTEGER NOT NULL,
    permission_id INTEGER NOT NULL,
    
    -- Constraints
    CONSTRAINT fk_role_permissions_role FOREIGN KEY (role_id) 
        REFERENCES roles(role_id) ON DELETE CASCADE,
    CONSTRAINT fk_role_permissions_permission FOREIGN KEY (permission_id) 
        REFERENCES permissions(permission_id) ON DELETE CASCADE,
    CONSTRAINT uq_role_permission UNIQUE (role_id, permission_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_role_permissions_role_id ON role_permissions(role_id);
CREATE INDEX IF NOT EXISTS idx_role_permissions_permission_id ON role_permissions(permission_id);

-- Audit Log Table (Track user actions)
CREATE TABLE IF NOT EXISTS audit_logs (
    -- Primary Key
    log_id SERIAL PRIMARY KEY,
    
    -- User Information
    user_id INTEGER,
    username VARCHAR(100),
    
    -- Action Details
    action VARCHAR(100) NOT NULL,  -- 'login', 'upload_data', 'view_prediction', etc.
    resource_type VARCHAR(50),  -- 'data', 'model', 'prediction', etc.
    resource_id VARCHAR(100),  -- ID of the resource accessed
    
    -- Request Details
    ip_address INET,
    user_agent TEXT,
    request_method VARCHAR(10),  -- 'GET', 'POST', 'PUT', 'DELETE'
    request_path TEXT,
    
    -- Result
    status_code INTEGER,
    success BOOLEAN,
    error_message TEXT,
    
    -- Metadata
    metadata JSONB,  -- Additional context
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for Audit Logs
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);

-- Comments
COMMENT ON TABLE users IS 'System users with authentication and account information';
COMMENT ON TABLE roles IS 'User roles for access control (e.g., admin, data_analyst, viewer)';
COMMENT ON TABLE user_roles IS 'Many-to-many mapping between users and roles';
COMMENT ON TABLE permissions IS 'Granular permissions for resources and actions';
COMMENT ON TABLE role_permissions IS 'Many-to-many mapping between roles and permissions';
COMMENT ON TABLE audit_logs IS 'Audit trail of user actions for security and compliance';
