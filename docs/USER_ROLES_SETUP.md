# User Roles and Permissions Setup

## Overview

This document describes the users that have been created with different roles and permissions for the MLOps Credit Scoring system. These users demonstrate role-based access control (RBAC) and can be used for testing and demonstration purposes.

## Created Users

### 1. Data Administrator
- **Username:** `data_admin`
- **Email:** `data.admin@batibank.com`
- **Password:** `DataAdmin@2024`
- **Full Name:** Sarah Johnson
- **Department:** Data Management
- **Position:** Data Administrator
- **Role:** Data Administrator (`data_admin`)

**Permissions:**
- Upload raw data
- Read raw data
- Delete raw data
- Process data
- View data metrics
- View predictions
- View customer predictions
- Export predictions
- View dashboard
- View business KPIs
- View monitoring dashboard

**Use Case:** This user can manage all data-related operations, upload new transaction data, and view all predictions and metrics.

---

### 2. Data Analyst
- **Username:** `data_analyst`
- **Email:** `analyst@batibank.com`
- **Password:** `Analyst@2024`
- **Full Name:** Michael Chen
- **Department:** Analytics
- **Position:** Senior Data Analyst
- **Role:** Data Analyst (`data_analyst`)

**Permissions:**
- Read raw data (view only)
- View data metrics
- View predictions
- View customer predictions
- Export predictions
- View model performance
- View model registry
- View dashboard
- View business KPIs
- View monitoring dashboard

**Use Case:** This user can analyze data, view predictions, and monitor model performance but cannot modify data or train new models.

---

### 3. Business User
- **Username:** `business_user`
- **Email:** `business.user@batibank.com`
- **Password:** `Business@2024`
- **Full Name:** Emily Rodriguez
- **Department:** Credit Operations
- **Position:** Credit Manager
- **Role:** Business User (`business_user`)

**Permissions:**
- View predictions
- View customer predictions
- View dashboard
- View business KPIs

**Use Case:** This user focuses on business operations and can view predictions and KPIs to make credit decisions but has limited access to technical details.

---

### 4. Model Developer
- **Username:** `model_dev`
- **Email:** `model.dev@batibank.com`
- **Password:** `ModelDev@2024`
- **Full Name:** David Kim
- **Department:** ML Engineering
- **Position:** ML Engineer
- **Role:** Model Developer (`model_developer`)

**Permissions:**
- Read raw data (view only)
- View data metrics
- View model performance
- Train models
- Deploy models
- View model registry
- Manage model registry
- View dashboard
- View monitoring dashboard

**Use Case:** This user can train new models, deploy them to production, and manage the model registry but has limited access to business operations.

---

## Role Hierarchy

The system includes the following roles (in order of permissions):

1. **Super Administrator** - Full system access (all permissions)
2. **Data Administrator** - Data management and predictions
3. **Model Developer** - Model training and deployment
4. **Data Analyst** - Read-only access to data and predictions
5. **Business User** - Business-focused predictions and KPIs
6. **Viewer** - Read-only dashboard access

## Access Control

The system implements role-based access control (RBAC) where:

1. **Users** are assigned one or more **Roles**
2. **Roles** are assigned multiple **Permissions**
3. **Permissions** control access to specific resources and actions

### Permission Format

Permissions follow the format: `resource:action`

Examples:
- `data:upload` - Upload data
- `prediction:read` - View predictions
- `model:train` - Train models
- `dashboard:read` - View dashboard

## Security Notes

⚠️ **IMPORTANT:** 
- These are demonstration passwords and should be changed in production
- Passwords are hashed using bcrypt with 12 rounds
- Users can be deactivated without deletion (soft delete)
- All user actions are logged in the audit_logs table

## Viewing Users in Dashboard

Users can be viewed in the dashboard by:

1. Navigate to the **Users** section in the sidebar
2. View all users with their roles and permissions
3. Search and filter users by status, department, etc.

## Viewing Roles in Dashboard

Roles can be viewed in the dashboard by:

1. Navigate to the **Roles & Permissions** section in the sidebar
2. View all roles with their assigned permissions
3. See which users are assigned to each role

## API Endpoints

The following API endpoints are available for user and role management:

- `GET /api/users` - Get all users
- `GET /api/users/{user_id}` - Get user by ID
- `GET /api/roles` - Get all roles
- `GET /api/roles/{role_id}` - Get role by ID
- `GET /api/permissions` - Get all permissions

## Re-running the Seed Script

To re-run the user seeding script:

```bash
cd /home/haben/Project/KAIM-Training-Portfolio/bati-bank-credit-scoring-mlops
source venv/bin/activate
python scripts/seed_users.py
```

The script will skip users that already exist, so it's safe to run multiple times.

## Next Steps

1. **Implement Authentication:** Add login functionality to the frontend
2. **Add Authorization Middleware:** Check user permissions before allowing API access
3. **Implement Session Management:** Use JWT tokens or session cookies
4. **Add User Management UI:** Allow admins to create/edit users from the dashboard
5. **Add Role Management UI:** Allow admins to assign roles to users

## Database Schema

Users are stored in the following tables:

- `users` - User accounts
- `roles` - Role definitions
- `permissions` - Permission definitions
- `user_roles` - User-Role mappings
- `role_permissions` - Role-Permission mappings
- `audit_logs` - User action logs

For more details, see the database schema files in `scripts/schema/`.
