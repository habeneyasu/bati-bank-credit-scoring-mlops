#!/usr/bin/env python3
"""
Seed Users Script
Purpose: Add three users with different roles and permissions for testing and demonstration

Usage:
    python scripts/seed_users.py

This script creates:
1. Data Administrator - Can upload data, manage data, view predictions
2. Data Analyst - Can view data, predictions, and model performance
3. Business User - Can view predictions and business KPIs
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_db_session
from src.database.repositories import UserRepository, RoleRepository
from src.database.models import User, Role, UserRole
from sqlalchemy.exc import IntegrityError
import bcrypt
from datetime import datetime, timezone

# Password hashing function
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def create_user(
    session,
    username: str,
    email: str,
    password: str,
    full_name: str,
    department: str,
    position: str,
    role_code: str,
    is_verified: bool = True
):
    """Create a user and assign a role."""
    try:
        # Check if user already exists
        user_repo = UserRepository(session)
        existing_user = user_repo.get_by_username(username)
        if existing_user:
            print(f"⚠️  User '{username}' already exists. Skipping...")
            return existing_user
        
        # Hash password
        password_hash = hash_password(password)
        
        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            department=department,
            position=position,
            is_active=True,
            is_verified=is_verified,
            is_superuser=False,
            created_by='system',
            created_at=datetime.now(timezone.utc)
        )
        
        session.add(user)
        session.flush()  # Get user_id
        
        # Get role
        role_repo = RoleRepository(session)
        role = role_repo.get_by_code(role_code)
        
        if not role:
            print(f"❌ Role '{role_code}' not found. Please run seed_roles_permissions.sql first.")
            session.rollback()
            return None
        
        # Assign role
        user_role = UserRole(
            user_id=user.user_id,
            role_id=role.role_id,
            assigned_by='system',
            assigned_at=datetime.now(timezone.utc)
        )
        
        session.add(user_role)
        session.commit()
        
        print(f"✅ Created user '{username}' with role '{role.role_name}'")
        print(f"   Email: {email}")
        print(f"   Password: {password}")
        print(f"   Department: {department}")
        print()
        
        return user
        
    except IntegrityError as e:
        session.rollback()
        print(f"❌ Integrity error creating user '{username}': {e}")
        return None
    except Exception as e:
        session.rollback()
        print(f"❌ Error creating user '{username}': {e}")
        return None


def main():
    """Main function to seed users."""
    print("=" * 70)
    print("Seeding Users with Roles and Permissions")
    print("=" * 70)
    print()
    
    try:
        with get_db_session() as session:
            # User 1: Data Administrator
            print("Creating Data Administrator...")
            create_user(
                session=session,
                username="data_admin",
                email="data.admin@batibank.com",
                password="DataAdmin@2024",
                full_name="Sarah Johnson",
                department="Data Management",
                position="Data Administrator",
                role_code="data_admin",
                is_verified=True
            )
            
            # User 2: Data Analyst
            print("Creating Data Analyst...")
            create_user(
                session=session,
                username="data_analyst",
                email="analyst@batibank.com",
                password="Analyst@2024",
                full_name="Michael Chen",
                department="Analytics",
                position="Senior Data Analyst",
                role_code="data_analyst",
                is_verified=True
            )
            
            # User 3: Business User
            print("Creating Business User...")
            create_user(
                session=session,
                username="business_user",
                email="business.user@batibank.com",
                password="Business@2024",
                full_name="Emily Rodriguez",
                department="Credit Operations",
                position="Credit Manager",
                role_code="business_user",
                is_verified=True
            )
            
            # Optional: Create a Model Developer user
            print("Creating Model Developer...")
            create_user(
                session=session,
                username="model_dev",
                email="model.dev@batibank.com",
                password="ModelDev@2024",
                full_name="David Kim",
                department="ML Engineering",
                position="ML Engineer",
                role_code="model_developer",
                is_verified=True
            )
            
            print("=" * 70)
            print("✅ User seeding completed successfully!")
            print("=" * 70)
            print()
            print("Summary of created users:")
            print()
            print("1. Data Administrator (data_admin)")
            print("   - Username: data_admin")
            print("   - Password: DataAdmin@2024")
            print("   - Permissions: Upload data, manage data, view predictions")
            print()
            print("2. Data Analyst (data_analyst)")
            print("   - Username: data_analyst")
            print("   - Password: Analyst@2024")
            print("   - Permissions: View data, predictions, model performance")
            print()
            print("3. Business User (business_user)")
            print("   - Username: business_user")
            print("   - Password: Business@2024")
            print("   - Permissions: View predictions and business KPIs")
            print()
            print("4. Model Developer (model_dev)")
            print("   - Username: model_dev")
            print("   - Password: ModelDev@2024")
            print("   - Permissions: Train models, deploy, manage model registry")
            print()
            print("⚠️  IMPORTANT: Change these passwords in production!")
            print()
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
