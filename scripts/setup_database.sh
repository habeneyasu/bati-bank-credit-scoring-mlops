#!/bin/bash
# Database Setup Script
# Purpose: Initialize the mlops_db database with all schemas

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "MLOps Database Setup Script"
echo "=========================================="
echo ""

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo -e "${RED}Error: PostgreSQL (psql) is not installed or not in PATH${NC}"
    echo "Please install PostgreSQL first:"
    echo "  Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    echo "  macOS: brew install postgresql"
    echo "  Or use Docker: docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:14"
    exit 1
fi

# Database configuration
DB_NAME="mlops_db"
DB_USER="${POSTGRES_USER:-postgres}"
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"

echo "Database Configuration:"
echo "  Database: $DB_NAME"
echo "  User: $DB_USER"
echo "  Host: $DB_HOST"
echo "  Port: $DB_PORT"
echo ""

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Project Root: $PROJECT_ROOT"
echo ""

# Check if database exists
echo "Checking if database exists..."
if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    echo -e "${YELLOW}Database '$DB_NAME' already exists${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Dropping existing database..."
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "DROP DATABASE IF EXISTS $DB_NAME;"
        echo "Creating new database..."
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "CREATE DATABASE $DB_NAME;"
    else
        echo "Using existing database..."
    fi
else
    echo "Creating database '$DB_NAME'..."
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "CREATE DATABASE $DB_NAME;"
fi

echo ""
echo "=========================================="
echo "Running initialization script..."
echo "=========================================="
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

# Run initialization script
if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f scripts/init_db.sql; then
    echo ""
    echo -e "${GREEN}✓ Database initialization completed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Change the default admin password:"
    echo "     UPDATE users SET password_hash = '<new_hash>' WHERE username = 'admin';"
    echo ""
    echo "  2. Verify setup:"
    echo "     psql -U $DB_USER -d $DB_NAME -c 'SELECT * FROM roles;'"
    echo ""
    echo "  3. Check default admin user:"
    echo "     psql -U $DB_USER -d $DB_NAME -c 'SELECT username, email, is_active FROM users WHERE username = '\''admin'\'';'"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Database initialization failed!${NC}"
    echo "Please check the error messages above."
    exit 1
fi
