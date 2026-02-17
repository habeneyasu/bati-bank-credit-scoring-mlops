#!/bin/bash

# Demo Setup Script for Bati Bank Credit Scoring MLOps Platform
# This script ensures the system is ready for client demonstration

set -e  # Exit on error

echo "=========================================="
echo "Bati Bank Credit Scoring - Demo Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
pip install -q -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}Please update .env with your database credentials${NC}"
    else
        echo -e "${YELLOW}Creating basic .env file...${NC}"
        cat > .env << EOF
# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/credit_scoring_db

# MLflow Configuration
MLFLOW_TRACKING_URI=file:./mlruns

# API Configuration
API_PORT=8000
ENVIRONMENT=development
LOG_LEVEL=INFO

# Security
SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
EOF
        echo -e "${YELLOW}Please update .env with your database credentials${NC}"
    fi
fi

# Check database connection
echo -e "${GREEN}Checking database connection...${NC}"
python3 -c "
from src.database.connection import get_db_session
try:
    with get_db_session() as session:
        session.execute('SELECT 1')
    print('✓ Database connection successful')
except Exception as e:
    print(f'✗ Database connection failed: {e}')
    print('Please ensure PostgreSQL is running and DATABASE_URL is correct in .env')
    exit(1)
" || exit 1

# Create database tables
echo -e "${GREEN}Creating database tables...${NC}"
python3 scripts/create_tables.py

# Seed roles and permissions
echo -e "${GREEN}Seeding roles and permissions...${NC}"
if [ -f "scripts/schema/seed_roles_permissions.sql" ]; then
    psql $DATABASE_URL -f scripts/schema/seed_roles_permissions.sql 2>/dev/null || echo "Roles may already exist, continuing..."
else
    echo -e "${YELLOW}Warning: seed_roles_permissions.sql not found${NC}"
fi

# Seed users
echo -e "${GREEN}Seeding demo users...${NC}"
python3 scripts/seed_users.py

# Check if models exist
echo -e "${GREEN}Checking model files...${NC}"
if [ ! -f "models/random_forest.pkl" ]; then
    echo -e "${YELLOW}Warning: Model files not found. You may need to train models first.${NC}"
    echo -e "${YELLOW}Run: python examples/complete_training_script.py${NC}"
else
    echo "✓ Model files found"
fi

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}Demo Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Demo Users Created:"
echo "  1. Data Administrator"
echo "     Username: data_admin"
echo "     Password: DataAdmin@2024"
echo ""
echo "  2. Data Analyst"
echo "     Username: data_analyst"
echo "     Password: Analyst@2024"
echo ""
echo "  3. Business User"
echo "     Username: business_user"
echo "     Password: Business@2024"
echo ""
echo "  4. Model Developer"
echo "     Username: model_dev"
echo "     Password: ModelDev@2024"
echo ""
echo "Next Steps:"
echo "  1. Start the backend API:"
echo "     docker-compose up -d"
echo "     OR"
echo "     uvicorn src.api.main:app --host 0.0.0.0 --port 8001"
echo ""
echo "  2. Start the frontend:"
echo "     cd frontend && npm install && npm run dev"
echo ""
echo "  3. Access the application:"
echo "     Frontend: http://localhost:3000"
echo "     Backend API: http://localhost:8001"
echo "     API Docs: http://localhost:8001/docs"
echo ""
echo "=========================================="
