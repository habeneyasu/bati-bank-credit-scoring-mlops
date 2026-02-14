#!/usr/bin/env python3
"""
Script to create database tables using SQLAlchemy.

This script will create all tables defined in the models, including the
data_lineage table with the correct lineage_metadata column name.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import DatabaseManager
from src.database.exceptions import DatabaseError

def main():
    """Create all database tables."""
    print("Creating database tables...")
    print("=" * 50)
    
    try:
        db_manager = DatabaseManager()
        db_manager.create_tables()
        print("\n✓ Database tables created successfully!")
        print("\nThe data_lineage table has been created with the 'lineage_metadata' column.")
        
    except DatabaseError as e:
        print(f"\n✗ Error creating tables: {e}")
        if hasattr(e, 'original_error'):
            print(f"  Original error: {e.original_error}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
