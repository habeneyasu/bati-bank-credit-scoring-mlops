#!/usr/bin/env python3
"""
Migration script to add id column to drift_metrics table.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import DatabaseManager
from sqlalchemy import text

def main():
    """Add id column to drift_metrics table."""
    print("Migrating drift_metrics table...")
    print("=" * 50)
    
    try:
        db_manager = DatabaseManager()
        
        with db_manager.get_session() as session:
            # Check if id column exists
            result = session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'drift_metrics' AND column_name = 'id'
            """))
            
            if result.fetchone():
                print("✓ Column 'id' already exists in drift_metrics table")
            else:
                # Add id column
                print("Adding 'id' column to drift_metrics table...")
                session.execute(text("""
                    ALTER TABLE drift_metrics 
                    ADD COLUMN id SERIAL PRIMARY KEY
                """))
                session.commit()
                print("✓ Successfully added 'id' column to drift_metrics table")
        
        print("\n✓ Migration complete!")
        
    except Exception as e:
        print(f"\n✗ Error during migration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
