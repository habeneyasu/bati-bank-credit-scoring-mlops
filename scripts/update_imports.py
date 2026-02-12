#!/usr/bin/env python3
"""
Script to update imports after refactoring to modular structure.
"""

import re
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    r'from src\.rfm_calculator import': 'from src.features.rfm import',
    r'from src\.customer_clustering import': 'from src.features.clustering import',
    r'from src\.high_risk_labeling import': 'from src.features.labeling import',
    r'from src\.data_processing import': 'from src.features.processing import',
    r'from src\.woe_calculator import': 'from src.features.woe import',
    r'from src\.data_splitting import': 'from src.features.splitting import',
    r'from src\.model_training import': 'from src.models.training import',
    r'from src\.hyperparameter_tuning import': 'from src.models.tuning import',
    r'from src\.mlflow_tracking import': 'from src.models.tracking import',
    r'from src\.config import': 'from src.utils.config import',
    r'from src\.logger import': 'from src.utils.logging import',
    r'import src\.rfm_calculator': 'import src.features.rfm',
    r'import src\.customer_clustering': 'import src.features.clustering',
    r'import src\.high_risk_labeling': 'import src.features.labeling',
    r'import src\.data_processing': 'import src.features.processing',
    r'import src\.woe_calculator': 'import src.features.woe',
    r'import src\.data_splitting': 'import src.features.splitting',
    r'import src\.model_training': 'import src.models.training',
    r'import src\.hyperparameter_tuning': 'import src.models.tuning',
    r'import src\.mlflow_tracking': 'import src.models.tracking',
    r'import src\.config': 'import src.utils.config',
    r'import src\.logger': 'import src.utils.logging',
}

def update_file(file_path: Path):
    """Update imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all mappings
        for old_pattern, new_import in IMPORT_MAPPINGS.items():
            content = re.sub(old_pattern, new_import, content)
        
        # Only write if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update all Python files in examples and tests directories."""
    base_dir = Path(__file__).parent.parent
    
    # Directories to update
    dirs_to_update = [
        base_dir / 'examples',
        base_dir / 'tests',
    ]
    
    updated_count = 0
    
    for dir_path in dirs_to_update:
        if not dir_path.exists():
            continue
        
        for py_file in dir_path.rglob('*.py'):
            if update_file(py_file):
                updated_count += 1
    
    print(f"\nUpdated {updated_count} files")

if __name__ == '__main__':
    main()
