"""
Example script for versioning data files.

This script demonstrates how to version datasets, features, and artifacts
using the DataVersioner utility.

Usage:
    python examples/version_data.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.versioning import DataVersioner
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main function to demonstrate data versioning."""
    
    print("=" * 80)
    print("Data Versioning Example")
    print("=" * 80)
    print()
    
    # Initialize versioner
    versioner = DataVersioner()
    print(f"Version directory: {versioner.version_dir}")
    print()
    
    # Example: Version a dataset
    data_dir = project_root / "data"
    
    # Check for existing data files
    raw_data = data_dir / "raw" / "transactions.csv"
    processed_data = data_dir / "processed" / "features.csv"
    splits_dir = data_dir / "processed" / "splits"
    
    print("Versioning Data Files")
    print("-" * 80)
    print()
    
    # Version raw dataset if exists
    if raw_data.exists():
        print(f"Versioning raw dataset: {raw_data}")
        version_info = versioner.version_data(
            data_path=raw_data,
            data_type="dataset",
            metadata={
                "source": "raw",
                "description": "Original transaction data"
            }
        )
        print(f"✓ Versioned as: {version_info['version']}")
        print(f"  Checksum: {version_info['checksum'][:16]}...")
        print()
    else:
        print(f"⚠ Raw dataset not found: {raw_data}")
        print()
    
    # Version processed features if exists
    if processed_data.exists():
        print(f"Versioning processed features: {processed_data}")
        version_info = versioner.version_data(
            data_path=processed_data,
            data_type="features",
            metadata={
                "source": "processed",
                "description": "Engineered features"
            },
            dependencies=["dataset:v1"]  # Depends on dataset v1
        )
        print(f"✓ Versioned as: {version_info['version']}")
        print(f"  Checksum: {version_info['checksum'][:16]}...")
        print()
    else:
        print(f"⚠ Processed features not found: {processed_data}")
        print()
    
    # Version splits directory if exists
    if splits_dir.exists() and splits_dir.is_dir():
        print(f"Versioning data splits: {splits_dir}")
        version_info = versioner.version_data(
            data_path=splits_dir,
            data_type="splits",
            metadata={
                "source": "processed",
                "description": "Train/test splits"
            },
            dependencies=["features:v1"]
        )
        print(f"✓ Versioned as: {version_info['version']}")
        print(f"  Checksum: {version_info['checksum'][:16]}...")
        print()
    else:
        print(f"⚠ Splits directory not found: {splits_dir}")
        print()
    
    # List all versions
    print("=" * 80)
    print("All Versions")
    print("=" * 80)
    print()
    
    all_versions = versioner.list_versions()
    
    for data_type, versions in all_versions.items():
        print(f"{data_type.upper()}:")
        for version, info in versions.items():
            print(f"  {version}:")
            print(f"    Path: {info['path']}")
            print(f"    Created: {info['created']}")
            print(f"    Checksum: {info['checksum'][:16]}...")
            if info.get('dependencies'):
                print(f"    Dependencies: {', '.join(info['dependencies'])}")
        print()
    
    # Get latest versions
    print("=" * 80)
    print("Latest Versions")
    print("=" * 80)
    print()
    
    data_types = ["dataset", "features", "splits", "artifacts"]
    for data_type in data_types:
        latest = versioner.get_latest_version(data_type)
        if latest:
            print(f"{data_type.upper()}: {latest['version']}")
            print(f"  Created: {latest['created']}")
        else:
            print(f"{data_type.upper()}: No versions")
        print()
    
    # Verify versions
    print("=" * 80)
    print("Version Verification")
    print("=" * 80)
    print()
    
    for data_type, versions in all_versions.items():
        for version in versions.keys():
            is_valid = versioner.verify_version(data_type, version)
            status = "✓ Valid" if is_valid else "✗ Invalid"
            print(f"{data_type} {version}: {status}")
    
    print()
    print("=" * 80)
    print("Data Versioning Complete!")
    print("=" * 80)
    print()
    print("Version metadata stored in:")
    print(f"  {versioner.metadata_file}")
    print()
    print("Next steps:")
    print("  1. View versions in dashboard: Click 'Versions' button")
    print("  2. Check API: GET /api/versions/data")
    print("  3. Use versioned data in model training")


if __name__ == "__main__":
    main()
