"""
Versioning utilities for models, data, and artifacts.

This module provides comprehensive versioning for:
- Model versions (MLflow integration)
- Data versions (file-based with metadata)
- Feature versions
- Artifact versions
"""

import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

from src.utils.logging import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


class DataVersioner:
    """
    Data versioning system using file-based versioning with metadata.
    
    Tracks versions of datasets, features, and artifacts with:
    - Version numbers
    - Checksums (MD5/SHA256)
    - Metadata (size, shape, creation date)
    - Dependencies
    """
    
    def __init__(self, version_dir: Optional[Path] = None):
        """
        Initialize data versioner.
        
        Args:
            version_dir: Directory to store version metadata (default: data/versions)
        """
        project_root = Path(__file__).parent.parent.parent
        self.version_dir = version_dir or (project_root / "data" / "versions")
        self.version_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.version_dir / "versions.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load version metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load version metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save version metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save version metadata: {e}")
    
    def _calculate_checksum(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """
        Calculate file checksum.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5' or 'sha256')
        
        Returns:
            Hexadecimal checksum string
        """
        hash_obj = hashlib.sha256() if algorithm == 'sha256' else hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get file information.
        
        Args:
            file_path: Path to file
        
        Returns:
            Dictionary with file information
        """
        stat = file_path.stat()
        
        return {
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
    
    def version_data(
        self,
        data_path: Path,
        data_type: str = "dataset",
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Version a data file or directory.
        
        Args:
            data_path: Path to data file or directory
            data_type: Type of data ('dataset', 'features', 'splits', 'artifacts')
            version: Version string (auto-generated if None)
            metadata: Additional metadata
            dependencies: List of dependency versions
        
        Returns:
            Version information dictionary
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Generate version if not provided
        if version is None:
            existing_versions = [
                v for v in self.metadata.get(data_type, {}).keys()
                if isinstance(v, str) and v.startswith('v')
            ]
            if existing_versions:
                # Extract numbers and increment
                version_nums = [
                    int(v[1:]) for v in existing_versions
                    if v[1:].isdigit()
                ]
                next_version = max(version_nums) + 1 if version_nums else 1
            else:
                next_version = 1
            version = f"v{next_version}"
        
        # Calculate checksum
        if data_path.is_file():
            checksum = self._calculate_checksum(data_path)
            file_info = self._get_file_info(data_path)
            
            # For CSV/Parquet files, get shape info
            shape_info = {}
            try:
                if data_path.suffix == '.csv':
                    df = pd.read_csv(data_path, nrows=0)  # Just read header
                    shape_info = {"columns": len(df.columns)}
                elif data_path.suffix == '.parquet':
                    df = pd.read_parquet(data_path, nrows=0)
                    shape_info = {"columns": len(df.columns)}
            except Exception:
                pass
        else:
            # For directories, calculate checksum of all files
            checksum = self._calculate_directory_checksum(data_path)
            file_info = {"type": "directory"}
            shape_info = {}
        
        # Create version entry
        version_info = {
            "version": version,
            "path": str(data_path.absolute()),
            "checksum": checksum,
            "data_type": data_type,
            "created": datetime.now().isoformat(),
            "file_info": file_info,
            "shape_info": shape_info,
            "metadata": metadata or {},
            "dependencies": dependencies or []
        }
        
        # Store in metadata
        if data_type not in self.metadata:
            self.metadata[data_type] = {}
        
        self.metadata[data_type][version] = version_info
        self._save_metadata()
        
        logger.info(
            f"Versioned {data_type} as {version}",
            extra={"data_type": data_type, "version": version, "path": str(data_path)}
        )
        
        return version_info
    
    def _calculate_directory_checksum(self, dir_path: Path) -> str:
        """Calculate checksum for directory (sum of all file checksums)."""
        hash_obj = hashlib.sha256()
        
        for file_path in sorted(dir_path.rglob('*')):
            if file_path.is_file():
                file_hash = self._calculate_checksum(file_path)
                hash_obj.update(f"{file_path.relative_to(dir_path)}:{file_hash}".encode())
        
        return hash_obj.hexdigest()
    
    def get_version(self, data_type: str, version: str) -> Optional[Dict[str, Any]]:
        """
        Get version information.
        
        Args:
            data_type: Type of data
            version: Version string
        
        Returns:
            Version information or None
        """
        return self.metadata.get(data_type, {}).get(version)
    
    def list_versions(self, data_type: Optional[str] = None) -> Dict[str, Any]:
        """
        List all versions.
        
        Args:
            data_type: Filter by data type (None for all)
        
        Returns:
            Dictionary of versions
        """
        if data_type:
            return self.metadata.get(data_type, {})
        return self.metadata
    
    def get_latest_version(self, data_type: str) -> Optional[Dict[str, Any]]:
        """
        Get latest version for a data type.
        
        Args:
            data_type: Type of data
        
        Returns:
            Latest version information or None
        """
        versions = self.metadata.get(data_type, {})
        if not versions:
            return None
        
        # Sort by version number
        sorted_versions = sorted(
            versions.items(),
            key=lambda x: int(x[0][1:]) if x[0][1:].isdigit() else 0,
            reverse=True
        )
        
        return sorted_versions[0][1] if sorted_versions else None
    
    def verify_version(self, data_type: str, version: str) -> bool:
        """
        Verify version integrity by checking checksum.
        
        Args:
            data_type: Type of data
            version: Version string
        
        Returns:
            True if checksum matches, False otherwise
        """
        version_info = self.get_version(data_type, version)
        if not version_info:
            return False
        
        data_path = Path(version_info["path"])
        if not data_path.exists():
            return False
        
        current_checksum = self._calculate_checksum(data_path) if data_path.is_file() else self._calculate_directory_checksum(data_path)
        return current_checksum == version_info["checksum"]


class ModelVersioner:
    """
    Enhanced model versioning with MLflow integration.
    
    Tracks model versions with:
    - MLflow run IDs
    - Model registry versions
    - Training metadata
    - Performance metrics
    - Data versions used
    """
    
    def __init__(self):
        """Initialize model versioner."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            self.mlflow_available = True
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            self.client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        except ImportError:
            self.mlflow_available = False
            logger.warning("MLflow not available, model versioning limited")
    
    def get_model_version_info(
        self,
        model_name: str,
        stage: Optional[str] = None,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get model version information from MLflow.
        
        Args:
            model_name: Registered model name
            stage: Model stage ('Production', 'Staging', etc.)
            version: Specific version number
        
        Returns:
            Model version information
        """
        if not self.mlflow_available:
            return {"error": "MLflow not available"}
        
        try:
            import mlflow
            
            if version:
                model_version = self.client.get_model_version(model_name, version)
                run_id = model_version.run_id
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
                # Get latest version in stage
                model_versions = [
                    mv for mv in self.client.search_model_versions(f"name='{model_name}'")
                    if mv.current_stage == stage
                ]
                if not model_versions:
                    return {"error": f"No model found in {stage} stage"}
                model_version = model_versions[0]
                run_id = model_version.run_id
            else:
                # Get latest version
                model_versions = self.client.search_model_versions(f"name='{model_name}'")
                if not model_versions:
                    return {"error": "No versions found"}
                model_version = model_versions[0]
                run_id = model_version.run_id
            
            # Get run information
            run = self.client.get_run(run_id)
            
            # Extract metrics
            metrics = run.data.metrics
            
            # Extract parameters
            params = run.data.params
            
            # Extract tags
            tags = run.data.tags
            
            return {
                "model_name": model_name,
                "version": model_version.version,
                "stage": model_version.current_stage,
                "run_id": run_id,
                "created_at": model_version.creation_timestamp,
                "metrics": metrics,
                "parameters": params,
                "tags": tags,
                "model_uri": f"models:/{model_name}/{model_version.version}"
            }
            
        except Exception as e:
            logger.error(f"Error getting model version info: {e}", exc_info=True)
            return {"error": str(e)}
    
    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model.
        
        Args:
            model_name: Registered model name
        
        Returns:
            List of model version information
        """
        if not self.mlflow_available:
            return []
        
        try:
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            versions = []
            for mv in model_versions:
                try:
                    run = self.client.get_run(mv.run_id)
                    versions.append({
                        "version": mv.version,
                        "stage": mv.current_stage,
                        "run_id": mv.run_id,
                        "created_at": mv.creation_timestamp,
                        "metrics": run.data.metrics,
                        "description": mv.description
                    })
                except Exception as e:
                    logger.warning(f"Could not get info for version {mv.version}: {e}")
            
            return sorted(versions, key=lambda x: int(x["version"]), reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing model versions: {e}", exc_info=True)
            return []
    
    def get_current_production_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get current production model information.
        
        Args:
            model_name: Registered model name
        
        Returns:
            Production model information or None
        """
        return self.get_model_version_info(model_name, stage="Production")


def get_system_versions() -> Dict[str, Any]:
    """
    Get comprehensive system version information.
    
    Returns:
        Dictionary with all version information
    """
    versions = {
        "timestamp": datetime.now().isoformat(),
        "data_versions": {},
        "model_versions": {},
        "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
    }
    
    # Get data versions
    try:
        data_versioner = DataVersioner()
        versions["data_versions"] = data_versioner.list_versions()
    except Exception as e:
        logger.warning(f"Could not get data versions: {e}")
        versions["data_versions"] = {"error": str(e)}
    
    # Get model versions
    try:
        model_versioner = ModelVersioner()
        if model_versioner.mlflow_available:
            # Try to get model info
            model_name = settings.model_name if hasattr(settings, 'model_name') else "credit_scoring_model"
            versions["model_versions"] = {
                "current": model_versioner.get_current_production_model(model_name),
                "all_versions": model_versioner.list_model_versions(model_name)
            }
    except Exception as e:
        logger.warning(f"Could not get model versions: {e}")
        versions["model_versions"] = {"error": str(e)}
    
    return versions
