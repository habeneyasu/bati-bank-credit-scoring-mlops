"""
Data Quality Monitoring Module

Monitors data quality metrics:
- Schema validation
- Missing value detection
- Outlier detection
- Data freshness checks
- Data completeness metrics
- Automated data quality reports
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from decimal import Decimal
import json

from src.utils.logging import get_logger
from src.database.connection import get_db_session
from src.database.models import RawTransaction

logger = get_logger(__name__)


class DataQualityChecker:
    """
    Checks data quality for uploaded transactions.
    """
    
    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        missing_threshold: float = 0.1,
        outlier_threshold: float = 3.0
    ):
        """
        Initialize data quality checker.
        
        Args:
            schema: Expected data schema
            missing_threshold: Maximum allowed missing value ratio
            outlier_threshold: Z-score threshold for outliers
        """
        self.schema = schema or self._get_default_schema()
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.logger = get_logger(__name__)
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Get default schema for raw transactions."""
        return {
            "required_fields": ["transaction_id", "customer_id", "amount", "transaction_start_time"],
            "optional_fields": [
                "batch_id", "account_id", "subscription_id", "currency_code",
                "country_code", "provider_id", "product_id", "product_category",
                "channel_id", "value", "pricing_strategy", "fraud_result"
            ],
            "field_types": {
                "transaction_id": str,
                "customer_id": str,
                "amount": (int, float),
                "transaction_start_time": (str, datetime),
                "batch_id": (str, type(None)),
                "account_id": (str, type(None)),
                "subscription_id": (str, type(None)),
                "currency_code": (str, type(None)),
                "country_code": (str, type(None)),
                "provider_id": (str, type(None)),
                "product_id": (str, type(None)),
                "product_category": (str, type(None)),
                "channel_id": (str, type(None)),
                "value": (int, float, type(None)),
                "pricing_strategy": (int, type(None)),
                "fraud_result": (int, type(None))
            },
            "constraints": {
                "amount": {"min": -1000000, "max": 1000000},
                "fraud_result": {"values": [0, 1]}
            }
        }
    
    def validate_schema(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate data against schema.
        
        Args:
            data: List of transaction dictionaries
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        if not data:
            return {
                "valid": False,
                "errors": ["Empty dataset"],
                "warnings": []
            }
        
        # Check required fields
        required_fields = self.schema.get("required_fields", [])
        for idx, record in enumerate(data):
            for field in required_fields:
                if field not in record or record[field] is None:
                    errors.append(f"Row {idx + 1}: Missing required field '{field}'")
        
        # Check field types
        field_types = self.schema.get("field_types", {})
        for idx, record in enumerate(data):
            for field, expected_type in field_types.items():
                if field in record and record[field] is not None:
                    if not isinstance(record[field], expected_type):
                        if not isinstance(expected_type, tuple):
                            expected_type = (expected_type,)
                        if type(record[field]) not in expected_type:
                            warnings.append(
                                f"Row {idx + 1}: Field '{field}' has unexpected type. "
                                f"Expected {expected_type}, got {type(record[field])}"
                            )
        
        # Check constraints
        constraints = self.schema.get("constraints", {})
        for idx, record in enumerate(data):
            for field, constraint in constraints.items():
                if field in record and record[field] is not None:
                    if "min" in constraint and record[field] < constraint["min"]:
                        errors.append(
                            f"Row {idx + 1}: Field '{field}' value {record[field]} "
                            f"below minimum {constraint['min']}"
                        )
                    if "max" in constraint and record[field] > constraint["max"]:
                        errors.append(
                            f"Row {idx + 1}: Field '{field}' value {record[field]} "
                            f"above maximum {constraint['max']}"
                        )
                    if "values" in constraint and record[field] not in constraint["values"]:
                        errors.append(
                            f"Row {idx + 1}: Field '{field}' value {record[field]} "
                            f"not in allowed values {constraint['values']}"
                        )
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_records": len(data),
            "error_count": len(errors),
            "warning_count": len(warnings)
        }
    
    def check_missing_values(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for missing values in data.
        
        Args:
            data: List of transaction dictionaries
            
        Returns:
            Missing value analysis
        """
        if not data:
            return {
                "total_records": 0,
                "missing_counts": {},
                "missing_percentages": {},
                "fields_above_threshold": []
            }
        
        df = pd.DataFrame(data)
        missing_counts = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Convert numpy/pandas types to native Python types for JSON serialization
        missing_counts = {k: int(v) for k, v in missing_counts.items()}
        missing_percentages = {k: float(v) for k, v in missing_percentages.items()}
        
        fields_above_threshold = [
            field for field, pct in missing_percentages.items()
            if pct > self.missing_threshold * 100
        ]
        
        return {
            "total_records": int(len(df)),
            "missing_counts": missing_counts,
            "missing_percentages": missing_percentages,
            "fields_above_threshold": fields_above_threshold,
            "threshold": float(self.missing_threshold * 100)
        }
    
    def detect_outliers(self, data: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
        """
        Detect outliers in a numeric field using Z-score.
        
        Args:
            data: List of transaction dictionaries
            field: Field name to check
            
        Returns:
            Outlier detection results
        """
        if not data:
            return {
                "field": field,
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "outlier_indices": []
            }
        
        df = pd.DataFrame(data)
        
        if field not in df.columns:
            return {
                "field": field,
                "error": f"Field '{field}' not found in data"
            }
        
        # Convert to numeric, handling non-numeric values
        numeric_data = pd.to_numeric(df[field], errors='coerce')
        numeric_data = numeric_data.dropna()
        
        if len(numeric_data) == 0:
            return {
                "field": field,
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "outlier_indices": [],
                "message": "No numeric values found"
            }
        
        # Calculate Z-scores
        mean = numeric_data.mean()
        std = numeric_data.std()
        
        if std == 0:
            return {
                "field": field,
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "outlier_indices": [],
                "message": "Standard deviation is zero"
            }
        
        z_scores = np.abs((numeric_data - mean) / std)
        outlier_mask = z_scores > self.outlier_threshold
        outlier_indices = [int(idx) for idx in numeric_data[outlier_mask].index.tolist()]
        
        return {
            "field": field,
            "outlier_count": int(len(outlier_indices)),
            "outlier_percentage": float((len(outlier_indices) / len(numeric_data)) * 100),
            "outlier_indices": outlier_indices,
            "mean": float(mean),
            "std": float(std),
            "threshold": float(self.outlier_threshold)
        }
    
    def check_data_freshness(
        self,
        latest_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Check data freshness (time since last update).
        
        Args:
            latest_timestamp: Latest transaction timestamp
            
        Returns:
            Freshness check results
        """
        if latest_timestamp is None:
            # Get latest from database
            try:
                with get_db_session() as session:
                    latest = session.query(RawTransaction).order_by(
                        RawTransaction.transaction_start_time.desc()
                    ).first()
                    
                    if latest:
                        latest_timestamp = latest.transaction_start_time
                    else:
                        return {
                            "fresh": False,
                            "message": "No data found",
                            "hours_since_update": None
                        }
            except Exception as e:
                logger.error(f"Error checking data freshness: {e}", exc_info=True)
                return {
                    "fresh": False,
                    "message": f"Error: {str(e)}",
                    "hours_since_update": None
                }
        
        now = datetime.now(timezone.utc)
        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
        
        time_diff = now - latest_timestamp
        hours_since_update = time_diff.total_seconds() / 3600
        
        # Consider data fresh if updated within last 24 hours
        is_fresh = hours_since_update < 24
        
        return {
            "fresh": is_fresh,
            "hours_since_update": hours_since_update,
            "latest_timestamp": latest_timestamp.isoformat(),
            "current_timestamp": now.isoformat()
        }
    
    def calculate_completeness(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate data completeness metrics.
        
        Args:
            data: List of transaction dictionaries
            
        Returns:
            Completeness metrics
        """
        if not data:
            return {
                "total_records": 0,
                "completeness_score": 0.0,
                "field_completeness": {}
            }
        
        df = pd.DataFrame(data)
        total_cells = len(df) * len(df.columns)
        non_null_cells = int(df.notna().sum().sum())
        completeness_score = float((non_null_cells / total_cells) * 100 if total_cells > 0 else 0)
        
        field_completeness = {}
        for col in df.columns:
            non_null_count = int(df[col].notna().sum())
            field_completeness[col] = {
                "non_null_count": non_null_count,
                "null_count": int(len(df) - non_null_count),
                "completeness_percentage": float((non_null_count / len(df)) * 100 if len(df) > 0 else 0)
            }
        
        return {
            "total_records": int(len(df)),
            "total_fields": int(len(df.columns)),
            "completeness_score": completeness_score,
            "field_completeness": field_completeness
        }
    
    def generate_quality_report(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            data: List of transaction dictionaries
            
        Returns:
            Complete quality report
        """
        schema_validation = self.validate_schema(data)
        missing_analysis = self.check_missing_values(data)
        completeness = self.calculate_completeness(data)
        freshness = self.check_data_freshness()
        
        # Check outliers for numeric fields
        outlier_results = {}
        if data:
            df = pd.DataFrame(data)
            numeric_fields = df.select_dtypes(include=[np.number]).columns.tolist()
            for field in numeric_fields:
                outlier_results[field] = self.detect_outliers(data, field)
        
        # Calculate overall quality score
        quality_score = 100.0
        if not schema_validation["valid"]:
            quality_score -= 30
        if missing_analysis["fields_above_threshold"]:
            quality_score -= 20
        if completeness["completeness_score"] < 90:
            quality_score -= 10
        if not freshness["fresh"]:
            quality_score -= 10
        
        quality_score = max(0, quality_score)
        
        # Ensure all values are JSON serializable (convert numpy/pandas types)
        def make_json_serializable(obj):
            """Recursively convert numpy/pandas types to native Python types."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quality_score": float(quality_score),
            "schema_validation": make_json_serializable(schema_validation),
            "missing_values": make_json_serializable(missing_analysis),
            "completeness": make_json_serializable(completeness),
            "freshness": make_json_serializable(freshness),
            "outliers": make_json_serializable(outlier_results),
            "summary": {
                "total_records": int(len(data)),
                "is_valid": bool(schema_validation["valid"]),
                "has_missing_issues": bool(len(missing_analysis["fields_above_threshold"]) > 0),
                "is_complete": bool(completeness["completeness_score"] >= 90),
                "is_fresh": bool(freshness["fresh"])
            }
        }
        
        return make_json_serializable(report)
