"""
Prediction Validation and Quality Assurance Module

This module provides comprehensive validation for:
- Feature vectors before model prediction
- Model predictions after inference
- Data quality checks
- Confidence and uncertainty measures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from src.utils.logging import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    message: str
    severity: str  # 'error', 'warning', 'info'
    details: Optional[Dict[str, Any]] = None


@dataclass
class PredictionQuality:
    """Quality metrics for a prediction."""
    confidence_score: float  # 0-1, higher is more confident
    uncertainty_level: str  # 'low', 'medium', 'high'
    data_quality_score: float  # 0-1, quality of input data
    feature_completeness: float  # 0-1, how complete the features are
    warnings: List[str]
    validation_results: List[ValidationResult]


class FeatureValidator:
    """Validates feature vectors before model prediction."""
    
    def __init__(
        self,
        expected_features: int = 26,
        feature_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
        allow_nan: bool = False,
        allow_inf: bool = False
    ):
        """
        Initialize feature validator.
        
        Args:
            expected_features: Expected number of features
            feature_ranges: Dict mapping feature index to (min, max) range
            allow_nan: Whether to allow NaN values
            allow_inf: Whether to allow infinite values
        """
        self.expected_features = expected_features
        self.feature_ranges = feature_ranges or {}
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf
    
    def validate_feature_vector(
        self,
        features: List[float],
        customer_id: Optional[str] = None
    ) -> List[ValidationResult]:
        """
        Validate a feature vector.
        
        Args:
            features: Feature vector to validate
            customer_id: Optional customer ID for logging
        
        Returns:
            List of validation results
        """
        results = []
        
        # Convert to numpy array for easier manipulation
        try:
            features_array = np.array(features, dtype=np.float64)
        except (ValueError, TypeError) as e:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Features cannot be converted to numeric array: {e}",
                severity='error',
                details={'error_type': 'type_conversion'}
            ))
            return results
        
        # Check feature count
        if len(features_array) != self.expected_features:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Expected {self.expected_features} features, got {len(features_array)}",
                severity='error',
                details={
                    'expected': self.expected_features,
                    'actual': len(features_array)
                }
            ))
        
        # Check for NaN values
        nan_count = np.isnan(features_array).sum()
        if nan_count > 0 and not self.allow_nan:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Found {nan_count} NaN values in features",
                severity='error',
                details={'nan_count': int(nan_count), 'nan_indices': np.where(np.isnan(features_array))[0].tolist()}
            ))
        elif nan_count > 0:
            results.append(ValidationResult(
                is_valid=True,
                message=f"Found {nan_count} NaN values (allowed)",
                severity='warning',
                details={'nan_count': int(nan_count)}
            ))
        
        # Check for infinite values
        inf_count = np.isinf(features_array).sum()
        if inf_count > 0 and not self.allow_inf:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Found {inf_count} infinite values in features",
                severity='error',
                details={'inf_count': int(inf_count), 'inf_indices': np.where(np.isinf(features_array))[0].tolist()}
            ))
        elif inf_count > 0:
            results.append(ValidationResult(
                is_valid=True,
                message=f"Found {inf_count} infinite values (allowed)",
                severity='warning',
                details={'inf_count': int(inf_count)}
            ))
        
        # Check feature ranges
        for idx, (min_val, max_val) in self.feature_ranges.items():
            if 0 <= idx < len(features_array):
                val = features_array[idx]
                if not (np.isnan(val) or np.isinf(val)):
                    if val < min_val or val > max_val:
                        results.append(ValidationResult(
                            is_valid=True,
                            message=f"Feature {idx} value {val:.4f} outside expected range [{min_val}, {max_val}]",
                            severity='warning',
                            details={
                                'feature_index': idx,
                                'value': float(val),
                                'expected_min': min_val,
                                'expected_max': max_val
                            }
                        ))
        
        # Check for constant features (all zeros or all same value)
        unique_values = np.unique(features_array[~np.isnan(features_array)])
        if len(unique_values) == 1:
            results.append(ValidationResult(
                is_valid=True,
                message="All features have the same value (constant feature vector)",
                severity='warning',
                details={'constant_value': float(unique_values[0])}
            ))
        
        # Check feature variance
        non_nan_features = features_array[~np.isnan(features_array)]
        if len(non_nan_features) > 1:
            variance = np.var(non_nan_features)
            if variance < 1e-10:
                results.append(ValidationResult(
                    is_valid=True,
                    message="Very low feature variance (near-constant features)",
                    severity='warning',
                    details={'variance': float(variance)}
                ))
        
        return results
    
    def fix_feature_vector(self, features: List[float]) -> Tuple[List[float], List[str]]:
        """
        Fix common issues in feature vectors.
        
        Args:
            features: Feature vector to fix
        
        Returns:
            Tuple of (fixed_features, fix_messages)
        """
        features_array = np.array(features, dtype=np.float64)
        fix_messages = []
        
        # Fix NaN values (replace with 0)
        nan_mask = np.isnan(features_array)
        if nan_mask.any():
            features_array[nan_mask] = 0.0
            fix_messages.append(f"Replaced {nan_mask.sum()} NaN values with 0.0")
        
        # Fix infinite values (clip to reasonable range)
        inf_mask = np.isinf(features_array)
        if inf_mask.any():
            # Replace positive inf with large value, negative inf with small value
            features_array[inf_mask & (features_array > 0)] = 9999.0
            features_array[inf_mask & (features_array < 0)] = -9999.0
            fix_messages.append(f"Replaced {inf_mask.sum()} infinite values with bounded values")
        
        # Ensure correct length
        if len(features_array) < self.expected_features:
            padding = np.zeros(self.expected_features - len(features_array))
            features_array = np.concatenate([features_array, padding])
            fix_messages.append(f"Padded features from {len(features)} to {self.expected_features}")
        elif len(features_array) > self.expected_features:
            features_array = features_array[:self.expected_features]
            fix_messages.append(f"Truncated features from {len(features)} to {self.expected_features}")
        
        return features_array.tolist(), fix_messages


class PredictionValidator:
    """Validates model predictions."""
    
    def __init__(
        self,
        risk_threshold_low: float = 0.30,
        risk_threshold_high: float = 0.60
    ):
        """
        Initialize prediction validator.
        
        Args:
            risk_threshold_low: Low risk threshold
            risk_threshold_high: High risk threshold
        """
        self.risk_threshold_low = risk_threshold_low
        self.risk_threshold_high = risk_threshold_high
    
    def validate_prediction(
        self,
        probability: float,
        prediction: int,
        customer_id: Optional[str] = None
    ) -> List[ValidationResult]:
        """
        Validate a model prediction.
        
        Args:
            probability: Probability of high risk (0-1)
            prediction: Binary prediction (0 or 1)
            customer_id: Optional customer ID for logging
        
        Returns:
            List of validation results
        """
        results = []
        
        # Validate probability range
        if not (0.0 <= probability <= 1.0):
            results.append(ValidationResult(
                is_valid=False,
                message=f"Probability {probability} outside valid range [0, 1]",
                severity='error',
                details={'probability': probability}
            ))
        
        # Validate prediction value
        if prediction not in [0, 1]:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Prediction {prediction} must be 0 or 1",
                severity='error',
                details={'prediction': prediction}
            ))
        
        # Check consistency between probability and prediction
        expected_prediction = 1 if probability > 0.5 else 0
        if prediction != expected_prediction:
            results.append(ValidationResult(
                is_valid=True,
                message=f"Prediction {prediction} inconsistent with probability {probability:.4f} (expected {expected_prediction})",
                severity='warning',
                details={
                    'prediction': prediction,
                    'probability': probability,
                    'expected_prediction': expected_prediction
                }
            ))
        
        # Check for edge cases
        if probability < 0.01:
            results.append(ValidationResult(
                is_valid=True,
                message="Very low probability (< 0.01) - prediction may be unreliable",
                severity='warning',
                details={'probability': probability}
            ))
        elif probability > 0.99:
            results.append(ValidationResult(
                is_valid=True,
                message="Very high probability (> 0.99) - prediction may be unreliable",
                severity='warning',
                details={'probability': probability}
            ))
        
        # Check for ambiguous predictions (near threshold)
        if self.risk_threshold_low <= probability <= self.risk_threshold_high:
            results.append(ValidationResult(
                is_valid=True,
                message=f"Probability {probability:.4f} in medium-risk range - consider manual review",
                severity='info',
                details={
                    'probability': probability,
                    'risk_range': 'medium'
                }
            ))
        
        return results


class PredictionQualityAssessor:
    """Assesses the quality and reliability of predictions."""
    
    def __init__(
        self,
        min_transactions_for_reliable: int = 5,
        feature_completeness_threshold: float = 0.8
    ):
        """
        Initialize quality assessor.
        
        Args:
            min_transactions_for_reliable: Minimum transactions for reliable prediction
            feature_completeness_threshold: Threshold for feature completeness (0-1)
        """
        self.min_transactions_for_reliable = min_transactions_for_reliable
        self.feature_completeness_threshold = feature_completeness_threshold
    
    def assess_quality(
        self,
        features: List[float],
        probability: float,
        transaction_count: int,
        prediction: int
    ) -> PredictionQuality:
        """
        Assess the quality of a prediction.
        
        Args:
            features: Feature vector used for prediction
            probability: Probability of high risk
            transaction_count: Number of transactions used
            prediction: Binary prediction
        
        Returns:
            PredictionQuality object
        """
        warnings = []
        validation_results = []
        
        # Calculate feature completeness
        # Note: Normalized features can legitimately be 0, so we check for meaningful variance
        # instead of just counting non-zero values
        features_array = np.array(features, dtype=np.float64)
        
        # Remove NaN and Inf for calculation
        valid_features = features_array[~np.isnan(features_array) & ~np.isinf(features_array)]
        
        if len(valid_features) == 0:
            feature_completeness = 0.0
        else:
            # Calculate feature completeness based on:
            # 1. How many features are not NaN/Inf (data availability)
            # 2. Feature variance (meaningful variation vs constant)
            data_availability = len(valid_features) / len(features_array)
            
            # Check variance - if all features are the same, completeness is low
            if len(valid_features) > 1:
                variance = np.var(valid_features)
                # Normalize variance (assuming features are normalized 0-1, max variance is 0.25)
                variance_score = min(1.0, variance / 0.25) if variance > 0 else 0.0
            else:
                variance_score = 0.0
            
            # Feature completeness is weighted average of availability and variance
            # Availability is more important (70%), variance is less (30%)
            feature_completeness = data_availability * 0.7 + variance_score * 0.3
        
        if feature_completeness < self.feature_completeness_threshold:
            warnings.append(f"Low feature completeness ({feature_completeness:.2%})")
        
        # Calculate data quality score
        data_quality_score = 1.0
        
        # Penalize for insufficient transactions
        if transaction_count < self.min_transactions_for_reliable:
            transaction_penalty = 1.0 - (transaction_count / self.min_transactions_for_reliable)
            data_quality_score -= transaction_penalty * 0.3
            warnings.append(f"Insufficient transaction history ({transaction_count} < {self.min_transactions_for_reliable})")
        
        # Penalize for low feature completeness
        if feature_completeness < self.feature_completeness_threshold:
            completeness_penalty = (self.feature_completeness_threshold - feature_completeness) * 0.2
            data_quality_score -= completeness_penalty
        
        # Ensure score is in [0, 1]
        data_quality_score = max(0.0, min(1.0, data_quality_score))
        
        # Calculate confidence score based on probability distance from 0.5
        # Predictions near 0.5 are less confident
        confidence_score = abs(probability - 0.5) * 2.0  # Maps [0, 0.5] to [0, 1]
        
        # Adjust confidence based on data quality
        confidence_score = confidence_score * 0.7 + data_quality_score * 0.3
        
        # Determine uncertainty level
        if confidence_score >= 0.8:
            uncertainty_level = 'low'
        elif confidence_score >= 0.5:
            uncertainty_level = 'medium'
        else:
            uncertainty_level = 'high'
        
        return PredictionQuality(
            confidence_score=confidence_score,
            uncertainty_level=uncertainty_level,
            data_quality_score=data_quality_score,
            feature_completeness=feature_completeness,
            warnings=warnings,
            validation_results=validation_results
        )


def validate_and_assess_prediction(
    features: List[float],
    probability: float,
    prediction: int,
    transaction_count: int,
    customer_id: Optional[str] = None,
    expected_features: int = 26
) -> Tuple[bool, PredictionQuality, List[ValidationResult]]:
    """
    Comprehensive validation and quality assessment for a prediction.
    
    Args:
        features: Feature vector
        probability: Probability of high risk
        prediction: Binary prediction
        transaction_count: Number of transactions
        customer_id: Optional customer ID
        expected_features: Expected number of features
    
    Returns:
        Tuple of (is_valid, quality, validation_results)
    """
    # Validate features
    feature_validator = FeatureValidator(expected_features=expected_features)
    feature_results = feature_validator.validate_feature_vector(features, customer_id)
    
    # Validate prediction
    prediction_validator = PredictionValidator(
        risk_threshold_low=settings.risk_threshold_low,
        risk_threshold_high=settings.risk_threshold_high
    )
    prediction_results = prediction_validator.validate_prediction(probability, prediction, customer_id)
    
    # Assess quality
    quality_assessor = PredictionQualityAssessor()
    quality = quality_assessor.assess_quality(features, probability, transaction_count, prediction)
    
    # Combine all results
    all_results = feature_results + prediction_results
    
    # Check if any errors exist
    has_errors = any(not r.is_valid and r.severity == 'error' for r in all_results)
    
    # Log validation results
    if all_results:
        error_count = sum(1 for r in all_results if r.severity == 'error')
        warning_count = sum(1 for r in all_results if r.severity == 'warning')
        
        logger.info(
            f"Prediction validation for customer {customer_id or 'unknown'}",
            extra={
                'customer_id': customer_id,
                'validation_errors': error_count,
                'validation_warnings': warning_count,
                'confidence_score': quality.confidence_score,
                'uncertainty_level': quality.uncertainty_level,
                'data_quality_score': quality.data_quality_score
            }
        )
    
    return not has_errors, quality, all_results
