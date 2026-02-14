"""
Drift Detection and Monitoring Module

Implements statistical drift detection methods:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov (KS) test
- Chi-square test
- Distribution monitoring
- Concept drift detection

This module is critical for production ML systems to detect when
data distributions change over time, which can degrade model performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from decimal import Decimal
from scipy import stats
import pandas as pd

from src.utils.logging import get_logger
from src.database.connection import get_db_session
from src.database.models import DriftMetric
from src.database.exceptions import DatabaseError

logger = get_logger(__name__)


class DriftDetector:
    """
    Drift detection using statistical tests.
    
    Detects data drift by comparing current data distributions
    against reference (training) distributions.
    """
    
    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        reference_distributions: Optional[Dict[str, Dict[str, Any]]] = None,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        chi_square_threshold: float = 0.05
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference dataset (training data)
            reference_distributions: Pre-computed reference distributions
            psi_threshold: PSI threshold for drift (0.2 = minor, 0.25 = major)
            ks_threshold: KS test p-value threshold
            chi_square_threshold: Chi-square test p-value threshold
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.chi_square_threshold = chi_square_threshold
        
        if reference_distributions:
            self.reference_distributions = reference_distributions
        elif reference_data is not None:
            self.reference_distributions = self._compute_distributions(reference_data)
        else:
            self.reference_distributions = {}
    
    def _compute_distributions(self, data: np.ndarray, bins: int = 10) -> Dict[str, Any]:
        """
        Compute distribution statistics for reference data.
        
        Args:
            data: Input data array
            bins: Number of bins for histogram
            
        Returns:
            Dictionary with distribution statistics
        """
        if len(data) == 0:
            return {}
        
        # Remove NaN and infinite values
        clean_data = data[np.isfinite(data)]
        
        if len(clean_data) == 0:
            return {}
        
        # Compute histogram
        hist, bin_edges = np.histogram(clean_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize to get probabilities
        hist_prob = hist / hist.sum() if hist.sum() > 0 else hist
        
        return {
            "histogram": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bin_centers": bin_centers.tolist(),
            "probabilities": hist_prob.tolist(),
            "mean": float(np.mean(clean_data)),
            "std": float(np.std(clean_data)),
            "min": float(np.min(clean_data)),
            "max": float(np.max(clean_data)),
            "median": float(np.median(clean_data)),
            "count": int(len(clean_data))
        }
    
    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures how much a distribution has shifted:
        - PSI < 0.1: No significant population change
        - 0.1 <= PSI < 0.2: Minor population change
        - PSI >= 0.2: Significant population change (drift detected)
        
        Args:
            reference: Reference distribution (training data)
            current: Current distribution (production data)
            bins: Number of bins for histogram
            
        Returns:
            PSI value
        """
        try:
            # Clean data
            ref_clean = reference[np.isfinite(reference)]
            curr_clean = current[np.isfinite(current)]
            
            if len(ref_clean) == 0 or len(curr_clean) == 0:
                return 0.0
            
            # Use same bin edges for both distributions
            all_values = np.concatenate([ref_clean, curr_clean])
            bin_edges = np.linspace(all_values.min(), all_values.max(), bins + 1)
            
            # Compute histograms
            ref_hist, _ = np.histogram(ref_clean, bins=bin_edges)
            curr_hist, _ = np.histogram(curr_clean, bins=bin_edges)
            
            # Normalize to probabilities
            ref_prob = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
            curr_prob = curr_hist / curr_hist.sum() if curr_hist.sum() > 0 else curr_hist
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-6
            ref_prob = ref_prob + epsilon
            curr_prob = curr_prob + epsilon
            
            # Normalize again after adding epsilon
            ref_prob = ref_prob / ref_prob.sum()
            curr_prob = curr_prob / curr_prob.sum()
            
            # Calculate PSI
            psi = np.sum((curr_prob - ref_prob) * np.log(curr_prob / ref_prob))
            
            return float(psi)
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}", exc_info=True)
            return 0.0
    
    def calculate_ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov test statistic and p-value.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            
        Returns:
            Tuple of (KS statistic, p-value)
        """
        try:
            ref_clean = reference[np.isfinite(reference)]
            curr_clean = current[np.isfinite(current)]
            
            if len(ref_clean) == 0 or len(curr_clean) == 0:
                return (0.0, 1.0)
            
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(ref_clean, curr_clean)
            
            return (float(ks_statistic), float(p_value))
            
        except Exception as e:
            logger.error(f"Error calculating KS test: {e}", exc_info=True)
            return (0.0, 1.0)
    
    def calculate_chi_square(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> Tuple[float, float]:
        """
        Calculate Chi-square test statistic and p-value.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for histogram
            
        Returns:
            Tuple of (chi-square statistic, p-value)
        """
        try:
            ref_clean = reference[np.isfinite(reference)]
            curr_clean = current[np.isfinite(current)]
            
            if len(ref_clean) == 0 or len(curr_clean) == 0:
                return (0.0, 1.0)
            
            # Use same bin edges
            all_values = np.concatenate([ref_clean, curr_clean])
            bin_edges = np.linspace(all_values.min(), all_values.max(), bins + 1)
            
            # Compute histograms
            ref_hist, _ = np.histogram(ref_clean, bins=bin_edges)
            curr_hist, _ = np.histogram(curr_clean, bins=bin_edges)
            
            # Normalize to expected frequencies
            ref_total = ref_hist.sum()
            curr_total = curr_hist.sum()
            total = ref_total + curr_total
            
            if total == 0:
                return (0.0, 1.0)
            
            # Expected frequencies (assuming same distribution)
            expected_ref = ref_hist * (ref_total / total)
            expected_curr = curr_hist * (curr_total / total)
            
            # Avoid division by zero
            epsilon = 1e-6
            expected_ref = expected_ref + epsilon
            expected_curr = expected_curr + epsilon
            
            # Calculate chi-square
            chi_square = np.sum(
                ((ref_hist - expected_ref) ** 2) / expected_ref +
                ((curr_hist - expected_curr) ** 2) / expected_curr
            )
            
            # Degrees of freedom
            df = bins - 1
            
            # Calculate p-value
            p_value = 1 - stats.chi2.cdf(chi_square, df)
            
            return (float(chi_square), float(p_value))
            
        except Exception as e:
            logger.error(f"Error calculating chi-square: {e}", exc_info=True)
            return (0.0, 1.0)
    
    def detect_drift(
        self,
        feature_name: str,
        current_data: np.ndarray,
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect drift for a single feature.
        
        Args:
            feature_name: Name of the feature
            current_data: Current feature values
            model_version: Model version identifier
            
        Returns:
            Dictionary with drift detection results
        """
        if feature_name not in self.reference_distributions:
            logger.warning(f"No reference distribution for feature: {feature_name}")
            return {
                "feature_name": feature_name,
                "drift_detected": False,
                "psi": 0.0,
                "ks_statistic": 0.0,
                "ks_p_value": 1.0,
                "chi_square": 0.0,
                "chi_square_p_value": 1.0,
                "drift_severity": "none",
                "message": "No reference distribution available"
            }
        
        # Get reference distribution
        ref_dist = self.reference_distributions[feature_name]
        ref_data = np.array(ref_dist.get("histogram", []))
        
        # Calculate metrics
        psi = self.calculate_psi(
            np.array(ref_dist.get("probabilities", [])),
            current_data,
            bins=len(ref_dist.get("bin_edges", [])) - 1
        )
        
        # For KS and Chi-square, we need actual data
        # Use bin centers weighted by probabilities as proxy
        ref_bin_centers = np.array(ref_dist.get("bin_centers", []))
        ref_probs = np.array(ref_dist.get("probabilities", []))
        
        # Create synthetic reference data from distribution
        if len(ref_bin_centers) > 0 and len(ref_probs) > 0:
            # Sample from reference distribution
            ref_sample_size = min(len(current_data), 1000)
            ref_synthetic = np.random.choice(
                ref_bin_centers,
                size=ref_sample_size,
                p=ref_probs / ref_probs.sum()
            )
            
            ks_stat, ks_p_value = self.calculate_ks_test(ref_synthetic, current_data)
            chi_sq, chi_p_value = self.calculate_chi_square(ref_synthetic, current_data)
        else:
            ks_stat, ks_p_value = (0.0, 1.0)
            chi_sq, chi_p_value = (0.0, 1.0)
        
        # Determine drift severity
        drift_detected = False
        drift_severity = "none"
        
        if psi >= 0.25:
            drift_detected = True
            drift_severity = "major"
        elif psi >= self.psi_threshold:
            drift_detected = True
            drift_severity = "minor"
        elif ks_p_value < self.ks_threshold:
            drift_detected = True
            drift_severity = "minor"
        elif chi_p_value < self.chi_square_threshold:
            drift_detected = True
            drift_severity = "minor"
        
        # Compute current distribution
        current_dist = self._compute_distributions(current_data)
        
        return {
            "feature_name": feature_name,
            "drift_detected": drift_detected,
            "psi": psi,
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p_value,
            "chi_square": chi_sq,
            "chi_square_p_value": chi_p_value,
            "drift_severity": drift_severity,
            "reference_distribution": ref_dist,
            "current_distribution": current_dist,
            "model_version": model_version,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def detect_batch_drift(
        self,
        features: Dict[str, np.ndarray],
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect drift for multiple features.
        
        Args:
            features: Dictionary mapping feature names to data arrays
            model_version: Model version identifier
            
        Returns:
            Dictionary with drift results for all features
        """
        results = {}
        drifted_features = []
        
        for feature_name, feature_data in features.items():
            result = self.detect_drift(feature_name, feature_data, model_version)
            results[feature_name] = result
            
            if result["drift_detected"]:
                drifted_features.append(feature_name)
        
        return {
            "overall_drift_detected": len(drifted_features) > 0,
            "drifted_features": drifted_features,
            "total_features": len(features),
            "drifted_count": len(drifted_features),
            "feature_results": results,
            "model_version": model_version,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class DriftMonitor:
    """
    Monitor and store drift metrics in database.
    """
    
    def __init__(self, detector: Optional[DriftDetector] = None):
        """
        Initialize drift monitor.
        
        Args:
            detector: DriftDetector instance
        """
        self.detector = detector or DriftDetector()
        self.logger = get_logger(__name__)
    
    def save_drift_metric(
        self,
        feature_name: str,
        psi: float,
        ks_statistic: Optional[float] = None,
        chi_square: Optional[float] = None,
        is_drifted: bool = False,
        drift_severity: str = "none",
        reference_distribution: Optional[Dict] = None,
        current_distribution: Optional[Dict] = None,
        model_version: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> DriftMetric:
        """
        Save drift metric to database.
        
        Args:
            feature_name: Feature name
            psi: PSI value
            ks_statistic: KS test statistic
            chi_square: Chi-square statistic
            is_drifted: Whether drift was detected
            drift_severity: Severity level
            reference_distribution: Reference distribution data
            current_distribution: Current distribution data
            model_version: Model version
            metadata: Additional metadata
            
        Returns:
            Created DriftMetric instance
        """
        try:
            from sqlalchemy import JSON
            
            with get_db_session() as session:
                drift_metric = DriftMetric(
                    time=datetime.now(timezone.utc),
                    feature_name=feature_name,
                    psi=Decimal(str(psi)),
                    ks_statistic=Decimal(str(ks_statistic)) if ks_statistic is not None else None,
                    chi_square=Decimal(str(chi_square)) if chi_square is not None else None,
                    is_drifted=is_drifted,
                    drift_severity=drift_severity,
                    reference_distribution=reference_distribution,
                    current_distribution=current_distribution,
                    model_version=model_version,
                    drift_metadata=metadata
                )
                
                session.add(drift_metric)
                session.commit()
                session.refresh(drift_metric)
                
                self.logger.info(
                    f"Saved drift metric for feature {feature_name}",
                    extra={
                        "feature_name": feature_name,
                        "psi": psi,
                        "is_drifted": is_drifted,
                        "drift_severity": drift_severity
                    }
                )
                
                return drift_metric
                
        except Exception as e:
            self.logger.error(f"Error saving drift metric: {e}", exc_info=True)
            raise DatabaseError(f"Failed to save drift metric: {str(e)}", original_error=e)
    
    def monitor_predictions(
        self,
        predictions: np.ndarray,
        reference_predictions: Optional[np.ndarray] = None,
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Monitor prediction distribution for drift.
        
        Args:
            predictions: Current prediction values
            reference_predictions: Reference prediction distribution
            model_version: Model version
            
        Returns:
            Drift detection results
        """
        if reference_predictions is None:
            # Use stored reference if available
            # For now, we'll need to load from database or use default
            logger.warning("No reference predictions provided")
            return {
                "drift_detected": False,
                "message": "No reference predictions available"
            }
        
        # Detect drift in prediction distribution
        psi = self.detector.calculate_psi(reference_predictions, predictions)
        ks_stat, ks_p_value = self.detector.calculate_ks_test(reference_predictions, predictions)
        
        drift_detected = psi >= self.detector.psi_threshold or ks_p_value < self.detector.ks_threshold
        drift_severity = "major" if psi >= 0.25 else ("minor" if drift_detected else "none")
        
        return {
            "drift_detected": drift_detected,
            "psi": psi,
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p_value,
            "drift_severity": drift_severity,
            "model_version": model_version,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
