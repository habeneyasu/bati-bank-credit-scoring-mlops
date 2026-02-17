"""
Feature Store Service

Provides centralized feature storage and retrieval for online serving.
Supports both online (real-time) and offline (batch) feature computation.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from src.utils.logging import get_logger
from src.database.connection import get_db_session
from src.database.repositories import CustomerFeatureRepository
from src.database.models import CustomerFeature
from src.features.processing import DataProcessor
from src.features.rfm import RFMCalculator

logger = get_logger(__name__)


class FeatureStore:
    """
    Feature Store for managing customer features.
    
    Provides:
    - Online feature serving (real-time retrieval)
    - Feature computation and storage
    - Batch feature operations
    - Feature versioning
    """
    
    def __init__(self, session=None):
        """
        Initialize feature store.
        
        Args:
            session: Optional database session (creates new if None)
        """
        self.session = session
        self.logger = get_logger(f"{__name__}.FeatureStore")
    
    def get_features(
        self,
        customer_id: str,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get features for a customer from feature store.
        
        Args:
            customer_id: Customer identifier
            use_cache: Whether to use cached features (default: True)
            
        Returns:
            Dictionary with feature data or None if not found
        """
        if not use_cache:
            return None
        
        try:
            with get_db_session() as session:
                repo = CustomerFeatureRepository(session)
                feature = repo.get_by_id(customer_id)
                
                if feature:
                    return {
                        "customer_id": feature.customer_id,
                        "feature_vector": [float(f) for f in feature.feature_vector] if feature.feature_vector else None,
                        "recency_normalized": float(feature.recency_normalized) if feature.recency_normalized else None,
                        "frequency_normalized": float(feature.frequency_normalized) if feature.frequency_normalized else None,
                        "monetary_normalized": float(feature.monetary_normalized) if feature.monetary_normalized else None,
                        "transaction_hour": float(feature.transaction_hour) if feature.transaction_hour else None,
                        "transaction_day": float(feature.transaction_day) if feature.transaction_day else None,
                        "transaction_month": float(feature.transaction_month) if feature.transaction_month else None,
                        "transaction_year": float(feature.transaction_year) if feature.transaction_year else None,
                        "transaction_dayofweek": float(feature.transaction_dayofweek) if feature.transaction_dayofweek else None,
                        "aggregate_features": feature.aggregate_features,
                        "categorical_features": feature.categorical_features,
                        "feature_version": feature.feature_version,
                        "data_version": feature.data_version,
                        "last_updated": feature.last_updated.isoformat() if feature.last_updated else None
                    }
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting features for customer {customer_id}: {e}", exc_info=True)
            return None
    
    def get_feature_vector(
        self,
        customer_id: str,
        use_cache: bool = True
    ) -> Optional[List[float]]:
        """
        Get feature vector for a customer (for model prediction).
        
        Args:
            customer_id: Customer identifier
            use_cache: Whether to use cached features
            
        Returns:
            List of feature values or None if not found
        """
        features = self.get_features(customer_id, use_cache)
        if features and features.get("feature_vector"):
            return features["feature_vector"]
        return None
    
    def compute_and_store_features(
        self,
        customer_id: str,
        transactions: List[Dict[str, Any]],
        feature_version: Optional[str] = None,
        data_version: Optional[str] = None,
        store_features: bool = True
    ) -> Dict[str, Any]:
        """
        Compute features from transactions and optionally store them.
        
        Args:
            customer_id: Customer identifier
            transactions: List of transaction dictionaries
            feature_version: Version of feature engineering pipeline
            data_version: Version of source data
            store_features: Whether to store features in feature store
            
        Returns:
            Dictionary with computed features
        """
        try:
            import pandas as pd
            from datetime import datetime, timezone
            
            if not transactions:
                raise ValueError("No transactions provided")
            
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            
            # Ensure required columns
            required_cols = ['customer_id', 'amount', 'transaction_start_time']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                # Try case-insensitive matching
                col_mapping = {}
                for col in required_cols:
                    for df_col in df.columns:
                        if df_col.lower() == col.lower() or df_col.lower().replace('_', '') == col.lower().replace('_', ''):
                            col_mapping[col] = df_col
                            break
                
                # Check if we found all required columns
                for col in required_cols:
                    if col not in col_mapping:
                        raise ValueError(f"Missing required column: {col}")
                
                # Rename columns
                df = df.rename(columns={v: k for k, v in col_mapping.items()})
            
            # Filter to this customer
            df = df[df['customer_id'] == customer_id].copy()
            
            if len(df) == 0:
                raise ValueError(f"No transactions found for customer {customer_id}")
            
            # Convert transaction_start_time to datetime
            if df['transaction_start_time'].dtype == 'object':
                df['transaction_start_time'] = pd.to_datetime(df['transaction_start_time'])
            
            # Ensure amount is numeric
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            
            # Initialize feature processor
            processor = DataProcessor(
                customer_col='customer_id',
                datetime_col='transaction_start_time',
                amount_col='amount'
            )
            
            # Process the data to generate features
            # Note: For online serving, we may not have a fitted processor.
            # The processor is mainly used for batch processing during training.
            # For online serving, we compute features manually (RFM, temporal, aggregate).
            processed_df = None
            try:
                # Try to transform if processor is already fitted
                if processor.pipeline_ is not None:
                    processed_df = processor.transform(df)
                else:
                    # If not fitted, fit_transform (but this may not work well on single customer)
                    processed_df = processor.fit_transform(df)
            except Exception as e:
                self.logger.warning(
                    f"Feature processing error: {e}, skipping processed features. "
                    f"Using manually computed features (RFM, temporal, aggregate) instead.",
                    exc_info=True
                )
                processed_df = None  # Will use fallback - manually computed features only
            
            # Calculate RFM features
            rfm_calc = RFMCalculator(
                customer_col='customer_id',
                datetime_col='transaction_start_time',
                amount_col='amount'
            )
            
            rfm_features = None
            rfm_values = {}
            try:
                rfm_df = rfm_calc.calculate_rfm(df)
                if len(rfm_df) > 0:
                    customer_rfm = rfm_df[rfm_df['customer_id'] == customer_id]
                    if len(customer_rfm) > 0:
                        rfm_features = customer_rfm.iloc[0]
                        # Get raw RFM values
                        rfm_values = {
                            'recency': float(rfm_features.get('recency', 0)),
                            'frequency': float(rfm_features.get('frequency', 0)),
                            'monetary': float(rfm_features.get('monetary', 0))
                        }
                        # Normalize RFM (if normalized columns exist, use them; otherwise normalize here)
                        # IMPORTANT: For online serving with single customer, we need to use fixed normalization ranges
                        # to avoid normalizing each customer to 1.0. Use reasonable global ranges.
                        
                        # Fixed normalization ranges (based on typical credit scoring data)
                        # These should match the ranges used during training
                        RECENCY_MAX = 365.0  # 1 year max recency
                        FREQUENCY_MAX = 1000.0  # Max transactions (matches normalize_count)
                        MONETARY_MAX = 100000.0  # Max monetary value (matches normalize_amount)
                        
                        if 'recency_normalized' in rfm_features:
                            rfm_values['recency_normalized'] = float(rfm_features.get('recency_normalized', 0))
                        else:
                            # Use fixed max for normalization (not per-customer max)
                            # For recency: lower is better, so we normalize and invert
                            # recency_normalized = 1 - (recency / RECENCY_MAX) clamped to [0, 1]
                            recency_norm = 1.0 - min(1.0, rfm_values['recency'] / RECENCY_MAX) if RECENCY_MAX > 0 else 0.0
                            rfm_values['recency_normalized'] = max(0.0, min(1.0, recency_norm))
                            self.logger.debug(
                                f"RFM normalization for {customer_id}: recency={rfm_values['recency']:.1f} days -> "
                                f"normalized={rfm_values['recency_normalized']:.4f} (using fixed max={RECENCY_MAX})"
                            )
                        
                        if 'frequency_normalized' in rfm_features:
                            rfm_values['frequency_normalized'] = float(rfm_features.get('frequency_normalized', 0))
                        else:
                            # Use fixed max for normalization (not per-customer max)
                            # For frequency: higher is better, so normalize directly
                            frequency_norm = rfm_values['frequency'] / FREQUENCY_MAX if FREQUENCY_MAX > 0 else 0.0
                            rfm_values['frequency_normalized'] = max(0.0, min(1.0, frequency_norm))
                            self.logger.debug(
                                f"RFM normalization for {customer_id}: frequency={rfm_values['frequency']:.0f} -> "
                                f"normalized={rfm_values['frequency_normalized']:.4f} (using fixed max={FREQUENCY_MAX})"
                            )
                        
                        if 'monetary_normalized' in rfm_features:
                            rfm_values['monetary_normalized'] = float(rfm_features.get('monetary_normalized', 0))
                        else:
                            # Use fixed max for normalization (not per-customer max)
                            # For monetary: higher is better, so normalize directly
                            monetary_norm = rfm_values['monetary'] / MONETARY_MAX if MONETARY_MAX > 0 else 0.0
                            rfm_values['monetary_normalized'] = max(0.0, min(1.0, monetary_norm))
                            self.logger.debug(
                                f"RFM normalization for {customer_id}: monetary={rfm_values['monetary']:.2f} -> "
                                f"normalized={rfm_values['monetary_normalized']:.4f} (using fixed max={MONETARY_MAX})"
                            )
            except Exception as e:
                self.logger.warning(f"RFM calculation error: {e}", exc_info=True)
                rfm_values = {'recency_normalized': 0.0, 'frequency_normalized': 0.0, 'monetary_normalized': 0.0}
            
            # Extract temporal features from transactions
            first_txn = df.iloc[0] if len(df) > 0 else None
            last_txn = df.iloc[-1] if len(df) > 0 else None
            
            temporal_features = {}
            if first_txn is not None:
                txn_time = first_txn['transaction_start_time']
                if isinstance(txn_time, pd.Timestamp):
                    temporal_features = {
                        'transaction_hour': float(txn_time.hour) / 23.0,  # Normalize to 0-1
                        'transaction_day': float(txn_time.day) / 31.0,  # Normalize to 0-1
                        'transaction_month': float(txn_time.month) / 12.0,  # Normalize to 0-1
                        'transaction_year': float(txn_time.year - 2019) / 10.0 if txn_time.year >= 2019 else 0.0,  # Normalize
                        'transaction_dayofweek': float(txn_time.dayofweek) / 6.0,  # Normalize to 0-1
                    }
                else:
                    temporal_features = {'transaction_hour': 0.0, 'transaction_day': 0.0, 'transaction_month': 0.0, 
                                       'transaction_year': 0.0, 'transaction_dayofweek': 0.0}
            else:
                temporal_features = {'transaction_hour': 0.0, 'transaction_day': 0.0, 'transaction_month': 0.0, 
                                   'transaction_year': 0.0, 'transaction_dayofweek': 0.0}
            
            # Helper function to clamp values to DECIMAL(10,6) range: -9999.999999 to 9999.999999
            def clamp_decimal(value, min_val=-9999.999999, max_val=9999.999999):
                """Clamp value to DECIMAL(10,6) range."""
                if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    return 0.0
                return max(min_val, min(max_val, float(value)))
            
            # Helper function to normalize large values (e.g., amounts) to 0-1 range
            def normalize_amount(value, max_amount=100000.0):
                """Normalize amount values to 0-1 range for feature vector."""
                if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    return 0.0
                normalized = float(value) / max_amount if max_amount > 0 else 0.0
                return clamp_decimal(normalized, 0.0, 1.0)
            
            # Helper function to normalize count values
            def normalize_count(value, max_count=1000.0):
                """Normalize count values to 0-1 range for feature vector."""
                if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    return 0.0
                normalized = float(value) / max_count if max_count > 0 else 0.0
                return clamp_decimal(normalized, 0.0, 1.0)
            
            # Calculate aggregate/statistical features from transactions
            # Note: These are normalized for the feature_vector (0-1 range), but raw values stored in metadata
            aggregate_features_list = []
            if len(df) > 0:
                # Get raw statistics for metadata
                total_amount = float(df['amount'].sum())
                avg_amount = float(df['amount'].mean())
                std_amount = float(df['amount'].std() if len(df) > 1 else 0.0)
                min_amount = float(df['amount'].min())
                max_amount = float(df['amount'].max())
                median_amount = float(df['amount'].median())
                transaction_count = float(len(df))
                
                # Normalize for feature vector (DECIMAL(10,6) compatible)
                aggregate_features_list.extend([
                    normalize_amount(total_amount),  # Normalized total amount
                    normalize_amount(avg_amount),  # Normalized average amount
                    normalize_amount(std_amount),  # Normalized std deviation
                    normalize_amount(min_amount),  # Normalized min amount
                    normalize_amount(max_amount),  # Normalized max amount
                    normalize_amount(median_amount),  # Normalized median amount
                    normalize_count(transaction_count),  # Normalized transaction count
                ])
                
                # Time-based features (normalized)
                if last_txn is not None and first_txn is not None:
                    time_diff = (pd.to_datetime(last_txn['transaction_start_time']) - 
                               pd.to_datetime(first_txn['transaction_start_time'])).days
                    aggregate_features_list.append(normalize_count(time_diff, max_count=365.0))  # Normalized days
                else:
                    aggregate_features_list.append(0.0)
                
                # Amount distribution features (normalized)
                if len(df) > 1:
                    q25 = float(df['amount'].quantile(0.25))
                    q75 = float(df['amount'].quantile(0.75))
                    iqr = q75 - q25
                    aggregate_features_list.extend([
                        normalize_amount(q25),
                        normalize_amount(q75),
                        normalize_amount(iqr)
                    ])
                else:
                    aggregate_features_list.extend([0.0, 0.0, 0.0])
                
                # Transaction frequency features (normalized)
                if len(df) > 1 and last_txn is not None and first_txn is not None:
                    time_span = (pd.to_datetime(last_txn['transaction_start_time']) - 
                                pd.to_datetime(first_txn['transaction_start_time'])).days
                    if time_span > 0:
                        transactions_per_day = len(df) / time_span
                        aggregate_features_list.append(clamp_decimal(transactions_per_day, 0.0, 100.0))  # Cap at 100
                    else:
                        aggregate_features_list.append(normalize_count(len(df)))
                else:
                    aggregate_features_list.append(normalize_count(len(df)))
            else:
                aggregate_features_list = [0.0] * 11
            
            # Extract processed features from DataProcessor output
            processed_features_list = []
            if processed_df is not None and len(processed_df) > 0:
                try:
                    customer_row = processed_df[processed_df['customer_id'] == customer_id]
                    if len(customer_row) > 0:
                        # Get numeric columns (excluding customer_id and target if present)
                        numeric_cols = customer_row.select_dtypes(include=[np.number]).columns.tolist()
                        numeric_cols = [c for c in numeric_cols if c not in ['customer_id'] and 'target' not in c.lower()]
                    
                        # Get processed features (limit to avoid too many)
                        if numeric_cols:
                            processed_values = customer_row[numeric_cols].iloc[0].values.tolist()
                            # Convert NumPy types to native Python types and clamp to DECIMAL(10,6)
                            processed_features_list = []
                            for v in processed_values[:7]:
                                if isinstance(v, (np.floating, np.integer, np.number)):
                                    val = float(v)
                                else:
                                    val = float(v) if isinstance(v, (int, float)) else 0.0
                                # Clamp to DECIMAL(10,6) range
                                if np.isnan(val) or np.isinf(val):
                                    processed_features_list.append(0.0)
                                else:
                                    processed_features_list.append(max(-9999.999999, min(9999.999999, val)))
                except Exception as e:
                    self.logger.warning(
                        f"Error extracting processed features for customer {customer_id}: {e}. "
                        f"Using zeros for processed features.",
                        exc_info=True
                    )
                    processed_features_list = []
            
            # If no processed features, fill with zeros (we have RFM, temporal, and aggregate features)
            if not processed_features_list:
                processed_features_list = [0.0] * 7
            
            # Combine all features into 26-feature vector
            # Order: RFM (3) + Temporal (5) + Aggregate (11) + Processed (7) = 26
            feature_vector = []
            
            # 1. RFM features (3) - clamp to ensure within DECIMAL(10,6) range
            rfm_recency = clamp_decimal(rfm_values.get('recency_normalized', 0.0), 0.0, 1.0)
            rfm_frequency = clamp_decimal(rfm_values.get('frequency_normalized', 0.0), 0.0, 1.0)
            rfm_monetary = clamp_decimal(rfm_values.get('monetary_normalized', 0.0), 0.0, 1.0)
            
            # Log RFM values for debugging
            self.logger.info(
                f"Feature vector construction for {customer_id}: "
                f"RFM=[recency={rfm_values.get('recency', 0):.1f}->{rfm_recency:.4f}, "
                f"frequency={rfm_values.get('frequency', 0):.0f}->{rfm_frequency:.4f}, "
                f"monetary={rfm_values.get('monetary', 0):.2f}->{rfm_monetary:.4f}], "
                f"transaction_count={len(df)}"
            )
            
            feature_vector.extend([rfm_recency, rfm_frequency, rfm_monetary])
            
            # 2. Temporal features (5) - clamp to ensure within DECIMAL(10,6) range
            feature_vector.extend([
                clamp_decimal(temporal_features.get('transaction_hour', 0.0), 0.0, 1.0),
                clamp_decimal(temporal_features.get('transaction_day', 0.0), 0.0, 1.0),
                clamp_decimal(temporal_features.get('transaction_month', 0.0), 0.0, 1.0),
                clamp_decimal(temporal_features.get('transaction_year', 0.0), 0.0, 1.0),
                clamp_decimal(temporal_features.get('transaction_dayofweek', 0.0), 0.0, 1.0)
            ])
            
            # 3. Aggregate features (11)
            feature_vector.extend(aggregate_features_list[:11])
            
            # 4. Processed features (7) - fill remaining slots
            feature_vector.extend(processed_features_list[:7])
            
            # Ensure exactly 26 features (pad if needed, truncate if too many)
            if len(feature_vector) < 26:
                feature_vector.extend([0.0] * (26 - len(feature_vector)))
            elif len(feature_vector) > 26:
                feature_vector = feature_vector[:26]
            
            # Convert all NumPy types to native Python types and clamp to DECIMAL(10,6) range
            # (clamp_decimal function already defined above)
            feature_vector = [clamp_decimal(v) for v in feature_vector]
            
            # Extract temporal features for metadata (non-normalized, but clamped to DECIMAL(10,6))
            def clamp_temporal(value):
                """Clamp temporal value to DECIMAL(10,6) range."""
                if value is None:
                    return None
                try:
                    val = float(value)
                    if np.isnan(val) or np.isinf(val) or val <= 0:
                        return None
                    clamped = max(-9999.999999, min(9999.999999, val))
                    return clamped if clamped > 0 else None
                except (ValueError, TypeError):
                    return None
            
            # Calculate temporal metadata values and clamp them
            hour_val = temporal_features.get('transaction_hour', 0.0) * 23.0 if temporal_features.get('transaction_hour', 0.0) > 0 else None
            day_val = temporal_features.get('transaction_day', 0.0) * 31.0 if temporal_features.get('transaction_day', 0.0) > 0 else None
            month_val = temporal_features.get('transaction_month', 0.0) * 12.0 if temporal_features.get('transaction_month', 0.0) > 0 else None
            year_val = 2019 + (temporal_features.get('transaction_year', 0.0) * 10.0) if temporal_features.get('transaction_year', 0.0) > 0 else None
            dayofweek_val = temporal_features.get('transaction_dayofweek', 0.0) * 6.0 if temporal_features.get('transaction_dayofweek', 0.0) > 0 else None
            
            transaction_hour = clamp_temporal(hour_val)
            transaction_day = clamp_temporal(day_val)
            transaction_month = clamp_temporal(month_val)
            transaction_year = clamp_temporal(year_val)
            transaction_dayofweek = clamp_temporal(dayofweek_val)
            
            # Prepare aggregate features for metadata
            aggregate_features = {
                "total_transactions": int(len(df)) if len(df) > 0 else 0,
                "total_amount": float(df['amount'].sum()) if len(df) > 0 else 0.0,
                "avg_amount": float(df['amount'].mean()) if len(df) > 0 else 0.0,
                "min_amount": float(df['amount'].min()) if len(df) > 0 else 0.0,
                "max_amount": float(df['amount'].max()) if len(df) > 0 else 0.0,
            }
            
            # Convert all numeric values to native Python types and clamp to DECIMAL(10,6) range
            def convert_to_native(value):
                """Convert NumPy/pandas types to native Python types and clamp to DECIMAL(10,6)."""
                if value is None:
                    return None
                if isinstance(value, (np.floating, np.float64, np.float32)):
                    val = float(value)
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    val = float(int(value))
                else:
                    try:
                        if pd.isna(value):
                            return None
                    except (TypeError, ValueError):
                        pass
                    val = float(value) if isinstance(value, (int, float)) else value
                
                # Clamp to DECIMAL(10,6) range: -9999.999999 to 9999.999999
                if isinstance(val, (int, float)):
                    if np.isnan(val) or np.isinf(val):
                        return 0.0
                    return max(-9999.999999, min(9999.999999, float(val)))
                return val
            
            # Prepare result with all values converted to native types
            result = {
                "customer_id": customer_id,
                "feature_vector": feature_vector,  # Already converted above
                "recency_normalized": convert_to_native(rfm_values.get('recency_normalized', 0.0)),
                "frequency_normalized": convert_to_native(rfm_values.get('frequency_normalized', 0.0)),
                "monetary_normalized": convert_to_native(rfm_values.get('monetary_normalized', 0.0)),
                "transaction_hour": convert_to_native(transaction_hour),
                "transaction_day": convert_to_native(transaction_day),
                "transaction_month": convert_to_native(transaction_month),
                "transaction_year": convert_to_native(transaction_year),
                "transaction_dayofweek": convert_to_native(transaction_dayofweek),
                "aggregate_features": aggregate_features,
                "categorical_features": {},
                "feature_version": feature_version or "v1.0",
                "data_version": data_version,
                "feature_breakdown": {
                    "rfm_features": 3,
                    "temporal_features": 5,
                    "aggregate_features": 11,
                    "processed_features": 7,
                    "total": len(feature_vector)
                }
            }
            
            # Store features if requested
            if store_features:
                self.store_features(
                    customer_id=customer_id,
                    feature_vector=feature_vector,
                    recency_normalized=result["recency_normalized"],
                    frequency_normalized=result["frequency_normalized"],
                    monetary_normalized=result["monetary_normalized"],
                    transaction_hour=transaction_hour,
                    transaction_day=transaction_day,
                    transaction_month=transaction_month,
                    transaction_year=transaction_year,
                    transaction_dayofweek=transaction_dayofweek,
                    aggregate_features=aggregate_features,
                    categorical_features={},
                    feature_version=result["feature_version"],
                    data_version=data_version
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error computing features for customer {customer_id}: {e}", exc_info=True)
            raise
    
    def store_features(
        self,
        customer_id: str,
        feature_vector: List[float],
        recency_normalized: Optional[float] = None,
        frequency_normalized: Optional[float] = None,
        monetary_normalized: Optional[float] = None,
        transaction_hour: Optional[float] = None,
        transaction_day: Optional[float] = None,
        transaction_month: Optional[float] = None,
        transaction_year: Optional[float] = None,
        transaction_dayofweek: Optional[float] = None,
        aggregate_features: Optional[Dict[str, Any]] = None,
        categorical_features: Optional[Dict[str, Any]] = None,
        feature_version: Optional[str] = None,
        data_version: Optional[str] = None
    ) -> CustomerFeature:
        """
        Store features in feature store.
        
        Args:
            customer_id: Customer identifier
            feature_vector: List of 26 feature values
            recency_normalized: RFM recency feature
            frequency_normalized: RFM frequency feature
            monetary_normalized: RFM monetary feature
            transaction_hour: Temporal hour feature
            transaction_day: Temporal day feature
            transaction_month: Temporal month feature
            transaction_year: Temporal year feature
            transaction_dayofweek: Temporal day of week feature
            aggregate_features: Dictionary of aggregate features
            categorical_features: Dictionary of categorical features
            feature_version: Version of feature engineering pipeline
            data_version: Version of source data
            
        Returns:
            Stored CustomerFeature instance
        """
        try:
            with get_db_session() as session:
                repo = CustomerFeatureRepository(session)
                feature = repo.upsert_feature(
                    customer_id=customer_id,
                    feature_vector=feature_vector,
                    recency_normalized=recency_normalized,
                    frequency_normalized=frequency_normalized,
                    monetary_normalized=monetary_normalized,
                    transaction_hour=transaction_hour,
                    transaction_day=transaction_day,
                    transaction_month=transaction_month,
                    transaction_year=transaction_year,
                    transaction_dayofweek=transaction_dayofweek,
                    aggregate_features=aggregate_features,
                    categorical_features=categorical_features,
                    feature_version=feature_version,
                    data_version=data_version
                )
                
                self.logger.info(f"Stored features for customer {customer_id}")
                return feature
                
        except Exception as e:
            self.logger.error(f"Error storing features for customer {customer_id}: {e}", exc_info=True)
            raise
    
    def batch_get_features(
        self,
        customer_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get features for multiple customers.
        
        Args:
            customer_ids: List of customer identifiers
            
        Returns:
            Dictionary mapping customer_id to feature data
        """
        try:
            with get_db_session() as session:
                repo = CustomerFeatureRepository(session)
                features = repo.batch_get_feature_vectors(customer_ids)
                
                result = {}
                for customer_id in customer_ids:
                    if customer_id in features:
                        feature = repo.get_by_id(customer_id)
                        if feature:
                            result[customer_id] = {
                                "customer_id": feature.customer_id,
                                "feature_vector": features[customer_id],
                                "recency_normalized": float(feature.recency_normalized) if feature.recency_normalized else None,
                                "frequency_normalized": float(feature.frequency_normalized) if feature.frequency_normalized else None,
                                "monetary_normalized": float(feature.monetary_normalized) if feature.monetary_normalized else None,
                                "last_updated": feature.last_updated.isoformat() if feature.last_updated else None,
                                "feature_version": feature.feature_version,
                                "data_version": feature.data_version
                            }
                    else:
                        result[customer_id] = None
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error batch getting features: {e}", exc_info=True)
            raise


def get_feature_store() -> FeatureStore:
    """Get a FeatureStore instance."""
    return FeatureStore()
