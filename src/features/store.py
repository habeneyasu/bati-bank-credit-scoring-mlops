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
            try:
                processed_df = processor.fit_transform(df)
            except Exception as e:
                self.logger.warning(f"Feature processing error: {e}, using fallback", exc_info=True)
                processed_df = df.copy()
            
            # Calculate RFM features
            rfm_calc = RFMCalculator(
                customer_col='customer_id',
                datetime_col='transaction_start_time',
                amount_col='amount'
            )
            
            rfm_features = None
            try:
                rfm_df = rfm_calc.calculate_rfm(df)
                if 'recency_normalized' in rfm_df.columns:
                    rfm_features = rfm_df[rfm_df['customer_id'] == customer_id].iloc[0] if len(rfm_df[rfm_df['customer_id'] == customer_id]) > 0 else None
            except Exception as e:
                self.logger.warning(f"RFM calculation error: {e}", exc_info=True)
            
            # Extract feature vector from processed data
            feature_vector = None
            if len(processed_df) > 0:
                customer_row = processed_df[processed_df['customer_id'] == customer_id]
                if len(customer_row) > 0:
                    # Get numeric columns (excluding customer_id and target if present)
                    numeric_cols = customer_row.select_dtypes(include=[np.number]).columns.tolist()
                    numeric_cols = [c for c in numeric_cols if c != 'customer_id' and 'target' not in c.lower()]
                    
                    if numeric_cols:
                        feature_vector = customer_row[numeric_cols].iloc[0].values.tolist()
                        # Ensure we have exactly 26 features (pad or truncate if needed)
                        if len(feature_vector) < 26:
                            feature_vector.extend([0.0] * (26 - len(feature_vector)))
                        elif len(feature_vector) > 26:
                            feature_vector = feature_vector[:26]
            
            # If feature vector is still None, create a basic one from transaction stats
            if feature_vector is None:
                customer_stats = df.groupby('customer_id').agg({
                    'amount': ['sum', 'mean', 'count', 'std'],
                    'transaction_start_time': ['min', 'max']
                }).reset_index()
                
                if len(customer_stats) > 0:
                    stats = customer_stats.iloc[0]
                    # Create basic feature vector (will be padded/truncated to 26)
                    feature_vector = [
                        float(stats.get(('amount', 'sum'), 0)),
                        float(stats.get(('amount', 'mean'), 0)),
                        float(stats.get(('amount', 'count'), 0)),
                        float(stats.get(('amount', 'std'), 0) or 0),
                    ]
                    # Pad to 26 features
                    feature_vector.extend([0.0] * (26 - len(feature_vector)))
                else:
                    feature_vector = [0.0] * 26
            
            # Extract temporal features from first transaction
            first_txn = df.iloc[0]
            txn_time = first_txn['transaction_start_time']
            if isinstance(txn_time, pd.Timestamp):
                transaction_hour = float(txn_time.hour)
                transaction_day = float(txn_time.day)
                transaction_month = float(txn_time.month)
                transaction_year = float(txn_time.year)
                transaction_dayofweek = float(txn_time.dayofweek)
            else:
                transaction_hour = transaction_day = transaction_month = transaction_year = transaction_dayofweek = None
            
            # Prepare aggregate features
            aggregate_features = {
                "total_transactions": int(len(df)),
                "total_amount": float(df['amount'].sum()),
                "avg_amount": float(df['amount'].mean()),
                "min_amount": float(df['amount'].min()),
                "max_amount": float(df['amount'].max()),
            }
            
            # Prepare result
            result = {
                "customer_id": customer_id,
                "feature_vector": feature_vector,
                "recency_normalized": float(rfm_features['recency_normalized']) if rfm_features is not None and 'recency_normalized' in rfm_features else None,
                "frequency_normalized": float(rfm_features['frequency_normalized']) if rfm_features is not None and 'frequency_normalized' in rfm_features else None,
                "monetary_normalized": float(rfm_features['monetary_normalized']) if rfm_features is not None and 'monetary_normalized' in rfm_features else None,
                "transaction_hour": transaction_hour,
                "transaction_day": transaction_day,
                "transaction_month": transaction_month,
                "transaction_year": transaction_year,
                "transaction_dayofweek": transaction_dayofweek,
                "aggregate_features": aggregate_features,
                "categorical_features": {},
                "feature_version": feature_version or "v1.0",
                "data_version": data_version
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
