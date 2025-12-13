"""
Data processing and feature engineering module for credit scoring.

This module provides a comprehensive feature engineering pipeline using sklearn Pipeline
to transform raw transaction data into model-ready features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import joblib
import warnings

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
    RobustScaler
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin

# WoE and IV libraries
try:
    from xverse.transformer import MonotonicBinning
    XVERSE_AVAILABLE = True
except ImportError:
    XVERSE_AVAILABLE = False

try:
    from woe import WoE
    WOE_LIB_AVAILABLE = True
except ImportError:
    WOE_LIB_AVAILABLE = False

# WoE is available if either library is available
WOE_AVAILABLE = XVERSE_AVAILABLE or WOE_LIB_AVAILABLE

if not WOE_AVAILABLE:
    warnings.warn("WoE libraries (xverse, woe) not available. WoE transformation will be skipped.")

warnings.filterwarnings('ignore')


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from datetime columns."""
    
    def __init__(self, datetime_col: str = 'TransactionStartTime'):
        self.datetime_col = datetime_col
        self.feature_names_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        if self.datetime_col in X.columns:
            # Convert to datetime if not already
            X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
            
            # Extract temporal features
            X['transaction_hour'] = X[self.datetime_col].dt.hour
            X['transaction_day'] = X[self.datetime_col].dt.day
            X['transaction_month'] = X[self.datetime_col].dt.month
            X['transaction_year'] = X[self.datetime_col].dt.year
            X['transaction_dayofweek'] = X[self.datetime_col].dt.dayofweek
            X['transaction_week'] = X[self.datetime_col].dt.isocalendar().week
            
            # Store feature names
            self.feature_names_ = [
                'transaction_hour', 'transaction_day', 'transaction_month',
                'transaction_year', 'transaction_dayofweek', 'transaction_week'
            ]
        
        return X


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Create aggregate features at customer level."""
    
    def __init__(self, customer_col: str = 'CustomerId', amount_col: str = 'Amount'):
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.aggregation_stats_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        if self.customer_col not in X.columns or self.amount_col not in X.columns:
            return X
        
        # Create aggregate features
        customer_stats = X.groupby(self.customer_col)[self.amount_col].agg([
            ('total_transaction_amount', 'sum'),
            ('avg_transaction_amount', 'mean'),
            ('transaction_count', 'count'),
            ('std_transaction_amount', 'std'),
            ('min_transaction_amount', 'min'),
            ('max_transaction_amount', 'max'),
            ('median_transaction_amount', 'median')
        ]).reset_index()
        
        # Fill NaN in std (for customers with single transaction)
        customer_stats['std_transaction_amount'] = customer_stats['std_transaction_amount'].fillna(0)
        
        # Merge back to original dataframe
        X = X.merge(customer_stats, on=self.customer_col, how='left')
        
        # Store aggregation stats
        self.aggregation_stats_ = customer_stats.set_index(self.customer_col).to_dict('index')
        
        return X


class WoETransformer(BaseEstimator, TransformerMixin):
    """Weight of Evidence (WoE) transformation for categorical and numerical features."""
    
    def __init__(self, target_col: Optional[str] = None, min_samples: int = 100):
        self.target_col = target_col
        self.min_samples = min_samples
        self.woe_transformers_ = {}
        self.feature_names_ = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not WOE_AVAILABLE:
            warnings.warn("WoE libraries not available. Skipping WoE transformation.")
            return self
        
        if y is None and self.target_col is None:
            warnings.warn("No target variable provided. Skipping WoE transformation.")
            return self
        
        # Get target variable
        if y is not None:
            target = y
        elif self.target_col in X.columns:
            target = X[self.target_col]
        else:
            warnings.warn("Target variable not found. Skipping WoE transformation.")
            return self
        
        # Select features for WoE transformation (categorical and numerical)
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        if self.target_col:
            categorical_cols = [c for c in categorical_cols if c != self.target_col]
            numerical_cols = [c for c in numerical_cols if c != self.target_col]
        
        id_cols = [c for c in X.columns if 'Id' in c or 'id' in c]
        categorical_cols = [c for c in categorical_cols if c not in id_cols]
        numerical_cols = [c for c in numerical_cols if c not in id_cols]
        
        # Apply WoE transformation
        try:
            if XVERSE_AVAILABLE:
                # Use xverse for monotonic binning and WoE
                for col in numerical_cols[:5]:  # Limit to first 5 to avoid memory issues
                    if X[col].nunique() > 10:  # Only for features with sufficient unique values
                        try:
                            woe_transformer = MonotonicBinning(col)
                            woe_transformer.fit(X[[col]], target)
                            self.woe_transformers_[col] = woe_transformer
                        except Exception as e:
                            warnings.warn(f"Could not fit WoE for {col}: {e}")
            else:
                warnings.warn("xverse not available. Using custom WoE calculator.")
        except Exception as e:
            warnings.warn(f"WoE transformation failed: {e}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        if not self.woe_transformers_:
            return X
        
        # Apply WoE transformations
        for col, transformer in self.woe_transformers_.items():
            if col in X.columns:
                try:
                    transformed = transformer.transform(X[[col]])
                    X[f'{col}_woe'] = transformed.iloc[:, 0] if len(transformed.columns) > 0 else transformed
                except Exception as e:
                    warnings.warn(f"Could not transform {col} with WoE: {e}")
        
        return X


class DataProcessor:
    """Main data processor class that orchestrates the feature engineering pipeline."""
    
    def __init__(
        self,
        customer_col: str = 'CustomerId',
        datetime_col: str = 'TransactionStartTime',
        amount_col: str = 'Amount',
        target_col: Optional[str] = None,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        encoding_method: str = 'onehot',  # 'onehot' or 'label'
        imputation_method: str = 'median',  # 'mean', 'median', 'mode', 'knn', 'remove'
        scaling_method: str = 'standardize',  # 'standardize', 'normalize', 'robust', None
        use_woe: bool = True
    ):
        """
        Initialize the data processor.
        
        Args:
            customer_col: Name of the customer ID column
            datetime_col: Name of the datetime column
            amount_col: Name of the transaction amount column
            target_col: Name of the target variable column (if present)
            categorical_cols: List of categorical column names (auto-detected if None)
            numerical_cols: List of numerical column names (auto-detected if None)
            encoding_method: Method for encoding categorical variables ('onehot' or 'label')
            imputation_method: Method for handling missing values
            scaling_method: Method for scaling numerical features
            use_woe: Whether to apply WoE transformation
        """
        self.customer_col = customer_col
        self.datetime_col = datetime_col
        self.amount_col = amount_col
        self.target_col = target_col
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.encoding_method = encoding_method
        self.imputation_method = imputation_method
        self.scaling_method = scaling_method
        self.use_woe = use_woe
        
        self.pipeline_ = None
        self.feature_names_ = None
    
    def _detect_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Auto-detect categorical and numerical columns."""
        # Exclude ID columns and target
        exclude_cols = [self.customer_col, self.datetime_col, self.target_col]
        exclude_cols.extend([c for c in df.columns if 'Id' in c or 'id' in c])
        
        if self.categorical_cols is None:
            categorical = df.select_dtypes(include=['object']).columns.tolist()
            categorical = [c for c in categorical if c not in exclude_cols]
        else:
            categorical = self.categorical_cols
        
        if self.numerical_cols is None:
            numerical = df.select_dtypes(include=[np.number]).columns.tolist()
            numerical = [c for c in numerical if c not in exclude_cols]
            # Remove target if it's numerical
            if self.target_col and self.target_col in numerical:
                numerical.remove(self.target_col)
        else:
            numerical = self.numerical_cols
        
        return categorical, numerical
    
    def _build_pipeline(self, categorical_cols: List[str], numerical_cols: List[str]):
        """Build the sklearn Pipeline for feature engineering."""
        
        # Step 1: Temporal feature extraction
        temporal_transformer = TemporalFeatureExtractor(datetime_col=self.datetime_col)
        
        # Step 2: Customer aggregation
        aggregator = CustomerAggregator(
            customer_col=self.customer_col,
            amount_col=self.amount_col
        )
        
        # Step 3: Missing value imputation
        if self.imputation_method == 'remove':
            # Will be handled separately
            numerical_imputer = None
            categorical_imputer = None
        elif self.imputation_method == 'knn':
            numerical_imputer = KNNImputer(n_neighbors=5)
            categorical_imputer = SimpleImputer(strategy='most_frequent')
        else:
            strategy_map = {
                'mean': 'mean',
                'median': 'median',
                'mode': 'most_frequent'
            }
            strategy = strategy_map.get(self.imputation_method, 'median')
            numerical_imputer = SimpleImputer(strategy=strategy)
            categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        # Step 4: Categorical encoding
        if self.encoding_method == 'onehot':
            categorical_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        else:  # label encoding
            categorical_encoder = LabelEncoder()
        
        # Step 5: Numerical scaling
        if self.scaling_method == 'standardize':
            numerical_scaler = StandardScaler()
        elif self.scaling_method == 'normalize':
            numerical_scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            numerical_scaler = RobustScaler()
        else:
            numerical_scaler = None
        
        # Build column transformer for preprocessing
        transformers = []
        
        if categorical_cols and categorical_imputer:
            transformers.append(('cat_impute', categorical_imputer, categorical_cols))
        
        if numerical_cols and numerical_imputer:
            transformers.append(('num_impute', numerical_imputer, numerical_cols))
        
        if categorical_cols and categorical_encoder:
            if self.encoding_method == 'onehot':
                transformers.append(('cat_encode', categorical_encoder, categorical_cols))
            # Label encoding handled differently
        
        if numerical_cols and numerical_scaler:
            transformers.append(('num_scale', numerical_scaler, numerical_cols))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        ) if transformers else None
        
        # Build final pipeline
        pipeline_steps = []
        
        # Add temporal extraction
        pipeline_steps.append(('temporal', temporal_transformer))
        
        # Add aggregation
        pipeline_steps.append(('aggregate', aggregator))
        
        # Add preprocessing
        if preprocessor:
            pipeline_steps.append(('preprocess', preprocessor))
        
        # Add WoE transformation (if enabled and target available)
        if self.use_woe and WOE_AVAILABLE:
            woe_transformer = WoETransformer(target_col=self.target_col)
            pipeline_steps.append(('woe', woe_transformer))
        
        self.pipeline_ = Pipeline(pipeline_steps)
        
        return self.pipeline_
    
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None):
        """
        Fit the data processing pipeline.
        
        Args:
            df: Input DataFrame
            target: Target variable (optional)
        """
        df = df.copy()
        
        # Detect column types
        categorical_cols, numerical_cols = self._detect_column_types(df)
        
        # Build pipeline
        pipeline = self._build_pipeline(categorical_cols, numerical_cols)
        
        # Fit pipeline
        if target is not None:
            pipeline.fit(df, target)
        else:
            pipeline.fit(df)
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        df = df.copy()
        
        # Handle missing value removal if specified
        if self.imputation_method == 'remove':
            # Remove rows with missing values
            df = df.dropna()
        
        # Transform using pipeline
        if self.pipeline_ is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Transform
        transformed = self.pipeline_.transform(df)
        
        # Convert to DataFrame if needed
        if isinstance(transformed, np.ndarray):
            # Get feature names from pipeline
            feature_names = self._get_feature_names()
            df_transformed = pd.DataFrame(transformed, columns=feature_names, index=df.index)
        else:
            df_transformed = transformed
        
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, target)
        return self.transform(df)
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names from the pipeline."""
        # This is a simplified version - in practice, you'd extract from each transformer
        # For now, return a placeholder that will be updated
        return self.feature_names_ if self.feature_names_ else []
    
    def process_step_by_step(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Process data step by step (alternative to Pipeline approach).
        More reliable and easier to debug.
        
        Args:
            df: Input DataFrame
            target: Target variable (optional)
        
        Returns:
            Processed DataFrame
        """
        df = df.copy()
        
        print("Step 1: Extracting temporal features...")
        temporal_extractor = TemporalFeatureExtractor(datetime_col=self.datetime_col)
        df = temporal_extractor.transform(df)
        
        print("Step 2: Creating customer aggregate features...")
        aggregator = CustomerAggregator(
            customer_col=self.customer_col,
            amount_col=self.amount_col
        )
        df = aggregator.transform(df)
        
        # Detect column types
        categorical_cols, numerical_cols = self._detect_column_types(df)
        
        print(f"Step 3: Handling missing values ({self.imputation_method})...")
        if self.imputation_method == 'remove':
            df = df.dropna()
        else:
            # Impute numerical columns
            if numerical_cols:
                if self.imputation_method == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
                else:
                    strategy_map = {
                        'mean': 'mean',
                        'median': 'median',
                        'mode': 'most_frequent'
                    }
                    strategy = strategy_map.get(self.imputation_method, 'median')
                    imputer = SimpleImputer(strategy=strategy)
                    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
            
            # Impute categorical columns
            if categorical_cols:
                imputer = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
        
        print(f"Step 4: Encoding categorical variables ({self.encoding_method})...")
        if categorical_cols:
            if self.encoding_method == 'onehot':
                encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[categorical_cols])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=encoder.get_feature_names_out(categorical_cols),
                    index=df.index
                )
                # Drop original categorical columns and add encoded
                df = df.drop(columns=categorical_cols)
                df = pd.concat([df, encoded_df], axis=1)
            else:  # label encoding
                for col in categorical_cols:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
        
        print(f"Step 5: Scaling numerical features ({self.scaling_method})...")
        if numerical_cols and self.scaling_method:
            if self.scaling_method == 'standardize':
                scaler = StandardScaler()
            elif self.scaling_method == 'normalize':
                scaler = MinMaxScaler()
            elif self.scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = None
            
            if scaler:
                # Only scale original numerical columns (not aggregated ones)
                original_num_cols = [c for c in numerical_cols if c in df.columns]
                if original_num_cols:
                    df[original_num_cols] = scaler.fit_transform(df[original_num_cols])
        
        print("Step 6: Applying WoE transformation...")
        if self.use_woe and target is not None:
            try:
                from src.woe_calculator import calculate_iv_for_features
                
                # Select features for WoE (exclude IDs and target)
                woe_features = [c for c in df.columns 
                              if c not in [self.customer_col, self.target_col] 
                              and 'Id' not in c 
                              and 'id' not in c
                              and df[c].nunique() > 5][:10]  # Limit to 10 features
                
                if woe_features:
                    woe_transformer = WoETransformer(target_col=self.target_col)
                    woe_transformer.fit(df, target)
                    df = woe_transformer.transform(df)
            except Exception as e:
                warnings.warn(f"WoE transformation failed: {e}")
        
        # Remove ID columns
        id_cols = [c for c in df.columns if 'Id' in c or 'id' in c or c == self.datetime_col]
        df = df.drop(columns=[c for c in id_cols if c in df.columns])
        
        # Remove target if present
        if self.target_col and self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])
        
        self.feature_names_ = df.columns.tolist()
        
        return df
    
    def save_processor(self, file_path: str):
        """Save the processor to disk."""
        joblib.dump(self, file_path)
    
    @staticmethod
    def load_processor(file_path: str) -> 'DataProcessor':
        """Load a processor from disk."""
        return joblib.load(file_path)


    def process_step_by_step(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Process data step by step (alternative to Pipeline approach).
        More reliable and easier to debug.
        
        Args:
            df: Input DataFrame
            target: Target variable (optional)
        
        Returns:
            Processed DataFrame
        """
        df = df.copy()
        
        print("Step 1: Extracting temporal features...")
        temporal_extractor = TemporalFeatureExtractor(datetime_col=self.datetime_col)
        df = temporal_extractor.transform(df)
        
        print("Step 2: Creating customer aggregate features...")
        aggregator = CustomerAggregator(
            customer_col=self.customer_col,
            amount_col=self.amount_col
        )
        df = aggregator.transform(df)
        
        # Detect column types
        categorical_cols, numerical_cols = self._detect_column_types(df)
        
        print(f"Step 3: Handling missing values ({self.imputation_method})...")
        if self.imputation_method == 'remove':
            df = df.dropna()
        else:
            # Impute numerical columns
            if numerical_cols:
                if self.imputation_method == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
                else:
                    strategy_map = {
                        'mean': 'mean',
                        'median': 'median',
                        'mode': 'most_frequent'
                    }
                    strategy = strategy_map.get(self.imputation_method, 'median')
                    imputer = SimpleImputer(strategy=strategy)
                    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
            
            # Impute categorical columns
            if categorical_cols:
                imputer = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
        
        print(f"Step 4: Encoding categorical variables ({self.encoding_method})...")
        if categorical_cols:
            if self.encoding_method == 'onehot':
                encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[categorical_cols])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=encoder.get_feature_names_out(categorical_cols),
                    index=df.index
                )
                # Drop original categorical columns and add encoded
                df = df.drop(columns=categorical_cols)
                df = pd.concat([df, encoded_df], axis=1)
            else:  # label encoding
                for col in categorical_cols:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
        
        print(f"Step 5: Scaling numerical features ({self.scaling_method})...")
        if numerical_cols and self.scaling_method:
            if self.scaling_method == 'standardize':
                scaler = StandardScaler()
            elif self.scaling_method == 'normalize':
                scaler = MinMaxScaler()
            elif self.scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = None
            
            if scaler:
                # Only scale original numerical columns (not aggregated ones)
                original_num_cols = [c for c in numerical_cols if c in df.columns]
                if original_num_cols:
                    df[original_num_cols] = scaler.fit_transform(df[original_num_cols])
        
        print("Step 6: Applying WoE transformation...")
        if self.use_woe and target is not None:
            try:
                from src.woe_calculator import calculate_iv_for_features
                
                # Select features for WoE (exclude IDs and target)
                woe_features = [c for c in df.columns 
                              if c not in [self.customer_col, self.target_col] 
                              and 'Id' not in c 
                              and 'id' not in c
                              and df[c].nunique() > 5][:10]  # Limit to 10 features
                
                if woe_features:
                    woe_transformer = WoETransformer(target_col=self.target_col)
                    woe_transformer.fit(df, target)
                    df = woe_transformer.transform(df)
            except Exception as e:
                warnings.warn(f"WoE transformation failed: {e}")
        
        # Remove ID columns
        id_cols = [c for c in df.columns if 'Id' in c or 'id' in c or c == self.datetime_col]
        df = df.drop(columns=[c for c in id_cols if c in df.columns])
        
        # Remove target if present
        if self.target_col and self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])
        
        self.feature_names_ = df.columns.tolist()
        
        return df


def main():
    """Main function for running data processing."""
    from pathlib import Path
    
    # Example usage
    data_path = Path("data/raw/data.csv")
    output_path = Path("data/processed/processed_data.csv")
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Initialize processor
    processor = DataProcessor(
        customer_col='CustomerId',
        datetime_col='TransactionStartTime',
        amount_col='Amount',
        encoding_method='onehot',
        imputation_method='median',
        scaling_method='standardize',
        use_woe=False  # Set to True if target variable is available
    )
    
    # Process data using step-by-step method (more reliable)
    print("Processing data...")
    processed_df = processor.process_step_by_step(df)
    
    # Save processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Processed shape: {processed_df.shape}")
    print(f"Features: {list(processed_df.columns)[:10]}...")  # Show first 10 features


if __name__ == "__main__":
    main()
