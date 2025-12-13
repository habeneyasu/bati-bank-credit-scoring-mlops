"""
Test script for feature engineering pipeline.

This script can be run directly to test the data processing functionality.
Usage: python test_feature_engineering.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import DataProcessor, TemporalFeatureExtractor, CustomerAggregator
from src.woe_calculator import calculate_woe_iv, calculate_iv_for_features


def test_temporal_feature_extraction():
    """Test temporal feature extraction."""
    print("\n" + "="*80)
    print("TEST 1: Temporal Feature Extraction")
    print("="*80)
    
    # Create sample data
    dates = pd.date_range('2018-11-15', periods=10, freq='2H')
    df = pd.DataFrame({
        'TransactionStartTime': dates,
        'Amount': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    })
    
    extractor = TemporalFeatureExtractor(datetime_col='TransactionStartTime')
    result = extractor.transform(df)
    
    print(f"Original columns: {list(df.columns)}")
    print(f"New columns: {[c for c in result.columns if c not in df.columns]}")
    print(f"\nSample temporal features:")
    print(result[['TransactionStartTime', 'transaction_hour', 'transaction_day', 
                  'transaction_month', 'transaction_year']].head())
    
    assert 'transaction_hour' in result.columns
    assert 'transaction_day' in result.columns
    assert 'transaction_month' in result.columns
    assert 'transaction_year' in result.columns
    print("✓ Temporal feature extraction test passed!")


def test_customer_aggregation():
    """Test customer aggregation."""
    print("\n" + "="*80)
    print("TEST 2: Customer Aggregation")
    print("="*80)
    
    # Create sample data
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C1', 'C2', 'C2'],
        'Amount': [100, 200, 300, 150, 250]
    })
    
    aggregator = CustomerAggregator(customer_col='CustomerId', amount_col='Amount')
    result = aggregator.transform(df)
    
    print(f"Original shape: {df.shape}")
    print(f"Result shape: {result.shape}")
    print(f"\nAggregate features created:")
    agg_features = [c for c in result.columns if c not in df.columns]
    print(agg_features)
    
    # Check C1 aggregations
    c1_data = result[result['CustomerId'] == 'C1'].iloc[0]
    print(f"\nCustomer C1 aggregations:")
    print(f"  Total: {c1_data['total_transaction_amount']} (expected: 600)")
    print(f"  Average: {c1_data['avg_transaction_amount']:.2f} (expected: 200.00)")
    print(f"  Count: {c1_data['transaction_count']} (expected: 3)")
    
    assert 'total_transaction_amount' in result.columns
    assert 'avg_transaction_amount' in result.columns
    assert 'transaction_count' in result.columns
    assert c1_data['total_transaction_amount'] == 600
    assert abs(c1_data['avg_transaction_amount'] - 200.0) < 0.01
    print("✓ Customer aggregation test passed!")


def test_full_pipeline():
    """Test the full data processing pipeline."""
    print("\n" + "="*80)
    print("TEST 3: Full Data Processing Pipeline")
    print("="*80)
    
    # Load actual data if available
    data_path = Path("data/raw/data.csv")
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Creating synthetic data for testing...")
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000
        
        dates = pd.date_range('2018-11-15', periods=n_samples, freq='1H')
        df = pd.DataFrame({
            'TransactionId': [f'T{i}' for i in range(n_samples)],
            'CustomerId': np.random.choice([f'C{i}' for i in range(100)], n_samples),
            'TransactionStartTime': dates[:n_samples],
            'Amount': np.random.normal(1000, 500, n_samples),
            'Value': np.random.normal(1200, 600, n_samples),
            'CurrencyCode': np.random.choice(['UGX', 'USD', 'EUR'], n_samples),
            'ProductCategory': np.random.choice(['airtime', 'financial_services', 'utility_bill'], n_samples),
            'PricingStrategy': np.random.choice([1, 2, 3, 4], n_samples),
            'FraudResult': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        })
    else:
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path, nrows=1000)  # Load first 1000 rows for testing
        print(f"Loaded {len(df)} rows")
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)[:10]}...")
    
    # Initialize processor
    processor = DataProcessor(
        customer_col='CustomerId',
        datetime_col='TransactionStartTime',
        amount_col='Amount',
        encoding_method='onehot',
        imputation_method='median',
        scaling_method='standardize',
        use_woe=False  # Disable WoE for initial test
    )
    
    # Process data
    print("\nProcessing data...")
    processed_df = processor.process_step_by_step(df)
    
    print(f"\nProcessed data shape: {processed_df.shape}")
    print(f"Number of features: {len(processed_df.columns)}")
    print(f"\nFirst 10 features: {list(processed_df.columns)[:10]}")
    print(f"\nLast 10 features: {list(processed_df.columns)[-10:]}")
    
    # Check for expected features
    expected_temporal = ['transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year']
    expected_aggregate = ['total_transaction_amount', 'avg_transaction_amount', 'transaction_count']
    
    temporal_found = [f for f in expected_temporal if f in processed_df.columns]
    aggregate_found = [f for f in expected_aggregate if f in processed_df.columns]
    
    print(f"\nTemporal features found: {temporal_found}")
    print(f"Aggregate features found: {aggregate_found}")
    
    # Basic validation
    assert processed_df.shape[0] > 0, "Processed dataframe is empty"
    assert processed_df.shape[1] > df.shape[1], "Should have more features after processing"
    assert len(temporal_found) > 0, "Should have temporal features"
    assert len(aggregate_found) > 0, "Should have aggregate features"
    
    print("\n✓ Full pipeline test passed!")
    
    return processed_df


def test_woe_calculation():
    """Test WoE and IV calculation."""
    print("\n" + "="*80)
    print("TEST 4: WoE and IV Calculation")
    print("="*80)
    
    # Create sample data with clear relationship
    np.random.seed(42)
    n_samples = 1000
    
    # Create feature with clear relationship to target
    feature = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.4, 0.3])
    # Make target depend on feature
    target = np.where(feature == 'A', 
                     np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                     np.random.choice([0, 1], n_samples, p=[0.5, 0.5]))
    
    feature_series = pd.Series(feature)
    target_series = pd.Series(target)
    
    # Calculate WoE and IV
    woe_df, iv = calculate_woe_iv(feature_series, target_series)
    
    print(f"Information Value (IV): {iv:.4f}")
    print(f"\nWoE DataFrame:")
    print(woe_df)
    
    assert iv >= 0, "IV should be non-negative"
    assert len(woe_df) > 0, "WoE dataframe should not be empty"
    
    print("\n✓ WoE calculation test passed!")


def test_different_configurations():
    """Test different processor configurations."""
    print("\n" + "="*80)
    print("TEST 5: Different Processor Configurations")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'CustomerId': np.random.choice(['C1', 'C2', 'C3'], 100),
        'TransactionStartTime': pd.date_range('2018-11-15', periods=100, freq='1H'),
        'Amount': np.random.normal(1000, 500, 100),
        'Category': np.random.choice(['A', 'B', 'C'], 100),
        'Value': np.random.normal(1200, 600, 100)
    })
    
    # Test different encoding methods
    configs = [
        {'encoding_method': 'onehot', 'scaling_method': 'standardize'},
        {'encoding_method': 'label', 'scaling_method': 'normalize'},
        {'encoding_method': 'onehot', 'scaling_method': 'robust'},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config}")
        processor = DataProcessor(
            customer_col='CustomerId',
            datetime_col='TransactionStartTime',
            amount_col='Amount',
            **config
        )
        
        result = processor.process_step_by_step(df)
        print(f"  Result shape: {result.shape}")
        print(f"  Features: {len(result.columns)}")
    
    print("\n✓ Configuration tests passed!")


def main():
    """Run all tests."""
    print("="*80)
    print("FEATURE ENGINEERING PIPELINE TESTS")
    print("="*80)
    
    try:
        # Run tests
        test_temporal_feature_extraction()
        test_customer_aggregation()
        processed_df = test_full_pipeline()
        test_woe_calculation()
        test_different_configurations()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        
        # Show sample of processed data
        if 'processed_df' in locals():
            print("\nSample of processed data:")
            print(processed_df.head())
            print(f"\nData types:")
            print(processed_df.dtypes.value_counts())
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

