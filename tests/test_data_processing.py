import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pytest
import pandas as pd

from src.data_processing import create_aggregated_features, DateTimeFeatureExtractor


def test_create_aggregated_features_basic():
    df = pd.DataFrame({
        'AccountId': [1, 1, 2, 2, 3],
        'Amount': [100, 200, 300, 400, 500],
        'Value': [10, 20, 30, 40, 50]
    })
    result = create_aggregated_features(df, id_col='AccountId', aggregate_cols=['Amount', 'Value'])

    # Use correct case for columns
    assert 'Amount_sum_accountid' in result.columns
    assert 'Value_mean_accountid' in result.columns
    # Add more assertions as needed


def test_create_aggregated_features_missing_id_col():
    df = pd.DataFrame({
        'WrongId': [1, 2, 3],
        'Amount': [100, 200, 300]
    })
    with pytest.raises(ValueError):
        create_aggregated_features(df, id_col='AccountId')

def test_datetime_feature_extractor():
    df = pd.DataFrame({
        'TransactionStartTime': [
            '2024-06-01 12:34:56',
            '2024-06-02 00:00:00',
            'invalid_date',
            None
        ],
        'OtherCol': [1, 2, 3, 4]
    })

    extractor = DateTimeFeatureExtractor(datetime_col='TransactionStartTime')
    result = extractor.fit_transform(df)

    # Should drop invalid and None rows
    assert result.shape[0] == 2

    # Check new columns exist
    for col in ['transaction_hour', 'transaction_day_of_week', 'transaction_day_of_month', 'transaction_month', 'transaction_year']:
        assert col in result.columns

    # The original datetime column should be dropped
    assert 'TransactionStartTime' not in result.columns

def test_datetime_feature_extractor_wrong_column():
    df = pd.DataFrame({'SomeOtherCol': [1, 2, 3]})
    extractor = DateTimeFeatureExtractor(datetime_col='TransactionStartTime')
    with pytest.raises(ValueError):
        extractor.fit_transform(df)
