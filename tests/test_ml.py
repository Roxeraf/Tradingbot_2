"""
Tests for Machine Learning Module
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.ml.feature_engineering import FeatureEngineer
from src.ml.model_trainer import MLModelTrainer, ModelType


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
    n = len(dates)

    data = pd.DataFrame({
        'open': 50000 + np.random.randn(n).cumsum() * 100,
        'high': 50000 + np.random.randn(n).cumsum() * 100 + 100,
        'low': 50000 + np.random.randn(n).cumsum() * 100 - 100,
        'close': 50000 + np.random.randn(n).cumsum() * 100,
        'volume': np.random.uniform(100, 1000, n)
    }, index=dates)

    # Ensure high/low are correct
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    return data


def test_feature_engineer_price_features(sample_ohlcv_data):
    """Test price feature engineering"""
    engineer = FeatureEngineer()

    features = engineer.create_price_features(
        sample_ohlcv_data,
        lookback_periods=[5, 10, 20]
    )

    assert 'return_5' in features.columns
    assert 'return_10' in features.columns
    assert 'momentum_5' in features.columns
    assert 'hl_spread' in features.columns


def test_feature_engineer_technical_features(sample_ohlcv_data):
    """Test technical feature engineering"""
    engineer = FeatureEngineer()

    features = engineer.create_technical_features(
        sample_ohlcv_data,
        lookback_periods=[10, 20]
    )

    assert 'sma_10' in features.columns
    assert 'ema_10' in features.columns
    assert 'rsi_14' in features.columns
    assert 'macd' in features.columns
    assert 'bb_upper_20' in features.columns


def test_feature_engineer_all_features(sample_ohlcv_data):
    """Test creating all features"""
    engineer = FeatureEngineer()

    features = engineer.create_all_features(sample_ohlcv_data)

    # Should have many features
    assert len(engineer.feature_names) > 20

    # Check some key features exist
    original_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in original_cols:
        assert col in features.columns


def test_create_target_variable(sample_ohlcv_data):
    """Test target variable creation"""
    engineer = FeatureEngineer()

    # Binary direction target
    data_with_target = engineer.create_target_variable(
        sample_ohlcv_data,
        target_type='direction',
        forward_periods=1
    )

    assert 'target' in data_with_target.columns
    assert 'forward_return' in data_with_target.columns

    # Target should be binary (0 or 1)
    unique_targets = data_with_target['target'].dropna().unique()
    assert set(unique_targets).issubset({0, 1})


def test_ml_model_trainer_random_forest(sample_ohlcv_data):
    """Test Random Forest model trainer"""
    # Prepare data with features and target
    engineer = FeatureEngineer()
    data = engineer.create_all_features(sample_ohlcv_data)
    data = engineer.create_target_variable(data, target_type='direction')

    # Drop NaN
    data = data.dropna()

    # Create trainer
    trainer = MLModelTrainer(
        model_type=ModelType.RANDOM_FOREST,
        task='classification',
        random_state=42
    )

    # Prepare data
    feature_cols = engineer.feature_names[:20]  # Use first 20 features
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        data=data,
        feature_columns=feature_cols,
        target_column='target',
        train_size=0.8,
        scale_features=True
    )

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Train model
    trainer.train(X_train, y_train, n_estimators=10, max_depth=5)

    assert trainer.model is not None

    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)

    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_model_type_enum():
    """Test model type enum"""
    assert ModelType.RANDOM_FOREST.value == 'random_forest'
    assert ModelType.XGBOOST.value == 'xgboost'
    assert ModelType.LSTM.value == 'lstm'


def test_feature_selection(sample_ohlcv_data):
    """Test feature selection"""
    engineer = FeatureEngineer()
    data = engineer.create_all_features(sample_ohlcv_data)
    data = engineer.create_target_variable(data, target_type='direction')
    data = data.dropna()

    # Select top features by correlation
    selected = engineer.select_features(
        data=data,
        target_column='target',
        method='correlation',
        top_n=10
    )

    assert len(selected) <= 10
    assert all(isinstance(f, str) for f in selected)
