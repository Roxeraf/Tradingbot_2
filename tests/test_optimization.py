"""
Tests for Strategy Optimization Module
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.optimization.strategy_optimizer import StrategyOptimizer, OptimizationMethod
from src.optimization.walk_forward import WalkForwardOptimizer


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
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


def test_grid_search(sample_data):
    """Test grid search optimization"""
    optimizer = StrategyOptimizer(
        data=sample_data,
        strategy_name='ma_crossover',
        symbol='BTC/USD',
        optimization_metric='sharpe_ratio'
    )

    param_grid = {
        'fast_period': [10, 20],
        'slow_period': [30, 50]
    }

    results = optimizer.grid_search(param_grid)

    assert results['method'] == 'grid_search'
    assert 'best_params' in results
    assert 'best_metric_value' in results
    assert results['total_tests'] == 4  # 2 x 2 combinations


def test_random_search(sample_data):
    """Test random search optimization"""
    optimizer = StrategyOptimizer(
        data=sample_data,
        strategy_name='ma_crossover',
        symbol='BTC/USD',
        optimization_metric='sharpe_ratio'
    )

    param_distributions = {
        'fast_period': (5, 30, 'int'),
        'slow_period': (30, 100, 'int')
    }

    results = optimizer.random_search(param_distributions, n_iterations=10)

    assert results['method'] == 'random_search'
    assert 'best_params' in results
    assert results['total_tests'] == 10


def test_optimization_method_enum():
    """Test optimization method enum"""
    assert OptimizationMethod.GRID_SEARCH.value == 'grid_search'
    assert OptimizationMethod.RANDOM_SEARCH.value == 'random_search'
    assert OptimizationMethod.BAYESIAN.value == 'bayesian'


def test_walk_forward_optimizer(sample_data):
    """Test walk-forward optimization"""
    wf_optimizer = WalkForwardOptimizer(
        data=sample_data,
        strategy_name='ma_crossover',
        symbol='BTC/USD',
        optimization_metric='sharpe_ratio'
    )

    param_space = {
        'fast_period': (5, 30, 'int'),
        'slow_period': (30, 100, 'int')
    }

    results = wf_optimizer.optimize(
        param_space=param_space,
        train_period_days=30,
        test_period_days=10,
        optimization_method=OptimizationMethod.RANDOM_SEARCH,
        n_iterations=5
    )

    assert 'strategy_name' in results
    assert 'num_splits' in results
    assert 'summary' in results
    assert results['num_splits'] > 0
