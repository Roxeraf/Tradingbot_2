"""
Tests for Advanced Portfolio Management Module
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.portfolio.allocator import PortfolioAllocator, AllocationStrategy
from src.portfolio.rebalancer import PortfolioRebalancer, RebalancingStrategy


@pytest.fixture
def sample_returns():
    """Create sample returns data"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    returns = pd.DataFrame({
        'BTC/USD': np.random.normal(0.001, 0.02, len(dates)),
        'ETH/USD': np.random.normal(0.0012, 0.025, len(dates)),
        'SOL/USD': np.random.normal(0.0015, 0.03, len(dates))
    }, index=dates)

    return returns


def test_equal_weight_allocation():
    """Test equal weight allocation"""
    allocator = PortfolioAllocator()

    symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']
    weights = allocator.equal_weight(symbols)

    assert len(weights) == 3
    assert all(w == pytest.approx(1/3, abs=0.01) for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)


def test_risk_parity_allocation(sample_returns):
    """Test risk parity allocation"""
    allocator = PortfolioAllocator()

    symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']
    weights = allocator.risk_parity(symbols, sample_returns)

    assert len(weights) == 3
    assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
    assert all(w >= 0 for w in weights.values())


def test_allocation_strategy_enum():
    """Test allocation strategy enum"""
    assert AllocationStrategy.EQUAL_WEIGHT.value == 'equal_weight'
    assert AllocationStrategy.RISK_PARITY.value == 'risk_parity'
    assert AllocationStrategy.MAX_SHARPE.value == 'max_sharpe'


def test_portfolio_rebalancer():
    """Test portfolio rebalancer"""
    target_weights = {
        'BTC/USD': 0.4,
        'ETH/USD': 0.4,
        'SOL/USD': 0.2
    }

    rebalancer = PortfolioRebalancer(
        target_weights=target_weights,
        rebalancing_cost=0.001
    )

    current_positions = {
        'BTC/USD': 0.5,
        'ETH/USD': 5.0,
        'SOL/USD': 10.0
    }

    current_prices = {
        'BTC/USD': 50000,
        'ETH/USD': 3000,
        'SOL/USD': 100
    }

    # Calculate current weights
    current_weights = rebalancer.calculate_current_weights(
        current_positions,
        current_prices
    )

    assert len(current_weights) == 3
    assert sum(current_weights.values()) == pytest.approx(1.0, abs=0.01)


def test_rebalance_threshold():
    """Test threshold-based rebalancing"""
    target_weights = {
        'BTC/USD': 0.5,
        'ETH/USD': 0.5
    }

    rebalancer = PortfolioRebalancer(target_weights)

    # Current weights significantly different
    current_weights = {
        'BTC/USD': 0.7,
        'ETH/USD': 0.3
    }

    should_rebalance, drift = rebalancer.should_rebalance_threshold(
        current_weights,
        threshold=0.05
    )

    assert should_rebalance == True
    assert 'BTC/USD' in drift
    assert 'ETH/USD' in drift


def test_compare_allocations(sample_returns):
    """Test comparing multiple allocation strategies"""
    allocator = PortfolioAllocator()

    symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']

    comparison = allocator.compare_allocations(
        symbols=symbols,
        returns=sample_returns,
        strategies=[
            AllocationStrategy.EQUAL_WEIGHT,
            AllocationStrategy.RISK_PARITY
        ]
    )

    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 3  # 3 symbols
    assert comparison.shape[1] == 2  # 2 strategies
