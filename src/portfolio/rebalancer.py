"""
Portfolio Rebalancing Module
Implements automatic portfolio rebalancing strategies
"""
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger


class RebalancingStrategy(Enum):
    """Rebalancing strategies"""
    PERIODIC = "periodic"  # Rebalance on fixed schedule
    THRESHOLD = "threshold"  # Rebalance when weights deviate by threshold
    TOLERANCE_BAND = "tolerance_band"  # Rebalance when outside tolerance bands
    CALENDAR = "calendar"  # Rebalance on specific calendar dates
    VOLATILITY_BASED = "volatility_based"  # Rebalance based on volatility changes


class PortfolioRebalancer:
    """
    Automatic portfolio rebalancing engine
    """

    def __init__(
        self,
        target_weights: Dict[str, float],
        rebalancing_cost: float = 0.001  # 0.1% transaction cost
    ):
        """
        Initialize rebalancer

        Args:
            target_weights: Target portfolio weights {symbol: weight}
            rebalancing_cost: Transaction cost for rebalancing (as fraction)
        """
        self.target_weights = target_weights
        self.rebalancing_cost = rebalancing_cost
        self.last_rebalance_date: Optional[datetime] = None
        self.rebalancing_history: List[Dict] = []

    def calculate_current_weights(
        self,
        positions: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate current portfolio weights

        Args:
            positions: Current position sizes {symbol: size}
            current_prices: Current prices {symbol: price}

        Returns:
            Current weights {symbol: weight}
        """
        # Calculate position values
        total_value = 0
        position_values = {}

        for symbol in self.target_weights.keys():
            size = positions.get(symbol, 0)
            price = current_prices.get(symbol, 0)
            value = size * price
            position_values[symbol] = value
            total_value += value

        # Calculate weights
        if total_value == 0:
            return {symbol: 0 for symbol in self.target_weights.keys()}

        current_weights = {
            symbol: value / total_value
            for symbol, value in position_values.items()
        }

        return current_weights

    def calculate_drift(
        self,
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate weight drift from target

        Args:
            current_weights: Current portfolio weights

        Returns:
            Drift for each asset {symbol: drift}
        """
        drift = {}

        for symbol in self.target_weights.keys():
            target = self.target_weights.get(symbol, 0)
            current = current_weights.get(symbol, 0)
            drift[symbol] = current - target

        return drift

    def should_rebalance_periodic(
        self,
        current_date: datetime,
        rebalance_frequency_days: int
    ) -> bool:
        """
        Check if should rebalance based on time period

        Args:
            current_date: Current date
            rebalance_frequency_days: Days between rebalances

        Returns:
            True if should rebalance
        """
        if self.last_rebalance_date is None:
            return True

        days_since_rebalance = (current_date - self.last_rebalance_date).days

        return days_since_rebalance >= rebalance_frequency_days

    def should_rebalance_threshold(
        self,
        current_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if should rebalance based on drift threshold

        Args:
            current_weights: Current portfolio weights
            threshold: Maximum allowed drift (e.g., 0.05 = 5%)

        Returns:
            (should_rebalance, drift_dict)
        """
        drift = self.calculate_drift(current_weights)

        # Check if any asset exceeds threshold
        max_drift = max(abs(d) for d in drift.values())

        should_rebalance = max_drift > threshold

        return should_rebalance, drift

    def should_rebalance_tolerance_band(
        self,
        current_weights: Dict[str, float],
        tolerance: float = 0.2
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if should rebalance using tolerance bands

        Args:
            current_weights: Current portfolio weights
            tolerance: Tolerance as fraction of target (e.g., 0.2 = 20%)

        Returns:
            (should_rebalance, drift_dict)
        """
        drift = self.calculate_drift(current_weights)

        should_rebalance = False

        for symbol, target_weight in self.target_weights.items():
            current_weight = current_weights.get(symbol, 0)

            # Calculate tolerance band
            lower_band = target_weight * (1 - tolerance)
            upper_band = target_weight * (1 + tolerance)

            # Check if outside bands
            if current_weight < lower_band or current_weight > upper_band:
                should_rebalance = True
                break

        return should_rebalance, drift

    def should_rebalance_volatility(
        self,
        returns: pd.DataFrame,
        current_weights: Dict[str, float],
        volatility_threshold: float = 1.5,
        lookback_days: int = 30
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if should rebalance based on volatility changes

        Args:
            returns: Historical returns DataFrame
            current_weights: Current portfolio weights
            volatility_threshold: Multiplier for volatility change
            lookback_days: Days to lookback for volatility calculation

        Returns:
            (should_rebalance, volatility_ratios)
        """
        if len(returns) < lookback_days * 2:
            return False, {}

        # Calculate recent and historical volatility
        recent_returns = returns.tail(lookback_days)
        historical_returns = returns.iloc[-lookback_days*2:-lookback_days]

        recent_vol = recent_returns.std()
        historical_vol = historical_returns.std()

        volatility_ratios = {}
        should_rebalance = False

        for symbol in self.target_weights.keys():
            if symbol in recent_vol.index and symbol in historical_vol.index:
                ratio = recent_vol[symbol] / historical_vol[symbol]
                volatility_ratios[symbol] = ratio

                # Check if volatility changed significantly
                if ratio > volatility_threshold or ratio < (1 / volatility_threshold):
                    should_rebalance = True

        return should_rebalance, volatility_ratios

    def calculate_rebalance_trades(
        self,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        total_portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance

        Args:
            current_positions: Current positions {symbol: size}
            current_prices: Current prices {symbol: price}
            total_portfolio_value: Total portfolio value

        Returns:
            Trades to execute {symbol: size_change}
            Positive = buy, Negative = sell
        """
        trades = {}

        for symbol, target_weight in self.target_weights.items():
            # Calculate target value
            target_value = total_portfolio_value * target_weight

            # Calculate current value
            current_size = current_positions.get(symbol, 0)
            current_price = current_prices.get(symbol, 0)
            current_value = current_size * current_price

            # Calculate value difference
            value_diff = target_value - current_value

            # Convert to size change
            if current_price > 0:
                size_change = value_diff / current_price
                trades[symbol] = size_change
            else:
                trades[symbol] = 0

        return trades

    def estimate_rebalancing_cost(
        self,
        trades: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> float:
        """
        Estimate cost of rebalancing

        Args:
            trades: Trades to execute {symbol: size_change}
            current_prices: Current prices {symbol: price}

        Returns:
            Total rebalancing cost
        """
        total_cost = 0

        for symbol, size_change in trades.items():
            price = current_prices.get(symbol, 0)
            trade_value = abs(size_change * price)
            cost = trade_value * self.rebalancing_cost
            total_cost += cost

        return total_cost

    def rebalance(
        self,
        current_date: datetime,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        strategy: RebalancingStrategy,
        **kwargs
    ) -> Optional[Dict[str, float]]:
        """
        Execute rebalancing logic

        Args:
            current_date: Current date
            current_positions: Current positions {symbol: size}
            current_prices: Current prices {symbol: price}
            strategy: Rebalancing strategy to use
            **kwargs: Strategy-specific parameters

        Returns:
            Trades to execute {symbol: size_change} or None if no rebalancing needed

        Example:
            trades = rebalancer.rebalance(
                current_date=datetime.now(),
                current_positions={'BTC/USD': 0.5, 'ETH/USD': 10},
                current_prices={'BTC/USD': 50000, 'ETH/USD': 3000},
                strategy=RebalancingStrategy.THRESHOLD,
                threshold=0.05
            )
        """
        # Calculate current weights
        current_weights = self.calculate_current_weights(
            current_positions,
            current_prices
        )

        # Calculate total portfolio value
        total_value = sum(
            current_positions.get(s, 0) * current_prices.get(s, 0)
            for s in self.target_weights.keys()
        )

        # Check if should rebalance based on strategy
        should_rebalance = False
        rebalance_reason = ""
        metadata = {}

        if strategy == RebalancingStrategy.PERIODIC:
            frequency_days = kwargs.get('rebalance_frequency_days', 30)
            should_rebalance = self.should_rebalance_periodic(
                current_date,
                frequency_days
            )
            rebalance_reason = f"Periodic ({frequency_days} days)"

        elif strategy == RebalancingStrategy.THRESHOLD:
            threshold = kwargs.get('threshold', 0.05)
            should_rebalance, drift = self.should_rebalance_threshold(
                current_weights,
                threshold
            )
            rebalance_reason = f"Threshold ({threshold*100}%)"
            metadata['drift'] = drift

        elif strategy == RebalancingStrategy.TOLERANCE_BAND:
            tolerance = kwargs.get('tolerance', 0.2)
            should_rebalance, drift = self.should_rebalance_tolerance_band(
                current_weights,
                tolerance
            )
            rebalance_reason = f"Tolerance Band ({tolerance*100}%)"
            metadata['drift'] = drift

        elif strategy == RebalancingStrategy.VOLATILITY_BASED:
            returns = kwargs.get('returns')
            volatility_threshold = kwargs.get('volatility_threshold', 1.5)

            if returns is not None:
                should_rebalance, vol_ratios = self.should_rebalance_volatility(
                    returns,
                    current_weights,
                    volatility_threshold
                )
                rebalance_reason = "Volatility Change"
                metadata['volatility_ratios'] = vol_ratios
            else:
                logger.warning("Returns required for volatility-based rebalancing")
                return None

        else:
            raise ValueError(f"Unknown rebalancing strategy: {strategy}")

        # If should not rebalance, return None
        if not should_rebalance:
            logger.info("Rebalancing not needed")
            return None

        # Calculate rebalancing trades
        trades = self.calculate_rebalance_trades(
            current_positions,
            current_prices,
            total_value
        )

        # Estimate cost
        rebalancing_cost = self.estimate_rebalancing_cost(trades, current_prices)
        cost_pct = (rebalancing_cost / total_value * 100) if total_value > 0 else 0

        logger.info(
            f"Rebalancing triggered: {rebalance_reason}\n"
            f"Estimated cost: ${rebalancing_cost:.2f} ({cost_pct:.3f}%)"
        )

        # Record rebalancing
        self.last_rebalance_date = current_date
        self.rebalancing_history.append({
            'date': current_date,
            'reason': rebalance_reason,
            'current_weights': current_weights.copy(),
            'target_weights': self.target_weights.copy(),
            'trades': trades.copy(),
            'cost': rebalancing_cost,
            'cost_pct': cost_pct,
            'metadata': metadata
        })

        return trades

    def update_target_weights(
        self,
        new_target_weights: Dict[str, float]
    ) -> None:
        """
        Update target portfolio weights

        Args:
            new_target_weights: New target weights
        """
        # Validate weights sum to 1
        total = sum(new_target_weights.values())
        if not np.isclose(total, 1.0, atol=0.01):
            logger.warning(f"Weights sum to {total}, normalizing to 1.0")
            new_target_weights = {
                k: v / total for k, v in new_target_weights.items()
            }

        self.target_weights = new_target_weights
        logger.info(f"Updated target weights: {self.target_weights}")

    def get_rebalancing_stats(self) -> Dict:
        """
        Get rebalancing statistics

        Returns:
            Statistics about rebalancing history
        """
        if not self.rebalancing_history:
            return {
                'total_rebalances': 0,
                'total_cost': 0,
                'avg_cost_pct': 0
            }

        total_rebalances = len(self.rebalancing_history)
        total_cost = sum(r['cost'] for r in self.rebalancing_history)
        avg_cost_pct = np.mean([r['cost_pct'] for r in self.rebalancing_history])

        # Calculate time between rebalances
        if total_rebalances > 1:
            dates = [r['date'] for r in self.rebalancing_history]
            time_diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            avg_days_between = np.mean(time_diffs)
        else:
            avg_days_between = 0

        return {
            'total_rebalances': total_rebalances,
            'total_cost': total_cost,
            'avg_cost_pct': avg_cost_pct,
            'avg_days_between_rebalances': avg_days_between,
            'last_rebalance_date': self.last_rebalance_date
        }

    def plot_rebalancing_history(self) -> None:
        """
        Visualize rebalancing history
        """
        if not self.rebalancing_history:
            logger.warning("No rebalancing history to plot")
            return

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style('darkgrid')
        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Rebalancing costs over time
        ax = axes[0, 0]
        dates = [r['date'] for r in self.rebalancing_history]
        costs = [r['cost_pct'] for r in self.rebalancing_history]

        ax.plot(dates, costs, marker='o', linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Rebalancing Cost %')
        ax.set_title('Rebalancing Costs Over Time')
        ax.grid(True, alpha=0.3)

        # 2. Weight drift heatmap (if drift data available)
        ax = axes[0, 1]
        drift_data = []

        for r in self.rebalancing_history:
            if 'drift' in r.get('metadata', {}):
                drift_data.append(r['metadata']['drift'])

        if drift_data:
            drift_df = pd.DataFrame(drift_data)
            sns.heatmap(drift_df.T, cmap='RdYlGn_r', center=0, ax=ax)
            ax.set_xlabel('Rebalance Event')
            ax.set_ylabel('Asset')
            ax.set_title('Weight Drift at Rebalancing')

        # 3. Cumulative rebalancing cost
        ax = axes[1, 0]
        cumulative_cost = np.cumsum([r['cost'] for r in self.rebalancing_history])

        ax.plot(dates, cumulative_cost, linewidth=2, color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Cost')
        ax.set_title('Cumulative Rebalancing Cost')
        ax.grid(True, alpha=0.3)

        # 4. Rebalancing frequency
        ax = axes[1, 1]

        if len(dates) > 1:
            time_diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            ax.hist(time_diffs, bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Days Between Rebalances')
            ax.set_ylabel('Frequency')
            ax.set_title('Rebalancing Frequency Distribution')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('rebalancing_history.png', dpi=300)
        logger.info("Saved plot to rebalancing_history.png")
        plt.close()
