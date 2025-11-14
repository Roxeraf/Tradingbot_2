"""
Advanced Portfolio Allocation Strategies
Implements various portfolio allocation methods
"""
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class AllocationStrategy(Enum):
    """Portfolio allocation strategies"""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    MAX_DIVERSIFICATION = "max_diversification"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"


class PortfolioAllocator:
    """
    Advanced portfolio allocation engine
    Calculates optimal asset weights using various strategies
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        max_position_size: float = 0.4,
        min_position_size: float = 0.01
    ):
        """
        Initialize portfolio allocator

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculations
            max_position_size: Maximum weight for any single asset
            min_position_size: Minimum weight for any single asset
        """
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size

    def equal_weight(
        self,
        symbols: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """
        Equal weight allocation

        Args:
            symbols: List of trading symbols

        Returns:
            Dict mapping symbols to weights
        """
        n = len(symbols)
        weight = 1.0 / n

        return {symbol: weight for symbol in symbols}

    def market_cap_weight(
        self,
        symbols: List[str],
        market_caps: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """
        Market cap weighted allocation

        Args:
            symbols: List of trading symbols
            market_caps: Dict mapping symbols to market capitalizations

        Returns:
            Dict mapping symbols to weights
        """
        total_market_cap = sum(market_caps.get(s, 0) for s in symbols)

        if total_market_cap == 0:
            logger.warning("Total market cap is zero, using equal weight")
            return self.equal_weight(symbols)

        weights = {}
        for symbol in symbols:
            weight = market_caps.get(symbol, 0) / total_market_cap
            weights[symbol] = weight

        return self._apply_constraints(weights)

    def risk_parity(
        self,
        symbols: List[str],
        returns: pd.DataFrame,
        **kwargs
    ) -> Dict[str, float]:
        """
        Risk Parity allocation
        Allocates based on inverse volatility (equal risk contribution)

        Args:
            symbols: List of trading symbols
            returns: DataFrame with returns for each symbol

        Returns:
            Dict mapping symbols to weights
        """
        # Calculate volatilities
        volatilities = returns[symbols].std()

        # Inverse volatility weights
        inv_vol = 1.0 / volatilities
        weights_raw = inv_vol / inv_vol.sum()

        weights = {symbol: weights_raw[symbol] for symbol in symbols}

        return self._apply_constraints(weights)

    def mean_variance_optimization(
        self,
        symbols: List[str],
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0,
        **kwargs
    ) -> Dict[str, float]:
        """
        Mean-Variance Optimization (Markowitz)

        Args:
            symbols: List of trading symbols
            returns: DataFrame with returns for each symbol
            target_return: Target return (if None, uses max Sharpe)
            risk_aversion: Risk aversion parameter (higher = more conservative)

        Returns:
            Dict mapping symbols to weights
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.error("scipy required for mean-variance optimization")
            return self.equal_weight(symbols)

        # Calculate expected returns and covariance
        mean_returns = returns[symbols].mean()
        cov_matrix = returns[symbols].cov()

        n = len(symbols)

        # Objective function (minimize variance for given return)
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # weights sum to 1
        ]

        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w.T @ mean_returns - target_return
            })

        # Bounds
        bounds = tuple(
            (self.min_position_size, self.max_position_size)
            for _ in range(n)
        )

        # Initial guess (equal weight)
        x0 = np.array([1.0 / n] * n)

        # Optimize
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning(
                f"Optimization failed: {result.message}, using equal weight"
            )
            return self.equal_weight(symbols)

        weights = {symbol: result.x[i] for i, symbol in enumerate(symbols)}

        return weights

    def min_variance(
        self,
        symbols: List[str],
        returns: pd.DataFrame,
        **kwargs
    ) -> Dict[str, float]:
        """
        Minimum Variance Portfolio

        Args:
            symbols: List of trading symbols
            returns: DataFrame with returns for each symbol

        Returns:
            Dict mapping symbols to weights
        """
        return self.mean_variance_optimization(
            symbols,
            returns,
            target_return=None
        )

    def max_sharpe(
        self,
        symbols: List[str],
        returns: pd.DataFrame,
        **kwargs
    ) -> Dict[str, float]:
        """
        Maximum Sharpe Ratio Portfolio

        Args:
            symbols: List of trading symbols
            returns: DataFrame with returns for each symbol

        Returns:
            Dict mapping symbols to weights
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.error("scipy required for max Sharpe optimization")
            return self.equal_weight(symbols)

        mean_returns = returns[symbols].mean()
        cov_matrix = returns[symbols].cov()

        n = len(symbols)

        # Negative Sharpe ratio (to minimize)
        def neg_sharpe(weights):
            portfolio_return = weights.T @ mean_returns
            portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights)

            if portfolio_std == 0:
                return 1e10

            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds
        bounds = tuple(
            (self.min_position_size, self.max_position_size)
            for _ in range(n)
        )

        # Initial guess
        x0 = np.array([1.0 / n] * n)

        # Optimize
        result = minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning(
                f"Optimization failed: {result.message}, using equal weight"
            )
            return self.equal_weight(symbols)

        weights = {symbol: result.x[i] for i, symbol in enumerate(symbols)}

        return weights

    def max_diversification(
        self,
        symbols: List[str],
        returns: pd.DataFrame,
        **kwargs
    ) -> Dict[str, float]:
        """
        Maximum Diversification Portfolio
        Maximizes diversification ratio

        Args:
            symbols: List of trading symbols
            returns: DataFrame with returns for each symbol

        Returns:
            Dict mapping symbols to weights
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.error("scipy required for max diversification")
            return self.equal_weight(symbols)

        volatilities = returns[symbols].std().values
        cov_matrix = returns[symbols].cov().values

        n = len(symbols)

        # Negative diversification ratio (to minimize)
        def neg_diversification_ratio(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

            if portfolio_vol == 0:
                return 1e10

            weighted_vol = weights.T @ volatilities
            div_ratio = weighted_vol / portfolio_vol

            return -div_ratio

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds
        bounds = tuple(
            (self.min_position_size, self.max_position_size)
            for _ in range(n)
        )

        # Initial guess
        x0 = np.array([1.0 / n] * n)

        # Optimize
        result = minimize(
            neg_diversification_ratio,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning(
                f"Optimization failed: {result.message}, using equal weight"
            )
            return self.equal_weight(symbols)

        weights = {symbol: result.x[i] for i, symbol in enumerate(symbols)}

        return weights

    def hierarchical_risk_parity(
        self,
        symbols: List[str],
        returns: pd.DataFrame,
        **kwargs
    ) -> Dict[str, float]:
        """
        Hierarchical Risk Parity (HRP)
        Uses hierarchical clustering for more stable portfolios

        Args:
            symbols: List of trading symbols
            returns: DataFrame with returns for each symbol

        Returns:
            Dict mapping symbols to weights
        """
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
        except ImportError:
            logger.error("scipy required for HRP")
            return self.equal_weight(symbols)

        # Calculate correlation matrix
        corr_matrix = returns[symbols].corr()

        # Convert correlation to distance
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)

        # Hierarchical clustering
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='single')

        # Get cluster order
        def get_quasi_diag(link):
            """Get quasi-diagonal order from linkage"""
            link = link.astype(int)
            sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
            num_items = link[-1, 3]

            while sort_ix.max() >= num_items:
                sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                df0 = sort_ix[sort_ix >= num_items]

                i = df0.index
                j = df0.values - num_items

                sort_ix[i] = link[j, 0]
                df0 = pd.Series(link[j, 1], index=i + 1)
                sort_ix = pd.concat([sort_ix, df0]).sort_index()
                sort_ix.index = range(sort_ix.shape[0])

            return sort_ix.tolist()

        sorted_indices = get_quasi_diag(linkage_matrix)
        sorted_symbols = [symbols[i] for i in sorted_indices]

        # Calculate inverse variance weights
        cov_matrix = returns[symbols].cov()
        ivp = 1.0 / np.diag(cov_matrix)
        ivp /= ivp.sum()

        # Recursive bisection
        weights = pd.Series(1.0, index=sorted_symbols)

        def recursive_bisection(items):
            """Recursive bisection for HRP weights"""
            if len(items) == 1:
                return

            # Split in half
            split = len(items) // 2
            left = items[:split]
            right = items[split:]

            # Calculate cluster variances
            left_var = self._get_cluster_variance(returns[left], cov_matrix.loc[left, left])
            right_var = self._get_cluster_variance(returns[right], cov_matrix.loc[right, right])

            # Allocate weight
            alpha = 1 - left_var / (left_var + right_var)

            weights[left] *= alpha
            weights[right] *= (1 - alpha)

            # Recurse
            recursive_bisection(left)
            recursive_bisection(right)

        recursive_bisection(sorted_symbols)

        # Normalize
        weights = weights / weights.sum()

        return weights.to_dict()

    def _get_cluster_variance(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame
    ) -> float:
        """Calculate cluster variance using inverse variance weighting"""
        ivp = 1.0 / np.diag(cov_matrix)
        ivp /= ivp.sum()

        variance = ivp @ cov_matrix @ ivp

        return variance

    def _apply_constraints(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply position size constraints

        Args:
            weights: Raw weights

        Returns:
            Constrained weights
        """
        # Apply max/min constraints
        constrained = {}
        for symbol, weight in weights.items():
            if weight < self.min_position_size:
                constrained[symbol] = 0
            elif weight > self.max_position_size:
                constrained[symbol] = self.max_position_size
            else:
                constrained[symbol] = weight

        # Renormalize to sum to 1
        total = sum(constrained.values())
        if total > 0:
            constrained = {k: v / total for k, v in constrained.items()}

        return constrained

    def allocate(
        self,
        symbols: List[str],
        strategy: AllocationStrategy,
        returns: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Allocate portfolio using specified strategy

        Args:
            symbols: List of trading symbols
            strategy: Allocation strategy to use
            returns: Historical returns DataFrame (required for most strategies)
            **kwargs: Strategy-specific arguments

        Returns:
            Dict mapping symbols to weights

        Example:
            allocator = PortfolioAllocator()
            weights = allocator.allocate(
                symbols=['BTC/USD', 'ETH/USD'],
                strategy=AllocationStrategy.RISK_PARITY,
                returns=returns_df
            )
        """
        if strategy == AllocationStrategy.EQUAL_WEIGHT:
            return self.equal_weight(symbols, **kwargs)

        elif strategy == AllocationStrategy.MARKET_CAP_WEIGHT:
            return self.market_cap_weight(symbols, **kwargs)

        elif strategy == AllocationStrategy.RISK_PARITY:
            if returns is None:
                raise ValueError("Returns required for risk parity")
            return self.risk_parity(symbols, returns, **kwargs)

        elif strategy == AllocationStrategy.MEAN_VARIANCE:
            if returns is None:
                raise ValueError("Returns required for mean-variance")
            return self.mean_variance_optimization(symbols, returns, **kwargs)

        elif strategy == AllocationStrategy.MIN_VARIANCE:
            if returns is None:
                raise ValueError("Returns required for min variance")
            return self.min_variance(symbols, returns, **kwargs)

        elif strategy == AllocationStrategy.MAX_SHARPE:
            if returns is None:
                raise ValueError("Returns required for max Sharpe")
            return self.max_sharpe(symbols, returns, **kwargs)

        elif strategy == AllocationStrategy.MAX_DIVERSIFICATION:
            if returns is None:
                raise ValueError("Returns required for max diversification")
            return self.max_diversification(symbols, returns, **kwargs)

        elif strategy == AllocationStrategy.HIERARCHICAL_RISK_PARITY:
            if returns is None:
                raise ValueError("Returns required for HRP")
            return self.hierarchical_risk_parity(symbols, returns, **kwargs)

        else:
            raise ValueError(f"Unknown allocation strategy: {strategy}")

    def compare_allocations(
        self,
        symbols: List[str],
        returns: pd.DataFrame,
        strategies: Optional[List[AllocationStrategy]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compare different allocation strategies

        Args:
            symbols: List of trading symbols
            returns: Historical returns DataFrame
            strategies: List of strategies to compare (if None, uses all)
            **kwargs: Strategy-specific arguments

        Returns:
            DataFrame comparing allocations
        """
        if strategies is None:
            strategies = [
                AllocationStrategy.EQUAL_WEIGHT,
                AllocationStrategy.RISK_PARITY,
                AllocationStrategy.MIN_VARIANCE,
                AllocationStrategy.MAX_SHARPE
            ]

        results = {}

        for strategy in strategies:
            try:
                weights = self.allocate(symbols, strategy, returns, **kwargs)
                results[strategy.value] = weights
            except Exception as e:
                logger.error(f"Error with {strategy.value}: {e}")
                continue

        return pd.DataFrame(results).fillna(0)
