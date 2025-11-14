"""
Backtesting Runner - Convenient interface for running backtests
"""
from typing import Dict, Any, Optional
import pandas as pd
from loguru import logger

from .backtest_engine import BacktestEngine
from .performance_metrics import PerformanceAnalyzer, PerformanceMetrics
from ..strategies.base_strategy import BaseStrategy
from ..strategies.strategy_factory import StrategyFactory


class BacktestRunner:
    """
    High-level interface for running backtests
    """

    @staticmethod
    def run_backtest(
        data: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str = 'UNKNOWN',
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        risk_per_trade: float = 0.02,
        enable_shorts: bool = False,
        risk_free_rate: float = 0.02,
        print_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run a complete backtest with performance analysis

        Args:
            data: OHLCV DataFrame with datetime index
            strategy: Trading strategy instance
            symbol: Trading symbol
            initial_capital: Starting capital
            commission_rate: Commission as percentage (0.001 = 0.1%)
            slippage_rate: Slippage as percentage (0.0005 = 0.05%)
            risk_per_trade: Risk percentage per trade
            enable_shorts: Allow short selling
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino
            print_report: Print performance report

        Returns:
            Complete backtest results with metrics
        """
        # Create backtest engine
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            risk_per_trade=risk_per_trade,
            enable_shorts=enable_shorts
        )

        # Run backtest
        results = engine.run(data, symbol)

        # Calculate performance metrics
        metrics = PerformanceAnalyzer.calculate_metrics(
            trades=results['trades'],
            equity_curve=results['equity_curve'],
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate
        )

        # Add metrics to results
        results['metrics'] = metrics
        results['metrics_dict'] = metrics.to_dict()

        # Generate and optionally print report
        report = PerformanceAnalyzer.generate_report(metrics, results['trades'])
        results['report'] = report

        if print_report:
            print(report)

        return results

    @staticmethod
    def run_backtest_from_config(
        data: pd.DataFrame,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        symbol: str = 'UNKNOWN',
        backtest_config: Optional[Dict[str, Any]] = None,
        print_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run backtest using strategy name and parameters

        Args:
            data: OHLCV DataFrame with datetime index
            strategy_name: Name of strategy to use
            strategy_params: Strategy parameters
            symbol: Trading symbol
            backtest_config: Backtest configuration (capital, commission, etc.)
            print_report: Print performance report

        Returns:
            Complete backtest results with metrics
        """
        # Create strategy
        strategy = StrategyFactory.create(strategy_name, strategy_params)

        # Default backtest configuration
        default_config = {
            'initial_capital': 10000.0,
            'commission_rate': 0.001,
            'slippage_rate': 0.0005,
            'risk_per_trade': 0.02,
            'enable_shorts': False,
            'risk_free_rate': 0.02
        }

        if backtest_config:
            default_config.update(backtest_config)

        # Run backtest
        return BacktestRunner.run_backtest(
            data=data,
            strategy=strategy,
            symbol=symbol,
            print_report=print_report,
            **default_config
        )

    @staticmethod
    def compare_strategies(
        data: pd.DataFrame,
        strategies: Dict[str, tuple],  # {name: (strategy_name, params)}
        symbol: str = 'UNKNOWN',
        backtest_config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on the same data

        Args:
            data: OHLCV DataFrame with datetime index
            strategies: Dict of strategy configurations
                       {display_name: (strategy_name, params)}
            symbol: Trading symbol
            backtest_config: Backtest configuration

        Returns:
            DataFrame comparing all strategies

        Example:
            strategies = {
                'MA Fast': ('ma_crossover', {'fast_period': 10, 'slow_period': 30}),
                'MA Slow': ('ma_crossover', {'fast_period': 20, 'slow_period': 50}),
                'RSI': ('rsi', {'rsi_period': 14})
            }
            results = BacktestRunner.compare_strategies(data, strategies)
        """
        logger.info(f"Comparing {len(strategies)} strategies on {symbol}")

        comparison_data = []

        for display_name, (strategy_name, params) in strategies.items():
            logger.info(f"Running backtest for: {display_name}")

            # Run backtest
            results = BacktestRunner.run_backtest_from_config(
                data=data,
                strategy_name=strategy_name,
                strategy_params=params,
                symbol=symbol,
                backtest_config=backtest_config,
                print_report=False
            )

            # Extract key metrics
            metrics_dict = results['metrics_dict']
            metrics_dict['strategy'] = display_name
            metrics_dict['strategy_name'] = strategy_name

            comparison_data.append(metrics_dict)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Set strategy as index
        comparison_df = comparison_df.set_index('strategy')

        # Select key columns for display
        key_columns = [
            'total_return_pct',
            'annualized_return',
            'num_trades',
            'win_rate',
            'max_drawdown_pct',
            'sharpe_ratio',
            'sortino_ratio',
            'profit_factor',
            'expectancy'
        ]

        # Only include columns that exist
        display_columns = [col for col in key_columns if col in comparison_df.columns]

        logger.info("Strategy comparison complete")

        return comparison_df[display_columns]

    @staticmethod
    def optimize_strategy(
        data: pd.DataFrame,
        strategy_name: str,
        param_grid: Dict[str, list],
        symbol: str = 'UNKNOWN',
        backtest_config: Optional[Dict[str, Any]] = None,
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search

        Args:
            data: OHLCV DataFrame with datetime index
            strategy_name: Name of strategy to optimize
            param_grid: Dictionary of parameters to test
                       {param_name: [value1, value2, ...]}
            symbol: Trading symbol
            backtest_config: Backtest configuration
            metric: Metric to optimize for (default: 'sharpe_ratio')

        Returns:
            Dict with best parameters and results

        Example:
            param_grid = {
                'fast_period': [10, 20, 30],
                'slow_period': [40, 50, 60],
            }
            best = BacktestRunner.optimize_strategy(
                data, 'ma_crossover', param_grid, metric='sharpe_ratio'
            )
        """
        from itertools import product

        logger.info(f"Optimizing {strategy_name} on {symbol}")

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"Testing {len(combinations)} parameter combinations")

        best_metric_value = float('-inf')
        best_params = None
        best_results = None
        all_results = []

        for i, combo in enumerate(combinations):
            # Create parameter dict
            params = dict(zip(param_names, combo))

            logger.debug(f"Testing combination {i+1}/{len(combinations)}: {params}")

            # Run backtest
            try:
                results = BacktestRunner.run_backtest_from_config(
                    data=data,
                    strategy_name=strategy_name,
                    strategy_params=params,
                    symbol=symbol,
                    backtest_config=backtest_config,
                    print_report=False
                )

                # Get metric value
                metric_value = results['metrics_dict'].get(metric, float('-inf'))

                # Store results
                result_entry = {
                    'params': params,
                    'metric_value': metric_value,
                    **results['metrics_dict']
                }
                all_results.append(result_entry)

                # Check if this is the best
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_params = params
                    best_results = results

            except Exception as e:
                logger.error(f"Error testing parameters {params}: {e}")
                continue

        logger.info(
            f"Optimization complete. Best {metric}: {best_metric_value:.2f} "
            f"with params: {best_params}"
        )

        return {
            'best_params': best_params,
            'best_metric_value': best_metric_value,
            'best_results': best_results,
            'all_results': pd.DataFrame(all_results),
            'optimization_metric': metric
        }
