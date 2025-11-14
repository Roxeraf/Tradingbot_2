"""
Walk-Forward Optimization Module
Implements walk-forward analysis to avoid overfitting
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from .strategy_optimizer import StrategyOptimizer, OptimizationMethod
from ..backtesting.runner import BacktestRunner


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization

    Divides data into multiple in-sample (training) and out-of-sample (testing) periods
    to validate strategy robustness and reduce overfitting.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        symbol: str = 'UNKNOWN',
        backtest_config: Optional[Dict[str, Any]] = None,
        optimization_metric: str = 'sharpe_ratio'
    ):
        """
        Initialize Walk-Forward Optimizer

        Args:
            data: Full OHLCV DataFrame with datetime index
            strategy_name: Name of strategy to optimize
            symbol: Trading symbol
            backtest_config: Backtest configuration
            optimization_metric: Metric to optimize for
        """
        self.data = data
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.backtest_config = backtest_config or {}
        self.optimization_metric = optimization_metric

    def _split_data(
        self,
        train_period_days: int,
        test_period_days: int,
        anchored: bool = False
    ) -> List[Dict[str, pd.DataFrame]]:
        """
        Split data into train/test periods

        Args:
            train_period_days: Days for training (in-sample)
            test_period_days: Days for testing (out-of-sample)
            anchored: If True, training window grows (anchored),
                     If False, training window slides (rolling)

        Returns:
            List of dicts with 'train' and 'test' DataFrames
        """
        splits = []
        start_idx = 0

        while True:
            # Calculate indices for this split
            train_end_idx = start_idx + train_period_days

            if train_end_idx >= len(self.data):
                break

            test_end_idx = train_end_idx + test_period_days

            if test_end_idx > len(self.data):
                # Last split might have shorter test period
                test_end_idx = len(self.data)

            # Extract train and test data
            if anchored:
                # Anchored: always start from beginning
                train_data = self.data.iloc[0:train_end_idx]
            else:
                # Rolling: sliding window
                train_data = self.data.iloc[start_idx:train_end_idx]

            test_data = self.data.iloc[train_end_idx:test_end_idx]

            if len(test_data) == 0:
                break

            splits.append({
                'train': train_data,
                'test': test_data,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1]
            })

            # Move to next window
            if anchored:
                start_idx = train_end_idx
            else:
                start_idx += test_period_days

        return splits

    def optimize(
        self,
        param_space: Dict[str, Any],
        train_period_days: int = 180,
        test_period_days: int = 60,
        optimization_method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH,
        anchored: bool = False,
        n_iterations: int = 50,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """
        Run walk-forward optimization

        Args:
            param_space: Parameter space for optimization
            train_period_days: Days for training period
            test_period_days: Days for testing period
            optimization_method: Optimization method to use
            anchored: Use anchored (expanding) or rolling window
            n_iterations: Number of iterations for optimization
            **optimizer_kwargs: Additional arguments for optimizer

        Returns:
            Complete walk-forward results

        Example:
            param_space = {
                'fast_period': (5, 30, 'int'),
                'slow_period': (30, 100, 'int')
            }
            results = wf_optimizer.optimize(
                param_space,
                train_period_days=180,
                test_period_days=60,
                optimization_method=OptimizationMethod.RANDOM_SEARCH,
                n_iterations=50
            )
        """
        logger.info(
            f"Starting Walk-Forward Optimization for {self.strategy_name}\n"
            f"Train period: {train_period_days} days, Test period: {test_period_days} days\n"
            f"Window type: {'Anchored' if anchored else 'Rolling'}"
        )

        # Split data into train/test periods
        splits = self._split_data(train_period_days, test_period_days, anchored)

        logger.info(f"Created {len(splits)} train/test splits")

        walk_forward_results = []
        all_test_trades = []
        combined_equity = []

        for i, split in enumerate(splits):
            logger.info(
                f"\n=== Split {i+1}/{len(splits)} ===\n"
                f"Train: {split['train_start']} to {split['train_end']}\n"
                f"Test:  {split['test_start']} to {split['test_end']}"
            )

            # Optimize on training data
            optimizer = StrategyOptimizer(
                data=split['train'],
                strategy_name=self.strategy_name,
                symbol=self.symbol,
                backtest_config=self.backtest_config,
                optimization_metric=self.optimization_metric
            )

            # Run optimization based on method
            if optimization_method == OptimizationMethod.GRID_SEARCH:
                opt_results = optimizer.grid_search(param_space, **optimizer_kwargs)
            elif optimization_method == OptimizationMethod.RANDOM_SEARCH:
                opt_results = optimizer.random_search(
                    param_space,
                    n_iterations=n_iterations,
                    **optimizer_kwargs
                )
            elif optimization_method == OptimizationMethod.BAYESIAN:
                opt_results = optimizer.bayesian_optimization(
                    param_space,
                    n_iterations=n_iterations,
                    **optimizer_kwargs
                )
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")

            best_params = opt_results['best_params']
            train_metric = opt_results['best_metric_value']

            logger.info(
                f"Best params on training: {best_params}\n"
                f"Training {self.optimization_metric}: {train_metric:.4f}"
            )

            # Test on out-of-sample data
            test_results = BacktestRunner.run_backtest_from_config(
                data=split['test'],
                strategy_name=self.strategy_name,
                strategy_params=best_params,
                symbol=self.symbol,
                backtest_config=self.backtest_config,
                print_report=False
            )

            test_metric = test_results['metrics_dict'].get(
                self.optimization_metric,
                float('-inf')
            )

            logger.info(f"Out-of-sample {self.optimization_metric}: {test_metric:.4f}")

            # Store results
            walk_forward_results.append({
                'split_number': i + 1,
                'train_start': split['train_start'],
                'train_end': split['train_end'],
                'test_start': split['test_start'],
                'test_end': split['test_end'],
                'best_params': best_params,
                'train_metric': train_metric,
                'test_metric': test_metric,
                'train_results': opt_results,
                'test_results': test_results,
                'degradation': train_metric - test_metric,
                'degradation_pct': ((train_metric - test_metric) / abs(train_metric) * 100)
                if train_metric != 0 else 0
            })

            # Collect test period trades
            all_test_trades.extend(test_results['trades'])

            # Collect equity curve
            if combined_equity:
                # Normalize to continue from previous period
                prev_final = combined_equity[-1]
                test_equity = test_results['equity_curve']
                test_equity = test_equity / test_equity.iloc[0] * prev_final
                combined_equity.extend(test_equity.iloc[1:].tolist())
            else:
                combined_equity = test_results['equity_curve'].tolist()

        # Calculate aggregate statistics
        avg_train_metric = np.mean([r['train_metric'] for r in walk_forward_results])
        avg_test_metric = np.mean([r['test_metric'] for r in walk_forward_results])
        avg_degradation = np.mean([r['degradation'] for r in walk_forward_results])
        avg_degradation_pct = np.mean([r['degradation_pct'] for r in walk_forward_results])

        # Calculate consistency (how often test > 0)
        positive_tests = sum(1 for r in walk_forward_results if r['test_metric'] > 0)
        consistency = (positive_tests / len(walk_forward_results)) * 100

        logger.info(
            f"\n{'='*60}\n"
            f"Walk-Forward Optimization Complete\n"
            f"{'='*60}\n"
            f"Average In-Sample {self.optimization_metric}: {avg_train_metric:.4f}\n"
            f"Average Out-of-Sample {self.optimization_metric}: {avg_test_metric:.4f}\n"
            f"Average Degradation: {avg_degradation:.4f} ({avg_degradation_pct:.2f}%)\n"
            f"Consistency (positive periods): {consistency:.1f}%\n"
            f"Total Out-of-Sample Trades: {len(all_test_trades)}\n"
            f"{'='*60}"
        )

        return {
            'strategy_name': self.strategy_name,
            'optimization_method': optimization_method.value,
            'num_splits': len(splits),
            'train_period_days': train_period_days,
            'test_period_days': test_period_days,
            'anchored': anchored,
            'splits': walk_forward_results,
            'summary': {
                'avg_train_metric': avg_train_metric,
                'avg_test_metric': avg_test_metric,
                'avg_degradation': avg_degradation,
                'avg_degradation_pct': avg_degradation_pct,
                'consistency': consistency,
                'total_test_trades': len(all_test_trades)
            },
            'all_test_trades': all_test_trades,
            'combined_equity_curve': pd.Series(combined_equity),
            'optimization_metric': self.optimization_metric
        }

    def plot_walk_forward_results(self, results: Dict[str, Any]) -> None:
        """
        Visualize walk-forward optimization results

        Args:
            results: Walk-forward optimization results
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style('darkgrid')
        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. In-sample vs Out-of-sample performance
        ax = axes[0, 0]
        splits = results['splits']
        split_numbers = [s['split_number'] for s in splits]
        train_metrics = [s['train_metric'] for s in splits]
        test_metrics = [s['test_metric'] for s in splits]

        x = np.arange(len(split_numbers))
        width = 0.35

        ax.bar(x - width/2, train_metrics, width, label='In-Sample', alpha=0.8)
        ax.bar(x + width/2, test_metrics, width, label='Out-of-Sample', alpha=0.8)

        ax.set_xlabel('Split Number')
        ax.set_ylabel(results['optimization_metric'])
        ax.set_title('In-Sample vs Out-of-Sample Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(split_numbers)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Performance degradation
        ax = axes[0, 1]
        degradations = [s['degradation_pct'] for s in splits]

        ax.bar(split_numbers, degradations, alpha=0.8, color='coral')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.axhline(
            results['summary']['avg_degradation_pct'],
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Avg: {results["summary"]["avg_degradation_pct"]:.1f}%'
        )

        ax.set_xlabel('Split Number')
        ax.set_ylabel('Degradation %')
        ax.set_title('Performance Degradation (In-Sample - Out-of-Sample)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Combined equity curve
        ax = axes[1, 0]
        equity = results['combined_equity_curve']

        ax.plot(equity.values, linewidth=2)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Equity')
        ax.set_title('Combined Out-of-Sample Equity Curve')
        ax.grid(True, alpha=0.3)

        # Add split boundaries
        cumulative_len = 0
        for split in splits:
            cumulative_len += len(split['test_results']['equity_curve'])
            ax.axvline(cumulative_len, color='red', alpha=0.3, linestyle='--')

        # 4. Parameter stability (if applicable)
        ax = axes[1, 1]

        # Extract parameter values across splits
        param_names = list(splits[0]['best_params'].keys())

        for param_name in param_names:
            param_values = []
            for split in splits:
                val = split['best_params'].get(param_name)
                if isinstance(val, (int, float)):
                    param_values.append(val)

            if param_values:
                ax.plot(split_numbers[:len(param_values)], param_values,
                       marker='o', label=param_name, linewidth=2)

        ax.set_xlabel('Split Number')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter Stability Across Splits')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'walk_forward_{self.strategy_name}.png', dpi=300)
        logger.info(f"Saved plot to walk_forward_{self.strategy_name}.png")
        plt.close()
