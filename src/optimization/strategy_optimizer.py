"""
Advanced Strategy Optimization Module
Supports Grid Search, Random Search, and Bayesian Optimization
"""
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import pandas as pd
import numpy as np
from itertools import product
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from ..backtesting.runner import BacktestRunner
from ..strategies.strategy_factory import StrategyFactory


class OptimizationMethod(Enum):
    """Optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"


class StrategyOptimizer:
    """
    Advanced strategy parameter optimization
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
        Initialize optimizer

        Args:
            data: OHLCV DataFrame with datetime index
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
        self.results_cache: List[Dict] = []

    def _run_single_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run backtest with given parameters

        Args:
            params: Strategy parameters

        Returns:
            Backtest results
        """
        try:
            results = BacktestRunner.run_backtest_from_config(
                data=self.data,
                strategy_name=self.strategy_name,
                strategy_params=params,
                symbol=self.symbol,
                backtest_config=self.backtest_config,
                print_report=False
            )

            metric_value = results['metrics_dict'].get(
                self.optimization_metric,
                float('-inf')
            )

            result = {
                'params': params.copy(),
                'metric_value': metric_value,
                **results['metrics_dict']
            }

            self.results_cache.append(result)
            return result

        except Exception as e:
            logger.error(f"Error testing parameters {params}: {e}")
            return {
                'params': params.copy(),
                'metric_value': float('-inf'),
                'error': str(e)
            }

    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        max_combinations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Grid search optimization

        Args:
            param_grid: Dictionary of parameters to test
                       {param_name: [value1, value2, ...]}
            max_combinations: Maximum number of combinations to test

        Returns:
            Optimization results

        Example:
            param_grid = {
                'fast_period': [10, 20, 30],
                'slow_period': [40, 50, 60],
            }
        """
        logger.info(f"Starting Grid Search optimization for {self.strategy_name}")

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        # Limit combinations if specified
        if max_combinations and len(combinations) > max_combinations:
            logger.warning(
                f"Limiting combinations from {len(combinations)} to {max_combinations}"
            )
            import random
            random.shuffle(combinations)
            combinations = combinations[:max_combinations]

        logger.info(f"Testing {len(combinations)} parameter combinations")

        best_metric_value = float('-inf')
        best_params = None
        best_result = None

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            logger.debug(f"Testing {i+1}/{len(combinations)}: {params}")

            result = self._run_single_backtest(params)
            metric_value = result['metric_value']

            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_params = params
                best_result = result

        logger.info(
            f"Grid Search complete. Best {self.optimization_metric}: "
            f"{best_metric_value:.4f} with params: {best_params}"
        )

        return {
            'method': 'grid_search',
            'best_params': best_params,
            'best_metric_value': best_metric_value,
            'best_result': best_result,
            'all_results': pd.DataFrame(self.results_cache),
            'optimization_metric': self.optimization_metric,
            'total_tests': len(combinations)
        }

    def random_search(
        self,
        param_distributions: Dict[str, Any],
        n_iterations: int = 50,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Random search optimization

        Args:
            param_distributions: Dictionary of parameter distributions
                Can be:
                - List: randomly sample from list
                - Tuple (min, max): uniform sampling for numeric values
                - Tuple (min, max, 'int'): uniform sampling for integers
                - Callable: custom sampling function
            n_iterations: Number of random combinations to test
            random_state: Random seed for reproducibility

        Returns:
            Optimization results

        Example:
            param_distributions = {
                'fast_period': (5, 30, 'int'),
                'slow_period': (30, 100, 'int'),
                'rsi_period': [10, 14, 20, 28],
                'threshold': (0.6, 0.9)
            }
        """
        logger.info(
            f"Starting Random Search optimization for {self.strategy_name} "
            f"({n_iterations} iterations)"
        )

        if random_state is not None:
            np.random.seed(random_state)

        best_metric_value = float('-inf')
        best_params = None
        best_result = None

        for i in range(n_iterations):
            # Sample random parameters
            params = {}
            for param_name, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    # Random choice from list
                    params[param_name] = np.random.choice(distribution)
                elif isinstance(distribution, tuple):
                    if len(distribution) == 3 and distribution[2] == 'int':
                        # Integer uniform distribution
                        params[param_name] = np.random.randint(
                            distribution[0],
                            distribution[1] + 1
                        )
                    elif len(distribution) == 2:
                        # Float uniform distribution
                        params[param_name] = np.random.uniform(
                            distribution[0],
                            distribution[1]
                        )
                elif callable(distribution):
                    # Custom sampling function
                    params[param_name] = distribution()

            logger.debug(f"Testing {i+1}/{n_iterations}: {params}")

            result = self._run_single_backtest(params)
            metric_value = result['metric_value']

            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_params = params
                best_result = result

        logger.info(
            f"Random Search complete. Best {self.optimization_metric}: "
            f"{best_metric_value:.4f} with params: {best_params}"
        )

        return {
            'method': 'random_search',
            'best_params': best_params,
            'best_metric_value': best_metric_value,
            'best_result': best_result,
            'all_results': pd.DataFrame(self.results_cache),
            'optimization_metric': self.optimization_metric,
            'total_tests': n_iterations
        }

    def bayesian_optimization(
        self,
        param_space: Dict[str, tuple],
        n_iterations: int = 50,
        n_initial_points: int = 10,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Bayesian optimization using scikit-optimize

        Args:
            param_space: Dictionary of parameter spaces
                Format: {param_name: (min, max, 'type')}
                Types: 'int', 'float', 'log-uniform'
            n_iterations: Total number of iterations
            n_initial_points: Number of random initial points
            random_state: Random seed for reproducibility

        Returns:
            Optimization results

        Example:
            param_space = {
                'fast_period': (5, 50, 'int'),
                'slow_period': (50, 200, 'int'),
                'threshold': (0.5, 0.9, 'float'),
                'learning_rate': (0.001, 0.1, 'log-uniform')
            }
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Integer, Real
            from skopt.utils import use_named_args
        except ImportError:
            logger.error(
                "scikit-optimize not installed. "
                "Install with: pip install scikit-optimize"
            )
            raise ImportError("scikit-optimize required for Bayesian optimization")

        logger.info(
            f"Starting Bayesian Optimization for {self.strategy_name} "
            f"({n_iterations} iterations)"
        )

        # Build search space
        dimensions = []
        param_names = []

        for param_name, space_def in param_space.items():
            param_names.append(param_name)

            if len(space_def) == 3:
                min_val, max_val, space_type = space_def

                if space_type == 'int':
                    dimensions.append(Integer(min_val, max_val, name=param_name))
                elif space_type == 'float':
                    dimensions.append(Real(min_val, max_val, name=param_name))
                elif space_type == 'log-uniform':
                    dimensions.append(
                        Real(min_val, max_val, prior='log-uniform', name=param_name)
                    )
                else:
                    raise ValueError(f"Unknown space type: {space_type}")
            else:
                raise ValueError(
                    f"Invalid space definition for {param_name}: {space_def}"
                )

        # Define objective function
        @use_named_args(dimensions)
        def objective(**params):
            """Objective function to minimize (negative metric)"""
            result = self._run_single_backtest(params)
            # Return negative because we want to maximize
            return -result['metric_value']

        # Run Bayesian optimization
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=n_iterations,
            n_initial_points=n_initial_points,
            random_state=random_state,
            verbose=False
        )

        # Extract best parameters
        best_params = dict(zip(param_names, result.x))
        best_metric_value = -result.fun

        logger.info(
            f"Bayesian Optimization complete. Best {self.optimization_metric}: "
            f"{best_metric_value:.4f} with params: {best_params}"
        )

        # Find best result in cache
        best_result = max(
            self.results_cache,
            key=lambda x: x['metric_value']
        )

        return {
            'method': 'bayesian',
            'best_params': best_params,
            'best_metric_value': best_metric_value,
            'best_result': best_result,
            'all_results': pd.DataFrame(self.results_cache),
            'optimization_metric': self.optimization_metric,
            'total_tests': n_iterations,
            'convergence': result.func_vals
        }

    def optimize(
        self,
        method: OptimizationMethod,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run optimization with specified method

        Args:
            method: Optimization method to use
            **kwargs: Method-specific arguments

        Returns:
            Optimization results
        """
        self.results_cache = []  # Reset cache

        if method == OptimizationMethod.GRID_SEARCH:
            return self.grid_search(**kwargs)
        elif method == OptimizationMethod.RANDOM_SEARCH:
            return self.random_search(**kwargs)
        elif method == OptimizationMethod.BAYESIAN:
            return self.bayesian_optimization(**kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def plot_optimization_results(
        self,
        results: Dict[str, Any],
        top_n: int = 10
    ) -> None:
        """
        Plot optimization results

        Args:
            results: Optimization results
            top_n: Number of top results to highlight
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style('darkgrid')
        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")
            return

        df = results['all_results']

        if df.empty:
            logger.warning("No results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Metric value distribution
        ax = axes[0, 0]
        df['metric_value'].hist(bins=30, ax=ax, edgecolor='black')
        ax.axvline(
            results['best_metric_value'],
            color='red',
            linestyle='--',
            label='Best'
        )
        ax.set_xlabel(self.optimization_metric)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{self.optimization_metric.replace("_", " ").title()} Distribution')
        ax.legend()

        # 2. Convergence plot (for iterative methods)
        ax = axes[0, 1]
        metric_values = df['metric_value'].values
        ax.plot(metric_values, alpha=0.6, label='All tests')

        # Running maximum
        running_max = np.maximum.accumulate(metric_values)
        ax.plot(running_max, 'r-', linewidth=2, label='Best so far')

        ax.set_xlabel('Iteration')
        ax.set_ylabel(self.optimization_metric)
        ax.set_title('Optimization Convergence')
        ax.legend()
        ax.grid(True)

        # 3. Top parameters comparison
        ax = axes[1, 0]
        top_results = df.nlargest(top_n, 'metric_value')

        # Get parameter columns (exclude metric columns)
        param_cols = [col for col in df.columns if col not in [
            'metric_value', 'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct',
            'win_rate', 'num_trades', 'profit_factor', 'sortino_ratio',
            'annualized_return', 'expectancy', 'error'
        ]]

        if param_cols and len(top_results) > 0:
            # Normalize parameters for comparison
            for col in param_cols:
                if col in top_results.columns:
                    try:
                        top_results[col] = pd.to_numeric(top_results[col], errors='coerce')
                    except:
                        pass

            numeric_params = top_results[param_cols].select_dtypes(include=[np.number])
            if not numeric_params.empty:
                numeric_params.plot(kind='bar', ax=ax)
                ax.set_title(f'Top {top_n} Parameter Combinations')
                ax.set_xlabel('Rank')
                ax.legend(title='Parameters', bbox_to_anchor=(1.05, 1))

        # 4. Return vs Drawdown scatter
        ax = axes[1, 1]
        if 'total_return_pct' in df.columns and 'max_drawdown_pct' in df.columns:
            scatter = ax.scatter(
                df['max_drawdown_pct'],
                df['total_return_pct'],
                c=df['metric_value'],
                cmap='viridis',
                alpha=0.6,
                s=50
            )
            ax.scatter(
                results['best_result']['max_drawdown_pct'],
                results['best_result']['total_return_pct'],
                color='red',
                s=200,
                marker='*',
                label='Best',
                edgecolors='black',
                linewidth=2
            )
            ax.set_xlabel('Max Drawdown %')
            ax.set_ylabel('Total Return %')
            ax.set_title('Return vs Drawdown')
            ax.legend()
            plt.colorbar(scatter, ax=ax, label=self.optimization_metric)

        plt.tight_layout()
        plt.savefig(f'optimization_results_{self.strategy_name}.png', dpi=300)
        logger.info(f"Saved plot to optimization_results_{self.strategy_name}.png")
        plt.close()
