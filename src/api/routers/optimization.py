"""
Strategy Optimization API Router
Endpoints for strategy parameter optimization
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import pandas as pd
from loguru import logger

from ...optimization.strategy_optimizer import StrategyOptimizer, OptimizationMethod
from ...optimization.walk_forward import WalkForwardOptimizer

router = APIRouter(
    prefix="/optimization",
    tags=["optimization"]
)


class OptimizationRequest(BaseModel):
    """Optimization request model"""
    strategy_name: str
    symbol: str
    data_source: str  # 'file' or 'exchange'
    data_path: Optional[str] = None
    optimization_method: str  # 'grid_search', 'random_search', 'bayesian'
    param_space: Dict[str, Any]
    optimization_metric: str = 'sharpe_ratio'
    n_iterations: Optional[int] = 50
    backtest_config: Optional[Dict[str, Any]] = None


class WalkForwardRequest(BaseModel):
    """Walk-forward optimization request"""
    strategy_name: str
    symbol: str
    data_source: str
    data_path: Optional[str] = None
    param_space: Dict[str, Any]
    train_period_days: int = 180
    test_period_days: int = 60
    optimization_method: str = 'random_search'
    anchored: bool = False
    n_iterations: int = 50
    optimization_metric: str = 'sharpe_ratio'


# Store optimization jobs
optimization_jobs: Dict[str, Dict] = {}


@router.post("/optimize")
async def optimize_strategy(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Run strategy parameter optimization

    Supports grid search, random search, and Bayesian optimization
    """
    try:
        # Load data
        if request.data_source == 'file' and request.data_path:
            data = pd.read_csv(request.data_path, index_col=0, parse_dates=True)
        else:
            raise HTTPException(
                status_code=400,
                detail="Currently only file data source is supported. Provide data_path."
            )

        # Create optimizer
        optimizer = StrategyOptimizer(
            data=data,
            strategy_name=request.strategy_name,
            symbol=request.symbol,
            backtest_config=request.backtest_config,
            optimization_metric=request.optimization_metric
        )

        # Parse optimization method
        try:
            method = OptimizationMethod(request.optimization_method)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid optimization method: {request.optimization_method}"
            )

        # Run optimization
        if method == OptimizationMethod.GRID_SEARCH:
            results = optimizer.grid_search(request.param_space)
        elif method == OptimizationMethod.RANDOM_SEARCH:
            results = optimizer.random_search(
                request.param_space,
                n_iterations=request.n_iterations or 50
            )
        elif method == OptimizationMethod.BAYESIAN:
            results = optimizer.bayesian_optimization(
                request.param_space,
                n_iterations=request.n_iterations or 50
            )

        # Convert DataFrame to dict for JSON serialization
        if 'all_results' in results and isinstance(results['all_results'], pd.DataFrame):
            results['all_results'] = results['all_results'].to_dict('records')

        return {
            "status": "success",
            "optimization_method": request.optimization_method,
            "best_params": results['best_params'],
            "best_metric_value": results['best_metric_value'],
            "optimization_metric": results['optimization_metric'],
            "total_tests": results['total_tests'],
            "results_summary": {
                'best_result': results.get('best_result'),
                'num_results': len(results.get('all_results', []))
            }
        }

    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/walk-forward")
async def walk_forward_optimization(request: WalkForwardRequest):
    """
    Run walk-forward optimization

    Divides data into train/test periods to validate strategy robustness
    """
    try:
        # Load data
        if request.data_source == 'file' and request.data_path:
            data = pd.read_csv(request.data_path, index_col=0, parse_dates=True)
        else:
            raise HTTPException(
                status_code=400,
                detail="Currently only file data source is supported. Provide data_path."
            )

        # Create walk-forward optimizer
        wf_optimizer = WalkForwardOptimizer(
            data=data,
            strategy_name=request.strategy_name,
            symbol=request.symbol,
            optimization_metric=request.optimization_metric
        )

        # Parse optimization method
        try:
            method = OptimizationMethod(request.optimization_method)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid optimization method: {request.optimization_method}"
            )

        # Run walk-forward optimization
        results = wf_optimizer.optimize(
            param_space=request.param_space,
            train_period_days=request.train_period_days,
            test_period_days=request.test_period_days,
            optimization_method=method,
            anchored=request.anchored,
            n_iterations=request.n_iterations
        )

        # Simplify results for JSON response
        simplified_results = {
            'strategy_name': results['strategy_name'],
            'optimization_method': results['optimization_method'],
            'num_splits': results['num_splits'],
            'train_period_days': results['train_period_days'],
            'test_period_days': results['test_period_days'],
            'anchored': results['anchored'],
            'summary': results['summary'],
            'splits_summary': [
                {
                    'split_number': s['split_number'],
                    'train_metric': s['train_metric'],
                    'test_metric': s['test_metric'],
                    'degradation': s['degradation'],
                    'degradation_pct': s['degradation_pct'],
                    'best_params': s['best_params']
                }
                for s in results['splits']
            ]
        }

        return {
            "status": "success",
            "results": simplified_results
        }

    except Exception as e:
        logger.error(f"Walk-forward optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods")
async def get_optimization_methods():
    """
    Get available optimization methods
    """
    return {
        "methods": [
            {
                "name": "grid_search",
                "description": "Exhaustive search over parameter grid",
                "use_case": "Small parameter spaces, guaranteed to find best combination"
            },
            {
                "name": "random_search",
                "description": "Random sampling from parameter distributions",
                "use_case": "Large parameter spaces, good balance of speed and quality"
            },
            {
                "name": "bayesian",
                "description": "Bayesian optimization using Gaussian Processes",
                "use_case": "Expensive evaluations, smart exploration of parameter space"
            }
        ]
    }


@router.get("/metrics")
async def get_optimization_metrics():
    """
    Get available optimization metrics
    """
    return {
        "metrics": [
            "sharpe_ratio",
            "sortino_ratio",
            "total_return_pct",
            "annualized_return",
            "max_drawdown_pct",
            "win_rate",
            "profit_factor",
            "expectancy"
        ]
    }
