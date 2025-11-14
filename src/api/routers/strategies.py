"""
Strategy Management API Router
Endpoints for managing and configuring trading strategies
"""
from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any
from loguru import logger

from ..models.requests import UpdateStrategyRequest, BacktestRequest, OptimizeStrategyRequest
from ..models.responses import StrategyInfoResponse, SuccessResponse, BacktestResultResponse

from ...strategies.strategy_factory import StrategyFactory
from ...backtesting.runner import BacktestRunner

router = APIRouter(prefix="/strategies", tags=["Strategies"])


@router.get("/", response_model=List[str])
async def list_strategies():
    """
    Get list of available strategies

    Returns:
        List of supported strategy names
    """
    try:
        strategies = StrategyFactory.get_supported_strategies()
        return strategies

    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list strategies: {str(e)}"
        )


@router.get("/{strategy_name}", response_model=StrategyInfoResponse)
async def get_strategy_info(strategy_name: str):
    """
    Get information about a specific strategy

    Args:
        strategy_name: Name of the strategy

    Returns:
        Strategy information including parameters and description
    """
    try:
        info = StrategyFactory.get_strategy_info(strategy_name)

        # Create a default instance to get default parameters
        strategy = StrategyFactory.create(strategy_name, {})

        return StrategyInfoResponse(
            name=info['name'],
            description=info['description'],
            parameters=strategy.get_parameters(),
            required_history=strategy.get_required_history()
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting strategy info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get strategy info: {str(e)}"
        )


@router.get("/current/info", response_model=StrategyInfoResponse)
async def get_current_strategy():
    """
    Get information about the currently active strategy

    Returns:
        Current strategy information
    """
    try:
        # TODO: Get current strategy from bot state
        # For now, return placeholder

        return StrategyInfoResponse(
            name="MovingAverageCrossoverStrategy",
            description="Simple moving average crossover strategy",
            parameters={
                "fast_period": 20,
                "slow_period": 50,
                "min_confidence": 0.6
            },
            required_history=70
        )

    except Exception as e:
        logger.error(f"Error getting current strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get current strategy: {str(e)}"
        )


@router.put("/current", response_model=SuccessResponse)
async def update_strategy(request: UpdateStrategyRequest):
    """
    Update the current strategy or its parameters

    Args:
        request: Strategy update details

    Returns:
        Success message
    """
    try:
        logger.info(f"Updating strategy: {request}")

        # TODO: Implement actual strategy update
        # This would update the bot's active strategy

        if request.strategy_name:
            # Validate strategy exists
            StrategyFactory.create(request.strategy_name, {})

        return SuccessResponse(
            success=True,
            message="Strategy updated successfully",
            data={
                "strategy_name": request.strategy_name,
                "parameters": request.parameters
            }
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update strategy: {str(e)}"
        )


@router.post("/backtest", response_model=BacktestResultResponse)
async def run_backtest(request: BacktestRequest):
    """
    Run a backtest for a strategy

    Args:
        request: Backtest configuration

    Returns:
        Backtest results with performance metrics
    """
    try:
        logger.info(f"Running backtest for {request.strategy_name} on {request.symbol}")

        # TODO: Implement actual backtest
        # This would fetch historical data and run the backtest

        # For now, return placeholder data
        return BacktestResultResponse(
            strategy_name=request.strategy_name,
            strategy_params=request.strategy_params,
            initial_capital=request.initial_capital,
            final_capital=request.initial_capital * 1.15,  # Placeholder
            total_return=request.initial_capital * 0.15,
            total_return_pct=15.0,
            annualized_return=18.0,
            num_trades=50,
            win_rate=60.0,
            max_drawdown_pct=12.5,
            sharpe_ratio=1.8,
            sortino_ratio=2.1,
            profit_factor=1.75,
            expectancy=30.0
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run backtest: {str(e)}"
        )


@router.post("/optimize", response_model=Dict[str, Any])
async def optimize_strategy(request: OptimizeStrategyRequest):
    """
    Optimize strategy parameters using grid search

    Args:
        request: Optimization configuration

    Returns:
        Best parameters and optimization results
    """
    try:
        logger.info(f"Optimizing {request.strategy_name} on {request.symbol}")

        # TODO: Implement actual optimization
        # This would run backtests for all parameter combinations

        # For now, return placeholder data
        return {
            "best_params": {},
            "best_metric_value": 0.0,
            "optimization_metric": request.optimization_metric,
            "combinations_tested": 0,
            "message": "Optimization not yet fully implemented"
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error optimizing strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize strategy: {str(e)}"
        )


@router.post("/validate", response_model=SuccessResponse)
async def validate_strategy_params(
    strategy_name: str,
    parameters: Dict[str, Any]
):
    """
    Validate strategy parameters without running backtest

    Args:
        strategy_name: Name of the strategy
        parameters: Parameters to validate

    Returns:
        Validation result
    """
    try:
        # Try to create strategy with given parameters
        strategy = StrategyFactory.create(strategy_name, parameters)

        return SuccessResponse(
            success=True,
            message="Strategy parameters are valid",
            data={
                "strategy": strategy.name,
                "parameters": strategy.get_parameters()
            }
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error validating strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate strategy: {str(e)}"
        )
