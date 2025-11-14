"""
Portfolio Management API Router
Endpoints for advanced portfolio allocation and rebalancing
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
from loguru import logger

from ...portfolio.allocator import PortfolioAllocator, AllocationStrategy
from ...portfolio.rebalancer import PortfolioRebalancer, RebalancingStrategy

router = APIRouter(
    prefix="/portfolio",
    tags=["portfolio"]
)


class AllocationRequest(BaseModel):
    """Portfolio allocation request"""
    symbols: List[str]
    strategy: str  # Allocation strategy name
    returns_data: Optional[Dict[str, List[float]]] = None  # Historical returns
    market_caps: Optional[Dict[str, float]] = None  # For market cap weighting
    risk_free_rate: float = 0.02
    max_position_size: float = 0.4
    min_position_size: float = 0.01


class CompareAllocationsRequest(BaseModel):
    """Compare allocation strategies request"""
    symbols: List[str]
    returns_data: Dict[str, List[float]]  # Historical returns by symbol
    strategies: Optional[List[str]] = None  # If None, compare all
    risk_free_rate: float = 0.02


class RebalanceRequest(BaseModel):
    """Rebalancing request"""
    target_weights: Dict[str, float]
    current_positions: Dict[str, float]  # Current position sizes
    current_prices: Dict[str, float]  # Current prices
    strategy: str  # Rebalancing strategy
    rebalancing_cost: float = 0.001
    # Strategy-specific parameters
    threshold: Optional[float] = None
    tolerance: Optional[float] = None
    rebalance_frequency_days: Optional[int] = None


@router.post("/allocate")
async def allocate_portfolio(request: AllocationRequest):
    """
    Calculate optimal portfolio allocation

    Supports multiple allocation strategies:
    - equal_weight: Equal allocation across all assets
    - market_cap_weight: Weighted by market capitalization
    - risk_parity: Equal risk contribution from each asset
    - mean_variance: Mean-variance optimization (Markowitz)
    - min_variance: Minimum variance portfolio
    - max_sharpe: Maximum Sharpe ratio portfolio
    - max_diversification: Maximum diversification ratio
    - hierarchical_risk_parity: HRP using hierarchical clustering
    """
    try:
        # Create allocator
        allocator = PortfolioAllocator(
            risk_free_rate=request.risk_free_rate,
            max_position_size=request.max_position_size,
            min_position_size=request.min_position_size
        )

        # Parse strategy
        try:
            strategy = AllocationStrategy(request.strategy)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid allocation strategy: {request.strategy}"
            )

        # Prepare returns data if provided
        returns_df = None
        if request.returns_data:
            returns_df = pd.DataFrame(request.returns_data)

        # Calculate allocation
        weights = allocator.allocate(
            symbols=request.symbols,
            strategy=strategy,
            returns=returns_df,
            market_caps=request.market_caps
        )

        return {
            "status": "success",
            "strategy": request.strategy,
            "weights": weights,
            "symbols": request.symbols,
            "total_weight": sum(weights.values())
        }

    except Exception as e:
        logger.error(f"Allocation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-allocations")
async def compare_allocations(request: CompareAllocationsRequest):
    """
    Compare multiple allocation strategies

    Returns allocation weights for each strategy
    """
    try:
        # Create allocator
        allocator = PortfolioAllocator(
            risk_free_rate=request.risk_free_rate
        )

        # Prepare returns data
        returns_df = pd.DataFrame(request.returns_data)

        # Parse strategies
        if request.strategies:
            strategies = [AllocationStrategy(s) for s in request.strategies]
        else:
            strategies = None

        # Compare allocations
        comparison_df = allocator.compare_allocations(
            symbols=request.symbols,
            returns=returns_df,
            strategies=strategies
        )

        return {
            "status": "success",
            "comparison": comparison_df.to_dict(),
            "symbols": request.symbols
        }

    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rebalance")
async def rebalance_portfolio(request: RebalanceRequest):
    """
    Calculate rebalancing trades

    Supports multiple rebalancing strategies:
    - periodic: Rebalance on fixed schedule
    - threshold: Rebalance when weights deviate by threshold
    - tolerance_band: Rebalance when outside tolerance bands
    - volatility_based: Rebalance based on volatility changes
    """
    try:
        # Create rebalancer
        rebalancer = PortfolioRebalancer(
            target_weights=request.target_weights,
            rebalancing_cost=request.rebalancing_cost
        )

        # Parse strategy
        try:
            strategy = RebalancingStrategy(request.strategy)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid rebalancing strategy: {request.strategy}"
            )

        # Calculate current weights
        current_weights = rebalancer.calculate_current_weights(
            request.current_positions,
            request.current_prices
        )

        # Prepare kwargs based on strategy
        kwargs = {}
        if strategy == RebalancingStrategy.PERIODIC:
            kwargs['rebalance_frequency_days'] = request.rebalance_frequency_days or 30
        elif strategy == RebalancingStrategy.THRESHOLD:
            kwargs['threshold'] = request.threshold or 0.05
        elif strategy == RebalancingStrategy.TOLERANCE_BAND:
            kwargs['tolerance'] = request.tolerance or 0.2

        # Calculate rebalancing
        trades = rebalancer.rebalance(
            current_date=datetime.now(),
            current_positions=request.current_positions,
            current_prices=request.current_prices,
            strategy=strategy,
            **kwargs
        )

        if trades is None:
            return {
                "status": "success",
                "rebalancing_needed": False,
                "message": "Portfolio is within target allocation"
            }

        # Calculate total portfolio value
        total_value = sum(
            request.current_positions.get(s, 0) * request.current_prices.get(s, 0)
            for s in request.target_weights.keys()
        )

        # Estimate cost
        cost = rebalancer.estimate_rebalancing_cost(trades, request.current_prices)
        cost_pct = (cost / total_value * 100) if total_value > 0 else 0

        return {
            "status": "success",
            "rebalancing_needed": True,
            "current_weights": current_weights,
            "target_weights": request.target_weights,
            "trades": trades,
            "estimated_cost": cost,
            "estimated_cost_pct": cost_pct,
            "total_portfolio_value": total_value
        }

    except Exception as e:
        logger.error(f"Rebalancing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/allocation-strategies")
async def get_allocation_strategies():
    """
    Get available allocation strategies
    """
    return {
        "strategies": [
            {
                "name": "equal_weight",
                "description": "Equal allocation across all assets",
                "requires_data": False
            },
            {
                "name": "market_cap_weight",
                "description": "Weighted by market capitalization",
                "requires_data": False,
                "requires": "market_caps"
            },
            {
                "name": "risk_parity",
                "description": "Equal risk contribution from each asset",
                "requires_data": True
            },
            {
                "name": "mean_variance",
                "description": "Mean-variance optimization (Markowitz)",
                "requires_data": True
            },
            {
                "name": "min_variance",
                "description": "Minimum variance portfolio",
                "requires_data": True
            },
            {
                "name": "max_sharpe",
                "description": "Maximum Sharpe ratio portfolio",
                "requires_data": True
            },
            {
                "name": "max_diversification",
                "description": "Maximum diversification ratio",
                "requires_data": True
            },
            {
                "name": "hierarchical_risk_parity",
                "description": "HRP using hierarchical clustering",
                "requires_data": True
            }
        ]
    }


@router.get("/rebalancing-strategies")
async def get_rebalancing_strategies():
    """
    Get available rebalancing strategies
    """
    return {
        "strategies": [
            {
                "name": "periodic",
                "description": "Rebalance on fixed schedule",
                "parameters": ["rebalance_frequency_days"]
            },
            {
                "name": "threshold",
                "description": "Rebalance when weights deviate by threshold",
                "parameters": ["threshold"]
            },
            {
                "name": "tolerance_band",
                "description": "Rebalance when outside tolerance bands",
                "parameters": ["tolerance"]
            },
            {
                "name": "volatility_based",
                "description": "Rebalance based on volatility changes",
                "parameters": ["volatility_threshold", "returns"]
            }
        ]
    }
