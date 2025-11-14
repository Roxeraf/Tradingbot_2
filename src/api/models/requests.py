"""
API Request Models
Pydantic models for API request validation
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class StartBotRequest(BaseModel):
    """Request to start the trading bot"""
    config_override: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional configuration overrides"
    )


class StopBotRequest(BaseModel):
    """Request to stop the trading bot"""
    cancel_orders: bool = Field(
        default=True,
        description="Cancel all open orders when stopping"
    )
    close_positions: bool = Field(
        default=False,
        description="Close all open positions when stopping"
    )


class PlaceOrderRequest(BaseModel):
    """Request to place a manual order"""
    symbol: str = Field(..., description="Trading pair symbol")
    order_type: str = Field(..., description="Order type: market, limit")
    side: str = Field(..., description="Order side: buy, sell")
    amount: float = Field(..., gt=0, description="Order amount")
    price: Optional[float] = Field(None, gt=0, description="Order price (for limit orders)")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")


class CancelOrderRequest(BaseModel):
    """Request to cancel an order"""
    order_id: str = Field(..., description="Order ID to cancel")
    symbol: str = Field(..., description="Trading pair symbol")


class UpdateStrategyRequest(BaseModel):
    """Request to update strategy parameters"""
    strategy_name: Optional[str] = Field(None, description="Change strategy")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy parameters to update"
    )


class BacktestRequest(BaseModel):
    """Request to run a backtest"""
    strategy_name: str = Field(..., description="Strategy to backtest")
    strategy_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy parameters"
    )
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Candle timeframe")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, gt=0)
    commission_rate: float = Field(default=0.001, ge=0)
    slippage_rate: float = Field(default=0.0005, ge=0)


class OptimizeStrategyRequest(BaseModel):
    """Request to optimize strategy parameters"""
    strategy_name: str = Field(..., description="Strategy to optimize")
    param_grid: Dict[str, List[Any]] = Field(
        ...,
        description="Parameter grid for optimization"
    )
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Candle timeframe")
    optimization_metric: str = Field(
        default="sharpe_ratio",
        description="Metric to optimize"
    )
    initial_capital: float = Field(default=10000.0, gt=0)


class UpdateConfigRequest(BaseModel):
    """Request to update bot configuration"""
    config_updates: Dict[str, Any] = Field(
        ...,
        description="Configuration values to update"
    )
