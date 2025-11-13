"""
Pydantic request models for API endpoints
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class PlaceOrderRequest(BaseModel):
    """Request model for placing an order"""
    symbol: str = Field(..., description="Trading symbol (e.g., BTC/EUR)")
    side: str = Field(..., description="Order side: buy or sell")
    amount: float = Field(..., gt=0, description="Order amount")
    order_type: str = Field(default="market", description="Order type: market or limit")
    price: Optional[float] = Field(None, description="Limit price (required for limit orders)")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")


class CancelOrderRequest(BaseModel):
    """Request model for canceling an order"""
    order_id: str = Field(..., description="Order ID to cancel")
    symbol: str = Field(..., description="Trading symbol")


class ClosePositionRequest(BaseModel):
    """Request model for closing a position"""
    symbol: str = Field(..., description="Trading symbol")
    reason: str = Field(default="manual", description="Reason for closing")


class UpdatePositionRequest(BaseModel):
    """Request model for updating position stops"""
    symbol: str = Field(..., description="Trading symbol")
    stop_loss: Optional[float] = Field(None, description="New stop loss price")
    take_profit: Optional[float] = Field(None, description="New take profit price")


class StrategyConfigRequest(BaseModel):
    """Request model for strategy configuration"""
    name: str = Field(..., description="Strategy name")
    strategy_type: str = Field(..., description="Strategy type")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    is_active: bool = Field(default=False, description="Whether strategy is active")
    description: Optional[str] = Field(None, description="Strategy description")


class BacktestRequest(BaseModel):
    """Request model for running a backtest"""
    strategy_name: str = Field(..., description="Strategy to backtest")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(default="1h", description="Candlestick timeframe")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000, gt=0, description="Initial capital")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    enable_stop_loss: bool = Field(default=True, description="Enable stop loss")
    enable_take_profit: bool = Field(default=True, description="Enable take profit")


class SettingsUpdateRequest(BaseModel):
    """Request model for updating settings"""
    trading_pairs: Optional[List[str]] = Field(None, description="Trading pairs")
    timeframe: Optional[str] = Field(None, description="Timeframe")
    max_position_size: Optional[float] = Field(None, ge=0, le=1, description="Max position size")
    max_portfolio_risk: Optional[float] = Field(None, ge=0, le=1, description="Max portfolio risk")
    strategy_name: Optional[str] = Field(None, description="Active strategy name")
    strategy_params: Optional[Dict[str, Any]] = Field(None, description="Strategy parameters")
    log_level: Optional[str] = Field(None, description="Logging level")
