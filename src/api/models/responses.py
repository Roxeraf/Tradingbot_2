"""
Pydantic response models for API endpoints
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class StatusResponse(BaseModel):
    """Bot status response"""
    is_running: bool
    exchange: str
    strategy: str
    trading_pairs: List[str]
    uptime_seconds: float
    last_update: str


class BalanceResponse(BaseModel):
    """Account balance response"""
    balances: Dict[str, float]
    total_value_eur: Optional[float] = None
    timestamp: str


class PositionResponse(BaseModel):
    """Position response"""
    symbol: str
    side: str
    entry_price: float
    current_price: Optional[float] = None
    size: float
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_percentage: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: str
    strategy_name: str


class OrderResponse(BaseModel):
    """Order response"""
    id: str
    symbol: str
    type: str
    side: str
    amount: float
    price: Optional[float] = None
    status: str
    timestamp: str


class TradeResponse(BaseModel):
    """Trade response"""
    id: int
    symbol: str
    side: str
    amount: float
    price: float
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    timestamp: str
    strategy_name: Optional[str] = None


class PerformanceResponse(BaseModel):
    """Performance metrics response"""
    total_value: float
    cash_balance: float
    initial_capital: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    total_return: float
    num_open_positions: int
    num_closed_trades: int
    num_winning_trades: int
    num_losing_trades: int
    win_rate: float
    current_exposure: float


class BacktestResultResponse(BaseModel):
    """Backtest result response"""
    id: int
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    num_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    created_at: str


class StrategyResponse(BaseModel):
    """Strategy configuration response"""
    id: int
    name: str
    strategy_type: str
    parameters: Dict[str, Any]
    is_active: bool
    description: Optional[str] = None
    created_at: str
    updated_at: str


class LogResponse(BaseModel):
    """Log entry response"""
    id: int
    timestamp: str
    level: str
    message: str
    module: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: str
