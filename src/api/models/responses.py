"""
API Response Models
Pydantic models for API responses
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class BotStatus(str, Enum):
    """Bot status enumeration"""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"


class StatusResponse(BaseModel):
    """Bot status response"""
    status: BotStatus
    uptime_seconds: Optional[float] = None
    current_strategy: Optional[str] = None
    active_pairs: List[str] = []
    open_positions: int = 0
    total_trades: int = 0
    current_equity: Optional[float] = None
    message: Optional[str] = None


class PerformanceResponse(BaseModel):
    """Performance metrics response"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percentage: float
    avg_trade_pnl: float
    best_trade: float
    worst_trade: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    current_equity: float
    initial_equity: float


class TradeResponse(BaseModel):
    """Trade information response"""
    id: Optional[int] = None
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float] = None
    amount: float
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    entry_time: datetime
    exit_time: Optional[datetime] = None
    status: str
    strategy: Optional[str] = None


class PositionResponse(BaseModel):
    """Position information response"""
    symbol: str
    side: str
    entry_price: float
    current_price: float
    amount: float
    unrealized_pnl: float
    unrealized_pnl_percentage: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime


class OrderResponse(BaseModel):
    """Order information response"""
    id: str
    symbol: str
    type: str
    side: str
    amount: float
    price: Optional[float] = None
    status: str
    filled: Optional[float] = None
    remaining: Optional[float] = None
    timestamp: datetime


class StrategyInfoResponse(BaseModel):
    """Strategy information response"""
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]
    required_history: int


class BacktestResultResponse(BaseModel):
    """Backtest results response"""
    strategy_name: str
    strategy_params: Dict[str, Any]
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    annualized_return: float
    num_trades: int
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    expectancy: float


class BalanceResponse(BaseModel):
    """Account balance response"""
    balances: Dict[str, float]
    total_value_usd: Optional[float] = None
    timestamp: datetime


class HealthResponse(BaseModel):
    """API health check response"""
    status: str = "healthy"
    timestamp: datetime
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
