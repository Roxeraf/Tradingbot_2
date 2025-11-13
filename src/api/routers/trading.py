"""
Trading endpoints for the API
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime

from src.api.models.requests import (
    PlaceOrderRequest, CancelOrderRequest,
    ClosePositionRequest, UpdatePositionRequest
)
from src.api.models.responses import (
    StatusResponse, BalanceResponse, PositionResponse,
    OrderResponse, TradeResponse, PerformanceResponse
)
from src.config.settings import Settings
from src.data.data_storage import DataStorage
from src.monitoring.logger import get_logger

router = APIRouter()
logger = get_logger()

# Global state (in production, use dependency injection or shared state management)
settings = Settings()
db = DataStorage(settings.DATABASE_URL)


@router.get("/status", response_model=StatusResponse)
async def get_bot_status():
    """Get trading bot status"""
    # This would connect to the actual running bot instance
    # For now, return mock data
    return StatusResponse(
        is_running=True,
        exchange=settings.EXCHANGE_NAME,
        strategy=settings.STRATEGY_NAME,
        trading_pairs=settings.get_trading_pairs(),
        uptime_seconds=3600.0,  # Mock
        last_update=datetime.now().isoformat()
    )


@router.post("/start")
async def start_bot():
    """Start the trading bot"""
    # This would start the bot instance
    logger.info("Start bot requested via API")
    return {"status": "started", "message": "Trading bot started successfully"}


@router.post("/stop")
async def stop_bot():
    """Stop the trading bot"""
    # This would stop the bot instance
    logger.info("Stop bot requested via API")
    return {"status": "stopped", "message": "Trading bot stopped successfully"}


@router.get("/balance", response_model=BalanceResponse)
async def get_balance():
    """Get account balance"""
    # This would fetch from the exchange
    # For now, return mock data
    return BalanceResponse(
        balances={"EUR": 10000.0, "BTC": 0.5, "ETH": 2.0},
        total_value_eur=45000.0,
        timestamp=datetime.now().isoformat()
    )


@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(symbol: Optional[str] = None):
    """
    Get open positions

    Args:
        symbol: Filter by symbol (optional)
    """
    positions = db.get_open_positions()

    result = []
    for pos in positions:
        # Calculate current PnL (would need current price from exchange)
        result.append(PositionResponse(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            current_price=None,  # Would fetch from exchange
            size=pos.size,
            unrealized_pnl=None,
            unrealized_pnl_percentage=None,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            entry_time=pos.entry_time.isoformat(),
            strategy_name=pos.strategy_name or "unknown"
        ))

    if symbol:
        result = [p for p in result if p.symbol == symbol]

    return result


@router.post("/positions/close")
async def close_position(request: ClosePositionRequest):
    """Close a position"""
    logger.info(f"Close position requested: {request.symbol}")

    # This would close the position via the trading bot
    db.close_position(request.symbol)

    return {
        "status": "success",
        "message": f"Position {request.symbol} closed",
        "symbol": request.symbol,
        "reason": request.reason
    }


@router.post("/positions/close-all")
async def close_all_positions():
    """Close all open positions"""
    logger.info("Close all positions requested")

    positions = db.get_open_positions()
    closed_count = 0

    for pos in positions:
        db.close_position(pos.symbol)
        closed_count += 1

    return {
        "status": "success",
        "message": f"Closed {closed_count} positions",
        "count": closed_count
    }


@router.put("/positions/update")
async def update_position(request: UpdatePositionRequest):
    """Update position stop loss and/or take profit"""
    logger.info(f"Update position requested: {request.symbol}")

    # This would update the position in the portfolio manager
    # For now, just return success

    return {
        "status": "success",
        "message": f"Position {request.symbol} updated",
        "symbol": request.symbol,
        "stop_loss": request.stop_loss,
        "take_profit": request.take_profit
    }


@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(symbol: Optional[str] = None, status: Optional[str] = None):
    """
    Get orders

    Args:
        symbol: Filter by symbol (optional)
        status: Filter by status (optional)
    """
    # This would fetch from order manager
    # Return mock data for now
    return []


@router.post("/orders", response_model=OrderResponse)
async def place_order(request: PlaceOrderRequest):
    """Place a new order"""
    logger.info(f"Place order requested: {request.symbol} {request.side} {request.amount}")

    # Validate request
    if request.order_type == "limit" and request.price is None:
        raise HTTPException(status_code=400, detail="Price required for limit orders")

    # This would place order via order manager
    # Return mock response for now

    return OrderResponse(
        id="mock_order_123",
        symbol=request.symbol,
        type=request.order_type,
        side=request.side,
        amount=request.amount,
        price=request.price,
        status="pending",
        timestamp=datetime.now().isoformat()
    )


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str, symbol: str):
    """Cancel an order"""
    logger.info(f"Cancel order requested: {order_id}")

    # This would cancel via order manager

    return {
        "status": "success",
        "message": f"Order {order_id} cancelled",
        "order_id": order_id
    }


@router.get("/trades", response_model=List[TradeResponse])
async def get_trades(
    symbol: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Get trade history

    Args:
        symbol: Filter by symbol (optional)
        limit: Maximum number of trades to return
        offset: Number of trades to skip
    """
    trades = db.get_trades(symbol=symbol, limit=limit)

    result = []
    for trade in trades:
        result.append(TradeResponse(
            id=trade.id,
            symbol=trade.symbol,
            side=trade.side,
            amount=trade.amount,
            price=trade.price,
            pnl=trade.pnl,
            pnl_percentage=trade.pnl_percentage,
            timestamp=trade.timestamp.isoformat(),
            strategy_name=trade.strategy_name
        ))

    return result[offset:offset+limit]


@router.get("/performance", response_model=PerformanceResponse)
async def get_performance():
    """Get performance metrics"""
    # This would calculate from portfolio manager
    # Return mock data for now

    return PerformanceResponse(
        total_value=50000.0,
        cash_balance=10000.0,
        initial_capital=40000.0,
        unrealized_pnl=5000.0,
        realized_pnl=5000.0,
        total_pnl=10000.0,
        total_return=25.0,
        num_open_positions=3,
        num_closed_trades=50,
        num_winning_trades=35,
        num_losing_trades=15,
        win_rate=70.0,
        current_exposure=0.15
    )


@router.get("/performance/history")
async def get_performance_history(days: int = 30):
    """
    Get performance history

    Args:
        days: Number of days of history to return
    """
    from datetime import timedelta

    since = datetime.now() - timedelta(days=days)
    history = db.get_performance_history(since=since, limit=days * 24)

    result = []
    for perf in history:
        result.append({
            "timestamp": perf.timestamp.isoformat(),
            "total_balance": perf.total_balance,
            "unrealized_pnl": perf.unrealized_pnl,
            "realized_pnl": perf.realized_pnl,
            "num_open_positions": perf.num_open_positions,
            "win_rate": perf.win_rate
        })

    return result
