"""
WebSocket endpoints for real-time updates
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import asyncio
import json
from datetime import datetime

from src.monitoring.logger import get_logger

router = APIRouter()
logger = get_logger()


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and store new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific client"""
        await websocket.send_json(message)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/live")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates

    Streams:
    - Price updates
    - Position updates
    - Order updates
    - Trade executions
    - Bot status
    - Logs
    - Performance metrics
    """
    await manager.connect(websocket)

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "WebSocket connection established",
            "timestamp": datetime.now().isoformat()
        })

        # Start broadcasting updates
        while True:
            # Send periodic updates
            await broadcast_updates()
            await asyncio.sleep(1)  # Update every second

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_updates():
    """Broadcast updates to all connected clients"""
    # This function would be called by the trading bot to send real-time updates

    # Example: Price update
    await broadcast_price_update("BTC/EUR", 45000.50)

    # Example: Portfolio update
    await broadcast_portfolio_update({
        "total_value": 50000.0,
        "unrealized_pnl": 5000.0,
        "realized_pnl": 2000.0
    })


async def broadcast_price_update(symbol: str, price: float):
    """Broadcast price update to all clients"""
    message = {
        "type": "price_update",
        "symbol": symbol,
        "price": price,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(message)


async def broadcast_trade_execution(trade: Dict[str, Any]):
    """Broadcast trade execution to all clients"""
    message = {
        "type": "trade_execution",
        "trade": trade,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(message)


async def broadcast_position_update(position: Dict[str, Any]):
    """Broadcast position update to all clients"""
    message = {
        "type": "position_update",
        "position": position,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(message)


async def broadcast_order_update(order: Dict[str, Any]):
    """Broadcast order update to all clients"""
    message = {
        "type": "order_update",
        "order": order,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(message)


async def broadcast_bot_status(status: Dict[str, Any]):
    """Broadcast bot status to all clients"""
    message = {
        "type": "bot_status",
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(message)


async def broadcast_log(level: str, log_message: str):
    """Broadcast log message to all clients"""
    message = {
        "type": "log",
        "level": level,
        "message": log_message,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(message)


async def broadcast_portfolio_update(portfolio: Dict[str, Any]):
    """Broadcast portfolio update to all clients"""
    message = {
        "type": "portfolio_update",
        "portfolio": portfolio,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(message)


async def broadcast_signal(signal: Dict[str, Any]):
    """Broadcast trading signal to all clients"""
    message = {
        "type": "signal",
        "signal": signal,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(message)


# Export manager for use in other modules
__all__ = ['router', 'manager', 'broadcast_price_update', 'broadcast_trade_execution',
           'broadcast_position_update', 'broadcast_order_update', 'broadcast_bot_status',
           'broadcast_log', 'broadcast_portfolio_update', 'broadcast_signal']
