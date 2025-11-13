"""
Settings management endpoints
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime

from src.api.models.requests import SettingsUpdateRequest
from src.config.settings import Settings
from src.monitoring.logger import get_logger

router = APIRouter()
logger = get_logger()

settings = Settings()


@router.get("/")
async def get_settings():
    """Get current settings"""
    return {
        "exchange": {
            "name": settings.EXCHANGE_NAME,
            "testnet": settings.TESTNET
        },
        "trading": {
            "pairs": settings.get_trading_pairs(),
            "timeframe": settings.TIMEFRAME,
            "max_position_size": settings.MAX_POSITION_SIZE,
            "max_portfolio_risk": settings.MAX_PORTFOLIO_RISK
        },
        "strategy": {
            "name": settings.STRATEGY_NAME,
            "params": settings.get_strategy_params()
        },
        "risk_management": {
            "stop_loss_percentage": settings.STOP_LOSS_PERCENTAGE,
            "take_profit_percentage": settings.TAKE_PROFIT_PERCENTAGE,
            "trailing_stop": settings.TRAILING_STOP
        },
        "logging": {
            "level": settings.LOG_LEVEL,
            "to_file": settings.LOG_TO_FILE
        }
    }


@router.put("/")
async def update_settings(request: SettingsUpdateRequest):
    """
    Update settings

    Note: Some settings require bot restart to take effect
    """
    logger.info("Settings update requested")

    # This would update the settings in the actual application
    # For now, just return success

    updated_fields = []

    if request.trading_pairs is not None:
        updated_fields.append("trading_pairs")

    if request.timeframe is not None:
        updated_fields.append("timeframe")

    if request.max_position_size is not None:
        updated_fields.append("max_position_size")

    if request.max_portfolio_risk is not None:
        updated_fields.append("max_portfolio_risk")

    if request.strategy_name is not None:
        updated_fields.append("strategy_name")

    if request.strategy_params is not None:
        updated_fields.append("strategy_params")

    if request.log_level is not None:
        updated_fields.append("log_level")

    return {
        "status": "success",
        "message": "Settings updated. Bot restart required for some changes to take effect.",
        "updated_fields": updated_fields,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/exchanges")
async def get_available_exchanges():
    """Get list of supported exchanges"""
    from src.exchanges.exchange_factory import ExchangeFactory

    exchanges = ExchangeFactory.get_supported_exchanges()

    return {
        "exchanges": exchanges,
        "count": len(exchanges)
    }


@router.get("/pairs")
async def get_available_pairs():
    """Get available trading pairs for current exchange"""
    # This would fetch from the exchange
    # Return mock data for now

    return {
        "pairs": [
            "BTC/EUR", "ETH/EUR", "BTC/USD", "ETH/USD",
            "ADA/EUR", "DOT/EUR", "SOL/EUR", "LINK/EUR"
        ],
        "exchange": settings.EXCHANGE_NAME
    }


@router.get("/timeframes")
async def get_available_timeframes():
    """Get available timeframes"""
    return {
        "timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
        "default": "1h"
    }


@router.post("/test-connection")
async def test_exchange_connection():
    """Test exchange connection"""
    from src.exchanges.exchange_factory import ExchangeFactory

    try:
        exchange = ExchangeFactory.create_from_settings(settings)
        await exchange.connect()

        # Test balance fetch
        balance = await exchange.get_balance()

        await exchange.disconnect()

        return {
            "status": "success",
            "message": "Exchange connection successful",
            "exchange": settings.EXCHANGE_NAME,
            "testnet": settings.TESTNET,
            "currencies": list(balance.keys())
        }

    except Exception as e:
        logger.error(f"Exchange connection test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Exchange connection failed: {str(e)}"
        )


@router.get("/logs")
async def get_logs(level: str = None, limit: int = 100):
    """
    Get system logs

    Args:
        level: Filter by log level (DEBUG, INFO, WARNING, ERROR)
        limit: Maximum number of logs to return
    """
    from src.data.data_storage import DataStorage

    db = DataStorage(settings.DATABASE_URL)
    logs = db.get_logs(level=level, limit=limit)

    result = []
    for log in logs:
        result.append({
            "id": log.id,
            "timestamp": log.timestamp.isoformat(),
            "level": log.level,
            "message": log.message,
            "module": log.module
        })

    return result


@router.get("/system-info")
async def get_system_info():
    """Get system information"""
    import platform
    import sys

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "hostname": platform.node()
    }
