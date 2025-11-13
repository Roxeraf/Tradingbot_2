"""
Strategy management endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime
import json

from src.api.models.requests import StrategyConfigRequest
from src.api.models.responses import StrategyResponse
from src.config.settings import Settings
from src.data.data_storage import DataStorage
from src.strategies.strategy_factory import StrategyFactory
from src.monitoring.logger import get_logger

router = APIRouter()
logger = get_logger()

settings = Settings()
db = DataStorage(settings.DATABASE_URL)


@router.get("/", response_model=List[StrategyResponse])
async def list_strategies(active_only: bool = False):
    """
    List all strategy configurations

    Args:
        active_only: Only return active strategies
    """
    configs = db.get_strategy_configs(active_only=active_only)

    result = []
    for config in configs:
        params = json.loads(config.parameters) if isinstance(config.parameters, str) else config.parameters

        result.append(StrategyResponse(
            id=config.id,
            name=config.name,
            strategy_type=config.strategy_type,
            parameters=params,
            is_active=config.is_active,
            description=config.description,
            created_at=config.created_at.isoformat(),
            updated_at=config.updated_at.isoformat()
        ))

    return result


@router.get("/available")
async def list_available_strategies():
    """List all available strategy types"""
    strategies = StrategyFactory.get_supported_strategies()

    result = []
    for strategy_name in set(strategies):
        try:
            info = StrategyFactory.get_strategy_info(strategy_name)
            result.append({
                "name": strategy_name,
                "class_name": info['name'],
                "description": info['description']
            })
        except:
            result.append({
                "name": strategy_name,
                "class_name": strategy_name,
                "description": "No description available"
            })

    return result


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(strategy_id: int):
    """Get strategy configuration by ID"""
    # Get from database by ID
    configs = db.get_strategy_configs()
    config = next((c for c in configs if c.id == strategy_id), None)

    if not config:
        raise HTTPException(status_code=404, detail="Strategy not found")

    params = json.loads(config.parameters) if isinstance(config.parameters, str) else config.parameters

    return StrategyResponse(
        id=config.id,
        name=config.name,
        strategy_type=config.strategy_type,
        parameters=params,
        is_active=config.is_active,
        description=config.description,
        created_at=config.created_at.isoformat(),
        updated_at=config.updated_at.isoformat()
    )


@router.get("/name/{strategy_name}", response_model=StrategyResponse)
async def get_strategy_by_name(strategy_name: str):
    """Get strategy configuration by name"""
    config = db.get_strategy_config(strategy_name)

    if not config:
        raise HTTPException(status_code=404, detail="Strategy not found")

    params = json.loads(config.parameters) if isinstance(config.parameters, str) else config.parameters

    return StrategyResponse(
        id=config.id,
        name=config.name,
        strategy_type=config.strategy_type,
        parameters=params,
        is_active=config.is_active,
        description=config.description,
        created_at=config.created_at.isoformat(),
        updated_at=config.updated_at.isoformat()
    )


@router.post("/", response_model=StrategyResponse)
async def create_strategy(request: StrategyConfigRequest):
    """Create new strategy configuration"""
    logger.info(f"Creating strategy configuration: {request.name}")

    # Validate strategy type
    if request.strategy_type.lower() not in [s.lower() for s in StrategyFactory.get_supported_strategies()]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported strategy type: {request.strategy_type}"
        )

    # Check if name already exists
    existing = db.get_strategy_config(request.name)
    if existing:
        raise HTTPException(status_code=400, detail="Strategy with this name already exists")

    # Save to database
    config_data = {
        "name": request.name,
        "strategy_type": request.strategy_type,
        "parameters": request.parameters,
        "is_active": request.is_active,
        "description": request.description
    }

    config = db.save_strategy_config(config_data)

    params = json.loads(config.parameters) if isinstance(config.parameters, str) else config.parameters

    return StrategyResponse(
        id=config.id,
        name=config.name,
        strategy_type=config.strategy_type,
        parameters=params,
        is_active=config.is_active,
        description=config.description,
        created_at=config.created_at.isoformat(),
        updated_at=config.updated_at.isoformat()
    )


@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(strategy_id: int, request: StrategyConfigRequest):
    """Update strategy configuration"""
    logger.info(f"Updating strategy configuration: {request.name}")

    # Update in database
    config_data = {
        "name": request.name,
        "strategy_type": request.strategy_type,
        "parameters": request.parameters,
        "is_active": request.is_active,
        "description": request.description
    }

    config = db.save_strategy_config(config_data)

    params = json.loads(config.parameters) if isinstance(config.parameters, str) else config.parameters

    return StrategyResponse(
        id=config.id,
        name=config.name,
        strategy_type=config.strategy_type,
        parameters=params,
        is_active=config.is_active,
        description=config.description,
        created_at=config.created_at.isoformat(),
        updated_at=config.updated_at.isoformat()
    )


@router.post("/{strategy_id}/activate")
async def activate_strategy(strategy_id: int):
    """Activate a strategy"""
    logger.info(f"Activating strategy: {strategy_id}")

    # Get strategy
    configs = db.get_strategy_configs()
    config = next((c for c in configs if c.id == strategy_id), None)

    if not config:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Deactivate all other strategies
    for c in configs:
        if c.id != strategy_id:
            db.save_strategy_config({
                "name": c.name,
                "strategy_type": c.strategy_type,
                "parameters": json.loads(c.parameters) if isinstance(c.parameters, str) else c.parameters,
                "is_active": False,
                "description": c.description
            })

    # Activate this strategy
    params = json.loads(config.parameters) if isinstance(config.parameters, str) else config.parameters
    db.save_strategy_config({
        "name": config.name,
        "strategy_type": config.strategy_type,
        "parameters": params,
        "is_active": True,
        "description": config.description
    })

    return {"status": "success", "message": f"Strategy {config.name} activated"}


@router.post("/{strategy_id}/deactivate")
async def deactivate_strategy(strategy_id: int):
    """Deactivate a strategy"""
    logger.info(f"Deactivating strategy: {strategy_id}")

    # Get strategy
    configs = db.get_strategy_configs()
    config = next((c for c in configs if c.id == strategy_id), None)

    if not config:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Deactivate
    params = json.loads(config.parameters) if isinstance(config.parameters, str) else config.parameters
    db.save_strategy_config({
        "name": config.name,
        "strategy_type": config.strategy_type,
        "parameters": params,
        "is_active": False,
        "description": config.description
    })

    return {"status": "success", "message": f"Strategy {config.name} deactivated"}


@router.post("/{strategy_id}/test")
async def test_strategy(strategy_id: int):
    """Test strategy with current market data"""
    logger.info(f"Testing strategy: {strategy_id}")

    # Get strategy configuration
    configs = db.get_strategy_configs()
    config = next((c for c in configs if c.id == strategy_id), None)

    if not config:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Create strategy instance
    params = json.loads(config.parameters) if isinstance(config.parameters, str) else config.parameters

    try:
        strategy = StrategyFactory.create(config.strategy_type, params)

        return {
            "status": "success",
            "message": f"Strategy {config.name} is valid",
            "strategy_name": strategy.name,
            "parameters": strategy.get_parameters()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Strategy test failed: {str(e)}")


@router.get("/{strategy_id}/performance")
async def get_strategy_performance(strategy_id: int, days: int = 30):
    """Get strategy performance metrics"""
    from datetime import timedelta

    # Get strategy
    configs = db.get_strategy_configs()
    config = next((c for c in configs if c.id == strategy_id), None)

    if not config:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Get trades for this strategy
    since = datetime.now() - timedelta(days=days)
    trades = db.get_trades(since=since)

    # Filter by strategy name
    strategy_trades = [t for t in trades if t.strategy_name == config.name]

    if not strategy_trades:
        return {
            "strategy_name": config.name,
            "num_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_pnl": 0
        }

    # Calculate metrics
    winning_trades = [t for t in strategy_trades if t.pnl and t.pnl > 0]
    total_pnl = sum(t.pnl for t in strategy_trades if t.pnl)
    win_rate = (len(winning_trades) / len(strategy_trades)) * 100 if strategy_trades else 0

    return {
        "strategy_name": config.name,
        "num_trades": len(strategy_trades),
        "num_winning_trades": len(winning_trades),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(strategy_trades) if strategy_trades else 0
    }
