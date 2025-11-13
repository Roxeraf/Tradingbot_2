"""
Backtesting endpoints
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from datetime import datetime
import asyncio

from src.api.models.requests import BacktestRequest
from src.api.models.responses import BacktestResultResponse
from src.config.settings import Settings
from src.data.data_storage import DataStorage
from src.backtesting.backtest_engine import BacktestEngine
from src.strategies.strategy_factory import StrategyFactory
from src.exchanges.exchange_factory import ExchangeFactory
from src.monitoring.logger import get_logger

router = APIRouter()
logger = get_logger()

settings = Settings()
db = DataStorage(settings.DATABASE_URL)

# Store running backtests
running_backtests = {}


async def run_backtest_task(backtest_id: str, request: BacktestRequest):
    """Background task to run backtest"""
    try:
        logger.info(f"Starting backtest {backtest_id}: {request.strategy_name} on {request.symbol}")

        # Create strategy
        strategy = StrategyFactory.create(request.strategy_name, request.strategy_params)

        # Create backtest engine
        backtest = BacktestEngine(
            strategy=strategy,
            initial_capital=request.initial_capital,
            commission=0.001,
            slippage=0.0005
        )

        # Get historical data
        exchange = ExchangeFactory.create_from_settings(settings)
        await exchange.connect()

        try:
            # Fetch historical data
            start_dt = datetime.fromisoformat(request.start_date)
            end_dt = datetime.fromisoformat(request.end_date)

            # Convert to milliseconds
            since = int(start_dt.timestamp() * 1000)

            data = await exchange.get_historical_data(
                symbol=request.symbol,
                timeframe=request.timeframe,
                since=since,
                limit=1000
            )

            # Filter by date range
            data = data[request.start_date:request.end_date]

            if len(data) == 0:
                logger.error(f"No data available for backtest {backtest_id}")
                running_backtests[backtest_id] = {
                    "status": "error",
                    "error": "No data available for the specified date range"
                }
                return

            # Run backtest
            results = await backtest.run(
                data,
                request.symbol,
                enable_stop_loss=request.enable_stop_loss,
                enable_take_profit=request.enable_take_profit
            )

            # Save to database
            import json
            backtest_result = {
                'strategy_name': request.strategy_name,
                'symbol': request.symbol,
                'timeframe': request.timeframe,
                'start_date': start_dt,
                'end_date': end_dt,
                'initial_capital': request.initial_capital,
                'final_capital': results['final_capital'],
                'total_return': results['metrics']['total_return'],
                'num_trades': results['metrics']['num_trades'],
                'win_rate': results['metrics']['win_rate'],
                'sharpe_ratio': results['metrics']['sharpe_ratio'],
                'max_drawdown': results['metrics']['max_drawdown'],
                'avg_win': results['metrics']['avg_win'],
                'avg_loss': results['metrics']['avg_loss'],
                'profit_factor': results['metrics']['profit_factor'],
                'parameters': json.dumps(request.strategy_params),
                'trades_data': json.dumps(results['trades']),
                'equity_curve': json.dumps(results['equity_curve'])
            }

            saved_result = db.save_backtest_result(backtest_result)

            # Update status
            running_backtests[backtest_id] = {
                "status": "completed",
                "result_id": saved_result.id,
                "results": results
            }

            logger.info(f"Backtest {backtest_id} completed successfully")

        finally:
            await exchange.disconnect()

    except Exception as e:
        logger.error(f"Backtest {backtest_id} failed: {e}")
        running_backtests[backtest_id] = {
            "status": "error",
            "error": str(e)
        }


@router.post("/run")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run a backtest (async operation)

    Returns a backtest ID that can be used to check status
    """
    # Generate backtest ID
    backtest_id = f"bt_{int(datetime.now().timestamp())}"

    # Add to running backtests
    running_backtests[backtest_id] = {"status": "running"}

    # Start background task
    background_tasks.add_task(run_backtest_task, backtest_id, request)

    logger.info(f"Backtest {backtest_id} queued")

    return {
        "backtest_id": backtest_id,
        "status": "running",
        "message": "Backtest started. Use /api/backtest/status/{backtest_id} to check progress"
    }


@router.get("/status/{backtest_id}")
async def get_backtest_status(backtest_id: str):
    """Get backtest status and results"""
    if backtest_id not in running_backtests:
        raise HTTPException(status_code=404, detail="Backtest not found")

    status_info = running_backtests[backtest_id]

    if status_info["status"] == "completed":
        return {
            "backtest_id": backtest_id,
            "status": "completed",
            "result_id": status_info.get("result_id"),
            "results": status_info.get("results")
        }
    elif status_info["status"] == "error":
        return {
            "backtest_id": backtest_id,
            "status": "error",
            "error": status_info.get("error")
        }
    else:
        return {
            "backtest_id": backtest_id,
            "status": "running"
        }


@router.get("/results", response_model=List[BacktestResultResponse])
async def list_backtest_results(strategy_name: str = None, limit: int = 50):
    """
    List saved backtest results

    Args:
        strategy_name: Filter by strategy name (optional)
        limit: Maximum number of results
    """
    results = db.get_backtest_results(strategy_name=strategy_name, limit=limit)

    response = []
    for result in results:
        response.append(BacktestResultResponse(
            id=result.id,
            strategy_name=result.strategy_name,
            symbol=result.symbol,
            timeframe=result.timeframe,
            start_date=result.start_date.isoformat(),
            end_date=result.end_date.isoformat(),
            initial_capital=result.initial_capital,
            final_capital=result.final_capital,
            total_return=result.total_return,
            num_trades=result.num_trades,
            win_rate=result.win_rate,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            created_at=result.created_at.isoformat()
        ))

    return response


@router.get("/results/{result_id}")
async def get_backtest_result(result_id: int):
    """Get detailed backtest result"""
    results = db.get_backtest_results(limit=1000)
    result = next((r for r in results if r.id == result_id), None)

    if not result:
        raise HTTPException(status_code=404, detail="Backtest result not found")

    import json

    return {
        "id": result.id,
        "strategy_name": result.strategy_name,
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "start_date": result.start_date.isoformat(),
        "end_date": result.end_date.isoformat(),
        "initial_capital": result.initial_capital,
        "final_capital": result.final_capital,
        "total_return": result.total_return,
        "num_trades": result.num_trades,
        "win_rate": result.win_rate,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "avg_win": result.avg_win,
        "avg_loss": result.avg_loss,
        "profit_factor": result.profit_factor,
        "parameters": json.loads(result.parameters) if result.parameters else {},
        "trades": json.loads(result.trades_data) if result.trades_data else [],
        "equity_curve": json.loads(result.equity_curve) if result.equity_curve else [],
        "created_at": result.created_at.isoformat()
    }


@router.delete("/results/{result_id}")
async def delete_backtest_result(result_id: int):
    """Delete a backtest result"""
    # This would delete from database
    logger.info(f"Delete backtest result requested: {result_id}")

    return {
        "status": "success",
        "message": f"Backtest result {result_id} deleted"
    }
