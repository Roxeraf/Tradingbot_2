#!/usr/bin/env python
"""
Script to run backtests on historical data
"""
import asyncio
import argparse
from datetime import datetime, timedelta
import json

from src.backtesting.backtest_engine import BacktestEngine
from src.strategies.strategy_factory import StrategyFactory
from src.data.data_storage import DataStorage
from src.exchanges.exchange_factory import ExchangeFactory
from src.config.settings import Settings
from src.monitoring.logger import setup_logger


async def download_historical_data(exchange, symbol: str, timeframe: str, days: int):
    """Download historical data from exchange"""
    print(f"Downloading {days} days of {timeframe} data for {symbol}...")

    # Calculate start time
    now = datetime.now()
    start_time = now - timedelta(days=days)
    since = int(start_time.timestamp() * 1000)

    # Fetch data
    data = await exchange.get_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        since=since,
        limit=1000
    )

    print(f"Downloaded {len(data)} candles")
    return data


async def run_backtest(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
    strategy_params: dict,
    save_results: bool = True
):
    """Run backtest"""

    # Setup logger
    logger = setup_logger("INFO", True)

    print("\n" + "="*80)
    print("BACKTEST CONFIGURATION")
    print("="*80)
    print(f"Strategy: {strategy_name}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: {initial_capital}")
    print(f"Strategy Params: {json.dumps(strategy_params, indent=2)}")
    print("="*80 + "\n")

    # Create strategy
    strategy = StrategyFactory.create(strategy_name, strategy_params)

    # Initialize backtest engine
    backtest = BacktestEngine(
        strategy=strategy,
        initial_capital=initial_capital,
        commission=0.001,  # 0.1%
        slippage=0.0005   # 0.05%
    )

    # Get historical data
    settings = Settings()
    exchange = ExchangeFactory.create_from_settings(settings)

    try:
        await exchange.connect()

        # Download data
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        days = (end_dt - start_dt).days

        data = await download_historical_data(exchange, symbol, timeframe, days)

        # Filter data by date range
        data = data[start_date:end_date]

        if len(data) == 0:
            print("ERROR: No data available for the specified date range")
            return

        # Run backtest
        print("\nRunning backtest...")
        results = await backtest.run(data, symbol)

        # Display results
        print_backtest_results(results)

        # Save results to database
        if save_results:
            db = DataStorage(settings.DATABASE_URL)

            backtest_result = {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_dt,
                'end_date': end_dt,
                'initial_capital': initial_capital,
                'final_capital': results['final_capital'],
                'total_return': results['metrics']['total_return'],
                'num_trades': results['metrics']['num_trades'],
                'win_rate': results['metrics']['win_rate'],
                'sharpe_ratio': results['metrics']['sharpe_ratio'],
                'max_drawdown': results['metrics']['max_drawdown'],
                'avg_win': results['metrics']['avg_win'],
                'avg_loss': results['metrics']['avg_loss'],
                'profit_factor': results['metrics']['profit_factor'],
                'parameters': json.dumps(strategy_params),
                'trades_data': json.dumps(results['trades']),
                'equity_curve': json.dumps(results['equity_curve'])
            }

            db.save_backtest_result(backtest_result)
            print("\n✅ Results saved to database")

    finally:
        await exchange.disconnect()


def print_backtest_results(results: dict):
    """Print backtest results in a nice format"""
    metrics = results['metrics']

    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)

    # Performance Summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("-" * 80)
    print(f"Initial Capital:    ${results['initial_capital']:,.2f}")
    print(f"Final Capital:      ${results['final_capital']:,.2f}")
    print(f"Total Return:       {metrics['total_return']:+.2f}%")
    print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio:      {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown:       {metrics['max_drawdown']:.2f}%")
    print(f"Profit Factor:      {metrics['profit_factor']:.2f}")

    # Trading Statistics
    print("\n📈 TRADING STATISTICS")
    print("-" * 80)
    print(f"Total Trades:       {metrics['num_trades']}")
    print(f"Winning Trades:     {metrics['num_winning_trades']}")
    print(f"Losing Trades:      {metrics['num_losing_trades']}")
    print(f"Win Rate:           {metrics['win_rate']:.2f}%")
    print(f"Average Win:        ${metrics['avg_win']:,.2f}")
    print(f"Average Loss:       ${metrics['avg_loss']:,.2f}")
    print(f"Average Trade:      ${metrics['avg_trade']:,.2f}")
    print(f"Best Trade:         ${metrics['best_trade']:,.2f}")
    print(f"Worst Trade:        ${metrics['worst_trade']:,.2f}")
    print(f"Avg Duration:       {metrics['avg_trade_duration']:.1f} hours")
    print(f"Total Commission:   ${metrics['total_commission']:,.2f}")

    # Trade List
    if results['trades']:
        print("\n📋 TRADE HISTORY (Last 10 trades)")
        print("-" * 80)
        print(f"{'Date':<20} {'Side':<6} {'Entry':<10} {'Exit':<10} {'PnL':<12} {'%':<8} {'Reason':<12}")
        print("-" * 80)

        for trade in results['trades'][-10:]:
            pnl_str = f"${trade['pnl']:+,.2f}"
            pct_str = f"{trade['pnl_percentage']:+.2f}%"
            print(
                f"{trade['exit_date'][:19]:<20} "
                f"{trade['side']:<6} "
                f"{trade['entry_price']:<10.2f} "
                f"{trade['exit_price']:<10.2f} "
                f"{pnl_str:<12} "
                f"{pct_str:<8} "
                f"{trade['exit_reason']:<12}"
            )

    print("\n" + "="*80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run strategy backtest')

    parser.add_argument(
        '--strategy',
        type=str,
        default='MovingAverageCrossover',
        help='Strategy name (default: MovingAverageCrossover)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/EUR',
        help='Trading symbol (default: BTC/EUR)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        help='Timeframe (default: 1h)'
    )
    parser.add_argument(
        '--start',
        type=str,
        default=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
        help='Start date (YYYY-MM-DD, default: 90 days ago)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Initial capital (default: 10000)'
    )
    parser.add_argument(
        '--fast-period',
        type=int,
        default=20,
        help='Fast MA period (default: 20)'
    )
    parser.add_argument(
        '--slow-period',
        type=int,
        default=50,
        help='Slow MA period (default: 50)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to database'
    )

    args = parser.parse_args()

    # Strategy parameters
    strategy_params = {
        'fast_period': args.fast_period,
        'slow_period': args.slow_period,
        'min_confidence': 0.6
    }

    # Run backtest
    asyncio.run(run_backtest(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        strategy_params=strategy_params,
        save_results=not args.no_save
    ))


if __name__ == '__main__':
    main()
