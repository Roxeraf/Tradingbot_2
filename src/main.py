"""
Main trading bot orchestrator
Coordinates all components and runs the trading loop
"""
import asyncio
from typing import Dict, Optional
from datetime import datetime

from src.config.settings import Settings
from src.exchanges.exchange_factory import ExchangeFactory
from src.strategies.strategy_factory import StrategyFactory
from src.execution.order_manager import OrderManager
from src.risk_management.position_sizer import PositionSizer
from src.risk_management.portfolio_manager import PortfolioManager, PositionSide
from src.data.data_storage import DataStorage
from src.monitoring.logger import setup_logger, get_logger, log_signal, log_performance, log_error
from src.strategies.base_strategy import SignalType


class TradingBot:
    """
    Main trading bot orchestrator
    Coordinates exchange connection, strategy execution, risk management, and order execution
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize trading bot

        Args:
            settings: Settings instance (creates new one if not provided)
        """
        self.settings = settings or Settings()

        # Setup logger
        self.logger = setup_logger(
            log_level=self.settings.LOG_LEVEL,
            log_to_file=self.settings.LOG_TO_FILE
        )

        self.logger.info("=" * 80)
        self.logger.info("Initializing Crypto Trading Bot")
        self.logger.info("=" * 80)

        # Initialize components
        try:
            # Exchange connection
            self.exchange = ExchangeFactory.create_from_settings(self.settings)
            self.logger.info(f"Exchange configured: {self.settings.EXCHANGE_NAME}")

            # Trading strategy
            self.strategy = StrategyFactory.create_from_settings(self.settings)
            self.logger.info(f"Strategy loaded: {self.strategy.name}")

            # Risk management
            self.position_sizer = PositionSizer(
                max_position_size=self.settings.MAX_POSITION_SIZE,
                max_risk_per_trade=self.settings.MAX_PORTFOLIO_RISK
            )

            # Portfolio manager (will be initialized after getting balance)
            self.portfolio_manager: Optional[PortfolioManager] = None

            # Order manager
            self.order_manager = OrderManager(self.exchange)

            # Database
            self.db = DataStorage(self.settings.DATABASE_URL)
            self.logger.info("Database connected")

            # Trading pairs
            self.trading_pairs = self.settings.get_trading_pairs()
            self.logger.info(f"Trading pairs: {', '.join(self.trading_pairs)}")

            # State
            self.is_running = False
            self.current_prices: Dict[str, float] = {}

        except Exception as e:
            self.logger.error(f"Failed to initialize trading bot: {e}")
            raise

    async def start(self):
        """
        Start the trading bot
        Connects to exchange and begins trading loop
        """
        try:
            self.logger.info("Starting trading bot...")
            self.is_running = True

            # Connect to exchange
            await self.exchange.connect()
            self.logger.info(f"Connected to {self.settings.EXCHANGE_NAME}")

            # Get initial balance
            balance = await self.exchange.get_balance()
            self.logger.info(f"Account balance: {balance}")

            # Initialize portfolio manager with initial balance
            # Get total balance in quote currency (EUR/USD/etc)
            quote_currency = self.trading_pairs[0].split('/')[1] if self.trading_pairs else 'EUR'
            initial_capital = balance.get(quote_currency, 10000.0)  # Default to 10k if not found

            self.portfolio_manager = PortfolioManager(initial_capital=initial_capital)
            self.logger.info(f"Portfolio initialized with {initial_capital} {quote_currency}")

            # Test mode warning
            if self.settings.TESTNET:
                self.logger.warning("⚠️  RUNNING IN TESTNET/PAPER TRADING MODE ⚠️")
            else:
                self.logger.warning("🔴 RUNNING IN LIVE TRADING MODE 🔴")

            # Start main trading loop
            self.logger.info("Starting trading loop...")
            await self.trading_loop()

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
            await self.stop()
        except Exception as e:
            log_error(e, "Critical error in trading bot")
            await self.stop()

    async def stop(self):
        """Stop the trading bot gracefully"""
        self.logger.info("Stopping trading bot...")
        self.is_running = False

        # Close all open positions (optional - comment out if you want to keep positions open)
        # await self.close_all_positions()

        # Cancel all open orders
        await self.order_manager.cancel_all_orders()

        # Disconnect from exchange
        await self.exchange.disconnect()

        self.logger.info("Trading bot stopped")

    async def trading_loop(self):
        """
        Main trading loop
        Continuously monitors markets and executes strategies
        """
        iteration = 0

        while self.is_running:
            iteration += 1
            self.logger.info(f"--- Trading Loop Iteration {iteration} ---")

            try:
                # Update current prices
                await self.update_prices()

                # Check and update order statuses
                await self.order_manager.sync_orders()

                # Check stop losses and take profits
                if self.portfolio_manager:
                    closed_trades = self.portfolio_manager.check_stops(self.current_prices)
                    for trade in closed_trades:
                        self.logger.info(
                            f"Position closed by {trade['reason']}: {trade['symbol']} | "
                            f"PnL: {trade['pnl']:+.2f}"
                        )
                        # Save trade to database
                        self.db.save_trade(trade)

                # Process each trading pair
                for symbol in self.trading_pairs:
                    await self.process_symbol(symbol)

                # Log performance
                if self.portfolio_manager:
                    stats = self.portfolio_manager.get_portfolio_stats(self.current_prices)
                    log_performance(
                        total_value=stats['total_value'],
                        pnl=stats['total_pnl'],
                        win_rate=stats['win_rate'],
                        num_positions=stats['num_open_positions']
                    )

                    # Save performance snapshot
                    self.db.save_performance_snapshot({
                        'total_balance': stats['total_value'],
                        'unrealized_pnl': stats['unrealized_pnl'],
                        'realized_pnl': stats['realized_pnl'],
                        'num_open_positions': stats['num_open_positions'],
                        'win_rate': stats['win_rate'],
                        'strategy_name': self.strategy.name
                    })

                # Wait before next iteration (e.g., 60 seconds)
                await asyncio.sleep(60)

            except Exception as e:
                log_error(e, f"Error in trading loop iteration {iteration}")
                await asyncio.sleep(60)

    async def update_prices(self):
        """Update current prices for all trading pairs"""
        for symbol in self.trading_pairs:
            try:
                ticker = await self.exchange.get_ticker(symbol)
                self.current_prices[symbol] = ticker['last']
            except Exception as e:
                self.logger.warning(f"Could not update price for {symbol}: {e}")

    async def process_symbol(self, symbol: str):
        """
        Process a trading pair: fetch data, generate signal, execute if valid

        Args:
            symbol: Trading pair symbol
        """
        try:
            # Skip if we already have a position in this symbol
            if self.portfolio_manager and self.portfolio_manager.has_position(symbol):
                self.logger.debug(f"Already have position in {symbol}, skipping")
                return

            # Fetch historical data
            required_history = self.strategy.get_required_history()
            data = await self.exchange.get_historical_data(
                symbol=symbol,
                timeframe=self.settings.TIMEFRAME,
                limit=required_history
            )

            if len(data) < required_history:
                self.logger.warning(
                    f"Insufficient data for {symbol}: got {len(data)}, need {required_history}"
                )
                return

            # Set symbol in dataframe attributes
            data.attrs['symbol'] = symbol

            # Generate trading signal
            signal = self.strategy.generate_signal(data)

            # Log signal
            log_signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                confidence=signal.confidence,
                entry_price=signal.entry_price
            )

            # Validate signal
            if not self.strategy.validate_signal(signal):
                self.logger.debug(f"Signal validation failed for {symbol}")
                return

            # Only process BUY signals (SELL signals are handled by stop loss/take profit)
            if signal.signal_type != SignalType.BUY:
                return

            # Get current account balance
            balance = await self.exchange.get_balance()
            quote_currency = symbol.split('/')[1]
            account_balance = balance.get(quote_currency, 0)

            if account_balance <= 0:
                self.logger.warning(f"Insufficient balance: {account_balance} {quote_currency}")
                return

            # Calculate current portfolio exposure
            current_exposure = 0.0
            if self.portfolio_manager:
                current_exposure = self.portfolio_manager.get_current_exposure(self.current_prices)

            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                account_balance=account_balance,
                entry_price=signal.entry_price,
                stop_loss_price=signal.stop_loss or signal.entry_price * 0.98,
                confidence=signal.confidence,
                current_exposure=current_exposure
            )

            if position_size <= 0:
                self.logger.info(f"Position size is 0 for {symbol}, skipping trade")
                return

            # Validate position size
            if not self.position_sizer.validate_position_size(
                position_size=position_size,
                entry_price=signal.entry_price,
                account_balance=account_balance,
                stop_loss_price=signal.stop_loss
            ):
                self.logger.warning(f"Position size validation failed for {symbol}")
                return

            # Execute the trade
            self.logger.info(
                f"Executing BUY signal for {symbol} | "
                f"Size: {position_size:.6f} | Price: {signal.entry_price:.2f}"
            )

            order = await self.order_manager.execute_signal(
                signal=signal,
                position_size=position_size,
                order_type='market'
            )

            if order and order['status'] == 'filled':
                # Open position in portfolio manager
                if self.portfolio_manager:
                    self.portfolio_manager.open_position(
                        symbol=symbol,
                        side=PositionSide.LONG,
                        entry_price=order.get('price', signal.entry_price),
                        size=position_size,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        strategy_name=self.strategy.name,
                        metadata=signal.metadata
                    )

                # Save position to database
                self.db.save_position({
                    'symbol': symbol,
                    'side': 'long',
                    'entry_price': order.get('price', signal.entry_price),
                    'size': position_size,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'strategy_name': self.strategy.name,
                    'is_open': True
                })

                self.logger.info(f"✅ Position opened for {symbol}")

        except Exception as e:
            log_error(e, f"Error processing {symbol}")

    async def close_all_positions(self):
        """Close all open positions"""
        if not self.portfolio_manager:
            return

        self.logger.info("Closing all positions...")

        for position in self.portfolio_manager.get_all_positions():
            try:
                symbol = position.symbol
                current_price = self.current_prices.get(symbol)

                if not current_price:
                    ticker = await self.exchange.get_ticker(symbol)
                    current_price = ticker['last']

                # Place market sell order
                await self.order_manager.place_market_order(
                    symbol=symbol,
                    side='sell',
                    amount=position.size
                )

                # Close in portfolio manager
                trade = self.portfolio_manager.close_position(symbol, current_price, reason='manual')

                # Save to database
                self.db.save_trade(trade)
                self.db.close_position(symbol)

                self.logger.info(f"Closed position: {symbol} | PnL: {trade['pnl']:+.2f}")

            except Exception as e:
                log_error(e, f"Failed to close position: {position.symbol}")


async def main():
    """
    Main entry point
    """
    # Load settings
    settings = Settings()

    # Create and start bot
    bot = TradingBot(settings)

    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
