"""
Backtesting Engine for Strategy Testing
Simulates strategy execution on historical data
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from ..strategies.base_strategy import BaseStrategy, TradingSignal, SignalType
from ..risk_management.position_sizer import PositionSizer


@dataclass
class BacktestTrade:
    """Represents a completed backtest trade"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float  # Position size in base currency
    pnl: float
    pnl_percentage: float
    commission: float
    exit_reason: str  # 'take_profit', 'stop_loss', 'signal', 'end_of_data'
    signal_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestPosition:
    """Represents an open backtest position"""
    entry_time: datetime
    symbol: str
    side: str
    entry_price: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    signal_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current unrealized PnL"""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.size
        else:  # short
            return (self.entry_price - current_price) * self.size

    def calculate_pnl_percentage(self, current_price: float) -> float:
        """Calculate current unrealized PnL percentage"""
        if self.side == 'long':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            return ((self.entry_price - current_price) / self.entry_price) * 100

    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is hit"""
        if self.stop_loss is None:
            return False

        if self.side == 'long':
            return current_price <= self.stop_loss
        else:  # short
            return current_price >= self.stop_loss

    def check_take_profit(self, current_price: float) -> bool:
        """Check if take profit is hit"""
        if self.take_profit is None:
            return False

        if self.side == 'long':
            return current_price >= self.take_profit
        else:  # short
            return current_price <= self.take_profit


class BacktestEngine:
    """
    Backtesting engine for strategy evaluation

    Features:
    - Simulate trades on historical data
    - Support for long and short positions
    - Commission and slippage modeling
    - Stop loss and take profit execution
    - Comprehensive performance metrics
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,   # 0.05%
        max_positions: int = 1,
        risk_per_trade: float = 0.02,    # 2% risk per trade
        enable_shorts: bool = False
    ):
        """
        Initialize backtest engine

        Args:
            strategy: Trading strategy to backtest
            initial_capital: Starting capital
            commission_rate: Commission as a percentage (0.001 = 0.1%)
            slippage_rate: Slippage as a percentage (0.0005 = 0.05%)
            max_positions: Maximum number of concurrent positions
            risk_per_trade: Risk percentage per trade
            enable_shorts: Allow short selling
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.enable_shorts = enable_shorts

        # Initialize position sizer
        self.position_sizer = PositionSizer(
            max_position_size=0.95,  # Use up to 95% of capital
            risk_per_trade=risk_per_trade,
            max_portfolio_risk=0.1
        )

        # State variables
        self.current_capital = initial_capital
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[BacktestTrade] = []
        self.open_positions: List[BacktestPosition] = []

        logger.info(
            f"Initialized backtest engine with {initial_capital} capital, "
            f"commission: {commission_rate*100}%, slippage: {slippage_rate*100}%"
        )

    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade"""
        return trade_value * self.commission_rate

    def apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price"""
        if side == 'buy' or side == 'long':
            return price * (1 + self.slippage_rate)
        else:  # sell or short
            return price * (1 - self.slippage_rate)

    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        return len(self.open_positions) < self.max_positions

    def calculate_position_size(
        self,
        signal: TradingSignal,
        current_capital: float
    ) -> float:
        """
        Calculate position size based on risk management

        Args:
            signal: Trading signal with entry and stop loss
            current_capital: Current available capital

        Returns:
            Position size in base currency
        """
        if signal.stop_loss is None:
            # Default position size if no stop loss
            return current_capital * 0.1

        # Calculate position size using position sizer
        risk_amount = current_capital * self.risk_per_trade

        # Calculate risk per unit
        if signal.signal_type == SignalType.BUY:
            risk_per_unit = abs(signal.entry_price - signal.stop_loss)
        else:
            risk_per_unit = abs(signal.stop_loss - signal.entry_price)

        if risk_per_unit == 0:
            return 0

        # Calculate base position size
        position_size = risk_amount / risk_per_unit

        # Apply confidence scaling
        position_size *= signal.confidence

        # Ensure we don't exceed available capital
        max_position_value = current_capital * 0.95
        max_size = max_position_value / signal.entry_price

        return min(position_size, max_size)

    def open_position(
        self,
        signal: TradingSignal,
        timestamp: datetime
    ) -> Optional[BacktestPosition]:
        """
        Open a new position based on signal

        Args:
            signal: Trading signal
            timestamp: Current timestamp

        Returns:
            BacktestPosition if opened, None otherwise
        """
        if not self.can_open_position():
            return None

        # Determine position side
        if signal.signal_type == SignalType.BUY:
            side = 'long'
        elif signal.signal_type == SignalType.SELL:
            if not self.enable_shorts:
                return None
            side = 'short'
        else:
            return None

        # Calculate position size
        position_size = self.calculate_position_size(signal, self.current_capital)

        if position_size <= 0:
            return None

        # Apply slippage to entry price
        entry_price = self.apply_slippage(signal.entry_price, side)

        # Calculate trade value
        trade_value = entry_price * position_size

        # Check if we have enough capital
        required_capital = trade_value + self.calculate_commission(trade_value)
        if required_capital > self.current_capital:
            # Reduce position size to fit capital
            position_size = (self.current_capital * 0.95) / entry_price

        # Deduct capital
        trade_value = entry_price * position_size
        commission = self.calculate_commission(trade_value)
        self.current_capital -= (trade_value + commission)

        # Create position
        position = BacktestPosition(
            entry_time=timestamp,
            symbol=signal.symbol,
            side=side,
            entry_price=entry_price,
            size=position_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal_confidence=signal.confidence,
            metadata=signal.metadata
        )

        self.open_positions.append(position)

        logger.debug(
            f"Opened {side} position: {signal.symbol} @ {entry_price:.2f}, "
            f"size: {position_size:.6f}, SL: {signal.stop_loss}, TP: {signal.take_profit}"
        )

        return position

    def close_position(
        self,
        position: BacktestPosition,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ) -> BacktestTrade:
        """
        Close an open position

        Args:
            position: Position to close
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: Reason for exit

        Returns:
            BacktestTrade record
        """
        # Apply slippage
        exit_price = self.apply_slippage(
            exit_price,
            'sell' if position.side == 'long' else 'buy'
        )

        # Calculate PnL
        pnl = position.calculate_pnl(exit_price)
        pnl_percentage = position.calculate_pnl_percentage(exit_price)

        # Calculate trade value and commission
        trade_value = exit_price * position.size
        commission = self.calculate_commission(trade_value)

        # Return capital plus PnL
        self.current_capital += trade_value + pnl - commission

        # Create trade record
        trade = BacktestTrade(
            entry_time=position.entry_time,
            exit_time=exit_time,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            pnl=pnl,
            pnl_percentage=pnl_percentage,
            commission=commission * 2,  # Entry + exit commission
            exit_reason=exit_reason,
            signal_confidence=position.signal_confidence,
            metadata=position.metadata
        )

        self.trades.append(trade)

        # Remove from open positions
        self.open_positions.remove(position)

        logger.debug(
            f"Closed {position.side} position: {position.symbol} @ {exit_price:.2f}, "
            f"PnL: {pnl:.2f} ({pnl_percentage:.2f}%), reason: {exit_reason}"
        )

        return trade

    def check_and_close_positions(
        self,
        current_bar: pd.Series,
        timestamp: datetime
    ) -> None:
        """
        Check all open positions for stop loss / take profit hits

        Args:
            current_bar: Current OHLCV bar
            timestamp: Current timestamp
        """
        positions_to_close = []

        for position in self.open_positions:
            # Check stop loss (using low for long, high for short)
            if position.side == 'long':
                if position.check_stop_loss(current_bar['low']):
                    positions_to_close.append((position, position.stop_loss, 'stop_loss'))
                    continue
                elif position.check_take_profit(current_bar['high']):
                    positions_to_close.append((position, position.take_profit, 'take_profit'))
                    continue
            else:  # short
                if position.check_stop_loss(current_bar['high']):
                    positions_to_close.append((position, position.stop_loss, 'stop_loss'))
                    continue
                elif position.check_take_profit(current_bar['low']):
                    positions_to_close.append((position, position.take_profit, 'take_profit'))
                    continue

        # Close positions
        for position, exit_price, exit_reason in positions_to_close:
            self.close_position(position, exit_price, timestamp, exit_reason)

    def record_equity(self, timestamp: datetime, current_bar: pd.Series) -> None:
        """
        Record current equity for equity curve

        Args:
            timestamp: Current timestamp
            current_bar: Current OHLCV bar
        """
        # Calculate unrealized PnL from open positions
        unrealized_pnl = sum(
            pos.calculate_pnl(current_bar['close'])
            for pos in self.open_positions
        )

        total_equity = self.current_capital + unrealized_pnl

        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.current_capital,
            'unrealized_pnl': unrealized_pnl,
            'open_positions': len(self.open_positions),
            'price': current_bar['close']
        })

    def run(
        self,
        data: pd.DataFrame,
        symbol: str = 'UNKNOWN'
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data

        Args:
            data: DataFrame with OHLCV data (indexed by datetime)
            symbol: Trading symbol

        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting backtest for {symbol} with {len(data)} bars")

        # Set symbol in data attributes
        data.attrs['symbol'] = symbol

        # Get required history
        required_history = self.strategy.get_required_history()

        # Ensure we have enough data
        if len(data) < required_history:
            raise ValueError(
                f"Insufficient data: need at least {required_history} bars, "
                f"got {len(data)}"
            )

        # Reset state
        self.current_capital = self.initial_capital
        self.equity_curve = []
        self.trades = []
        self.open_positions = []

        # Run through historical data
        for i in range(required_history, len(data)):
            # Get historical window for strategy
            historical_data = data.iloc[:i+1].copy()
            historical_data.attrs['symbol'] = symbol

            current_bar = data.iloc[i]
            timestamp = current_bar.name  # Assuming datetime index

            # Check and close existing positions (stop loss / take profit)
            self.check_and_close_positions(current_bar, timestamp)

            # Generate signal
            try:
                signal = self.strategy.generate_signal(historical_data)
            except Exception as e:
                logger.error(f"Error generating signal at {timestamp}: {e}")
                continue

            # Open new position if signal is valid
            if signal.is_actionable(self.strategy.params.get('min_confidence', 0.5)):
                self.open_position(signal, timestamp)

            # Record equity
            self.record_equity(timestamp, current_bar)

        # Close all remaining positions at end of data
        final_bar = data.iloc[-1]
        final_timestamp = final_bar.name

        for position in list(self.open_positions):
            self.close_position(
                position,
                final_bar['close'],
                final_timestamp,
                'end_of_data'
            )

        # Record final equity
        self.record_equity(final_timestamp, final_bar)

        logger.info(f"Backtest complete: {len(self.trades)} trades executed")

        # Calculate and return results
        return self._compile_results(data)

    def _compile_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compile backtest results

        Args:
            data: Original OHLCV data

        Returns:
            Results dictionary
        """
        results = {
            'trades': self.trades,
            'equity_curve': pd.DataFrame(self.equity_curve),
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': self.current_capital - self.initial_capital,
            'total_return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            'num_trades': len(self.trades),
            'strategy_name': self.strategy.name,
            'strategy_params': self.strategy.params
        }

        return results
