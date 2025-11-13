"""
Portfolio management module
Tracks positions, calculates PnL, and manages overall portfolio risk
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """
    Represents an open trading position
    """
    symbol: str
    side: PositionSide
    entry_price: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    strategy_name: str = "unknown"
    metadata: Dict = field(default_factory=dict)

    @property
    def position_value(self) -> float:
        """Calculate position value"""
        return self.size * self.entry_price

    def calculate_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized PnL

        Args:
            current_price: Current market price

        Returns:
            Unrealized PnL
        """
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.size
        else:  # SHORT
            return (self.entry_price - current_price) * self.size

    def calculate_pnl_percentage(self, current_price: float) -> float:
        """
        Calculate unrealized PnL as percentage

        Args:
            current_price: Current market price

        Returns:
            PnL percentage
        """
        pnl = self.calculate_pnl(current_price)
        return (pnl / self.position_value) * 100

    def should_close(self, current_price: float) -> tuple[bool, str]:
        """
        Check if position should be closed based on stop loss or take profit

        Args:
            current_price: Current market price

        Returns:
            Tuple of (should_close, reason)
        """
        if self.side == PositionSide.LONG:
            if self.stop_loss and current_price <= self.stop_loss:
                return True, "stop_loss"
            if self.take_profit and current_price >= self.take_profit:
                return True, "take_profit"
        else:  # SHORT
            if self.stop_loss and current_price >= self.stop_loss:
                return True, "stop_loss"
            if self.take_profit and current_price <= self.take_profit:
                return True, "take_profit"

        return False, ""

    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'entry_price': self.entry_price,
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time.isoformat(),
            'strategy_name': self.strategy_name,
            'metadata': self.metadata
        }


class PortfolioManager:
    """
    Manages trading portfolio, tracks positions and calculates metrics
    """

    def __init__(self, initial_capital: float):
        """
        Initialize portfolio manager

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict] = []
        self.realized_pnl = 0.0

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value

        Args:
            current_prices: Dict mapping symbols to current prices

        Returns:
            Total portfolio value (cash + position values)
        """
        total = self.cash_balance

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                unrealized_pnl = position.calculate_pnl(current_price)
                total += position.position_value + unrealized_pnl

        return total

    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total unrealized PnL

        Args:
            current_prices: Dict mapping symbols to current prices

        Returns:
            Total unrealized PnL
        """
        total_pnl = 0.0

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_pnl += position.calculate_pnl(current_prices[symbol])

        return total_pnl

    def get_current_exposure(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio risk exposure as percentage

        Args:
            current_prices: Dict mapping symbols to current prices

        Returns:
            Risk exposure percentage (0-1)
        """
        if self.initial_capital == 0:
            return 0

        total_risk = 0.0

        for symbol, position in self.positions.items():
            if symbol in current_prices and position.stop_loss:
                current_price = current_prices[symbol]
                risk_amount = abs(position.calculate_pnl(position.stop_loss))
                total_risk += risk_amount

        return total_risk / self.initial_capital

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy_name: str = "unknown",
        metadata: Optional[Dict] = None
    ) -> Position:
        """
        Open a new position

        Args:
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            entry_price: Entry price
            size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy_name: Strategy that opened the position
            metadata: Additional position metadata

        Returns:
            Created Position object

        Raises:
            ValueError: If position already exists or insufficient balance
        """
        if symbol in self.positions:
            raise ValueError(f"Position for {symbol} already exists")

        position_value = size * entry_price

        if position_value > self.cash_balance:
            raise ValueError(
                f"Insufficient balance. Required: {position_value}, "
                f"Available: {self.cash_balance}"
            )

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name,
            metadata=metadata or {}
        )

        self.positions[symbol] = position
        self.cash_balance -= position_value

        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual"
    ) -> Dict:
        """
        Close an existing position

        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing (e.g., "stop_loss", "take_profit", "manual")

        Returns:
            Dict with trade details

        Raises:
            ValueError: If position doesn't exist
        """
        if symbol not in self.positions:
            raise ValueError(f"No open position for {symbol}")

        position = self.positions[symbol]

        # Calculate PnL
        pnl = position.calculate_pnl(exit_price)
        pnl_percentage = position.calculate_pnl_percentage(exit_price)

        # Update cash balance
        position_value_at_exit = position.size * exit_price
        self.cash_balance += position_value_at_exit
        self.realized_pnl += pnl

        # Record closed trade
        trade = {
            'symbol': symbol,
            'side': position.side.value,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'duration': (datetime.now() - position.entry_time).total_seconds(),
            'reason': reason,
            'strategy_name': position.strategy_name,
            'metadata': position.metadata
        }

        self.closed_trades.append(trade)

        # Remove position
        del self.positions[symbol]

        return trade

    def update_position_stops(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> None:
        """
        Update stop loss and/or take profit for a position

        Args:
            symbol: Trading symbol
            stop_loss: New stop loss price
            take_profit: New take profit price

        Raises:
            ValueError: If position doesn't exist
        """
        if symbol not in self.positions:
            raise ValueError(f"No open position for {symbol}")

        position = self.positions[symbol]

        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit

    def check_stops(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check all positions for stop loss or take profit triggers

        Args:
            current_prices: Dict mapping symbols to current prices

        Returns:
            List of trades that were closed
        """
        closed_trades = []

        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                position = self.positions[symbol]
                current_price = current_prices[symbol]

                should_close, reason = position.should_close(current_price)

                if should_close:
                    trade = self.close_position(symbol, current_price, reason)
                    closed_trades.append(trade)

        return closed_trades

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol"""
        return symbol in self.positions

    def get_all_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self.positions.values())

    def get_portfolio_stats(self, current_prices: Dict[str, float]) -> Dict:
        """
        Get comprehensive portfolio statistics

        Args:
            current_prices: Dict mapping symbols to current prices

        Returns:
            Dict with portfolio statistics
        """
        total_value = self.get_total_value(current_prices)
        unrealized_pnl = self.get_unrealized_pnl(current_prices)
        total_pnl = self.realized_pnl + unrealized_pnl
        total_return = ((total_value - self.initial_capital) / self.initial_capital) * 100

        # Calculate win rate
        winning_trades = [t for t in self.closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in self.closed_trades if t['pnl'] < 0]
        win_rate = (
            len(winning_trades) / len(self.closed_trades) * 100
            if self.closed_trades else 0
        )

        return {
            'total_value': total_value,
            'cash_balance': self.cash_balance,
            'initial_capital': self.initial_capital,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'num_open_positions': len(self.positions),
            'num_closed_trades': len(self.closed_trades),
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'current_exposure': self.get_current_exposure(current_prices)
        }
