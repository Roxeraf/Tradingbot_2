"""
Backtesting engine for strategy evaluation
Tests strategies on historical data to evaluate performance
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from src.strategies.base_strategy import BaseStrategy, SignalType
from src.risk_management.position_sizer import PositionSizer
from src.monitoring.logger import get_logger

logger = get_logger()


@dataclass
class BacktestTrade:
    """Represents a backtest trade"""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    size: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_percentage: float
    exit_reason: str
    commission: float
    metadata: Dict[str, Any]


class BacktestEngine:
    """
    Backtesting framework for strategy evaluation
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 10000,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
        position_sizer: Optional[PositionSizer] = None
    ):
        """
        Initialize backtest engine

        Args:
            strategy: Trading strategy to backtest
            initial_capital: Starting capital
            commission: Commission rate (as decimal, e.g., 0.001 = 0.1%)
            slippage: Slippage rate (as decimal)
            position_sizer: Position sizer for risk management
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_sizer = position_sizer or PositionSizer()

        # State
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        self.current_position = None
        self.capital = initial_capital

    async def run(
        self,
        data: pd.DataFrame,
        symbol: str,
        enable_stop_loss: bool = True,
        enable_take_profit: bool = True
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data

        Args:
            data: Historical OHLCV data
            symbol: Trading pair symbol
            enable_stop_loss: Whether to honor stop loss
            enable_take_profit: Whether to honor take profit

        Returns:
            Backtest results with performance metrics
        """
        logger.info(f"Starting backtest for {symbol}")
        logger.info(f"Data range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Number of candles: {len(data)}")

        # Reset state
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        self.current_position = None
        self.capital = self.initial_capital

        # Ensure data has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must have columns: {required_columns}")

        # Get minimum required history
        min_history = self.strategy.get_required_history()

        # Iterate through historical data
        for i in range(len(data)):
            if i < min_history:
                # Not enough data for indicators yet
                self.equity_curve.append(self.capital)
                self.timestamps.append(data.index[i])
                continue

            # Get data up to current point (avoid lookahead bias)
            current_data = data.iloc[:i+1].copy()
            current_data.attrs['symbol'] = symbol
            current_bar = data.iloc[i]

            # Check stop loss and take profit first
            if self.current_position and (enable_stop_loss or enable_take_profit):
                self._check_stops(current_bar, enable_stop_loss, enable_take_profit)

            # Generate signal only if no position
            if self.current_position is None:
                signal = self.strategy.generate_signal(current_data)

                # Validate signal
                if self.strategy.validate_signal(signal):
                    # Execute signal
                    if signal.signal_type == SignalType.BUY:
                        self._open_position(
                            entry_bar=current_bar,
                            signal=signal,
                            side='long'
                        )
                    elif signal.signal_type == SignalType.SELL:
                        # For short positions (if supported)
                        pass

            # Record equity
            current_equity = self._calculate_equity(current_bar)
            self.equity_curve.append(current_equity)
            self.timestamps.append(current_bar.name)

        # Close any open position at end
        if self.current_position:
            final_bar = data.iloc[-1]
            self._close_position(
                exit_bar=final_bar,
                reason='backtest_end'
            )

        # Calculate performance metrics
        metrics = self.calculate_metrics()

        logger.info(f"Backtest completed: {len(self.trades)} trades executed")
        logger.info(f"Final capital: {self.capital:.2f}")
        logger.info(f"Total return: {metrics['total_return']:.2f}%")
        logger.info(f"Win rate: {metrics['win_rate']:.2f}%")

        return {
            'trades': [self._trade_to_dict(t) for t in self.trades],
            'equity_curve': self.equity_curve,
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'metrics': metrics,
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'strategy_name': self.strategy.name,
            'symbol': symbol
        }

    def _open_position(self, entry_bar, signal, side: str):
        """Open a position"""
        # Calculate position size
        position_size = self.position_sizer.calculate_position_size(
            account_balance=self.capital,
            entry_price=entry_bar['close'],
            stop_loss_price=signal.stop_loss or entry_bar['close'] * 0.98,
            confidence=signal.confidence
        )

        if position_size <= 0:
            return

        # Apply slippage
        entry_price = entry_bar['close'] * (1 + self.slippage if side == 'long' else 1 - self.slippage)

        # Calculate commission
        position_value = position_size * entry_price
        commission_cost = position_value * self.commission

        # Check if we have enough capital
        if position_value + commission_cost > self.capital:
            # Adjust position size
            position_size = (self.capital * 0.95) / entry_price  # Use 95% of capital

        self.current_position = {
            'entry_date': entry_bar.name,
            'entry_price': entry_price,
            'size': position_size,
            'side': side,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'commission': commission_cost,
            'metadata': signal.metadata
        }

        # Deduct capital
        self.capital -= commission_cost

        logger.debug(
            f"Opened {side} position: {position_size:.6f} @ {entry_price:.2f} "
            f"| SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f}"
        )

    def _close_position(self, exit_bar, reason: str):
        """Close current position"""
        if not self.current_position:
            return

        # Apply slippage
        side = self.current_position['side']
        exit_price = exit_bar['close'] * (1 - self.slippage if side == 'long' else 1 + self.slippage)

        # Calculate PnL
        if side == 'long':
            pnl = (exit_price - self.current_position['entry_price']) * self.current_position['size']
        else:  # short
            pnl = (self.current_position['entry_price'] - exit_price) * self.current_position['size']

        # Calculate commission
        exit_value = self.current_position['size'] * exit_price
        exit_commission = exit_value * self.commission

        # Net PnL after commissions
        total_commission = self.current_position['commission'] + exit_commission
        net_pnl = pnl - exit_commission

        # Update capital
        self.capital += exit_value + net_pnl

        # Calculate PnL percentage
        position_value = self.current_position['size'] * self.current_position['entry_price']
        pnl_percentage = (net_pnl / position_value) * 100 if position_value > 0 else 0

        # Record trade
        trade = BacktestTrade(
            entry_date=self.current_position['entry_date'],
            exit_date=exit_bar.name,
            entry_price=self.current_position['entry_price'],
            exit_price=exit_price,
            size=self.current_position['size'],
            side=side,
            pnl=net_pnl,
            pnl_percentage=pnl_percentage,
            exit_reason=reason,
            commission=total_commission,
            metadata=self.current_position['metadata']
        )

        self.trades.append(trade)

        logger.debug(
            f"Closed {side} position: PnL: {net_pnl:+.2f} ({pnl_percentage:+.2f}%) | Reason: {reason}"
        )

        self.current_position = None

    def _check_stops(self, current_bar, enable_stop_loss: bool, enable_take_profit: bool):
        """Check if stop loss or take profit hit"""
        if not self.current_position:
            return

        side = self.current_position['side']
        current_price = current_bar['close']

        # Check stop loss
        if enable_stop_loss and self.current_position['stop_loss']:
            if side == 'long' and current_bar['low'] <= self.current_position['stop_loss']:
                self._close_position(current_bar, 'stop_loss')
                return
            elif side == 'short' and current_bar['high'] >= self.current_position['stop_loss']:
                self._close_position(current_bar, 'stop_loss')
                return

        # Check take profit
        if enable_take_profit and self.current_position['take_profit']:
            if side == 'long' and current_bar['high'] >= self.current_position['take_profit']:
                self._close_position(current_bar, 'take_profit')
                return
            elif side == 'short' and current_bar['low'] <= self.current_position['take_profit']:
                self._close_position(current_bar, 'take_profit')
                return

    def _calculate_equity(self, current_bar) -> float:
        """Calculate current equity including open positions"""
        equity = self.capital

        if self.current_position:
            current_price = current_bar['close']
            side = self.current_position['side']

            # Calculate unrealized PnL
            if side == 'long':
                unrealized_pnl = (current_price - self.current_position['entry_price']) * self.current_position['size']
            else:  # short
                unrealized_pnl = (self.current_position['entry_price'] - current_price) * self.current_position['size']

            # Add position value and unrealized PnL
            position_value = self.current_position['size'] * current_price
            equity += position_value + unrealized_pnl - self.current_position['commission']

        return equity

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'avg_trade': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'avg_trade_duration': 0.0
            }

        # Basic metrics
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        num_trades = len(self.trades)

        # Win/loss statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        break_even_trades = [t for t in self.trades if t.pnl == 0]

        win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))

        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

        avg_trade = np.mean([t.pnl for t in self.trades])
        best_trade = max(t.pnl for t in self.trades) if self.trades else 0
        worst_trade = min(t.pnl for t in self.trades) if self.trades else 0

        # Trade duration
        durations = [(t.exit_date - t.entry_date).total_seconds() / 3600 for t in self.trades]  # in hours
        avg_trade_duration = np.mean(durations) if durations else 0

        # Calculate returns series for Sharpe/Sortino
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()

        # Sharpe Ratio (annualized, assuming 252 trading days)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24)  # Assuming hourly data
        else:
            sharpe_ratio = 0.0

        # Sortino Ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            sortino_ratio = (returns.mean() / downside_std) * np.sqrt(252 * 24) if downside_std > 0 else 0
        else:
            sortino_ratio = 0.0

        # Maximum Drawdown
        cumulative_max = equity_series.cummax()
        drawdown = (equity_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100  # as percentage

        # Maximum drawdown duration
        drawdown_duration = 0
        current_dd_duration = 0
        for i in range(len(equity_series)):
            if equity_series.iloc[i] < cumulative_max.iloc[i]:
                current_dd_duration += 1
            else:
                drawdown_duration = max(drawdown_duration, current_dd_duration)
                current_dd_duration = 0

        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': drawdown_duration,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_trade_duration': avg_trade_duration,
            'total_commission': sum(t.commission for t in self.trades)
        }

    def _trade_to_dict(self, trade: BacktestTrade) -> Dict[str, Any]:
        """Convert BacktestTrade to dictionary"""
        return {
            'entry_date': trade.entry_date.isoformat(),
            'exit_date': trade.exit_date.isoformat(),
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'size': trade.size,
            'side': trade.side,
            'pnl': trade.pnl,
            'pnl_percentage': trade.pnl_percentage,
            'exit_reason': trade.exit_reason,
            'commission': trade.commission,
            'duration_hours': (trade.exit_date - trade.entry_date).total_seconds() / 3600,
            'metadata': trade.metadata
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'equity': self.equity_curve
        }).set_index('timestamp')

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([self._trade_to_dict(t) for t in self.trades])
