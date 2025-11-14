"""
Performance Metrics Calculation for Backtesting
Comprehensive analysis of strategy performance
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .backtest_engine import BacktestTrade


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics"""

    # Return metrics
    total_return: float
    total_return_pct: float
    annualized_return: float
    avg_trade_return: float
    avg_winning_trade: float
    avg_losing_trade: float

    # Trade statistics
    num_trades: int
    num_winning_trades: int
    num_losing_trades: int
    win_rate: float
    loss_rate: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk/Reward
    profit_factor: float
    expectancy: float
    avg_win_loss_ratio: float

    # Time metrics
    avg_trade_duration: float  # in hours
    max_trade_duration: float
    min_trade_duration: float

    # Consecutive stats
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Equity metrics
    final_equity: float
    peak_equity: float
    recovery_factor: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'annualized_return': self.annualized_return,
            'avg_trade_return': self.avg_trade_return,
            'avg_winning_trade': self.avg_winning_trade,
            'avg_losing_trade': self.avg_losing_trade,
            'num_trades': self.num_trades,
            'num_winning_trades': self.num_winning_trades,
            'num_losing_trades': self.num_losing_trades,
            'win_rate': self.win_rate,
            'loss_rate': self.loss_rate,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'avg_win_loss_ratio': self.avg_win_loss_ratio,
            'avg_trade_duration': self.avg_trade_duration,
            'max_trade_duration': self.max_trade_duration,
            'min_trade_duration': self.min_trade_duration,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'final_equity': self.final_equity,
            'peak_equity': self.peak_equity,
            'recovery_factor': self.recovery_factor
        }


class PerformanceAnalyzer:
    """
    Analyzes backtest results and calculates performance metrics
    """

    @staticmethod
    def calculate_metrics(
        trades: List[BacktestTrade],
        equity_curve: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float = 0.02  # 2% annual risk-free rate
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics

        Args:
            trades: List of executed trades
            equity_curve: DataFrame with equity over time
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino

        Returns:
            PerformanceMetrics object
        """
        if len(trades) == 0:
            # Return zero metrics if no trades
            return PerformanceMetrics(
                total_return=0, total_return_pct=0, annualized_return=0,
                avg_trade_return=0, avg_winning_trade=0, avg_losing_trade=0,
                num_trades=0, num_winning_trades=0, num_losing_trades=0,
                win_rate=0, loss_rate=0,
                max_drawdown=0, max_drawdown_pct=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                profit_factor=0, expectancy=0, avg_win_loss_ratio=0,
                avg_trade_duration=0, max_trade_duration=0, min_trade_duration=0,
                max_consecutive_wins=0, max_consecutive_losses=0,
                final_equity=initial_capital, peak_equity=initial_capital,
                recovery_factor=0
            )

        # Convert trades to DataFrame for easier analysis
        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_percentage,
                'commission': t.commission,
                'duration': (t.exit_time - t.entry_time).total_seconds() / 3600,  # hours
                'is_winner': t.pnl > 0
            }
            for t in trades
        ])

        # Basic return metrics
        total_return = trades_df['pnl'].sum()
        final_equity = initial_capital + total_return
        total_return_pct = (total_return / initial_capital) * 100

        # Calculate annualized return
        if len(equity_curve) > 0:
            total_days = (equity_curve['timestamp'].iloc[-1] - equity_curve['timestamp'].iloc[0]).days
            total_years = max(total_days / 365.25, 1/365.25)  # Minimum 1 day
            annualized_return = (((final_equity / initial_capital) ** (1 / total_years)) - 1) * 100
        else:
            annualized_return = 0

        # Trade statistics
        num_trades = len(trades)
        winning_trades = trades_df[trades_df['is_winner']]
        losing_trades = trades_df[~trades_df['is_winner']]

        num_winning_trades = len(winning_trades)
        num_losing_trades = len(losing_trades)
        win_rate = (num_winning_trades / num_trades) * 100 if num_trades > 0 else 0
        loss_rate = (num_losing_trades / num_trades) * 100 if num_trades > 0 else 0

        avg_trade_return = trades_df['pnl'].mean()
        avg_winning_trade = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_losing_trade = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

        # Risk/Reward metrics
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        expectancy = (win_rate / 100 * avg_winning_trade) + (loss_rate / 100 * avg_losing_trade)

        avg_win_loss_ratio = abs(avg_winning_trade / avg_losing_trade) if avg_losing_trade != 0 else 0

        # Drawdown calculation
        max_dd, max_dd_pct = PerformanceAnalyzer._calculate_drawdown(equity_curve)

        # Sharpe and Sortino ratios
        sharpe = PerformanceAnalyzer._calculate_sharpe_ratio(
            equity_curve, initial_capital, risk_free_rate
        )
        sortino = PerformanceAnalyzer._calculate_sortino_ratio(
            equity_curve, initial_capital, risk_free_rate
        )

        # Calmar ratio
        calmar = annualized_return / abs(max_dd_pct) if max_dd_pct != 0 else 0

        # Time metrics
        avg_trade_duration = trades_df['duration'].mean()
        max_trade_duration = trades_df['duration'].max()
        min_trade_duration = trades_df['duration'].min()

        # Consecutive wins/losses
        max_consecutive_wins = PerformanceAnalyzer._calculate_max_consecutive(
            trades_df['is_winner'], True
        )
        max_consecutive_losses = PerformanceAnalyzer._calculate_max_consecutive(
            trades_df['is_winner'], False
        )

        # Equity metrics
        peak_equity = equity_curve['equity'].max() if len(equity_curve) > 0 else initial_capital
        recovery_factor = total_return / abs(max_dd) if max_dd != 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            avg_trade_return=avg_trade_return,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            num_trades=num_trades,
            num_winning_trades=num_winning_trades,
            num_losing_trades=num_losing_trades,
            win_rate=win_rate,
            loss_rate=loss_rate,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_win_loss_ratio=avg_win_loss_ratio,
            avg_trade_duration=avg_trade_duration,
            max_trade_duration=max_trade_duration,
            min_trade_duration=min_trade_duration,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            final_equity=final_equity,
            peak_equity=peak_equity,
            recovery_factor=recovery_factor
        )

    @staticmethod
    def _calculate_drawdown(equity_curve: pd.DataFrame) -> tuple:
        """
        Calculate maximum drawdown

        Args:
            equity_curve: DataFrame with equity over time

        Returns:
            Tuple of (max_drawdown_value, max_drawdown_percentage)
        """
        if len(equity_curve) == 0:
            return 0, 0

        equity = equity_curve['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = running_max - equity
        drawdown_pct = (drawdown / running_max) * 100

        max_dd = drawdown.max()
        max_dd_pct = drawdown_pct.max()

        return max_dd, max_dd_pct

    @staticmethod
    def _calculate_sharpe_ratio(
        equity_curve: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float
    ) -> float:
        """
        Calculate Sharpe ratio

        Args:
            equity_curve: DataFrame with equity over time
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(equity_curve) < 2:
            return 0

        # Calculate returns
        returns = equity_curve['equity'].pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0

        # Annualize the metrics
        # Assume daily data for simplification
        trading_days_per_year = 252
        avg_return = returns.mean() * trading_days_per_year
        std_return = returns.std() * np.sqrt(trading_days_per_year)

        sharpe = (avg_return - risk_free_rate) / std_return

        return sharpe

    @staticmethod
    def _calculate_sortino_ratio(
        equity_curve: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float
    ) -> float:
        """
        Calculate Sortino ratio (uses downside deviation)

        Args:
            equity_curve: DataFrame with equity over time
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio
        """
        if len(equity_curve) < 2:
            return 0

        # Calculate returns
        returns = equity_curve['equity'].pct_change().dropna()

        if len(returns) == 0:
            return 0

        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return 0

        # Annualize the metrics
        trading_days_per_year = 252
        avg_return = returns.mean() * trading_days_per_year
        downside_std = downside_returns.std() * np.sqrt(trading_days_per_year)

        if downside_std == 0:
            return 0

        sortino = (avg_return - risk_free_rate) / downside_std

        return sortino

    @staticmethod
    def _calculate_max_consecutive(series: pd.Series, value: bool) -> int:
        """
        Calculate maximum consecutive occurrences of a value

        Args:
            series: Boolean series
            value: Value to count consecutively

        Returns:
            Maximum consecutive count
        """
        if len(series) == 0:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for v in series:
            if v == value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    @staticmethod
    def generate_report(
        metrics: PerformanceMetrics,
        trades: List[BacktestTrade]
    ) -> str:
        """
        Generate a formatted performance report

        Args:
            metrics: Performance metrics
            trades: List of trades

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("BACKTEST PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append("")

        # Returns section
        report.append("RETURNS")
        report.append("-" * 80)
        report.append(f"Total Return:           ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2f}%)")
        report.append(f"Annualized Return:      {metrics.annualized_return:.2f}%")
        report.append(f"Final Equity:           ${metrics.final_equity:,.2f}")
        report.append(f"Peak Equity:            ${metrics.peak_equity:,.2f}")
        report.append("")

        # Trade statistics
        report.append("TRADE STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Trades:           {metrics.num_trades}")
        report.append(f"Winning Trades:         {metrics.num_winning_trades} ({metrics.win_rate:.2f}%)")
        report.append(f"Losing Trades:          {metrics.num_losing_trades} ({metrics.loss_rate:.2f}%)")
        report.append(f"Average Trade Return:   ${metrics.avg_trade_return:,.2f}")
        report.append(f"Average Winning Trade:  ${metrics.avg_winning_trade:,.2f}")
        report.append(f"Average Losing Trade:   ${metrics.avg_losing_trade:,.2f}")
        report.append(f"Avg Win/Loss Ratio:     {metrics.avg_win_loss_ratio:.2f}")
        report.append("")

        # Risk metrics
        report.append("RISK METRICS")
        report.append("-" * 80)
        report.append(f"Max Drawdown:           ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.2f}%)")
        report.append(f"Sharpe Ratio:           {metrics.sharpe_ratio:.2f}")
        report.append(f"Sortino Ratio:          {metrics.sortino_ratio:.2f}")
        report.append(f"Calmar Ratio:           {metrics.calmar_ratio:.2f}")
        report.append(f"Profit Factor:          {metrics.profit_factor:.2f}")
        report.append(f"Expectancy:             ${metrics.expectancy:,.2f}")
        report.append(f"Recovery Factor:        {metrics.recovery_factor:.2f}")
        report.append("")

        # Time metrics
        report.append("TIME METRICS")
        report.append("-" * 80)
        report.append(f"Avg Trade Duration:     {metrics.avg_trade_duration:.2f} hours")
        report.append(f"Max Trade Duration:     {metrics.max_trade_duration:.2f} hours")
        report.append(f"Min Trade Duration:     {metrics.min_trade_duration:.2f} hours")
        report.append("")

        # Consecutive stats
        report.append("CONSECUTIVE STATISTICS")
        report.append("-" * 80)
        report.append(f"Max Consecutive Wins:   {metrics.max_consecutive_wins}")
        report.append(f"Max Consecutive Losses: {metrics.max_consecutive_losses}")
        report.append("")

        # Trade distribution
        if trades:
            report.append("TRADE DISTRIBUTION")
            report.append("-" * 80)
            pnl_values = [t.pnl for t in trades]
            report.append(f"Best Trade:             ${max(pnl_values):,.2f}")
            report.append(f"Worst Trade:            ${min(pnl_values):,.2f}")
            report.append(f"Median Trade:           ${np.median(pnl_values):,.2f}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)
