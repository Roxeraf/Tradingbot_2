"""
MACD (Moving Average Convergence Divergence) Strategy Implementation
Buy when MACD line crosses above signal line
Sell when MACD line crosses below signal line
"""
from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy, TradingSignal, SignalType


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) momentum strategy

    Strategy Logic:
    - BUY: When MACD line crosses above signal line (bullish crossover)
    - SELL: When MACD line crosses below signal line (bearish crossover)
    - HOLD: No clear crossover signal
    - Additional confirmation from histogram and zero line crossovers

    Parameters:
    - fast_period: Fast EMA period (default: 12)
    - slow_period: Slow EMA period (default: 26)
    - signal_period: Signal line EMA period (default: 9)
    - min_confidence: Minimum confidence threshold (default: 0.6)
    - stop_loss_pct: Stop loss percentage (default: 0.02 = 2%)
    - take_profit_pct: Take profit percentage (default: 0.04 = 4%)
    - use_histogram: Consider histogram strength in signals (default: True)
    - use_zero_cross: Consider zero line crossovers (default: True)
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize MACD strategy

        Args:
            params: Strategy parameters
        """
        # Set default parameters
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'min_confidence': 0.6,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'use_histogram': True,
            'use_zero_cross': True
        }
        default_params.update(params)
        super().__init__(default_params)

        # Validate parameters
        if self.params['fast_period'] >= self.params['slow_period']:
            raise ValueError(
                f"Fast period ({self.params['fast_period']}) must be less than "
                f"slow period ({self.params['slow_period']})"
            )

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average

        Args:
            series: Price series
            period: EMA period

        Returns:
            EMA values as pandas Series
        """
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD indicators

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with MACD indicators added
        """
        df = data.copy()

        # Calculate EMAs
        ema_fast = self.calculate_ema(df['close'], self.params['fast_period'])
        ema_slow = self.calculate_ema(df['close'], self.params['slow_period'])

        # Calculate MACD line
        df['macd'] = ema_fast - ema_slow

        # Calculate Signal line
        df['macd_signal'] = self.calculate_ema(df['macd'], self.params['signal_period'])

        # Calculate MACD histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Calculate histogram momentum
        df['histogram_momentum'] = df['macd_histogram'].diff()

        # Identify zero line crossovers
        df['macd_above_zero'] = df['macd'] > 0
        df['macd_below_zero'] = df['macd'] < 0

        return df

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal based on MACD

        Args:
            data: DataFrame with OHLCV data

        Returns:
            TradingSignal object
        """
        # Validate data
        self.validate_data(data)

        # Calculate indicators
        df = self.calculate_indicators(data)

        # Get symbol from dataframe attributes
        symbol = data.attrs.get('symbol', 'UNKNOWN')

        # Need at least 2 rows to detect crossover
        if len(df) < 2:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                confidence=0.0,
                entry_price=df['close'].iloc[-1]
            )

        # Get last two rows
        current = df.iloc[-1]
        previous = df.iloc[-2]

        # Check if we have valid MACD values
        if pd.isna(current['macd']) or pd.isna(current['macd_signal']):
            return TradingSignal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                confidence=0.0,
                entry_price=current['close']
            )

        # Initialize signal
        signal_type = SignalType.HOLD
        confidence = 0.0
        metadata = {
            'macd': float(current['macd']),
            'macd_signal': float(current['macd_signal']),
            'macd_histogram': float(current['macd_histogram'])
        }

        # Calculate crossover signals
        macd_diff_current = current['macd'] - current['macd_signal']
        macd_diff_previous = previous['macd'] - previous['macd_signal']

        # Bullish crossover: MACD crosses above signal line
        if macd_diff_previous < 0 and macd_diff_current > 0:
            signal_type = SignalType.BUY

            # Base confidence on crossover strength
            crossover_strength = abs(macd_diff_current) / abs(previous['close']) * 1000
            crossover_strength = min(crossover_strength, 1.0)

            # Histogram confirmation (growing)
            histogram_conf = 0.5
            if not pd.isna(current['histogram_momentum']) and current['histogram_momentum'] > 0:
                histogram_conf = min(0.5 + abs(current['histogram_momentum']) * 100, 1.0)

            # Calculate base confidence
            if self.params['use_histogram']:
                confidence = (crossover_strength * 0.6 + histogram_conf * 0.4)
            else:
                confidence = crossover_strength

            # Boost confidence if MACD is crossing above zero (stronger signal)
            if self.params['use_zero_cross'] and current['macd'] > 0:
                confidence = min(confidence * 1.15, 1.0)
                metadata['zero_line_position'] = 'above'
            else:
                metadata['zero_line_position'] = 'below'

            # Volume confirmation
            if len(df) >= 3:
                recent_volume = df['volume'].iloc[-3:].mean()
                if current['volume'] > recent_volume:
                    volume_ratio = current['volume'] / recent_volume
                    confidence = min(confidence * (1 + (volume_ratio - 1) * 0.1), 1.0)
                    metadata['volume_ratio'] = float(volume_ratio)

            metadata['crossover_type'] = 'bullish'
            metadata['crossover_strength'] = float(crossover_strength)

        # Bearish crossover: MACD crosses below signal line
        elif macd_diff_previous > 0 and macd_diff_current < 0:
            signal_type = SignalType.SELL

            # Base confidence on crossover strength
            crossover_strength = abs(macd_diff_current) / abs(previous['close']) * 1000
            crossover_strength = min(crossover_strength, 1.0)

            # Histogram confirmation (declining)
            histogram_conf = 0.5
            if not pd.isna(current['histogram_momentum']) and current['histogram_momentum'] < 0:
                histogram_conf = min(0.5 + abs(current['histogram_momentum']) * 100, 1.0)

            # Calculate base confidence
            if self.params['use_histogram']:
                confidence = (crossover_strength * 0.6 + histogram_conf * 0.4)
            else:
                confidence = crossover_strength

            # Boost confidence if MACD is crossing below zero (stronger signal)
            if self.params['use_zero_cross'] and current['macd'] < 0:
                confidence = min(confidence * 1.15, 1.0)
                metadata['zero_line_position'] = 'below'
            else:
                metadata['zero_line_position'] = 'above'

            # Volume confirmation
            if len(df) >= 3:
                recent_volume = df['volume'].iloc[-3:].mean()
                if current['volume'] > recent_volume:
                    volume_ratio = current['volume'] / recent_volume
                    confidence = min(confidence * (1 + (volume_ratio - 1) * 0.1), 1.0)
                    metadata['volume_ratio'] = float(volume_ratio)

            metadata['crossover_type'] = 'bearish'
            metadata['crossover_strength'] = float(crossover_strength)

        # Additional signal: Strong histogram divergence
        elif self.params['use_histogram'] and len(df) >= 3:
            # Check for strong histogram momentum without crossover
            histogram_values = df['macd_histogram'].iloc[-3:]

            # Bullish: Histogram growing and becoming less negative or more positive
            if (histogram_values.iloc[-1] > histogram_values.iloc[-2] > histogram_values.iloc[-3] and
                current['histogram_momentum'] > 0 and
                macd_diff_current < 0):  # Still below signal but improving

                signal_type = SignalType.BUY

                # Lower confidence since no crossover yet
                momentum_strength = abs(current['histogram_momentum']) * 100
                confidence = min(momentum_strength * 0.6, 0.7)  # Cap at 0.7

                metadata['signal_reason'] = 'histogram_divergence_bullish'
                metadata['histogram_momentum'] = float(current['histogram_momentum'])

            # Bearish: Histogram declining and becoming less positive or more negative
            elif (histogram_values.iloc[-1] < histogram_values.iloc[-2] < histogram_values.iloc[-3] and
                  current['histogram_momentum'] < 0 and
                  macd_diff_current > 0):  # Still above signal but weakening

                signal_type = SignalType.SELL

                # Lower confidence since no crossover yet
                momentum_strength = abs(current['histogram_momentum']) * 100
                confidence = min(momentum_strength * 0.6, 0.7)  # Cap at 0.7

                metadata['signal_reason'] = 'histogram_divergence_bearish'
                metadata['histogram_momentum'] = float(current['histogram_momentum'])

        # Calculate stop loss and take profit
        entry_price = current['close']
        stop_loss = None
        take_profit = None

        if signal_type == SignalType.BUY:
            stop_loss = entry_price * (1 - self.params['stop_loss_pct'])
            take_profit = entry_price * (1 + self.params['take_profit_pct'])
        elif signal_type == SignalType.SELL:
            stop_loss = entry_price * (1 + self.params['stop_loss_pct'])
            take_profit = entry_price * (1 - self.params['take_profit_pct'])

        return TradingSignal(
            signal_type=signal_type,
            symbol=symbol,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata,
            timestamp=current.name if hasattr(current, 'name') else None
        )

    def get_required_history(self) -> int:
        """
        Get minimum number of historical candles required

        Returns:
            Slow period + signal period + buffer for calculation
        """
        return self.params['slow_period'] + self.params['signal_period'] + 20
