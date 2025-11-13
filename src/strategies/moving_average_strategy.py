"""
Moving Average Crossover Strategy Implementation
Buy when fast MA crosses above slow MA
Sell when fast MA crosses below slow MA
"""
from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy, TradingSignal, SignalType


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Simple moving average crossover strategy

    Strategy Logic:
    - BUY: When fast MA crosses above slow MA (bullish crossover)
    - SELL: When fast MA crosses below slow MA (bearish crossover)
    - HOLD: No crossover detected

    Parameters:
    - fast_period: Period for fast moving average (default: 20)
    - slow_period: Period for slow moving average (default: 50)
    - min_confidence: Minimum confidence threshold (default: 0.6)
    - stop_loss_pct: Stop loss percentage (default: 0.02 = 2%)
    - take_profit_pct: Take profit percentage (default: 0.04 = 4%)
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize MA Crossover strategy

        Args:
            params: Strategy parameters
        """
        # Set default parameters
        default_params = {
            'fast_period': 20,
            'slow_period': 50,
            'min_confidence': 0.6,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04
        }
        default_params.update(params)
        super().__init__(default_params)

        # Validate parameters
        if self.params['fast_period'] >= self.params['slow_period']:
            raise ValueError(
                f"Fast period ({self.params['fast_period']}) must be less than "
                f"slow period ({self.params['slow_period']})"
            )

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving averages

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with MA indicators added
        """
        df = data.copy()

        # Calculate simple moving averages
        df['ma_fast'] = df['close'].rolling(
            window=self.params['fast_period'],
            min_periods=self.params['fast_period']
        ).mean()

        df['ma_slow'] = df['close'].rolling(
            window=self.params['slow_period'],
            min_periods=self.params['slow_period']
        ).mean()

        # Calculate difference between MAs
        df['ma_diff'] = df['ma_fast'] - df['ma_slow']

        # Calculate percentage difference (normalized)
        df['ma_diff_pct'] = (df['ma_diff'] / df['close']) * 100

        return df

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal based on MA crossover

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

        # Check if we have valid MA values
        if pd.isna(current['ma_fast']) or pd.isna(current['ma_slow']):
            return TradingSignal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                confidence=0.0,
                entry_price=current['close']
            )

        # Detect crossover
        signal_type = SignalType.HOLD
        confidence = 0.0
        metadata = {}

        # Bullish crossover: Fast MA crosses above Slow MA
        if previous['ma_diff'] < 0 and current['ma_diff'] > 0:
            signal_type = SignalType.BUY

            # Calculate confidence based on:
            # 1. Magnitude of the difference
            # 2. Volume confirmation (if volume is increasing)
            diff_confidence = min(abs(current['ma_diff_pct']) * 10, 1.0)

            # Volume confirmation
            volume_confidence = 0.5
            if len(df) >= 3:
                recent_volume = df['volume'].iloc[-3:].mean()
                if current['volume'] > recent_volume:
                    volume_confidence = min(current['volume'] / recent_volume, 1.0)

            confidence = (diff_confidence * 0.7 + volume_confidence * 0.3)

            metadata = {
                'ma_fast': float(current['ma_fast']),
                'ma_slow': float(current['ma_slow']),
                'ma_diff': float(current['ma_diff']),
                'crossover_type': 'bullish',
                'volume_ratio': float(current['volume'] / recent_volume) if len(df) >= 3 else 1.0
            }

        # Bearish crossover: Fast MA crosses below Slow MA
        elif previous['ma_diff'] > 0 and current['ma_diff'] < 0:
            signal_type = SignalType.SELL

            # Calculate confidence
            diff_confidence = min(abs(current['ma_diff_pct']) * 10, 1.0)

            # Volume confirmation
            volume_confidence = 0.5
            if len(df) >= 3:
                recent_volume = df['volume'].iloc[-3:].mean()
                if current['volume'] > recent_volume:
                    volume_confidence = min(current['volume'] / recent_volume, 1.0)

            confidence = (diff_confidence * 0.7 + volume_confidence * 0.3)

            metadata = {
                'ma_fast': float(current['ma_fast']),
                'ma_slow': float(current['ma_slow']),
                'ma_diff': float(current['ma_diff']),
                'crossover_type': 'bearish',
                'volume_ratio': float(current['volume'] / recent_volume) if len(df) >= 3 else 1.0
            }

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
            Slow period + buffer for indicator calculation
        """
        return self.params['slow_period'] + 20
