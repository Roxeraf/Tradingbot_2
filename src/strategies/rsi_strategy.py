"""
RSI (Relative Strength Index) Strategy Implementation
Buy when RSI is oversold (< 30)
Sell when RSI is overbought (> 70)
"""
from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy, TradingSignal, SignalType


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) momentum strategy

    Strategy Logic:
    - BUY: When RSI crosses above oversold threshold (bullish reversal)
    - SELL: When RSI crosses below overbought threshold (bearish reversal)
    - HOLD: RSI in neutral zone or no clear signal

    Parameters:
    - rsi_period: Period for RSI calculation (default: 14)
    - oversold_threshold: RSI level considered oversold (default: 30)
    - overbought_threshold: RSI level considered overbought (default: 70)
    - min_confidence: Minimum confidence threshold (default: 0.6)
    - stop_loss_pct: Stop loss percentage (default: 0.02 = 2%)
    - take_profit_pct: Take profit percentage (default: 0.04 = 4%)
    - use_divergence: Enable divergence detection (default: False)
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize RSI strategy

        Args:
            params: Strategy parameters
        """
        # Set default parameters
        default_params = {
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'min_confidence': 0.6,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'use_divergence': False
        }
        default_params.update(params)
        super().__init__(default_params)

        # Validate parameters
        if not 0 < self.params['oversold_threshold'] < 50:
            raise ValueError(
                f"Oversold threshold must be between 0 and 50, "
                f"got {self.params['oversold_threshold']}"
            )

        if not 50 < self.params['overbought_threshold'] < 100:
            raise ValueError(
                f"Overbought threshold must be between 50 and 100, "
                f"got {self.params['overbought_threshold']}"
            )

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI indicator

        Args:
            series: Price series (typically close prices)
            period: RSI period

        Returns:
            RSI values as pandas Series
        """
        # Calculate price changes
        delta = series.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI and related indicators

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with RSI indicators added
        """
        df = data.copy()

        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.params['rsi_period'])

        # Calculate RSI moving average for smoothing
        df['rsi_ma'] = df['rsi'].rolling(window=3).mean()

        # Identify overbought/oversold zones
        df['rsi_oversold'] = df['rsi'] < self.params['oversold_threshold']
        df['rsi_overbought'] = df['rsi'] > self.params['overbought_threshold']

        # Calculate RSI momentum (rate of change)
        df['rsi_momentum'] = df['rsi'].diff()

        return df

    def detect_divergence(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect bullish/bearish divergence between price and RSI

        Args:
            data: DataFrame with price and RSI data

        Returns:
            Dict with bullish_divergence and bearish_divergence flags
        """
        if len(data) < 20:
            return {'bullish_divergence': False, 'bearish_divergence': False}

        # Get recent data (last 20 periods)
        recent = data.tail(20)

        # Find local lows and highs
        price_lows = recent['close'] == recent['close'].rolling(5, center=True).min()
        price_highs = recent['close'] == recent['close'].rolling(5, center=True).max()
        rsi_lows = recent['rsi'] == recent['rsi'].rolling(5, center=True).min()
        rsi_highs = recent['rsi'] == recent['rsi'].rolling(5, center=True).max()

        bullish_div = False
        bearish_div = False

        # Bullish divergence: Price making lower lows, RSI making higher lows
        if price_lows.sum() >= 2 and rsi_lows.sum() >= 2:
            price_low_values = recent.loc[price_lows, 'close'].values
            rsi_low_values = recent.loc[rsi_lows, 'rsi'].values

            if len(price_low_values) >= 2 and len(rsi_low_values) >= 2:
                if price_low_values[-1] < price_low_values[-2] and rsi_low_values[-1] > rsi_low_values[-2]:
                    bullish_div = True

        # Bearish divergence: Price making higher highs, RSI making lower highs
        if price_highs.sum() >= 2 and rsi_highs.sum() >= 2:
            price_high_values = recent.loc[price_highs, 'close'].values
            rsi_high_values = recent.loc[rsi_highs, 'rsi'].values

            if len(price_high_values) >= 2 and len(rsi_high_values) >= 2:
                if price_high_values[-1] > price_high_values[-2] and rsi_high_values[-1] < rsi_high_values[-2]:
                    bearish_div = True

        return {'bullish_divergence': bullish_div, 'bearish_divergence': bearish_div}

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal based on RSI

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

        # Check if we have valid RSI values
        if pd.isna(current['rsi']) or pd.isna(previous['rsi']):
            return TradingSignal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                confidence=0.0,
                entry_price=current['close']
            )

        # Detect divergences if enabled
        divergence = {'bullish_divergence': False, 'bearish_divergence': False}
        if self.params['use_divergence']:
            divergence = self.detect_divergence(df)

        # Initialize signal
        signal_type = SignalType.HOLD
        confidence = 0.0
        metadata = {
            'rsi': float(current['rsi']),
            'rsi_momentum': float(current['rsi_momentum']) if not pd.isna(current['rsi_momentum']) else 0.0
        }

        # Bullish signal: RSI crosses above oversold threshold
        if (previous['rsi'] < self.params['oversold_threshold'] and
            current['rsi'] >= self.params['oversold_threshold']):

            signal_type = SignalType.BUY

            # Base confidence on how oversold it was and momentum
            oversold_strength = (self.params['oversold_threshold'] - previous['rsi']) / self.params['oversold_threshold']
            oversold_strength = max(0, min(oversold_strength, 1.0))

            # RSI momentum confirmation
            momentum_conf = 0.5
            if current['rsi_momentum'] > 0:
                momentum_conf = min(0.5 + (current['rsi_momentum'] / 10), 1.0)

            # Base confidence calculation
            confidence = (oversold_strength * 0.5 + momentum_conf * 0.5)

            # Boost confidence if divergence detected
            if divergence['bullish_divergence']:
                confidence = min(confidence * 1.2, 1.0)
                metadata['bullish_divergence'] = True

            metadata['signal_reason'] = 'rsi_oversold_bounce'
            metadata['oversold_strength'] = float(oversold_strength)

        # Also consider already oversold conditions with positive momentum
        elif (current['rsi'] < self.params['oversold_threshold'] and
              current['rsi_momentum'] > 1):

            signal_type = SignalType.BUY

            # Calculate confidence based on extreme oversold and momentum
            oversold_strength = (self.params['oversold_threshold'] - current['rsi']) / self.params['oversold_threshold']
            oversold_strength = max(0, min(oversold_strength, 1.0))

            momentum_conf = min(current['rsi_momentum'] / 10, 1.0)

            confidence = (oversold_strength * 0.6 + momentum_conf * 0.4) * 0.8  # Slightly lower base confidence

            if divergence['bullish_divergence']:
                confidence = min(confidence * 1.2, 1.0)
                metadata['bullish_divergence'] = True

            metadata['signal_reason'] = 'rsi_extreme_oversold'
            metadata['oversold_strength'] = float(oversold_strength)

        # Bearish signal: RSI crosses below overbought threshold
        elif (previous['rsi'] > self.params['overbought_threshold'] and
              current['rsi'] <= self.params['overbought_threshold']):

            signal_type = SignalType.SELL

            # Base confidence on how overbought it was and momentum
            overbought_strength = (previous['rsi'] - self.params['overbought_threshold']) / (100 - self.params['overbought_threshold'])
            overbought_strength = max(0, min(overbought_strength, 1.0))

            # RSI momentum confirmation
            momentum_conf = 0.5
            if current['rsi_momentum'] < 0:
                momentum_conf = min(0.5 + (abs(current['rsi_momentum']) / 10), 1.0)

            confidence = (overbought_strength * 0.5 + momentum_conf * 0.5)

            # Boost confidence if divergence detected
            if divergence['bearish_divergence']:
                confidence = min(confidence * 1.2, 1.0)
                metadata['bearish_divergence'] = True

            metadata['signal_reason'] = 'rsi_overbought_reversal'
            metadata['overbought_strength'] = float(overbought_strength)

        # Also consider already overbought conditions with negative momentum
        elif (current['rsi'] > self.params['overbought_threshold'] and
              current['rsi_momentum'] < -1):

            signal_type = SignalType.SELL

            # Calculate confidence based on extreme overbought and momentum
            overbought_strength = (current['rsi'] - self.params['overbought_threshold']) / (100 - self.params['overbought_threshold'])
            overbought_strength = max(0, min(overbought_strength, 1.0))

            momentum_conf = min(abs(current['rsi_momentum']) / 10, 1.0)

            confidence = (overbought_strength * 0.6 + momentum_conf * 0.4) * 0.8

            if divergence['bearish_divergence']:
                confidence = min(confidence * 1.2, 1.0)
                metadata['bearish_divergence'] = True

            metadata['signal_reason'] = 'rsi_extreme_overbought'
            metadata['overbought_strength'] = float(overbought_strength)

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
            RSI period + buffer for indicator calculation
        """
        # Need extra candles for RSI calculation and divergence detection
        return self.params['rsi_period'] * 3 + 20
