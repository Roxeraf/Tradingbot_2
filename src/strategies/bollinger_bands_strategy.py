"""
Bollinger Bands Strategy Implementation
Buy when price touches lower band and shows reversal
Sell when price touches upper band and shows reversal
"""
from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy, TradingSignal, SignalType


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands mean reversion strategy

    Strategy Logic:
    - BUY: When price touches/crosses below lower band and shows reversal signal
    - SELL: When price touches/crosses above upper band and shows reversal signal
    - HOLD: Price within bands or no clear reversal signal
    - Additional confirmation from bandwidth and %B indicator

    Parameters:
    - period: Moving average period (default: 20)
    - std_dev: Number of standard deviations (default: 2.0)
    - min_confidence: Minimum confidence threshold (default: 0.6)
    - stop_loss_pct: Stop loss percentage (default: 0.02 = 2%)
    - take_profit_pct: Take profit percentage (default: 0.04 = 4%)
    - use_bandwidth: Consider band width in signals (default: True)
    - min_bandwidth_pct: Minimum bandwidth to consider (default: 2.0%)
    - use_squeeze: Detect Bollinger squeeze patterns (default: True)
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize Bollinger Bands strategy

        Args:
            params: Strategy parameters
        """
        # Set default parameters
        default_params = {
            'period': 20,
            'std_dev': 2.0,
            'min_confidence': 0.6,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'use_bandwidth': True,
            'min_bandwidth_pct': 2.0,
            'use_squeeze': True
        }
        default_params.update(params)
        super().__init__(default_params)

        # Validate parameters
        if self.params['std_dev'] <= 0:
            raise ValueError(
                f"Standard deviation must be positive, got {self.params['std_dev']}"
            )

        if self.params['period'] < 2:
            raise ValueError(
                f"Period must be at least 2, got {self.params['period']}"
            )

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and related indicators

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Bollinger Bands indicators added
        """
        df = data.copy()

        # Calculate middle band (SMA)
        df['bb_middle'] = df['close'].rolling(
            window=self.params['period'],
            min_periods=self.params['period']
        ).mean()

        # Calculate standard deviation
        df['bb_std'] = df['close'].rolling(
            window=self.params['period'],
            min_periods=self.params['period']
        ).std()

        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (self.params['std_dev'] * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.params['std_dev'] * df['bb_std'])

        # Calculate %B (position within bands)
        # %B = (Price - Lower Band) / (Upper Band - Lower Band)
        df['bb_bandwidth'] = df['bb_upper'] - df['bb_lower']
        df['percent_b'] = (df['close'] - df['bb_lower']) / df['bb_bandwidth']

        # Calculate bandwidth percentage
        df['bb_bandwidth_pct'] = (df['bb_bandwidth'] / df['bb_middle']) * 100

        # Detect squeeze (narrow bands)
        df['bb_squeeze'] = df['bb_bandwidth_pct'] < self.params['min_bandwidth_pct']

        # Distance from bands
        df['distance_upper'] = (df['bb_upper'] - df['close']) / df['close'] * 100
        df['distance_lower'] = (df['close'] - df['bb_lower']) / df['close'] * 100

        # Price position relative to bands
        df['above_upper'] = df['close'] > df['bb_upper']
        df['below_lower'] = df['close'] < df['bb_lower']
        df['touching_upper'] = (df['high'] >= df['bb_upper']) & (df['close'] <= df['bb_upper'])
        df['touching_lower'] = (df['low'] <= df['bb_lower']) & (df['close'] >= df['bb_lower'])

        return df

    def detect_reversal(self, data: pd.DataFrame, direction: str = 'bullish') -> float:
        """
        Detect reversal signals using candlestick patterns and price action

        Args:
            data: DataFrame with OHLCV and BB data
            direction: 'bullish' or 'bearish'

        Returns:
            Reversal confidence (0.0 to 1.0)
        """
        if len(data) < 3:
            return 0.0

        current = data.iloc[-1]
        previous = data.iloc[-2]

        reversal_conf = 0.0

        if direction == 'bullish':
            # Check for bullish reversal patterns

            # Hammer or bullish engulfing
            body_size = abs(current['close'] - current['open'])
            lower_wick = min(current['open'], current['close']) - current['low']
            upper_wick = current['high'] - max(current['open'], current['close'])

            # Hammer: Small body, long lower wick
            if body_size > 0 and lower_wick > body_size * 2 and upper_wick < body_size:
                reversal_conf = 0.7

            # Bullish engulfing
            if (previous['close'] < previous['open'] and  # Previous candle bearish
                current['close'] > current['open'] and    # Current candle bullish
                current['open'] <= previous['close'] and
                current['close'] >= previous['open']):
                reversal_conf = 0.8

            # Price closing above previous high (momentum shift)
            if current['close'] > previous['high']:
                reversal_conf = max(reversal_conf, 0.6)

        elif direction == 'bearish':
            # Check for bearish reversal patterns

            # Shooting star or bearish engulfing
            body_size = abs(current['close'] - current['open'])
            lower_wick = min(current['open'], current['close']) - current['low']
            upper_wick = current['high'] - max(current['open'], current['close'])

            # Shooting star: Small body, long upper wick
            if body_size > 0 and upper_wick > body_size * 2 and lower_wick < body_size:
                reversal_conf = 0.7

            # Bearish engulfing
            if (previous['close'] > previous['open'] and  # Previous candle bullish
                current['close'] < current['open'] and    # Current candle bearish
                current['open'] >= previous['close'] and
                current['close'] <= previous['open']):
                reversal_conf = 0.8

            # Price closing below previous low (momentum shift)
            if current['close'] < previous['low']:
                reversal_conf = max(reversal_conf, 0.6)

        return reversal_conf

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal based on Bollinger Bands

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

        # Need sufficient data
        if len(df) < 3:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                confidence=0.0,
                entry_price=df['close'].iloc[-1]
            )

        # Get last rows
        current = df.iloc[-1]
        previous = df.iloc[-2]

        # Check if we have valid BB values
        if pd.isna(current['bb_upper']) or pd.isna(current['bb_lower']):
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
            'bb_upper': float(current['bb_upper']),
            'bb_middle': float(current['bb_middle']),
            'bb_lower': float(current['bb_lower']),
            'percent_b': float(current['percent_b']),
            'bb_bandwidth_pct': float(current['bb_bandwidth_pct'])
        }

        # Check for squeeze breakout
        squeeze_breakout = False
        if self.params['use_squeeze'] and len(df) >= 5:
            # Was in squeeze and now breaking out
            recent_squeeze = df['bb_squeeze'].iloc[-5:-1].sum() >= 3
            if recent_squeeze and not current['bb_squeeze']:
                squeeze_breakout = True
                metadata['squeeze_breakout'] = True

        # BULLISH SIGNAL: Price at/below lower band
        if (current['below_lower'] or current['touching_lower'] or
            (previous['below_lower'] and current['close'] > current['bb_lower'])):

            # Detect bullish reversal
            reversal_conf = self.detect_reversal(df, direction='bullish')

            if reversal_conf > 0:
                signal_type = SignalType.BUY

                # Base confidence on %B (how far below lower band)
                # %B < 0 means below lower band
                if current['percent_b'] < 0:
                    oversold_strength = min(abs(current['percent_b']), 1.0)
                else:
                    oversold_strength = min(1.0 - current['percent_b'], 0.5)

                # Combine oversold strength and reversal
                confidence = (oversold_strength * 0.5 + reversal_conf * 0.5)

                # Boost confidence if bandwidth is sufficient (not too tight)
                if self.params['use_bandwidth']:
                    if current['bb_bandwidth_pct'] >= self.params['min_bandwidth_pct']:
                        confidence = min(confidence * 1.1, 1.0)
                    else:
                        confidence *= 0.9  # Reduce confidence for tight bands

                # Boost confidence on squeeze breakout
                if squeeze_breakout and current['close'] > current['open']:
                    confidence = min(confidence * 1.2, 1.0)

                # Volume confirmation
                if len(df) >= 3:
                    recent_volume = df['volume'].iloc[-3:].mean()
                    if current['volume'] > recent_volume:
                        volume_ratio = current['volume'] / recent_volume
                        confidence = min(confidence * (1 + (volume_ratio - 1) * 0.1), 1.0)
                        metadata['volume_ratio'] = float(volume_ratio)

                metadata['signal_reason'] = 'lower_band_bounce'
                metadata['reversal_confidence'] = float(reversal_conf)
                metadata['oversold_strength'] = float(oversold_strength)

        # BEARISH SIGNAL: Price at/above upper band
        elif (current['above_upper'] or current['touching_upper'] or
              (previous['above_upper'] and current['close'] < current['bb_upper'])):

            # Detect bearish reversal
            reversal_conf = self.detect_reversal(df, direction='bearish')

            if reversal_conf > 0:
                signal_type = SignalType.SELL

                # Base confidence on %B (how far above upper band)
                # %B > 1 means above upper band
                if current['percent_b'] > 1:
                    overbought_strength = min(current['percent_b'] - 1, 1.0)
                else:
                    overbought_strength = min(current['percent_b'], 0.5)

                # Combine overbought strength and reversal
                confidence = (overbought_strength * 0.5 + reversal_conf * 0.5)

                # Boost confidence if bandwidth is sufficient
                if self.params['use_bandwidth']:
                    if current['bb_bandwidth_pct'] >= self.params['min_bandwidth_pct']:
                        confidence = min(confidence * 1.1, 1.0)
                    else:
                        confidence *= 0.9

                # Boost confidence on squeeze breakout
                if squeeze_breakout and current['close'] < current['open']:
                    confidence = min(confidence * 1.2, 1.0)

                # Volume confirmation
                if len(df) >= 3:
                    recent_volume = df['volume'].iloc[-3:].mean()
                    if current['volume'] > recent_volume:
                        volume_ratio = current['volume'] / recent_volume
                        confidence = min(confidence * (1 + (volume_ratio - 1) * 0.1), 1.0)
                        metadata['volume_ratio'] = float(volume_ratio)

                metadata['signal_reason'] = 'upper_band_reversal'
                metadata['reversal_confidence'] = float(reversal_conf)
                metadata['overbought_strength'] = float(overbought_strength)

        # MEAN REVERSION: Price moving back to middle band
        elif current['percent_b'] > 0 and current['percent_b'] < 1:
            # Check for strong momentum toward middle
            if len(df) >= 3:
                price_momentum = df['close'].diff().iloc[-1]
                distance_to_middle = current['close'] - current['bb_middle']

                # If far from middle and moving toward it
                if abs(current['percent_b'] - 0.5) > 0.3:
                    # Price below middle, moving up
                    if distance_to_middle < 0 and price_momentum > 0:
                        signal_type = SignalType.BUY
                        confidence = min((0.5 - current['percent_b']) * 1.5, 0.6)
                        metadata['signal_reason'] = 'mean_reversion_bullish'

                    # Price above middle, moving down
                    elif distance_to_middle > 0 and price_momentum < 0:
                        signal_type = SignalType.SELL
                        confidence = min((current['percent_b'] - 0.5) * 1.5, 0.6)
                        metadata['signal_reason'] = 'mean_reversion_bearish'

        # Calculate stop loss and take profit
        entry_price = current['close']
        stop_loss = None
        take_profit = None

        if signal_type == SignalType.BUY:
            # Use lower band as stop loss reference
            stop_loss = max(
                entry_price * (1 - self.params['stop_loss_pct']),
                current['bb_lower'] * 0.98
            )
            # Use middle or upper band as take profit
            take_profit = min(
                entry_price * (1 + self.params['take_profit_pct']),
                current['bb_middle']
            )

        elif signal_type == SignalType.SELL:
            # Use upper band as stop loss reference
            stop_loss = min(
                entry_price * (1 + self.params['stop_loss_pct']),
                current['bb_upper'] * 1.02
            )
            # Use middle or lower band as take profit
            take_profit = max(
                entry_price * (1 - self.params['take_profit_pct']),
                current['bb_middle']
            )

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
            Period + buffer for calculation and reversal detection
        """
        return self.params['period'] * 2 + 20
