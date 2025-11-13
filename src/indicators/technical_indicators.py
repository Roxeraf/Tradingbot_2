"""
Technical indicators calculation module
Provides commonly used trading indicators
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class TechnicalIndicators:
    """
    Collection of technical indicator calculations
    """

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average

        Args:
            data: Price series
            period: Moving average period

        Returns:
            SMA series
        """
        return data.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average

        Args:
            data: Price series
            period: EMA period

        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index

        Args:
            data: Price series
            period: RSI period (default: 14)

        Returns:
            RSI series (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence

        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        fast_ema = TechnicalIndicators.ema(data, fast_period)
        slow_ema = TechnicalIndicators.ema(data, slow_period)

        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands

        Args:
            data: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle_band = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()

        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        return upper_band, middle_band, lower_band

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period

        Returns:
            ATR series
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period
            d_period: %D period

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()

        return k, d

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume

        Args:
            close: Close price series
            volume: Volume series

        Returns:
            OBV series
        """
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Average Price

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series

        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()

        return vwap

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average Directional Index

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ADX period

        Returns:
            ADX series
        """
        # Calculate True Range
        tr = TechnicalIndicators.atr(high, low, close, 1)

        # Calculate directional movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Smooth the values
        atr = tr.rolling(window=period).mean()
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)

        # Calculate DX
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)

        # Calculate ADX
        adx = dx.rolling(window=period).mean()

        return adx

    @staticmethod
    def ichimoku(
        high: pd.Series,
        low: pd.Series,
        conversion_period: int = 9,
        base_period: int = 26,
        span_b_period: int = 52,
        displacement: int = 26
    ) -> dict:
        """
        Ichimoku Cloud

        Args:
            high: High price series
            low: Low price series
            conversion_period: Conversion line period
            base_period: Base line period
            span_b_period: Span B period
            displacement: Displacement for cloud

        Returns:
            Dict with Ichimoku components
        """
        # Conversion Line (Tenkan-sen)
        conversion_line = (
            high.rolling(window=conversion_period).max() +
            low.rolling(window=conversion_period).min()
        ) / 2

        # Base Line (Kijun-sen)
        base_line = (
            high.rolling(window=base_period).max() +
            low.rolling(window=base_period).min()
        ) / 2

        # Leading Span A (Senkou Span A)
        span_a = ((conversion_line + base_line) / 2).shift(displacement)

        # Leading Span B (Senkou Span B)
        span_b = (
            (high.rolling(window=span_b_period).max() +
             low.rolling(window=span_b_period).min()) / 2
        ).shift(displacement)

        # Lagging Span (Chikou Span)
        lagging_span = high.shift(-displacement)

        return {
            'conversion_line': conversion_line,
            'base_line': base_line,
            'span_a': span_a,
            'span_b': span_b,
            'lagging_span': lagging_span
        }

    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> dict:
        """
        Calculate Fibonacci retracement levels

        Args:
            high: Highest price in range
            low: Lowest price in range

        Returns:
            Dict with Fibonacci levels
        """
        diff = high - low

        return {
            'level_0': high,
            'level_236': high - 0.236 * diff,
            'level_382': high - 0.382 * diff,
            'level_500': high - 0.500 * diff,
            'level_618': high - 0.618 * diff,
            'level_786': high - 0.786 * diff,
            'level_100': low
        }
