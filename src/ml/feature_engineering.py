"""
Feature Engineering for Machine Learning Trading Models
Creates technical and statistical features from OHLCV data
"""
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger


class FeatureEngineer:
    """
    Comprehensive feature engineering for trading ML models
    """

    def __init__(self):
        """Initialize feature engineer"""
        self.feature_names: List[str] = []

    def create_all_features(
        self,
        data: pd.DataFrame,
        include_price_features: bool = True,
        include_technical_features: bool = True,
        include_statistical_features: bool = True,
        include_time_features: bool = True,
        lookback_periods: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Create all features

        Args:
            data: OHLCV DataFrame with datetime index
            include_price_features: Include price-based features
            include_technical_features: Include technical indicators
            include_statistical_features: Include statistical features
            include_time_features: Include time-based features
            lookback_periods: Lookback periods for features

        Returns:
            DataFrame with all features
        """
        df = data.copy()

        if include_price_features:
            df = self.create_price_features(df, lookback_periods)

        if include_technical_features:
            df = self.create_technical_features(df, lookback_periods)

        if include_statistical_features:
            df = self.create_statistical_features(df, lookback_periods)

        if include_time_features:
            df = self.create_time_features(df)

        # Track feature names (exclude original OHLCV columns)
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        self.feature_names = [col for col in df.columns if col not in original_cols]

        logger.info(f"Created {len(self.feature_names)} features")

        return df

    def create_price_features(
        self,
        data: pd.DataFrame,
        lookback_periods: List[int]
    ) -> pd.DataFrame:
        """
        Create price-based features

        Args:
            data: OHLCV DataFrame
            lookback_periods: Lookback periods

        Returns:
            DataFrame with price features
        """
        df = data.copy()

        # Returns
        for period in lookback_periods:
            df[f'return_{period}'] = df['close'].pct_change(period)

        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Price momentum
        for period in lookback_periods:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        # Price acceleration (rate of change of momentum)
        df['momentum_accel'] = df['momentum_5'].diff()

        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']

        # Close position in range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # Gap (open vs previous close)
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Intraday range
        df['intraday_range'] = (df['high'] - df['low']) / df['open']

        # Upper and lower shadows (candlestick patterns)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

        return df

    def create_technical_features(
        self,
        data: pd.DataFrame,
        lookback_periods: List[int]
    ) -> pd.DataFrame:
        """
        Create technical indicator features

        Args:
            data: OHLCV DataFrame
            lookback_periods: Lookback periods

        Returns:
            DataFrame with technical features
        """
        df = data.copy()

        # Moving Averages
        for period in lookback_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # MA Crossovers (relative position)
        if 'sma_10' in df.columns and 'sma_20' in df.columns:
            df['sma_10_20_cross'] = df['sma_10'] / df['sma_20'] - 1

        if 'ema_10' in df.columns and 'ema_20' in df.columns:
            df['ema_10_20_cross'] = df['ema_10'] / df['ema_20'] - 1

        # Relative position to moving averages
        for period in lookback_periods:
            if f'sma_{period}' in df.columns:
                df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1

        # RSI
        for period in [14, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        for period in [20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()

            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (
                df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10
            )

        # ATR (Average True Range)
        for period in [14]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = true_range.rolling(period).mean()
            df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close']

        # Stochastic Oscillator
        for period in [14]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()

            df[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
            df[f'stoch_{period}_d'] = df[f'stoch_{period}'].rolling(3).mean()

        # Volume indicators
        for period in lookback_periods:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_sma_{period}'] + 1e-10)

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()

        # Volume Price Trend
        df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()

        return df

    def create_statistical_features(
        self,
        data: pd.DataFrame,
        lookback_periods: List[int]
    ) -> pd.DataFrame:
        """
        Create statistical features

        Args:
            data: OHLCV DataFrame
            lookback_periods: Lookback periods

        Returns:
            DataFrame with statistical features
        """
        df = data.copy()

        # Volatility
        for period in lookback_periods:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()

        # Skewness and Kurtosis
        for period in [20, 50]:
            returns = df['close'].pct_change()
            df[f'skew_{period}'] = returns.rolling(period).skew()
            df[f'kurtosis_{period}'] = returns.rolling(period).kurt()

        # Percentile rank
        for period in [20, 50]:
            df[f'percentile_rank_{period}'] = (
                df['close'].rolling(period).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                    raw=False
                )
            )

        # Z-score
        for period in [20, 50]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'zscore_{period}'] = (df['close'] - mean) / (std + 1e-10)

        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'autocorr_lag_{lag}'] = df['close'].rolling(20).apply(
                lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else np.nan,
                raw=False
            )

        # Hurst exponent (simplified)
        for period in [50]:
            df[f'hurst_{period}'] = df['close'].rolling(period).apply(
                self._calculate_hurst,
                raw=False
            )

        return df

    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features

        Args:
            data: DataFrame with datetime index

        Returns:
            DataFrame with time features
        """
        df = data.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping time features")
            return df

        # Day of week (0 = Monday, 6 = Sunday)
        df['day_of_week'] = df.index.dayofweek

        # Hour of day
        df['hour'] = df.index.hour

        # Day of month
        df['day_of_month'] = df.index.day

        # Month
        df['month'] = df.index.month

        # Quarter
        df['quarter'] = df.index.quarter

        # Is weekend
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # Cyclical encoding for time features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def create_target_variable(
        self,
        data: pd.DataFrame,
        target_type: str = 'direction',
        forward_periods: int = 1,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Create target variable for supervised learning

        Args:
            data: OHLCV DataFrame
            target_type: Type of target
                - 'direction': Binary (up/down)
                - 'returns': Continuous returns
                - 'classification': Multi-class (strong_up, up, neutral, down, strong_down)
            forward_periods: Periods to look forward
            threshold: Threshold for classification (e.g., 0.01 = 1%)

        Returns:
            DataFrame with target variable
        """
        df = data.copy()

        # Calculate forward returns
        df['forward_return'] = df['close'].pct_change(forward_periods).shift(-forward_periods)

        if target_type == 'direction':
            # Binary: 1 if price goes up, 0 if down
            df['target'] = (df['forward_return'] > 0).astype(int)

        elif target_type == 'returns':
            # Continuous returns
            df['target'] = df['forward_return']

        elif target_type == 'classification':
            # Multi-class classification
            df['target'] = pd.cut(
                df['forward_return'],
                bins=[-np.inf, -threshold*2, -threshold, threshold, threshold*2, np.inf],
                labels=['strong_down', 'down', 'neutral', 'up', 'strong_up']
            )

        else:
            raise ValueError(f"Unknown target type: {target_type}")

        return df

    @staticmethod
    def _calculate_hurst(series: pd.Series) -> float:
        """
        Calculate Hurst exponent (simplified R/S method)

        Args:
            series: Price series

        Returns:
            Hurst exponent
        """
        try:
            if len(series) < 20:
                return np.nan

            lags = range(2, min(20, len(series) // 2))
            tau = []
            lagvec = []

            for lag in lags:
                #  Standard deviation of  lagged differences
                pp = np.subtract(series[lag:].values, series[:-lag].values)
                lagvec.append(lag)
                tau.append(np.std(pp))

            if len(lagvec) < 2:
                return np.nan

            # Linear fit
            poly = np.polyfit(np.log(lagvec), np.log(tau), 1)
            hurst = poly[0]

            return hurst

        except:
            return np.nan

    def get_feature_importance_names(self) -> List[str]:
        """
        Get list of feature names (for model training)

        Returns:
            List of feature names
        """
        return self.feature_names.copy()

    def select_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        method: str = 'correlation',
        top_n: int = 50
    ) -> List[str]:
        """
        Select most important features

        Args:
            data: DataFrame with features and target
            target_column: Name of target column
            method: Selection method ('correlation', 'mutual_info', 'variance')
            top_n: Number of top features to select

        Returns:
            List of selected feature names
        """
        feature_cols = [col for col in self.feature_names if col in data.columns]

        if method == 'correlation':
            # Select features most correlated with target
            correlations = data[feature_cols + [target_column]].corr()[target_column].abs()
            correlations = correlations.drop(target_column)
            selected = correlations.nlargest(top_n).index.tolist()

        elif method == 'variance':
            # Select features with highest variance
            variances = data[feature_cols].var()
            selected = variances.nlargest(top_n).index.tolist()

        elif method == 'mutual_info':
            try:
                from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

                # Determine if classification or regression
                if data[target_column].dtype in ['object', 'category']:
                    mi_scores = mutual_info_classif(
                        data[feature_cols].fillna(0),
                        data[target_column]
                    )
                else:
                    mi_scores = mutual_info_regression(
                        data[feature_cols].fillna(0),
                        data[target_column]
                    )

                mi_df = pd.Series(mi_scores, index=feature_cols)
                selected = mi_df.nlargest(top_n).index.tolist()

            except ImportError:
                logger.warning("sklearn not available, using correlation instead")
                return self.select_features(data, target_column, 'correlation', top_n)

        else:
            raise ValueError(f"Unknown selection method: {method}")

        logger.info(f"Selected {len(selected)} features using {method}")

        return selected
