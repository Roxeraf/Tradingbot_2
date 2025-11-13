"""
Abstract base class for trading strategies
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
import pandas as pd


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """
    Trading signal data structure
    """
    signal_type: SignalType
    symbol: str
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[pd.Timestamp] = None

    def __post_init__(self):
        """Validate signal after initialization"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {self.entry_price}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            'signal_type': self.signal_type.value,
            'symbol': self.symbol,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata,
            'timestamp': str(self.timestamp) if self.timestamp else None
        }

    def is_actionable(self, min_confidence: float = 0.5) -> bool:
        """Check if signal is strong enough to act on"""
        return (
            self.signal_type != SignalType.HOLD and
            self.confidence >= min_confidence
        )


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    All custom strategies must inherit from this class.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize strategy with parameters

        Args:
            params: Strategy-specific parameters
        """
        self.params = params
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal based on market data

        Args:
            data: DataFrame with OHLCV data and indicators
                 Must have columns: open, high, low, close, volume
                 Index should be datetime

        Returns:
            TradingSignal object with trading recommendation
        """
        pass

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators needed for the strategy

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        pass

    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate signal before execution
        Override this method for custom validation logic

        Args:
            signal: Trading signal to validate

        Returns:
            bool: True if signal is valid
        """
        min_confidence = self.params.get('min_confidence', 0.5)
        return signal.is_actionable(min_confidence)

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns and sufficient rows

        Args:
            data: DataFrame to validate

        Returns:
            bool: True if data is valid

        Raises:
            ValueError: If data is invalid
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(
                f"Data is missing required columns: {', '.join(missing_columns)}"
            )

        if len(data) < 2:
            raise ValueError(
                f"Insufficient data: need at least 2 rows, got {len(data)}"
            )

        return True

    def get_required_history(self) -> int:
        """
        Get minimum number of historical candles required

        Returns:
            int: Minimum number of candles needed
        """
        # Default to 100, override in subclasses for specific requirements
        return 100

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters

        Returns:
            Dict of parameters
        """
        return self.params.copy()

    def update_parameters(self, new_params: Dict[str, Any]) -> None:
        """
        Update strategy parameters

        Args:
            new_params: New parameters to merge with existing
        """
        self.params.update(new_params)

    def calculate_position_size(
        self,
        signal: TradingSignal,
        account_balance: float,
        risk_percentage: float = 0.02
    ) -> float:
        """
        Calculate position size based on risk management

        Args:
            signal: Trading signal with entry and stop loss
            account_balance: Available account balance
            risk_percentage: Percentage of account to risk (default 2%)

        Returns:
            Position size in base currency
        """
        if not signal.stop_loss or signal.stop_loss <= 0:
            # If no stop loss, use default position size
            return account_balance * 0.1

        # Calculate risk amount
        risk_amount = account_balance * risk_percentage

        # Calculate risk per unit
        if signal.signal_type == SignalType.BUY:
            risk_per_unit = abs(signal.entry_price - signal.stop_loss)
        else:
            risk_per_unit = abs(signal.stop_loss - signal.entry_price)

        if risk_per_unit == 0:
            return 0

        # Calculate position size
        position_size = risk_amount / risk_per_unit

        # Apply confidence scaling
        position_size *= signal.confidence

        return position_size

    def __repr__(self) -> str:
        return f"{self.name}(params={self.params})"

    def __str__(self) -> str:
        return self.name
