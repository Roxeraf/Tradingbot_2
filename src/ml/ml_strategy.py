"""
ML-Based Trading Strategy
Uses machine learning models for trading decisions
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from ..strategies.base_strategy import BaseStrategy, TradingSignal, SignalType
from .prediction_service import PredictionService
from .model_trainer import ModelType


class MLStrategy(BaseStrategy):
    """
    Machine Learning-based trading strategy
    Uses pre-trained ML models to generate trading signals
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize ML strategy

        Args:
            params: Strategy parameters
                Required:
                    - model_path: Path to trained model
                    - model_type: Type of model (random_forest, xgboost, etc.)
                Optional:
                    - confidence_threshold: Minimum confidence for signals (default: 0.6)
                    - stop_loss_pct: Stop loss percentage (default: 0.02)
                    - take_profit_pct: Take profit percentage (default: 0.04)
        """
        super().__init__(params)

        # Load model
        model_path = params.get('model_path')
        if model_path is None:
            raise ValueError("model_path required in params")

        model_type_str = params.get('model_type', 'random_forest')
        model_type = ModelType(model_type_str)

        task = params.get('task', 'classification')

        self.prediction_service = PredictionService(
            model_path=model_path,
            model_type=model_type,
            task=task
        )

        self.confidence_threshold = params.get('confidence_threshold', 0.6)
        self.stop_loss_pct = params.get('stop_loss_pct', 0.02)
        self.take_profit_pct = params.get('take_profit_pct', 0.04)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators (features are calculated by prediction service)

        Args:
            data: OHLCV DataFrame

        Returns:
            Same DataFrame (features calculated during prediction)
        """
        # Features are created by the PredictionService
        return data

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal using ML model

        Args:
            data: OHLCV DataFrame with recent price data

        Returns:
            Trading signal
        """
        # Validate data
        self.validate_data(data)

        # Get current price
        current_price = data['close'].iloc[-1]
        symbol = getattr(data, 'symbol', 'UNKNOWN')

        # Get ML prediction
        prediction_result = self.prediction_service.predict_direction(
            data,
            confidence_threshold=self.confidence_threshold
        )

        # Extract prediction details
        signal_str = prediction_result.get('signal', 'hold')
        confidence = prediction_result.get('confidence', 0.5)
        is_actionable = prediction_result.get('is_actionable', False)

        # Determine signal type
        if not is_actionable:
            signal_type = SignalType.HOLD
        elif signal_str == 'buy':
            signal_type = SignalType.BUY
        elif signal_str == 'sell':
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        # Calculate stop loss and take profit
        if signal_type == SignalType.BUY:
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        elif signal_type == SignalType.SELL:
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
        else:
            stop_loss = None
            take_profit = None

        # Create trading signal
        signal = TradingSignal(
            signal_type=signal_type,
            symbol=symbol,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'strategy': 'MLStrategy',
                'model_prediction': prediction_result.get('prediction'),
                'raw_confidence': confidence,
                'confidence_threshold': self.confidence_threshold,
                'probabilities': prediction_result.get('probabilities')
            }
        )

        return signal

    def get_required_history(self) -> int:
        """
        Get minimum number of historical candles required

        Returns:
            Minimum candles needed
        """
        # Need enough data for feature engineering
        return 100

    def update_model(self, new_model_path: str) -> None:
        """
        Update ML model

        Args:
            new_model_path: Path to new model
        """
        self.prediction_service.load_model(new_model_path)
        self.params['model_path'] = new_model_path
        logger.info(f"Model updated to {new_model_path}")

    def get_prediction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about model predictions

        Returns:
            Prediction statistics
        """
        return self.prediction_service.get_prediction_stats()

    def __repr__(self) -> str:
        return (
            f"MLStrategy(model_type={self.params.get('model_type')}, "
            f"confidence_threshold={self.confidence_threshold})"
        )
