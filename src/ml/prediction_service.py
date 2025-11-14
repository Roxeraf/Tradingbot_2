"""
ML Prediction Service
Real-time prediction service for trading decisions
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from .model_trainer import MLModelTrainer, ModelType
from .feature_engineering import FeatureEngineer


class PredictionService:
    """
    Real-time ML prediction service for trading
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: ModelType = ModelType.RANDOM_FOREST,
        task: str = 'classification'
    ):
        """
        Initialize prediction service

        Args:
            model_path: Path to trained model (if None, model must be trained first)
            model_type: Type of ML model
            task: Task type (classification or regression)
        """
        self.model_trainer = MLModelTrainer(model_type, task)
        self.feature_engineer = FeatureEngineer()

        if model_path is not None:
            self.load_model(model_path)

        self.prediction_history: List[Dict] = []

    def load_model(self, model_path: str) -> None:
        """
        Load trained model

        Args:
            model_path: Path to model file
        """
        self.model_trainer.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

    def predict(
        self,
        data: pd.DataFrame,
        return_proba: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction on new data

        Args:
            data: OHLCV DataFrame with recent price data
            return_proba: Return probabilities (classification only)

        Returns:
            Prediction result
        """
        if self.model_trainer.model is None:
            raise ValueError("Model not loaded or trained")

        # Engineer features
        data_with_features = self.feature_engineer.create_all_features(data)

        # Get last row (most recent)
        latest_data = data_with_features.iloc[[-1]]

        # Extract feature values
        feature_values = latest_data[self.model_trainer.feature_names].values

        # Check for NaN
        if np.isnan(feature_values).any():
            logger.warning("NaN values in features, filling with 0")
            feature_values = np.nan_to_num(feature_values, 0)

        # Make prediction
        prediction = self.model_trainer.predict(feature_values)[0]

        result = {
            'prediction': prediction,
            'timestamp': latest_data.index[0],
            'current_price': latest_data['close'].iloc[0]
        }

        # Add probability if classification
        if return_proba and self.model_trainer.task == 'classification':
            try:
                probabilities = self.model_trainer.predict_proba(feature_values)[0]
                result['probabilities'] = probabilities.tolist()
                result['confidence'] = float(np.max(probabilities))
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")

        # Store prediction
        self.prediction_history.append(result.copy())

        return result

    def predict_direction(
        self,
        data: pd.DataFrame,
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Predict price direction with confidence

        Args:
            data: OHLCV DataFrame
            confidence_threshold: Minimum confidence for actionable signal

        Returns:
            Direction prediction with confidence
        """
        result = self.predict(data, return_proba=True)

        if 'confidence' not in result:
            result['confidence'] = 0.5

        # Determine signal
        if self.model_trainer.task == 'classification':
            # Binary classification (0 = down, 1 = up)
            if result['prediction'] == 1:
                signal = 'buy'
            else:
                signal = 'sell'
        else:
            # Regression (predict returns)
            if result['prediction'] > 0:
                signal = 'buy'
            else:
                signal = 'sell'

        # Check confidence
        is_actionable = result['confidence'] >= confidence_threshold

        result.update({
            'signal': signal,
            'is_actionable': is_actionable,
            'confidence_threshold': confidence_threshold
        })

        return result

    def batch_predict(
        self,
        data: pd.DataFrame,
        return_proba: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions on batch of data

        Args:
            data: OHLCV DataFrame
            return_proba: Return probabilities (classification only)

        Returns:
            DataFrame with predictions
        """
        if self.model_trainer.model is None:
            raise ValueError("Model not loaded or trained")

        # Engineer features
        data_with_features = self.feature_engineer.create_all_features(data)

        # Extract features
        feature_values = data_with_features[self.model_trainer.feature_names].fillna(0).values

        # Make predictions
        predictions = self.model_trainer.predict(feature_values)

        # Create result DataFrame
        result_df = data.copy()
        result_df['prediction'] = predictions

        # Add probabilities if classification
        if return_proba and self.model_trainer.task == 'classification':
            try:
                probabilities = self.model_trainer.predict_proba(feature_values)
                result_df['confidence'] = np.max(probabilities, axis=1)

                # Add individual class probabilities
                for i in range(probabilities.shape[1]):
                    result_df[f'prob_class_{i}'] = probabilities[:, i]
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")

        return result_df

    def get_feature_values(
        self,
        data: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get engineered feature values

        Args:
            data: OHLCV DataFrame
            feature_names: Specific features to return (if None, returns all)

        Returns:
            DataFrame with feature values
        """
        # Engineer features
        data_with_features = self.feature_engineer.create_all_features(data)

        if feature_names is None:
            feature_names = self.model_trainer.feature_names

        return data_with_features[feature_names]

    def get_prediction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about predictions

        Returns:
            Prediction statistics
        """
        if not self.prediction_history:
            return {'total_predictions': 0}

        df = pd.DataFrame(self.prediction_history)

        stats = {
            'total_predictions': len(df),
            'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else None,
            'latest_prediction': self.prediction_history[-1]
        }

        if self.model_trainer.task == 'classification':
            # Count predictions by class
            stats['prediction_distribution'] = df['prediction'].value_counts().to_dict()

        return stats

    def clear_history(self) -> None:
        """Clear prediction history"""
        self.prediction_history = []
        logger.info("Prediction history cleared")
