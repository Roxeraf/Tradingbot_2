"""
ML Model Training Pipeline
Supports Random Forest, XGBoost, and LSTM models
"""
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelType(Enum):
    """Supported ML model types"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    LIGHTGBM = "lightgbm"
    GRADIENT_BOOSTING = "gradient_boosting"


class MLModelTrainer:
    """
    Machine Learning model training pipeline for trading
    """

    def __init__(
        self,
        model_type: ModelType,
        task: str = 'classification',  # 'classification' or 'regression'
        random_state: int = 42
    ):
        """
        Initialize ML model trainer

        Args:
            model_type: Type of model to train
            task: Task type (classification or regression)
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.task = task
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.training_metrics: Dict[str, Any] = {}

    def prepare_data(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        train_size: float = 0.8,
        scale_features: bool = True,
        handle_nan: str = 'drop'  # 'drop', 'fill_zero', 'fill_mean'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training

        Args:
            data: DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Name of target column
            train_size: Proportion of data for training
            scale_features: Whether to scale features
            handle_nan: How to handle NaN values

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Handle missing values
        if handle_nan == 'drop':
            data = data.dropna(subset=feature_columns + [target_column])
        elif handle_nan == 'fill_zero':
            data[feature_columns] = data[feature_columns].fillna(0)
        elif handle_nan == 'fill_mean':
            data[feature_columns] = data[feature_columns].fillna(data[feature_columns].mean())

        # Extract features and target
        X = data[feature_columns].values
        y = data[target_column].values

        # Store feature names
        self.feature_names = feature_columns

        # Split data (time series split - no shuffle!)
        split_idx = int(len(X) * train_size)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(
            f"Data prepared: Train size: {len(X_train)}, Test size: {len(X_test)}"
        )

        # Scale features
        if scale_features:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> Any:
        """
        Train Random Forest model

        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Model hyperparameters

        Returns:
            Trained model
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        default_params.update(kwargs)

        if self.task == 'classification':
            model = RandomForestClassifier(**default_params)
        else:
            model = RandomForestRegressor(**default_params)

        logger.info(f"Training Random Forest ({self.task})...")
        model.fit(X_train, y_train)

        return model

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> Any:
        """
        Train XGBoost model

        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Model hyperparameters

        Returns:
            Trained model
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")

        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        default_params.update(kwargs)

        if self.task == 'classification':
            model = xgb.XGBClassifier(**default_params)
        else:
            model = xgb.XGBRegressor(**default_params)

        logger.info(f"Training XGBoost ({self.task})...")
        model.fit(X_train, y_train)

        return model

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> Any:
        """
        Train LightGBM model

        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Model hyperparameters

        Returns:
            Trained model
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Install with: pip install lightgbm")

        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        default_params.update(kwargs)

        if self.task == 'classification':
            model = lgb.LGBMClassifier(**default_params)
        else:
            model = lgb.LGBMRegressor(**default_params)

        logger.info(f"Training LightGBM ({self.task})...")
        model.fit(X_train, y_train)

        return model

    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> Any:
        """
        Train Gradient Boosting model

        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Model hyperparameters

        Returns:
            Trained model
        """
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': self.random_state
        }
        default_params.update(kwargs)

        if self.task == 'classification':
            model = GradientBoostingClassifier(**default_params)
        else:
            model = GradientBoostingRegressor(**default_params)

        logger.info(f"Training Gradient Boosting ({self.task})...")
        model.fit(X_train, y_train)

        return model

    def train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sequence_length: int = 20,
        **kwargs
    ) -> Any:
        """
        Train LSTM model

        Args:
            X_train: Training features
            y_train: Training target
            sequence_length: Length of input sequences
            **kwargs: Model hyperparameters

        Returns:
            Trained model
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError(
                "tensorflow not installed. Install with: pip install tensorflow"
            )

        # Reshape data for LSTM (samples, timesteps, features)
        X_sequences = []
        y_sequences = []

        for i in range(len(X_train) - sequence_length):
            X_sequences.append(X_train[i:i+sequence_length])
            y_sequences.append(y_train[i+sequence_length])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        logger.info(f"LSTM input shape: {X_sequences.shape}")

        # Build LSTM model
        default_params = {
            'lstm_units': 50,
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        default_params.update(kwargs)

        model = keras.Sequential([
            layers.LSTM(
                default_params['lstm_units'],
                input_shape=(sequence_length, X_train.shape[1]),
                return_sequences=True
            ),
            layers.Dropout(default_params['dropout']),
            layers.LSTM(default_params['lstm_units'] // 2),
            layers.Dropout(default_params['dropout']),
            layers.Dense(32, activation='relu'),
            layers.Dense(
                1 if self.task == 'regression' else len(np.unique(y_train)),
                activation='linear' if self.task == 'regression' else 'softmax'
            )
        ])

        # Compile model
        if self.task == 'classification':
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=default_params['learning_rate']),
            loss=loss,
            metrics=metrics
        )

        # Train model
        logger.info(f"Training LSTM ({self.task})...")

        history = model.fit(
            X_sequences,
            y_sequences,
            epochs=default_params['epochs'],
            batch_size=default_params['batch_size'],
            validation_split=0.2,
            verbose=0
        )

        self.training_metrics['lstm_history'] = history.history

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> None:
        """
        Train model based on model_type

        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Model-specific hyperparameters
        """
        if self.model_type == ModelType.RANDOM_FOREST:
            self.model = self.train_random_forest(X_train, y_train, **kwargs)

        elif self.model_type == ModelType.XGBOOST:
            self.model = self.train_xgboost(X_train, y_train, **kwargs)

        elif self.model_type == ModelType.LIGHTGBM:
            self.model = self.train_lightgbm(X_train, y_train, **kwargs)

        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            self.model = self.train_gradient_boosting(X_train, y_train, **kwargs)

        elif self.model_type == ModelType.LSTM:
            self.model = self.train_lstm(X_train, y_train, **kwargs)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        logger.info(f"Training complete: {self.model_type.value}")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Make predictions
        if self.model_type == ModelType.LSTM:
            # Handle LSTM sequence prediction
            sequence_length = kwargs.get('sequence_length', 20)
            # Simplified evaluation - in production would need proper sequence handling
            y_pred = self.model.predict(X_test[:len(X_test)//sequence_length*sequence_length].reshape(-1, sequence_length, X_test.shape[1]))
            y_pred = y_pred.flatten()
            y_test = y_test[:len(y_pred)]
        else:
            y_pred = self.model.predict(X_test)

        metrics = {}

        if self.task == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

            metrics['accuracy'] = accuracy_score(y_test, y_pred)

            try:
                # Probability predictions for AUC
                if hasattr(self.model, 'predict_proba'):
                    y_pred_proba = self.model.predict_proba(X_test)

                    if y_pred_proba.shape[1] == 2:
                        # Binary classification
                        metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                        metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
                        metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
                        metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
                    else:
                        # Multi-class
                        metrics['auc'] = roc_auc_score(
                            y_test,
                            y_pred_proba,
                            multi_class='ovr',
                            average='weighted'
                        )
                        metrics['precision'] = precision_score(
                            y_test, y_pred, average='weighted', zero_division=0
                        )
                        metrics['recall'] = recall_score(
                            y_test, y_pred, average='weighted', zero_division=0
                        )
                        metrics['f1'] = f1_score(
                            y_test, y_pred, average='weighted', zero_division=0
                        )
            except Exception as e:
                logger.warning(f"Error calculating some metrics: {e}")

        else:  # regression
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)

            # Direction accuracy (important for trading)
            direction_correct = np.sum(
                (y_test > 0) == (y_pred > 0)
            ) / len(y_test)
            metrics['direction_accuracy'] = direction_correct

        self.training_metrics.update(metrics)

        logger.info(f"Evaluation metrics: {metrics}")

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if self.model_type == ModelType.LSTM:
            logger.warning("Feature importance not available for LSTM")
            return pd.DataFrame()

        # Get feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            logger.warning("Model does not have feature_importances_")
            return pd.DataFrame()

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save_model(self, filepath: str) -> None:
        """
        Save trained model

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.model_type == ModelType.LSTM:
            # Save Keras model
            self.model.save(str(filepath))
        else:
            # Save sklearn/xgboost model
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)

        # Save scaler
        if self.scaler is not None:
            scaler_path = filepath.parent / f"{filepath.stem}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

        # Save metadata
        metadata = {
            'model_type': self.model_type.value,
            'task': self.task,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }

        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load trained model

        Args:
            filepath: Path to load model from
        """
        filepath = Path(filepath)

        # Load model
        if self.model_type == ModelType.LSTM:
            try:
                from tensorflow import keras
                self.model = keras.models.load_model(str(filepath))
            except ImportError:
                raise ImportError("tensorflow required to load LSTM model")
        else:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)

        # Load scaler
        scaler_path = filepath.parent / f"{filepath.stem}_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        # Load metadata
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.training_metrics = metadata.get('training_metrics', {})

        logger.info(f"Model loaded from {filepath}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Scale if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (classification only)

        Args:
            X: Features

        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        # Scale if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")
