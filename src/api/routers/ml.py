"""
Machine Learning API Router
Endpoints for ML model training, prediction, and ML strategies
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import tempfile

from ...ml.feature_engineering import FeatureEngineer
from ...ml.model_trainer import MLModelTrainer, ModelType
from ...ml.prediction_service import PredictionService

router = APIRouter(
    prefix="/ml",
    tags=["machine-learning"]
)


class TrainModelRequest(BaseModel):
    """ML model training request"""
    model_type: str  # 'random_forest', 'xgboost', 'lstm', etc.
    task: str  # 'classification' or 'regression'
    data_path: str  # Path to training data CSV
    feature_columns: List[str]
    target_column: str
    train_size: float = 0.8
    scale_features: bool = True
    model_params: Optional[Dict[str, Any]] = None
    save_path: Optional[str] = None


class PredictionRequest(BaseModel):
    """Prediction request"""
    model_path: str
    model_type: str
    task: str = 'classification'
    data: Dict[str, List[float]]  # OHLCV data
    return_proba: bool = False


class FeatureEngineeringRequest(BaseModel):
    """Feature engineering request"""
    data: Dict[str, List[float]]  # OHLCV data
    include_price_features: bool = True
    include_technical_features: bool = True
    include_statistical_features: bool = True
    include_time_features: bool = True
    create_target: bool = False
    target_type: Optional[str] = 'direction'
    forward_periods: int = 1


# Store active prediction services
prediction_services: Dict[str, PredictionService] = {}


@router.post("/train")
async def train_model(request: TrainModelRequest):
    """
    Train a machine learning model

    Supports Random Forest, XGBoost, LightGBM, Gradient Boosting, and LSTM
    """
    try:
        # Parse model type
        try:
            model_type = ModelType(request.model_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type: {request.model_type}"
            )

        # Load training data
        try:
            data = pd.read_csv(request.data_path, index_col=0, parse_dates=True)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error loading data: {str(e)}"
            )

        # Create trainer
        trainer = MLModelTrainer(
            model_type=model_type,
            task=request.task
        )

        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            data=data,
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            train_size=request.train_size,
            scale_features=request.scale_features
        )

        # Train model
        model_params = request.model_params or {}
        trainer.train(X_train, y_train, **model_params)

        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)

        # Get feature importance
        try:
            feature_importance = trainer.get_feature_importance(top_n=20)
            top_features = feature_importance.to_dict('records')
        except:
            top_features = []

        # Save model if path provided
        model_path = None
        if request.save_path:
            trainer.save_model(request.save_path)
            model_path = request.save_path

        return {
            "status": "success",
            "model_type": request.model_type,
            "task": request.task,
            "metrics": metrics,
            "top_features": top_features,
            "model_path": model_path,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict(request: PredictionRequest):
    """
    Make predictions using a trained model
    """
    try:
        # Parse model type
        try:
            model_type = ModelType(request.model_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type: {request.model_type}"
            )

        # Create or get prediction service
        service_key = f"{request.model_path}_{request.model_type}"

        if service_key not in prediction_services:
            prediction_services[service_key] = PredictionService(
                model_path=request.model_path,
                model_type=model_type,
                task=request.task
            )

        service = prediction_services[service_key]

        # Convert data to DataFrame
        data_df = pd.DataFrame(request.data)

        # Make prediction
        result = service.predict(data_df, return_proba=request.return_proba)

        return {
            "status": "success",
            "prediction": result
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-direction")
async def predict_direction(
    model_path: str,
    model_type: str,
    data: Dict[str, List[float]],
    confidence_threshold: float = 0.6
):
    """
    Predict price direction with confidence

    Returns buy/sell signal with confidence level
    """
    try:
        # Parse model type
        try:
            model_type_enum = ModelType(model_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type: {model_type}"
            )

        # Create or get prediction service
        service_key = f"{model_path}_{model_type}"

        if service_key not in prediction_services:
            prediction_services[service_key] = PredictionService(
                model_path=model_path,
                model_type=model_type_enum,
                task='classification'
            )

        service = prediction_services[service_key]

        # Convert data to DataFrame
        data_df = pd.DataFrame(data)

        # Predict direction
        result = service.predict_direction(data_df, confidence_threshold)

        return {
            "status": "success",
            "signal": result['signal'],
            "confidence": result['confidence'],
            "is_actionable": result['is_actionable'],
            "current_price": result['current_price']
        }

    except Exception as e:
        logger.error(f"Direction prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/engineer-features")
async def engineer_features(request: FeatureEngineeringRequest):
    """
    Engineer features from OHLCV data

    Creates technical, statistical, and time-based features
    """
    try:
        # Create feature engineer
        engineer = FeatureEngineer()

        # Convert data to DataFrame
        data_df = pd.DataFrame(request.data)

        # Engineer features
        features_df = engineer.create_all_features(
            data=data_df,
            include_price_features=request.include_price_features,
            include_technical_features=request.include_technical_features,
            include_statistical_features=request.include_statistical_features,
            include_time_features=request.include_time_features
        )

        # Create target if requested
        if request.create_target:
            features_df = engineer.create_target_variable(
                features_df,
                target_type=request.target_type,
                forward_periods=request.forward_periods
            )

        # Get feature names
        feature_names = engineer.get_feature_importance_names()

        return {
            "status": "success",
            "num_features": len(feature_names),
            "feature_names": feature_names,
            "sample_features": features_df.tail(5).to_dict('records')
        }

    except Exception as e:
        logger.error(f"Feature engineering error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-types")
async def get_model_types():
    """
    Get available ML model types
    """
    return {
        "model_types": [
            {
                "name": "random_forest",
                "description": "Random Forest - ensemble of decision trees",
                "tasks": ["classification", "regression"],
                "pros": "Robust, handles non-linear relationships, feature importance"
            },
            {
                "name": "xgboost",
                "description": "XGBoost - gradient boosted trees",
                "tasks": ["classification", "regression"],
                "pros": "High performance, handles missing values, regularization"
            },
            {
                "name": "lightgbm",
                "description": "LightGBM - fast gradient boosting",
                "tasks": ["classification", "regression"],
                "pros": "Very fast, memory efficient, handles large datasets"
            },
            {
                "name": "gradient_boosting",
                "description": "Gradient Boosting - sklearn implementation",
                "tasks": ["classification", "regression"],
                "pros": "Strong baseline, well-tested, interpretable"
            },
            {
                "name": "lstm",
                "description": "LSTM - Long Short-Term Memory neural network",
                "tasks": ["classification", "regression"],
                "pros": "Captures temporal patterns, sequence modeling"
            }
        ]
    }


@router.get("/feature-types")
async def get_feature_types():
    """
    Get available feature types
    """
    return {
        "feature_types": [
            {
                "name": "price_features",
                "examples": ["returns", "momentum", "gaps", "intraday_range"]
            },
            {
                "name": "technical_features",
                "examples": ["moving_averages", "rsi", "macd", "bollinger_bands", "atr"]
            },
            {
                "name": "statistical_features",
                "examples": ["volatility", "skewness", "kurtosis", "z-score", "hurst_exponent"]
            },
            {
                "name": "time_features",
                "examples": ["day_of_week", "hour", "month", "is_weekend"]
            }
        ]
    }


@router.delete("/prediction-service/{service_key}")
async def clear_prediction_service(service_key: str):
    """
    Clear a prediction service from memory
    """
    if service_key in prediction_services:
        del prediction_services[service_key]
        return {"status": "success", "message": f"Service {service_key} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Service not found")


@router.get("/active-services")
async def get_active_services():
    """
    Get list of active prediction services
    """
    return {
        "active_services": list(prediction_services.keys()),
        "count": len(prediction_services)
    }
