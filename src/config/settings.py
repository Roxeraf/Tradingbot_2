"""
Configuration management using Pydantic for type-safe settings
"""
from typing import List, Dict, Any, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


class Settings(BaseSettings):
    """
    Application settings with validation and type checking
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Exchange Configuration
    EXCHANGE_NAME: str = Field(default="bitpanda", description="Exchange name (e.g., bitpanda, binance)")
    API_KEY: str = Field(default="", description="Exchange API key")
    API_SECRET: str = Field(default="", description="Exchange API secret")
    API_PASSPHRASE: Optional[str] = Field(default=None, description="API passphrase (if required)")
    TESTNET: bool = Field(default=True, description="Use testnet/sandbox mode")

    # Trading Parameters
    TRADING_PAIRS: str = Field(default="BTC/EUR,ETH/EUR", description="Comma-separated trading pairs")
    TIMEFRAME: str = Field(default="1h", description="Candlestick timeframe")
    MAX_POSITION_SIZE: float = Field(default=0.1, ge=0.01, le=1.0, description="Max position size as % of portfolio")
    MAX_PORTFOLIO_RISK: float = Field(default=0.02, ge=0.001, le=0.1, description="Max risk per trade")

    # Strategy Configuration
    STRATEGY_NAME: str = Field(default="MovingAverageCrossover", description="Strategy to use")
    STRATEGY_PARAMS: str = Field(
        default='{"fast_period": 20, "slow_period": 50, "min_confidence": 0.6}',
        description="Strategy parameters as JSON string"
    )

    # Risk Management
    STOP_LOSS_PERCENTAGE: float = Field(default=0.02, ge=0.001, le=0.5, description="Stop loss %")
    TAKE_PROFIT_PERCENTAGE: float = Field(default=0.04, ge=0.001, le=1.0, description="Take profit %")
    TRAILING_STOP: bool = Field(default=False, description="Enable trailing stop")

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///data/trading_bot.db",
        description="Database connection URL"
    )

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, ge=1, le=65535, description="API port")
    SECRET_KEY: str = Field(default="change-this-secret-key", description="JWT secret key")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=1, description="Token expiry time")

    # Frontend Configuration
    VITE_API_URL: str = Field(default="http://localhost:8000/api", description="Frontend API URL")
    VITE_WS_URL: str = Field(default="ws://localhost:8000/ws", description="Frontend WebSocket URL")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_TO_FILE: bool = Field(default=True, description="Enable file logging")

    # Redis Configuration (optional)
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")

    # Notifications (optional)
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(default=None, description="Telegram bot token")
    TELEGRAM_CHAT_ID: Optional[str] = Field(default=None, description="Telegram chat ID")
    DISCORD_WEBHOOK_URL: Optional[str] = Field(default=None, description="Discord webhook URL")
    EMAIL_SMTP_HOST: Optional[str] = Field(default=None, description="Email SMTP host")
    EMAIL_SMTP_PORT: Optional[int] = Field(default=587, description="Email SMTP port")
    EMAIL_FROM: Optional[str] = Field(default=None, description="Email from address")
    EMAIL_PASSWORD: Optional[str] = Field(default=None, description="Email password")
    EMAIL_TO: Optional[str] = Field(default=None, description="Email to address")

    @field_validator("TIMEFRAME")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe format"""
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        if v not in valid_timeframes:
            raise ValueError(f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {', '.join(valid_levels)}")
        return v_upper

    def get_trading_pairs(self) -> List[str]:
        """Parse trading pairs from comma-separated string"""
        return [pair.strip() for pair in self.TRADING_PAIRS.split(",") if pair.strip()]

    def get_strategy_params(self) -> Dict[str, Any]:
        """Parse strategy parameters from JSON string"""
        try:
            return json.loads(self.STRATEGY_PARAMS)
        except json.JSONDecodeError:
            return {}

    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.TESTNET

    def has_notifications_enabled(self) -> bool:
        """Check if any notification channel is configured"""
        return any([
            self.TELEGRAM_BOT_TOKEN,
            self.DISCORD_WEBHOOK_URL,
            self.EMAIL_FROM
        ])


# Global settings instance
settings = Settings()
