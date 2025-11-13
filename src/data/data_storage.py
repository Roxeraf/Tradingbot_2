"""
Data storage layer for database operations
"""
from sqlalchemy import create_engine, and_, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import json

from .models import (
    Base, Trade, HistoricalData, PerformanceMetrics,
    Position, BacktestResult, SystemLog, StrategyConfig
)


class DataStorage:
    """
    Handles all database operations
    """

    def __init__(self, database_url: str = "sqlite:///data/trading_bot.db"):
        """
        Initialize database connection

        Args:
            database_url: SQLAlchemy database URL
        """
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.create_tables()

    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    # Trade operations
    def save_trade(self, trade_data: Dict[str, Any]) -> Trade:
        """
        Save a trade to database

        Args:
            trade_data: Trade information dict

        Returns:
            Created Trade object
        """
        session = self.get_session()
        try:
            # Convert metadata to JSON string if it's a dict
            if 'metadata' in trade_data and isinstance(trade_data['metadata'], dict):
                trade_data['metadata'] = json.dumps(trade_data['metadata'])

            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            session.refresh(trade)
            return trade
        finally:
            session.close()

    def get_trade(self, trade_id: int) -> Optional[Trade]:
        """Get trade by ID"""
        session = self.get_session()
        try:
            return session.query(Trade).filter(Trade.id == trade_id).first()
        finally:
            session.close()

    def get_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Trade]:
        """
        Get trades with optional filters

        Args:
            symbol: Filter by symbol
            since: Get trades after this timestamp
            limit: Maximum number of trades

        Returns:
            List of trades
        """
        session = self.get_session()
        try:
            query = session.query(Trade)

            if symbol:
                query = query.filter(Trade.symbol == symbol)
            if since:
                query = query.filter(Trade.timestamp >= since)

            query = query.order_by(desc(Trade.timestamp)).limit(limit)

            return query.all()
        finally:
            session.close()

    # Historical data operations
    def save_historical_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Save historical OHLCV data to database

        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1h')
        """
        session = self.get_session()
        try:
            for timestamp, row in data.iterrows():
                hist_data = HistoricalData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    timeframe=timeframe
                )

                # Try to add, skip if already exists
                try:
                    session.add(hist_data)
                    session.commit()
                except IntegrityError:
                    session.rollback()
                    continue
        finally:
            session.close()

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical data from database

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            since: Start date
            until: End date

        Returns:
            DataFrame with OHLCV data
        """
        session = self.get_session()
        try:
            query = session.query(HistoricalData).filter(
                and_(
                    HistoricalData.symbol == symbol,
                    HistoricalData.timeframe == timeframe
                )
            )

            if since:
                query = query.filter(HistoricalData.timestamp >= since)
            if until:
                query = query.filter(HistoricalData.timestamp <= until)

            query = query.order_by(HistoricalData.timestamp)

            data = query.all()

            # Convert to DataFrame
            if data:
                df = pd.DataFrame([
                    {
                        'timestamp': d.timestamp,
                        'open': d.open,
                        'high': d.high,
                        'low': d.low,
                        'close': d.close,
                        'volume': d.volume
                    }
                    for d in data
                ])
                df.set_index('timestamp', inplace=True)
                return df
            else:
                return pd.DataFrame()
        finally:
            session.close()

    # Performance metrics operations
    def save_performance_snapshot(self, metrics: Dict[str, Any]) -> PerformanceMetrics:
        """Save performance metrics snapshot"""
        session = self.get_session()
        try:
            if 'metadata' in metrics and isinstance(metrics['metadata'], dict):
                metrics['metadata'] = json.dumps(metrics['metadata'])

            perf = PerformanceMetrics(**metrics)
            session.add(perf)
            session.commit()
            session.refresh(perf)
            return perf
        finally:
            session.close()

    def get_performance_history(
        self,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[PerformanceMetrics]:
        """Get performance history"""
        session = self.get_session()
        try:
            query = session.query(PerformanceMetrics)

            if since:
                query = query.filter(PerformanceMetrics.timestamp >= since)

            query = query.order_by(desc(PerformanceMetrics.timestamp)).limit(limit)

            return query.all()
        finally:
            session.close()

    # Position operations
    def save_position(self, position_data: Dict[str, Any]) -> Position:
        """Save or update position"""
        session = self.get_session()
        try:
            if 'metadata' in position_data and isinstance(position_data['metadata'], dict):
                position_data['metadata'] = json.dumps(position_data['metadata'])

            # Check if position exists
            existing = session.query(Position).filter(
                Position.symbol == position_data['symbol']
            ).first()

            if existing:
                for key, value in position_data.items():
                    setattr(existing, key, value)
                session.commit()
                session.refresh(existing)
                return existing
            else:
                position = Position(**position_data)
                session.add(position)
                session.commit()
                session.refresh(position)
                return position
        finally:
            session.close()

    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        session = self.get_session()
        try:
            return session.query(Position).filter(Position.is_open == True).all()
        finally:
            session.close()

    def close_position(self, symbol: str) -> None:
        """Mark position as closed"""
        session = self.get_session()
        try:
            position = session.query(Position).filter(Position.symbol == symbol).first()
            if position:
                position.is_open = False
                session.commit()
        finally:
            session.close()

    # Backtest results operations
    def save_backtest_result(self, result_data: Dict[str, Any]) -> BacktestResult:
        """Save backtest result"""
        session = self.get_session()
        try:
            # Convert dict fields to JSON strings
            for field in ['parameters', 'trades_data', 'equity_curve']:
                if field in result_data and isinstance(result_data[field], (dict, list)):
                    result_data[field] = json.dumps(result_data[field])

            result = BacktestResult(**result_data)
            session.add(result)
            session.commit()
            session.refresh(result)
            return result
        finally:
            session.close()

    def get_backtest_results(
        self,
        strategy_name: Optional[str] = None,
        limit: int = 50
    ) -> List[BacktestResult]:
        """Get backtest results"""
        session = self.get_session()
        try:
            query = session.query(BacktestResult)

            if strategy_name:
                query = query.filter(BacktestResult.strategy_name == strategy_name)

            query = query.order_by(desc(BacktestResult.created_at)).limit(limit)

            return query.all()
        finally:
            session.close()

    # System logs operations
    def save_log(self, level: str, message: str, module: str = None, metadata: Dict = None) -> None:
        """Save system log"""
        session = self.get_session()
        try:
            log = SystemLog(
                level=level,
                message=message,
                module=module,
                metadata=json.dumps(metadata) if metadata else None
            )
            session.add(log)
            session.commit()
        finally:
            session.close()

    def get_logs(
        self,
        level: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 500
    ) -> List[SystemLog]:
        """Get system logs"""
        session = self.get_session()
        try:
            query = session.query(SystemLog)

            if level:
                query = query.filter(SystemLog.level == level)
            if since:
                query = query.filter(SystemLog.timestamp >= since)

            query = query.order_by(desc(SystemLog.timestamp)).limit(limit)

            return query.all()
        finally:
            session.close()

    # Strategy configuration operations
    def save_strategy_config(self, config_data: Dict[str, Any]) -> StrategyConfig:
        """Save or update strategy configuration"""
        session = self.get_session()
        try:
            if 'parameters' in config_data and isinstance(config_data['parameters'], dict):
                config_data['parameters'] = json.dumps(config_data['parameters'])

            # Check if config exists
            existing = session.query(StrategyConfig).filter(
                StrategyConfig.name == config_data['name']
            ).first()

            if existing:
                for key, value in config_data.items():
                    setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
                session.commit()
                session.refresh(existing)
                return existing
            else:
                config = StrategyConfig(**config_data)
                session.add(config)
                session.commit()
                session.refresh(config)
                return config
        finally:
            session.close()

    def get_strategy_configs(self, active_only: bool = False) -> List[StrategyConfig]:
        """Get strategy configurations"""
        session = self.get_session()
        try:
            query = session.query(StrategyConfig)

            if active_only:
                query = query.filter(StrategyConfig.is_active == True)

            return query.all()
        finally:
            session.close()

    def get_strategy_config(self, name: str) -> Optional[StrategyConfig]:
        """Get strategy configuration by name"""
        session = self.get_session()
        try:
            return session.query(StrategyConfig).filter(StrategyConfig.name == name).first()
        finally:
            session.close()
