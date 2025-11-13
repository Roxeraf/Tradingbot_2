"""
Database models for storing trading data
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


class Trade(Base):
    """
    Represents a completed trade
    """
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(100), unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    order_type = Column(String(20), nullable=False)  # 'market', 'limit', etc.
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    commission_asset = Column(String(10))
    status = Column(String(20), nullable=False)  # 'pending', 'filled', 'cancelled'
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    strategy_name = Column(String(100))
    pnl = Column(Float, nullable=True)
    pnl_percentage = Column(Float, nullable=True)
    metadata = Column(Text)  # JSON string for additional data

    def __repr__(self):
        return f"<Trade(id={self.id}, symbol={self.symbol}, side={self.side}, price={self.price})>"


class HistoricalData(Base):
    """
    Stores historical OHLCV data
    """
    __tablename__ = 'historical_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    timeframe = Column(String(10), nullable=False)  # '1h', '1d', etc.

    def __repr__(self):
        return f"<HistoricalData(symbol={self.symbol}, timestamp={self.timestamp}, close={self.close})>"

    class Meta:
        # Composite unique constraint
        unique_together = ('symbol', 'timestamp', 'timeframe')


class PerformanceMetrics(Base):
    """
    Stores performance metrics snapshots
    """
    __tablename__ = 'performance_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    total_balance = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    num_open_positions = Column(Integer, default=0)
    num_trades_today = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    strategy_name = Column(String(100))
    metadata = Column(Text)  # JSON string

    def __repr__(self):
        return f"<PerformanceMetrics(timestamp={self.timestamp}, balance={self.total_balance})>"


class Position(Base):
    """
    Stores open positions
    """
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    side = Column(String(10), nullable=False)  # 'long' or 'short'
    entry_price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    entry_time = Column(DateTime, default=datetime.utcnow)
    strategy_name = Column(String(100))
    is_open = Column(Boolean, default=True, index=True)
    metadata = Column(Text)  # JSON string

    def __repr__(self):
        return f"<Position(symbol={self.symbol}, side={self.side}, size={self.size})>"


class BacktestResult(Base):
    """
    Stores backtest results
    """
    __tablename__ = 'backtest_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float)
    num_trades = Column(Integer)
    win_rate = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    profit_factor = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    parameters = Column(Text)  # JSON string of strategy parameters
    trades_data = Column(Text)  # JSON string of all trades
    equity_curve = Column(Text)  # JSON string of equity curve

    def __repr__(self):
        return f"<BacktestResult(strategy={self.strategy_name}, return={self.total_return}%)>"


class SystemLog(Base):
    """
    Stores system logs and events
    """
    __tablename__ = 'system_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(20), nullable=False)  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    message = Column(Text, nullable=False)
    module = Column(String(100))
    function = Column(String(100))
    metadata = Column(Text)  # JSON string

    def __repr__(self):
        return f"<SystemLog(level={self.level}, timestamp={self.timestamp})>"


class StrategyConfig(Base):
    """
    Stores strategy configurations
    """
    __tablename__ = 'strategy_configs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    strategy_type = Column(String(100), nullable=False)
    parameters = Column(Text, nullable=False)  # JSON string
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    description = Column(Text)

    def __repr__(self):
        return f"<StrategyConfig(name={self.name}, type={self.strategy_type}, active={self.is_active})>"
