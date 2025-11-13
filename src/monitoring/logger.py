"""
Logging configuration using loguru
"""
from loguru import logger
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "data/logs"
) -> logger:
    """
    Configure logging with loguru

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # File handler (if enabled)
    if log_to_file:
        # Create log directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # General log file (rotated daily)
        logger.add(
            f"{log_dir}/trading_bot_{{time:YYYY-MM-DD}}.log",
            rotation="00:00",  # Rotate at midnight
            retention="30 days",  # Keep logs for 30 days
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            compression="zip"  # Compress old logs
        )

        # Error log file (separate file for errors)
        logger.add(
            f"{log_dir}/errors_{{time:YYYY-MM-DD}}.log",
            rotation="00:00",
            retention="90 days",  # Keep error logs longer
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            compression="zip"
        )

        # Trading-specific log file (for trades and signals)
        logger.add(
            f"{log_dir}/trading_{{time:YYYY-MM-DD}}.log",
            rotation="00:00",
            retention="90 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            filter=lambda record: "TRADE" in record["message"] or "SIGNAL" in record["message"],
            compression="zip"
        )

    logger.info(f"Logger initialized with level: {log_level}")
    logger.info(f"File logging: {'enabled' if log_to_file else 'disabled'}")

    return logger


def get_logger() -> logger:
    """
    Get the logger instance

    Returns:
        Logger instance
    """
    return logger


# Create a custom logging function for trades
def log_trade(symbol: str, side: str, amount: float, price: float, pnl: Optional[float] = None):
    """
    Log trade execution

    Args:
        symbol: Trading symbol
        side: Trade side (buy/sell)
        amount: Trade amount
        price: Trade price
        pnl: Profit/Loss (if closing position)
    """
    pnl_str = f" | PnL: {pnl:+.2f}" if pnl is not None else ""
    logger.info(f"TRADE | {symbol} | {side.upper()} | Amount: {amount:.6f} | Price: {price:.2f}{pnl_str}")


def log_signal(symbol: str, signal_type: str, confidence: float, entry_price: float):
    """
    Log trading signal

    Args:
        symbol: Trading symbol
        signal_type: Signal type (BUY/SELL/HOLD)
        confidence: Signal confidence
        entry_price: Entry price
    """
    logger.info(f"SIGNAL | {symbol} | {signal_type.upper()} | Confidence: {confidence:.2%} | Entry: {entry_price:.2f}")


def log_error(error: Exception, context: str = ""):
    """
    Log error with context

    Args:
        error: Exception object
        context: Additional context information
    """
    context_str = f" | Context: {context}" if context else ""
    logger.error(f"ERROR | {type(error).__name__}: {str(error)}{context_str}")
    logger.exception(error)  # Log full traceback


def log_performance(total_value: float, pnl: float, win_rate: float, num_positions: int):
    """
    Log performance metrics

    Args:
        total_value: Total portfolio value
        pnl: Profit/Loss
        win_rate: Win rate percentage
        num_positions: Number of open positions
    """
    logger.info(
        f"PERFORMANCE | Value: {total_value:.2f} | PnL: {pnl:+.2f} | "
        f"Win Rate: {win_rate:.1f}% | Positions: {num_positions}"
    )
