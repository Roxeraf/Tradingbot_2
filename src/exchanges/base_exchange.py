"""
Abstract base class for exchange implementations
All exchange connectors must implement these methods
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd


class BaseExchange(ABC):
    """
    Abstract base class for exchange implementations.
    All exchange classes must implement these methods.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, passphrase: Optional[str] = None):
        """
        Initialize exchange connection

        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use testnet/sandbox mode
            passphrase: API passphrase (if required by exchange)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.passphrase = passphrase
        self.exchange = None
        self.name = self.__class__.__name__

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to exchange

        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close exchange connection and cleanup"""
        pass

    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """
        Get account balance for all assets

        Returns:
            Dict mapping currency symbols to available balance
            Example: {"BTC": 0.5, "EUR": 1000.0, "ETH": 2.0}
        """
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data for a symbol

        Args:
            symbol: Trading pair symbol (e.g., "BTC/EUR")

        Returns:
            Dict with ticker data including:
            - symbol: str
            - last: float (last price)
            - bid: float
            - ask: float
            - high: float (24h high)
            - low: float (24h low)
            - volume: float (24h volume)
            - timestamp: int
        """
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data

        Args:
            symbol: Trading pair symbol (e.g., "BTC/EUR")
            timeframe: Candlestick timeframe (e.g., "1h", "1d")
            since: Fetch data from this timestamp (milliseconds)
            limit: Maximum number of candles to fetch

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Place an order

        Args:
            symbol: Trading pair symbol
            order_type: Order type ("market", "limit", "stop_loss", "take_profit")
            side: Order side ("buy" or "sell")
            amount: Order amount in base currency
            price: Order price (required for limit orders)
            params: Additional exchange-specific parameters

        Returns:
            Dict with order information:
            - id: str (order ID)
            - symbol: str
            - type: str
            - side: str
            - amount: float
            - price: float
            - status: str ("open", "closed", "canceled")
            - timestamp: int
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an open order

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol

        Returns:
            Dict with cancellation confirmation
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders

        Args:
            symbol: Filter by specific symbol (optional)

        Returns:
            List of open orders
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Check order status

        Args:
            order_id: Order ID to check
            symbol: Trading pair symbol

        Returns:
            Dict with order status information
        """
        pass

    @abstractmethod
    async def get_closed_orders(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get closed/filled orders

        Args:
            symbol: Filter by specific symbol
            since: Fetch orders from this timestamp
            limit: Maximum number of orders to fetch

        Returns:
            List of closed orders
        """
        pass

    @abstractmethod
    async def get_my_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trade history

        Args:
            symbol: Filter by specific symbol
            since: Fetch trades from this timestamp
            limit: Maximum number of trades to fetch

        Returns:
            List of executed trades
        """
        pass

    async def get_markets(self) -> List[Dict[str, Any]]:
        """
        Get available trading markets/pairs

        Returns:
            List of available markets
        """
        if self.exchange:
            return await self.exchange.fetch_markets()
        return []

    async def get_trading_fees(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get trading fees

        Args:
            symbol: Specific symbol to get fees for

        Returns:
            Dict with fee information
        """
        if self.exchange:
            return await self.exchange.fetch_trading_fees()
        return {}

    def format_symbol(self, symbol: str) -> str:
        """
        Format symbol to exchange-specific format

        Args:
            symbol: Standard symbol format (e.g., "BTC/EUR")

        Returns:
            Exchange-specific symbol format
        """
        return symbol

    def parse_ohlcv(self, ohlcv_data: List[List]) -> pd.DataFrame:
        """
        Parse OHLCV data into DataFrame

        Args:
            ohlcv_data: Raw OHLCV data from exchange

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        df = pd.DataFrame(
            ohlcv_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    async def test_connection(self) -> bool:
        """
        Test if exchange connection is working

        Returns:
            bool: True if connection is working
        """
        try:
            await self.get_balance()
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"{self.name}(testnet={self.testnet})"
