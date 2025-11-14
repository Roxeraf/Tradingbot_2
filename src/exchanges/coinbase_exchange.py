"""
Coinbase (Coinbase Advanced Trade) exchange implementation using CCXT
"""
from typing import Dict, List, Optional, Any
import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger

from .base_exchange import BaseExchange


class CoinbaseExchange(BaseExchange):
    """
    Coinbase Advanced Trade (formerly Coinbase Pro) connector using CCXT library
    Note: This uses Coinbase Advanced Trade API, not the basic Coinbase API
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, passphrase: Optional[str] = None):
        """
        Initialize Coinbase exchange connection

        Args:
            api_key: Coinbase API key
            api_secret: Coinbase API secret
            testnet: Use sandbox mode (default: True)
            passphrase: API passphrase (required for Coinbase)
        """
        super().__init__(api_key, api_secret, testnet, passphrase)
        self.exchange_id = "coinbase"
        self.exchange = None

        if not passphrase and not testnet:
            logger.warning("Coinbase requires a passphrase for API authentication")

    async def connect(self) -> bool:
        """
        Establish connection to Coinbase exchange

        Returns:
            bool: True if connection successful
        """
        try:
            # Initialize CCXT Coinbase exchange
            # Note: CCXT uses 'coinbase' for Coinbase Advanced Trade
            self.exchange = ccxt.coinbase({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.passphrase,  # Coinbase requires passphrase
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            })

            # Enable sandbox if specified
            if self.testnet:
                self.exchange.set_sandbox_mode(True)
                logger.info("Coinbase sandbox mode enabled")

            # Load markets
            await self.exchange.load_markets()

            # Test connection
            await self.exchange.fetch_balance()

            logger.info(f"Successfully connected to Coinbase ({'sandbox' if self.testnet else 'live'})")
            return True

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Coinbase: {str(e)}")

    async def disconnect(self) -> None:
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
            logger.info("Disconnected from Coinbase")

    async def get_balance(self) -> Dict[str, float]:
        """
        Get account balance

        Returns:
            Dict mapping currency symbols to available balance
        """
        try:
            balance = await self.exchange.fetch_balance()
            # Extract only available (free) balances
            result = {}
            for currency, amounts in balance.items():
                if isinstance(amounts, dict) and 'free' in amounts:
                    free_balance = amounts['free']
                    if free_balance and free_balance > 0:
                        result[currency] = free_balance
            return result
        except Exception as e:
            raise Exception(f"Failed to fetch balance: {str(e)}")

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data

        Args:
            symbol: Trading pair (e.g., "BTC/USD", "ETH/USD")

        Returns:
            Dict with ticker information
        """
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'symbol': ticker['symbol'],
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': ticker['baseVolume'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            raise Exception(f"Failed to fetch ticker for {symbol}: {str(e)}")

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: int = 300
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data

        Args:
            symbol: Trading pair (e.g., "BTC/USD")
            timeframe: Candlestick timeframe (e.g., "1h", "1d", "5m", "15m")
            since: Fetch data from this timestamp (milliseconds)
            limit: Maximum number of candles (max 300 for Coinbase)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Coinbase has a max limit of 300
            limit = min(limit, 300)

            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )

            df = self.parse_ohlcv(ohlcv)
            return df

        except Exception as e:
            raise Exception(f"Failed to fetch historical data for {symbol}: {str(e)}")

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
            symbol: Trading pair (e.g., "BTC/USD")
            order_type: Order type ("market", "limit")
            side: Order side ("buy" or "sell")
            amount: Order amount in base currency
            price: Order price (required for limit orders)
            params: Additional parameters

        Returns:
            Dict with order information
        """
        try:
            params = params or {}

            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )

            return {
                'id': order['id'],
                'symbol': order['symbol'],
                'type': order['type'],
                'side': order['side'],
                'amount': order['amount'],
                'price': order['price'],
                'status': order['status'],
                'timestamp': order['timestamp']
            }

        except Exception as e:
            raise Exception(f"Failed to place order: {str(e)}")

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an open order

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair

        Returns:
            Dict with cancellation confirmation
        """
        try:
            result = await self.exchange.cancel_order(order_id, symbol)
            return result
        except Exception as e:
            raise Exception(f"Failed to cancel order {order_id}: {str(e)}")

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders

        Args:
            symbol: Filter by specific symbol (optional)

        Returns:
            List of open orders
        """
        try:
            orders = await self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            raise Exception(f"Failed to fetch open orders: {str(e)}")

    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Check order status

        Args:
            order_id: Order ID to check
            symbol: Trading pair

        Returns:
            Dict with order status information
        """
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            raise Exception(f"Failed to fetch order status: {str(e)}")

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
        try:
            orders = await self.exchange.fetch_closed_orders(
                symbol=symbol,
                since=since,
                limit=limit
            )
            return orders
        except Exception as e:
            raise Exception(f"Failed to fetch closed orders: {str(e)}")

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
        try:
            trades = await self.exchange.fetch_my_trades(
                symbol=symbol,
                since=since,
                limit=limit
            )
            return trades
        except Exception as e:
            raise Exception(f"Failed to fetch trades: {str(e)}")

    def format_symbol(self, symbol: str) -> str:
        """
        Format symbol to Coinbase format
        Coinbase uses format like "BTC/USD" or "BTC-USD"

        Args:
            symbol: Standard symbol format

        Returns:
            Coinbase-specific symbol format
        """
        # CCXT handles the conversion automatically
        return symbol
