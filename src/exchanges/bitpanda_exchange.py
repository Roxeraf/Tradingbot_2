"""
Bitpanda exchange implementation using CCXT
"""
from typing import Dict, List, Optional, Any
import ccxt.async_support as ccxt
import pandas as pd
from datetime import datetime

from .base_exchange import BaseExchange


class BitpandaExchange(BaseExchange):
    """
    Bitpanda exchange connector using CCXT library
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, passphrase: Optional[str] = None):
        """
        Initialize Bitpanda exchange connection

        Args:
            api_key: Bitpanda API key
            api_secret: Bitpanda API secret
            testnet: Use sandbox mode (Bitpanda doesn't have testnet, this will enable paper trading mode)
            passphrase: Not used for Bitpanda
        """
        super().__init__(api_key, api_secret, testnet, passphrase)
        self.exchange_id = "bitpanda"
        self.exchange = None

    async def connect(self) -> bool:
        """
        Establish connection to Bitpanda exchange

        Returns:
            bool: True if connection successful
        """
        try:
            # Initialize CCXT Bitpanda exchange
            # Note: Bitpanda Pro is the exchange for API trading
            self.exchange = ccxt.bitpanda({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            })

            # Load markets
            await self.exchange.load_markets()

            # Test connection
            await self.exchange.fetch_balance()

            return True

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Bitpanda: {str(e)}")

    async def disconnect(self) -> None:
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()

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
            symbol: Trading pair (e.g., "BTC/EUR")

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
                'volume': ticker['quoteVolume'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            raise Exception(f"Failed to fetch ticker for {symbol}: {str(e)}")

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
            symbol: Trading pair symbol
            timeframe: Candle timeframe (e.g., "1h", "1d")
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )

            # Convert to DataFrame
            df = self.parse_ohlcv(ohlcv)
            df.attrs['symbol'] = symbol
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
            symbol: Trading pair
            order_type: "market" or "limit"
            side: "buy" or "sell"
            amount: Order amount
            price: Limit price (required for limit orders)
            params: Additional parameters

        Returns:
            Dict with order information
        """
        try:
            if params is None:
                params = {}

            # Place order based on type
            if order_type == 'market':
                order = await self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    params=params
                )
            elif order_type == 'limit':
                if price is None:
                    raise ValueError("Price is required for limit orders")
                order = await self.exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    price=price,
                    params=params
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            return {
                'id': order['id'],
                'symbol': order['symbol'],
                'type': order['type'],
                'side': order['side'],
                'amount': order['amount'],
                'price': order.get('price'),
                'status': order['status'],
                'timestamp': order['timestamp'],
                'raw': order
            }

        except Exception as e:
            raise Exception(f"Failed to place order: {str(e)}")

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an open order

        Args:
            order_id: Order ID
            symbol: Trading pair

        Returns:
            Cancellation confirmation
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
            symbol: Filter by symbol (optional)

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
            order_id: Order ID
            symbol: Trading pair

        Returns:
            Order status information
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
        Get closed orders

        Args:
            symbol: Filter by symbol
            since: Start timestamp
            limit: Number of orders

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
            symbol: Filter by symbol
            since: Start timestamp
            limit: Number of trades

        Returns:
            List of trades
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
        Format symbol for Bitpanda

        Args:
            symbol: Standard format (e.g., "BTC/EUR")

        Returns:
            Bitpanda format
        """
        # Bitpanda uses standard CCXT format
        return symbol

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
