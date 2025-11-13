"""
Order management and execution system
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from src.exchanges.base_exchange import BaseExchange
from src.strategies.base_strategy import TradingSignal, SignalType
from src.monitoring.logger import get_logger, log_trade, log_error

logger = get_logger()


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderManager:
    """
    Manages order placement, tracking, and execution
    """

    def __init__(self, exchange: BaseExchange):
        """
        Initialize order manager

        Args:
            exchange: Exchange instance for order execution
        """
        self.exchange = exchange
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place a market order

        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            amount: Order amount
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)

        Returns:
            Order result dictionary
        """
        try:
            logger.info(f"Placing market {side} order: {symbol} | Amount: {amount}")

            # Place the main market order
            order = await self.exchange.place_order(
                symbol=symbol,
                order_type='market',
                side=side,
                amount=amount
            )

            # Log the trade
            log_trade(
                symbol=symbol,
                side=side,
                amount=amount,
                price=order.get('price', 0)
            )

            # Track the order
            order_id = order['id']
            self.open_orders[order_id] = {
                **order,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'created_at': datetime.now()
            }

            # If stop loss or take profit specified, place additional orders
            if stop_loss and order['status'] == 'filled':
                await self._place_stop_loss(symbol, amount, stop_loss, side)

            if take_profit and order['status'] == 'filled':
                await self._place_take_profit(symbol, amount, take_profit, side)

            return order

        except Exception as e:
            log_error(e, f"Failed to place market order: {symbol} {side}")
            raise

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place a limit order

        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Limit price
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)

        Returns:
            Order result dictionary
        """
        try:
            logger.info(f"Placing limit {side} order: {symbol} | Amount: {amount} | Price: {price}")

            order = await self.exchange.place_order(
                symbol=symbol,
                order_type='limit',
                side=side,
                amount=amount,
                price=price
            )

            # Track the order
            order_id = order['id']
            self.open_orders[order_id] = {
                **order,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'created_at': datetime.now()
            }

            return order

        except Exception as e:
            log_error(e, f"Failed to place limit order: {symbol} {side} @ {price}")
            raise

    async def _place_stop_loss(
        self,
        symbol: str,
        amount: float,
        stop_price: float,
        original_side: str
    ) -> Optional[Dict[str, Any]]:
        """
        Place a stop loss order

        Args:
            symbol: Trading symbol
            amount: Order amount
            stop_price: Stop loss price
            original_side: Original position side

        Returns:
            Order result or None
        """
        try:
            # Stop loss side is opposite of position side
            side = 'sell' if original_side == 'buy' else 'buy'

            logger.info(f"Placing stop loss: {symbol} @ {stop_price}")

            # Note: Not all exchanges support stop loss orders via CCXT
            # This is a simplified implementation
            order = await self.exchange.place_order(
                symbol=symbol,
                order_type='stop_loss',
                side=side,
                amount=amount,
                price=stop_price
            )

            return order

        except Exception as e:
            logger.warning(f"Could not place stop loss order: {e}")
            return None

    async def _place_take_profit(
        self,
        symbol: str,
        amount: float,
        take_profit_price: float,
        original_side: str
    ) -> Optional[Dict[str, Any]]:
        """
        Place a take profit order

        Args:
            symbol: Trading symbol
            amount: Order amount
            take_profit_price: Take profit price
            original_side: Original position side

        Returns:
            Order result or None
        """
        try:
            # Take profit side is opposite of position side
            side = 'sell' if original_side == 'buy' else 'buy'

            logger.info(f"Placing take profit: {symbol} @ {take_profit_price}")

            order = await self.exchange.place_order(
                symbol=symbol,
                order_type='take_profit',
                side=side,
                amount=amount,
                price=take_profit_price
            )

            return order

        except Exception as e:
            logger.warning(f"Could not place take profit order: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol

        Returns:
            True if cancelled successfully
        """
        try:
            logger.info(f"Cancelling order: {order_id}")

            await self.exchange.cancel_order(order_id, symbol)

            if order_id in self.open_orders:
                order = self.open_orders[order_id]
                order['status'] = OrderStatus.CANCELLED.value
                self.order_history.append(order)
                del self.open_orders[order_id]

            return True

        except Exception as e:
            log_error(e, f"Failed to cancel order: {order_id}")
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders

        Args:
            symbol: Cancel orders for specific symbol only (optional)

        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0

        orders_to_cancel = list(self.open_orders.values())
        if symbol:
            orders_to_cancel = [o for o in orders_to_cancel if o['symbol'] == symbol]

        for order in orders_to_cancel:
            success = await self.cancel_order(order['id'], order['symbol'])
            if success:
                cancelled_count += 1

        logger.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count

    async def execute_signal(
        self,
        signal: TradingSignal,
        position_size: float,
        order_type: str = 'market'
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a trading signal

        Args:
            signal: Trading signal to execute
            position_size: Size of position to open
            order_type: Order type ('market' or 'limit')

        Returns:
            Order result or None
        """
        if signal.signal_type == SignalType.HOLD:
            logger.info(f"Signal is HOLD for {signal.symbol}, not executing")
            return None

        try:
            side = 'buy' if signal.signal_type == SignalType.BUY else 'sell'

            if order_type == 'market':
                order = await self.place_market_order(
                    symbol=signal.symbol,
                    side=side,
                    amount=position_size,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
            else:  # limit order
                order = await self.place_limit_order(
                    symbol=signal.symbol,
                    side=side,
                    amount=position_size,
                    price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )

            return order

        except Exception as e:
            log_error(e, f"Failed to execute signal: {signal.symbol} {signal.signal_type.value}")
            return None

    async def sync_orders(self) -> None:
        """
        Synchronize order status with exchange
        Updates order statuses and moves filled orders to history
        """
        try:
            for order_id in list(self.open_orders.keys()):
                order = self.open_orders[order_id]
                symbol = order['symbol']

                # Fetch current order status
                current_status = await self.exchange.get_order_status(order_id, symbol)

                # Update order
                self.open_orders[order_id].update(current_status)

                # Move to history if filled or cancelled
                if current_status['status'] in ['filled', 'cancelled', 'rejected']:
                    self.order_history.append(self.open_orders[order_id])
                    del self.open_orders[order_id]

        except Exception as e:
            log_error(e, "Failed to sync orders")

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of open orders
        """
        if symbol:
            return [o for o in self.open_orders.values() if o['symbol'] == symbol]
        return list(self.open_orders.values())

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get order history

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of orders to return

        Returns:
            List of historical orders
        """
        history = self.order_history
        if symbol:
            history = [o for o in history if o['symbol'] == symbol]

        return history[-limit:]
