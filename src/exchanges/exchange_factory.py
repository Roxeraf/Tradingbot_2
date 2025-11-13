"""
Factory pattern for creating exchange instances
"""
from typing import Optional
from .base_exchange import BaseExchange
from .bitpanda_exchange import BitpandaExchange


class ExchangeFactory:
    """
    Factory class for creating exchange instances
    """

    _exchanges = {
        'bitpanda': BitpandaExchange,
        # Add more exchanges here as they are implemented
        # 'binance': BinanceExchange,
        # 'coinbase': CoinbaseExchange,
        # 'kraken': KrakenExchange,
    }

    @classmethod
    def create(
        cls,
        exchange_name: str,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        passphrase: Optional[str] = None
    ) -> BaseExchange:
        """
        Create an exchange instance

        Args:
            exchange_name: Name of the exchange (e.g., 'bitpanda', 'binance')
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use testnet/sandbox mode
            passphrase: API passphrase (if required)

        Returns:
            Exchange instance

        Raises:
            ValueError: If exchange is not supported
        """
        exchange_name_lower = exchange_name.lower()

        if exchange_name_lower not in cls._exchanges:
            available = ', '.join(cls._exchanges.keys())
            raise ValueError(
                f"Exchange '{exchange_name}' is not supported. "
                f"Available exchanges: {available}"
            )

        exchange_class = cls._exchanges[exchange_name_lower]
        return exchange_class(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            passphrase=passphrase
        )

    @classmethod
    def create_from_settings(cls, settings) -> BaseExchange:
        """
        Create exchange instance from settings object

        Args:
            settings: Settings instance with exchange configuration

        Returns:
            Exchange instance
        """
        return cls.create(
            exchange_name=settings.EXCHANGE_NAME,
            api_key=settings.API_KEY,
            api_secret=settings.API_SECRET,
            testnet=settings.TESTNET,
            passphrase=settings.API_PASSPHRASE
        )

    @classmethod
    def get_supported_exchanges(cls) -> list:
        """
        Get list of supported exchanges

        Returns:
            List of exchange names
        """
        return list(cls._exchanges.keys())

    @classmethod
    def register_exchange(cls, name: str, exchange_class: type):
        """
        Register a new exchange implementation

        Args:
            name: Exchange name
            exchange_class: Exchange class (must inherit from BaseExchange)

        Raises:
            TypeError: If exchange_class doesn't inherit from BaseExchange
        """
        if not issubclass(exchange_class, BaseExchange):
            raise TypeError(
                f"Exchange class must inherit from BaseExchange, "
                f"got {exchange_class.__name__}"
            )

        cls._exchanges[name.lower()] = exchange_class
