"""
Factory pattern for creating strategy instances
"""
from typing import Dict, Any
from .base_strategy import BaseStrategy
from .moving_average_strategy import MovingAverageCrossoverStrategy


class StrategyFactory:
    """
    Factory class for creating strategy instances
    """

    _strategies = {
        'movingaveragecrossover': MovingAverageCrossoverStrategy,
        'ma_crossover': MovingAverageCrossoverStrategy,
        'sma_crossover': MovingAverageCrossoverStrategy,
        # Add more strategies here as they are implemented
        # 'rsi': RSIStrategy,
        # 'bollinger': BollingerBandsStrategy,
        # 'macd': MACDStrategy,
    }

    @classmethod
    def create(cls, strategy_name: str, params: Dict[str, Any]) -> BaseStrategy:
        """
        Create a strategy instance

        Args:
            strategy_name: Name of the strategy
            params: Strategy parameters

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy is not supported
        """
        strategy_name_lower = strategy_name.lower().replace('_', '').replace('-', '')

        if strategy_name_lower not in cls._strategies:
            available = ', '.join(set(cls._strategies.keys()))
            raise ValueError(
                f"Strategy '{strategy_name}' is not supported. "
                f"Available strategies: {available}"
            )

        strategy_class = cls._strategies[strategy_name_lower]
        return strategy_class(params)

    @classmethod
    def create_from_settings(cls, settings) -> BaseStrategy:
        """
        Create strategy instance from settings object

        Args:
            settings: Settings instance with strategy configuration

        Returns:
            Strategy instance
        """
        params = settings.get_strategy_params()
        return cls.create(settings.STRATEGY_NAME, params)

    @classmethod
    def get_supported_strategies(cls) -> list:
        """
        Get list of supported strategies

        Returns:
            List of unique strategy names
        """
        return list(set(cls._strategies.keys()))

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """
        Register a new strategy implementation

        Args:
            name: Strategy name
            strategy_class: Strategy class (must inherit from BaseStrategy)

        Raises:
            TypeError: If strategy_class doesn't inherit from BaseStrategy
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(
                f"Strategy class must inherit from BaseStrategy, "
                f"got {strategy_class.__name__}"
            )

        cls._strategies[name.lower()] = strategy_class

    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict[str, Any]:
        """
        Get information about a strategy

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dict with strategy information
        """
        strategy_name_lower = strategy_name.lower().replace('_', '').replace('-', '')

        if strategy_name_lower not in cls._strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")

        strategy_class = cls._strategies[strategy_name_lower]

        return {
            'name': strategy_class.__name__,
            'description': strategy_class.__doc__,
            'module': strategy_class.__module__
        }
