"""
Strategy registry for novelty detection.

Provides a central registry for all novelty detection strategies,
allowing dynamic strategy registration and instantiation.
"""

from typing import Dict, List, Type
from ..strategies.base import NoveltyStrategy


class StrategyRegistry:
    """
    Registry for novelty detection strategies.

    Strategies are registered using the @StrategyRegistry.register decorator.
    Once registered, they can be instantiated by their strategy_id.
    """

    _strategies: Dict[str, Type[NoveltyStrategy]] = {}

    @classmethod
    def register(cls, strategy_cls: Type[NoveltyStrategy]) -> Type[NoveltyStrategy]:
        """
        Register a strategy class.

        Usage:
            @StrategyRegistry.register
            class MyStrategy(NoveltyStrategy):
                strategy_id = "my_strategy"
                ...

        Args:
            strategy_cls: Strategy class to register

        Returns:
            The same strategy class (for decorator use)
        """
        if not hasattr(strategy_cls, "strategy_id"):
            raise ValueError(
                f"Strategy class {strategy_cls.__name__} must have a 'strategy_id' attribute"
            )

        strategy_id = strategy_cls.strategy_id
        if strategy_id in cls._strategies:
            raise ValueError(
                f"Strategy ID '{strategy_id}' is already registered "
                f"(existing: {cls._strategies[strategy_id].__name__}, "
                f"new: {strategy_cls.__name__})"
            )

        cls._strategies[strategy_id] = strategy_cls
        return strategy_cls

    @classmethod
    def get(cls, strategy_id: str) -> Type[NoveltyStrategy]:
        """
        Get a strategy class by ID.

        Args:
            strategy_id: Unique strategy identifier

        Returns:
            Strategy class

        Raises:
            ValueError: If strategy_id is not registered
        """
        if strategy_id not in cls._strategies:
            available = ", ".join(cls.list_strategies())
            raise ValueError(
                f"Unknown strategy: '{strategy_id}'. Available strategies: {available}"
            )
        return cls._strategies[strategy_id]

    @classmethod
    def create(cls, strategy_id: str) -> NoveltyStrategy:
        """
        Create an instance of a strategy.

        Args:
            strategy_id: Unique strategy identifier

        Returns:
            Instantiated strategy object
        """
        strategy_cls = cls.get(strategy_id)
        return strategy_cls()

    @classmethod
    def list_strategies(cls, maturity: str | None = None) -> List[str]:
        """
        List all registered strategy IDs, optionally filtered by maturity.

        Args:
            maturity: Optional maturity filter ("production", "experimental", "internal").

        Returns:
            List of strategy IDs in registration order
        """
        if maturity is None:
            return list(cls._strategies.keys())
        return [
            sid
            for sid, scls in cls._strategies.items()
            if getattr(scls, "maturity", "experimental") == maturity
        ]

    @classmethod
    def is_registered(cls, strategy_id: str) -> bool:
        """
        Check if a strategy is registered.

        Args:
            strategy_id: Strategy identifier to check

        Returns:
            True if strategy is registered
        """
        return strategy_id in cls._strategies

    @classmethod
    def unregister(cls, strategy_id: str) -> None:
        """
        Unregister a strategy.

        This is primarily useful for testing.

        Args:
            strategy_id: Strategy identifier to unregister

        Raises:
            ValueError: If strategy_id is not registered
        """
        if strategy_id not in cls._strategies:
            raise ValueError(f"Cannot unregister unknown strategy: '{strategy_id}'")
        del cls._strategies[strategy_id]

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered strategies.

        This is primarily useful for testing.
        """
        cls._strategies.clear()
