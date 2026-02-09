from .base import Strategy
from .ema_filter import EMAFilterStrategy

STRATEGY_REGISTRY = {
    "ema_filter": EMAFilterStrategy,
}

__all__ = ["Strategy", "EMAFilterStrategy", "STRATEGY_REGISTRY"]