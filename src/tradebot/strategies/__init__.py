from __future__ import annotations

from .ema_filter import EMAFilterStrategy  # your existing one

from .ema140_ema20_50 import EMA140EMA2050Strategy
from .mtf_weekly_ema26_daily_20_50 import MTFWeeklyEMA26Daily2050Strategy

STRATEGY_REGISTRY = {
    "ema_filter": EMAFilterStrategy,
    "ema140_ema20_50": EMA140EMA2050Strategy,
    "mtf_weeklyema26_daily20_50": MTFWeeklyEMA26Daily2050Strategy,
}