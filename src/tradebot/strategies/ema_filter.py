from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from .base import Signal


def ema_last(values: list[float], span: int) -> float:
    """
    Compute EMA last value using standard alpha = 2/(span+1).
    Requires at least 'span' values for a stable result.
    """
    if len(values) < span:
        raise ValueError(f"Need at least {span} values for EMA, got {len(values)}")

    alpha = 2.0 / (span + 1.0)

    # seed with SMA(span)
    seed = sum(values[:span]) / span
    e = seed
    for x in values[span:]:
        e = alpha * x + (1.0 - alpha) * e
    return e


@dataclass
class EMAFilterStrategy:
    ema_length: int = 140
    name: str = "EMAFilter(140)"

    @property
    def lookback_bars(self) -> int:
        return self.ema_length + 30

    def generate_signal(self, candles: pd.DataFrame) -> Signal:
        close = candles["close"].astype(float)
        ema = close.ewm(span=self.ema_length, adjust=False, min_periods=self.ema_length).mean()

        last_close = float(close.iloc[-1])
        last_ema = float(ema.iloc[-1])
        target = 1 if last_close > last_ema else 0

        return Signal(
            target_position=target,
            info={"ema_length": self.ema_length, "last_close": last_close, "last_ema": last_ema},
        )