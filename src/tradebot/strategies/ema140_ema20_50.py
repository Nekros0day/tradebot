from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Signal:
    target_position: int  # 0 or 1
    info: dict[str, str]


class EMA140EMA2050Strategy:
    """
    Long if (Close > EMA140) AND (EMA20 > EMA50).
    Signals computed on latest closed daily candle.
    """

    key = "ema140_ema20_50"
    name = "EMA140+EMA20/50"

    def __init__(self, *, ema_long: int = 140, ema_fast: int = 20, ema_slow: int = 50):
        self.ema_long = int(ema_long)
        self.ema_fast = int(ema_fast)
        self.ema_slow = int(ema_slow)

        # enough history to stabilize EMAs
        self.lookback_bars = max(self.ema_long, self.ema_slow) + 60

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    def generate_signal(self, candles: pd.DataFrame) -> Signal:
        # candles: DataFrame indexed by dt, with columns: open/high/low/close/volume (+start)
        close = candles["close"].astype(float)

        ema_long = self._ema(close, self.ema_long)
        ema_fast = self._ema(close, self.ema_fast)
        ema_slow = self._ema(close, self.ema_slow)

        last_close = float(close.iloc[-1])
        last_ema_long = float(ema_long.iloc[-1])
        last_ema_fast = float(ema_fast.iloc[-1])
        last_ema_slow = float(ema_slow.iloc[-1])

        regime_on = last_close > last_ema_long
        trigger_on = last_ema_fast > last_ema_slow

        target = 1 if (regime_on and trigger_on) else 0

        return Signal(
            target_position=target,
            info={
                "last_close": str(last_close),
                "ema_long": str(last_ema_long),
                "ema_fast": str(last_ema_fast),
                "ema_slow": str(last_ema_slow),
                "regime_on": str(int(regime_on)),
                "trigger_on": str(int(trigger_on)),
            },
        )