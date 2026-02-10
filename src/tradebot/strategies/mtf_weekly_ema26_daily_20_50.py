from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Signal:
    target_position: int  # 0 or 1
    info: dict[str, str]


class MTFWeeklyEMA26Daily2050Strategy:
    """
    Weekly regime + daily trigger:
      weekly_regime_on = WeeklyClose > WeeklyEMA(26)
      daily_trigger_on = EMA20(daily close) > EMA50(daily close)
    Long iff both are true.
    """

    key = "mtf_weeklyema26_daily20_50"
    name = "MTF(WeeklyEMA26)+Daily20/50"

    def __init__(self, *, weekly_ema_weeks: int = 26, ema_fast: int = 20, ema_slow: int = 50):
        self.weekly_ema_weeks = int(weekly_ema_weeks)
        self.ema_fast = int(ema_fast)
        self.ema_slow = int(ema_slow)

        # Need enough daily bars for weekly EMA stabilization:
        # weekly_ema_weeks of daily data + warmup buffer.
        self.lookback_bars = max(self.weekly_ema_weeks * 7 + 90, self.ema_slow + 120)

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    def generate_signal(self, candles: pd.DataFrame) -> Signal:
        close = candles["close"].astype(float)

        # --- Daily trigger ---
        ema_fast = self._ema(close, self.ema_fast)
        ema_slow = self._ema(close, self.ema_slow)
        daily_trigger_on = float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1])

        # --- Weekly regime ---
        # Use last close of each week. W-SUN matches your backtest.
        weekly = close.resample("W-SUN").last().dropna()
        weekly_ema = self._ema(weekly, self.weekly_ema_weeks)

        # Forward-fill weekly regime onto daily index
        weekly_regime = (weekly > weekly_ema).astype(float)
        weekly_regime_daily = weekly_regime.reindex(close.index, method="ffill").fillna(0.0)

        weekly_regime_on = float(weekly_regime_daily.iloc[-1]) > 0.5

        enough_weekly_history = len(weekly) >= self.weekly_ema_weeks
        target = 1 if (enough_weekly_history and weekly_regime_on and daily_trigger_on) else 0

        return Signal(
            target_position=target,
            info={
                "last_close": str(float(close.iloc[-1])),
                "ema_fast": str(float(ema_fast.iloc[-1])),
                "ema_slow": str(float(ema_slow.iloc[-1])),
                "weekly_close": str(float(weekly.iloc[-1])) if len(weekly) else "nan",
                "weekly_ema": str(float(weekly_ema.iloc[-1])) if len(weekly_ema) else "nan",
                "weekly_regime_on": str(int(weekly_regime_on)),
                "daily_trigger_on": str(int(daily_trigger_on)),
                "enough_weekly_history": str(int(enough_weekly_history)),
            },
        )