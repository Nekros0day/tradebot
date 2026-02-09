from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import pandas as pd


@dataclass(frozen=True)
class Signal:
    target_position: int  # 0 = flat, 1 = long
    info: dict


class Strategy(Protocol):
    name: str
    lookback_bars: int

    def generate_signal(self, candles: pd.DataFrame) -> Signal:
        ...