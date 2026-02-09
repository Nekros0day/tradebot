from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd

from ..exchange.coinbase_client import CoinbaseClient
from ..utils.time_utils import now_ts, ts_to_dt_utc

log = logging.getLogger("tradebot.candles")


@dataclass(frozen=True)
class CandleFetchResult:
    df: pd.DataFrame
    latest_closed_start: int


def _candles_any_to_df(candles: list[Any]) -> pd.DataFrame:
    """
    Supports two common Coinbase candle formats:
      A) list of dicts: {start, low, high, open, close, volume}
      B) list of lists/tuples: [start, low, high, open, close, volume]
    Returns DataFrame indexed by UTC datetime with columns:
      start, open, high, low, close, volume
    """
    if not candles:
        return pd.DataFrame(columns=["start", "open", "high", "low", "close", "volume"])

    first = candles[0]

    if isinstance(first, dict):
        df = pd.DataFrame(candles).copy()
        # normalize keys just in case
        # expected: start, low, high, open, close, volume
        needed = ["start", "open", "high", "low", "close", "volume"]
        # some responses might use capitalized keys
        df.columns = [str(c).lower() for c in df.columns]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise RuntimeError(f"Unexpected candle dict keys. Missing {missing}. Got {list(df.columns)}")
        df = df[needed]

    else:
        # assume array shape: [start, low, high, open, close, volume]
        df = pd.DataFrame(candles, columns=["start", "low", "high", "open", "close", "volume"]).copy()
        df = df[["start", "open", "high", "low", "close", "volume"]]

    # ensure numeric
    for c in ["start", "open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    df = df.sort_values("start").drop_duplicates("start", keep="last")
    df["dt"] = df["start"].astype(int).apply(ts_to_dt_utc)
    df = df.set_index("dt")

    return df[["start", "open", "high", "low", "close", "volume"]]


def get_latest_closed_daily_candles(
    client: CoinbaseClient,
    product_id: str,
    *,
    lookback_days: int,
    granularity: str = "ONE_DAY",
) -> CandleFetchResult:
    end = now_ts()
    start = end - int(lookback_days * 86400)

    raw = client.get_product_candles(
        product_id=product_id,
        start=start,
        end=end,
        granularity=granularity,
        limit=min(350, lookback_days + 5),
    )

    candles = raw.get("candles")
    if candles is None:
        raise RuntimeError(f"Unexpected candles response shape: {raw}")

    df = _candles_any_to_df(candles)

    if df.empty:
        raise RuntimeError("No candles parsed from Coinbase response.")

    # Drop the most recent candle if it likely isn't closed yet (daily)
    last_start = int(df["start"].iloc[-1])
    last_dt = ts_to_dt_utc(last_start)
    if datetime.now(timezone.utc) - last_dt < timedelta(hours=23):
        df = df.iloc[:-1]
        if df.empty:
            raise RuntimeError("Only an in-progress candle returned; try again later.")
        last_start = int(df["start"].iloc[-1])

    return CandleFetchResult(df=df, latest_closed_start=last_start)