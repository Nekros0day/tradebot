# btc_intraday_suite.py
# pip install yfinance pandas numpy

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance") from e


# =========================
# Config defaults
# =========================
DEFAULT_SYMBOL = "BTC-USD"
DEFAULT_INTERVAL = "1h"     # yfinance supports up to ~730d for 1h; 30m is limited (~60d)
DEFAULT_LOOKBACK_DAYS = 180*4 # ~6 months
DEFAULT_START_EQUITY = 10_000.0

DEFAULT_FEE_BPS = 10.0       # 0.10% per side
DEFAULT_SLIPPAGE_BPS = 10.0  # 0.10% per side

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers
# =========================
def bps_to_frac(bps: float) -> float:
    return bps / 10000.0

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    return mid, upper, lower

def annualization_factor_from_interval(interval: str) -> float:
    # For Sharpe/Vol annualization:
    # 1h -> 24*365 = 8760 periods/year
    # 30m -> 48*365 = 17520
    # 15m -> 96*365 = 35040
    # 1d -> 365
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return 24.0 / hours * 365.0
    if interval.endswith("m"):
        mins = int(interval[:-1])
        return (24.0 * 60.0) / mins * 365.0
    if interval.endswith("d"):
        days = int(interval[:-1])
        return 365.0 / days
    # fallback
    return 365.0


# =========================
# Data loading (with cache)
# =========================
def cache_path(symbol: str, interval: str, lookback_days: int) -> Path:
    safe = symbol.replace("-", "").replace("=", "")
    return DATA_DIR / f"{safe}_{interval}_{lookback_days}d.csv"

def download_or_load_intraday(symbol: str, interval: str, lookback_days: int, refresh: bool = False) -> pd.DataFrame:
    """
    yfinance limitations:
      - 30m / 15m often limited to ~60 days
      - 1h often supports up to ~730 days
    We'll enforce sanity: if interval != 1h and days > 60, warn and cap to 60 unless refresh is forced.
    """
    if interval != "1h" and lookback_days > 60:
        print(f"[WARN] yfinance often limits {interval} data to ~60 days. Capping lookback_days to 60.")
        lookback_days = 60

    p = cache_path(symbol, interval, lookback_days)
    if p.exists() and not refresh:
        df = pd.read_csv(p)
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
        df = df.set_index("Datetime").sort_index()
        return df

    df = yf.download(
        symbol,
        period=f"{lookback_days}d",
        interval=interval,
        auto_adjust=False,
        progress=True,
    )
    if df is None or df.empty:
        raise RuntimeError("No data downloaded. Try again or check symbol/interval.")

    # flatten columns if multiindex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename_axis("Datetime").reset_index()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)

    df = df[["Datetime", "open", "high", "low", "close", "volume"]].dropna().copy()
    df = df.set_index("Datetime").sort_index()

    # enforce numeric
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    df.reset_index().to_csv(p, index=False)
    return df


# =========================
# Backtester (market orders)
# =========================
@dataclass
class BacktestResult:
    equity: pd.Series
    position_close: pd.Series
    trades: int
    win_rate: float
    avg_trade: float

def backtest_long_flat_market(
    df: pd.DataFrame,
    target_pos_close: pd.Series,
    *,
    start_equity: float,
    fee_bps: float,
    slippage_bps: float,
) -> BacktestResult:
    """
    Long/flat (spot-style).
    Signal computed at bar CLOSE.
    Execution at NEXT bar OPEN (avoids lookahead).
    Applies fee+slippage on entries/exits.
    """
    fee = bps_to_frac(fee_bps)
    slip = bps_to_frac(slippage_bps)

    target_pos_close = target_pos_close.reindex(df.index).fillna(0).clip(0, 1).astype(int)
    exec_pos_open = target_pos_close.shift(1).fillna(0).astype(int)

    cash = float(start_equity)
    btc = 0.0
    in_pos = 0

    entry_equity = None
    trade_returns = []

    equity = []
    pos_close = []

    for i in range(len(df)):
        o = float(df["open"].iloc[i])
        c = float(df["close"].iloc[i])

        desired = int(exec_pos_open.iloc[i])

        if desired == 1 and in_pos == 0:
            buy_px = o * (1 + slip)
            qty = (cash * (1 - fee)) / buy_px
            btc = qty
            cash = 0.0
            in_pos = 1
            entry_equity = btc * buy_px

        elif desired == 0 and in_pos == 1:
            sell_px = o * (1 - slip)
            cash = btc * sell_px * (1 - fee)
            btc = 0.0
            in_pos = 0
            if entry_equity and entry_equity > 0:
                trade_returns.append(cash / entry_equity - 1.0)
            entry_equity = None

        equity.append(cash + btc * c)
        pos_close.append(in_pos)

    equity_s = pd.Series(equity, index=df.index, name="equity")
    pos_s = pd.Series(pos_close, index=df.index, name="pos")

    tr = np.array(trade_returns, dtype=float)
    trades = int(len(tr))
    win_rate = float(np.mean(tr > 0)) if trades else 0.0
    avg_trade = float(np.mean(tr)) if trades else 0.0

    return BacktestResult(equity=equity_s, position_close=pos_s, trades=trades, win_rate=win_rate, avg_trade=avg_trade)


# =========================
# Metrics
# =========================
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def perf_stats(equity: pd.Series, pos: pd.Series, trades: int, win_rate: float, avg_trade: float, interval: str) -> Dict[str, float]:
    rets = equity.pct_change().fillna(0.0)

    ann_periods = annualization_factor_from_interval(interval)
    # approximate elapsed years from number of bars
    years = len(equity) / ann_periods if ann_periods > 0 else np.nan

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    vol = float(rets.std(ddof=0) * math.sqrt(ann_periods)) if ann_periods > 0 else np.nan
    sharpe = float((rets.mean() / rets.std(ddof=0)) * math.sqrt(ann_periods)) if rets.std(ddof=0) > 0 else np.nan

    return {
        "FinalEquity": float(equity.iloc[-1]),
        "TotalReturn": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Vol(ann.)": vol,
        "MaxDD": max_drawdown(equity),
        "Exposure": float(pos.mean()),
        "Trades": float(trades),
        "WinRate": float(win_rate),
        "AvgTrade": float(avg_trade),
    }


# =========================
# Strategies (positions at CLOSE)
# =========================
class Strategy:
    name: str
    def generate_position(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

@dataclass
class BuyHold(Strategy):
    name: str = "BuyHold"
    def generate_position(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index)

@dataclass
class MACrossover(Strategy):
    fast: int = 20
    slow: int = 50
    use_ema: bool = True
    name: str = "MA_Crossover"

    def generate_position(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        fast_ma = ema(close, self.fast) if self.use_ema else sma(close, self.fast)
        slow_ma = ema(close, self.slow) if self.use_ema else sma(close, self.slow)

        pos = (fast_ma > slow_ma).astype(int)
        pos[(fast_ma.isna()) | (slow_ma.isna())] = 0
        return pos

@dataclass
class RSIBollingerMR(Strategy):
    rsi_n: int = 14
    bb_n: int = 20
    bb_k: float = 2.0
    entry_rsi: float = 30.0
    exit_rsi: float = 50.0
    # long/flat only here (spot). You can extend to short later.
    name: str = "RSI+BB_MR"

    def generate_position(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        r = rsi(close, self.rsi_n)
        mid, upper, lower = bollinger(close, self.bb_n, self.bb_k)

        pos = np.zeros(len(df), dtype=int)
        in_pos = 0

        for i in range(len(df)):
            if np.isnan(r.iloc[i]) or np.isnan(mid.iloc[i]) or np.isnan(lower.iloc[i]):
                pos[i] = 0
                continue

            if in_pos == 0:
                # Enter on "extreme + oversold"
                if close.iloc[i] < lower.iloc[i] and r.iloc[i] < self.entry_rsi:
                    in_pos = 1
            else:
                # Exit when reverts toward mean or RSI recovers
                if close.iloc[i] >= mid.iloc[i] or r.iloc[i] > self.exit_rsi:
                    in_pos = 0

            pos[i] = in_pos

        return pd.Series(pos, index=df.index)

@dataclass
class BreakoutVolume(Strategy):
    lookback: int = 48            # ~2 days on 1h
    vol_mult: float = 1.5
    exit_on_fail: bool = True     # exit if it falls back into range
    name: str = "Breakout+Volume"

    def generate_position(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        vol = df["volume"].astype(float)

        recent_high = high.rolling(self.lookback, min_periods=self.lookback).max().shift(1)
        recent_low = low.rolling(self.lookback, min_periods=self.lookback).min().shift(1)
        avg_vol = vol.rolling(self.lookback, min_periods=self.lookback).mean().shift(1)

        pos = np.zeros(len(df), dtype=int)
        in_pos = 0
        breakout_level = np.nan

        for i in range(len(df)):
            if np.isnan(recent_high.iloc[i]) or np.isnan(recent_low.iloc[i]) or np.isnan(avg_vol.iloc[i]):
                pos[i] = 0
                continue

            if in_pos == 0:
                up_break = (close.iloc[i] > recent_high.iloc[i]) and (vol.iloc[i] > self.vol_mult * avg_vol.iloc[i])
                if up_break:
                    in_pos = 1
                    breakout_level = float(recent_high.iloc[i])
            else:
                if self.exit_on_fail and close.iloc[i] < breakout_level:
                    in_pos = 0
                    breakout_level = np.nan

            pos[i] = in_pos

        return pd.Series(pos, index=df.index)

@dataclass
class RangeRSI(Strategy):
    lookback: int = 48          # support/resistance window
    band_pct: float = 0.003     # "near support/resistance" threshold (0.3%)
    rsi_n: int = 14
    entry_rsi: float = 30.0
    exit_rsi: float = 55.0
    name: str = "Range+RSI"

    def generate_position(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        r = rsi(close, self.rsi_n)

        # rolling bounds based on prior bars (shifted to avoid lookahead)
        res = high.rolling(self.lookback, min_periods=self.lookback).max().shift(1)
        sup = low.rolling(self.lookback, min_periods=self.lookback).min().shift(1)

        pos = np.zeros(len(df), dtype=int)
        in_pos = 0

        for i in range(len(df)):
            if np.isnan(res.iloc[i]) or np.isnan(sup.iloc[i]) or np.isnan(r.iloc[i]):
                pos[i] = 0
                continue

            support = float(sup.iloc[i])
            resist = float(res.iloc[i])

            near_support = close.iloc[i] <= support * (1.0 + self.band_pct)
            near_resist = close.iloc[i] >= resist * (1.0 - self.band_pct)

            if in_pos == 0:
                # enter long near support when oversold
                if near_support and (r.iloc[i] < self.entry_rsi):
                    in_pos = 1
            else:
                # exit long near resistance or when RSI recovers
                if near_resist or (r.iloc[i] > self.exit_rsi):
                    in_pos = 0

            pos[i] = in_pos

        return pd.Series(pos, index=df.index)


# =========================
# Suite runner
# =========================
def run_suite(
    df: pd.DataFrame,
    *,
    interval: str,
    start_equity: float,
    fee_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    strategies = [
        BuyHold(),
        MACrossover(fast=20, slow=50, use_ema=True, name="EMA_Cross(20/50)"),
        RSIBollingerMR(rsi_n=14, bb_n=20, bb_k=2.0, entry_rsi=30.0, exit_rsi=50.0, name="RSI+BB_MR"),
        BreakoutVolume(lookback=48, vol_mult=1.5, exit_on_fail=True, name="Breakout(48)+Vol1.5x"),
        RangeRSI(lookback=48, band_pct=0.003, rsi_n=14, entry_rsi=30.0, exit_rsi=55.0, name="Range(48)+RSI"),
    ]

    rows = []
    for strat in strategies:
        pos = strat.generate_position(df)

        # BuyHold baseline: model without fees/slippage for pure benchmark
        if strat.name == "BuyHold":
            res = backtest_long_flat_market(df, pos, start_equity=start_equity, fee_bps=0.0, slippage_bps=0.0)
        else:
            res = backtest_long_flat_market(df, pos, start_equity=start_equity, fee_bps=fee_bps, slippage_bps=slippage_bps)

        stats = perf_stats(res.equity, res.position_close, res.trades, res.win_rate, res.avg_trade, interval)
        stats["Strategy"] = strat.name
        rows.append(stats)

    out = pd.DataFrame(rows).set_index("Strategy")

    # Compare vs BuyHold
    if "BuyHold" in out.index:
        bh = out.loc["BuyHold"]
        out["TotalReturn_vs_BH"] = out["TotalReturn"] - float(bh["TotalReturn"])
        out["CAGR_vs_BH"] = out["CAGR"] - float(bh["CAGR"])
        out["Sharpe_vs_BH"] = out["Sharpe"] - float(bh["Sharpe"])
        out["MaxDD_vs_BH"] = out["MaxDD"] - float(bh["MaxDD"])  # negative => better drawdown

    # nice column order
    cols = [
        "FinalEquity","TotalReturn","CAGR","Sharpe","Vol(ann.)","MaxDD","Exposure","Trades","WinRate","AvgTrade",
        "TotalReturn_vs_BH","CAGR_vs_BH","Sharpe_vs_BH","MaxDD_vs_BH",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values("CAGR", ascending=False)


# =========================
# CLI
# =========================
def main():
    p = argparse.ArgumentParser(description="BTC intraday multi-trade/day strategy backtest suite (cached).")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--interval", default=DEFAULT_INTERVAL, help="yfinance interval, e.g. 1h, 30m (30m usually limited to ~60d).")
    p.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Lookback days (~180=6 months).")
    p.add_argument("--equity", type=float, default=DEFAULT_START_EQUITY, help="Starting equity (quote currency).")
    p.add_argument("--fee_bps", type=float, default=DEFAULT_FEE_BPS, help="Fee per side, in bps (10=0.10%).")
    p.add_argument("--slip_bps", type=float, default=DEFAULT_SLIPPAGE_BPS, help="Slippage per side, in bps.")
    p.add_argument("--refresh", action="store_true", help="Force refresh data (ignore cache).")
    args = p.parse_args()

    df = download_or_load_intraday(args.symbol, args.interval, args.days, refresh=args.refresh)
    if len(df) < 200:
        print("[WARN] Very few rows. Consider increasing --days or use --interval 1h.")

    print("\n" + "=" * 110)
    print(f"{args.symbol} | interval={args.interval} | rows={len(df)} | from={df.index.min()} to={df.index.max()}")
    print(f"Cache: {cache_path(args.symbol, args.interval, min(args.days, 60) if args.interval != '1h' and args.days > 60 else args.days).resolve()}")
    print(f"Costs: fee={args.fee_bps} bps/side, slippage={args.slip_bps} bps/side (applied to non-BH strategies)")
    print("=" * 110)

    summary = run_suite(
        df,
        interval=args.interval,
        start_equity=args.equity,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps,
    )

    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(summary.round({
            "FinalEquity": 2,
            "TotalReturn": 4,
            "CAGR": 4,
            "Sharpe": 3,
            "Vol(ann.)": 4,
            "MaxDD": 4,
            "Exposure": 3,
            "Trades": 0,
            "WinRate": 3,
            "AvgTrade": 4,
            "TotalReturn_vs_BH": 4,
            "CAGR_vs_BH": 4,
            "Sharpe_vs_BH": 3,
            "MaxDD_vs_BH": 4,
        }))


if __name__ == "__main__":
    main()