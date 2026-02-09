# btc_ema140_momentum_suite.py
# pip install yfinance pandas numpy

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Literal

import numpy as np
import pandas as pd
import yfinance as yf

ExecutionMode = Literal["market", "limit"]


# =============================
# Indicators
# =============================
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


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    sig_line = ema(macd_line, signal)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = true_range(df["High"], df["Low"], df["Close"])
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Wilder-style ADX(14). Trend-strength (no direction).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)

    atr_w = pd.Series(tr).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_dm_w = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    minus_dm_w = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    plus_di = 100 * (plus_dm_w / atr_w.replace(0, np.nan))
    minus_di = 100 * (minus_dm_w / atr_w.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_s = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return adx_s


def realized_vol(close: pd.Series, n: int = 20, annual_days: int = 365) -> pd.Series:
    r = close.pct_change()
    return r.rolling(n, min_periods=n).std(ddof=0) * math.sqrt(float(annual_days))


# =============================
# Data
# =============================
def download_crypto_daily(ticker: str, years: int = 10, auto_adjust: bool = True) -> pd.DataFrame:
    end = datetime.utcnow().date()
    start = end - timedelta(days=int(years * 365.25) + 30)

    df = yf.download(
        ticker,
        start=str(start),
        end=str(end + timedelta(days=1)),
        interval="1d",
        auto_adjust=auto_adjust,
        progress=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        else:
            df.columns = df.columns.get_level_values(0)

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def slice_last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    end = df.index.max()
    start = end - pd.DateOffset(years=years)
    return df[df.index >= start].copy()


# =============================
# Order sim (market or 1-day limit, cancel/replace style)
# =============================
@dataclass
class Order:
    side: Literal["BUY", "SELL"]
    qty: float
    limit_price: float | None
    submitted_i: int
    status: Literal["OPEN", "FILLED", "CANCELED"] = "OPEN"
    fill_price: float | None = None


@dataclass
class BacktestResult:
    equity: pd.Series
    weight: pd.Series
    trades: int
    win_rate: float
    avg_trade: float
    trade_returns: np.ndarray
    orders_submitted: int
    orders_filled: int
    orders_canceled: int


def _maybe_fill_limit(order: Order, o: float, h: float, l: float) -> tuple[bool, float | None]:
    lp = order.limit_price
    if lp is None:
        return False, None

    if order.side == "BUY":
        if o <= lp:
            return True, o
        if l <= lp <= h:
            return True, lp
        return False, None

    # SELL
    if o >= lp:
        return True, o
    if l <= lp <= h:
        return True, lp
    return False, None


def backtest_target_weight(
    df: pd.DataFrame,
    target_weight_close: pd.Series,
    *,
    fee_bps: float = 20.0,
    slippage_bps: float = 10.0,
    execution: ExecutionMode = "market",
    limit_offset_bps: float = 10.0,
    initial_cash: float = 10_000.0,
) -> BacktestResult:
    """
    Signal at CLOSE, rebalance attempt at NEXT OPEN.
    - market: fill at open +/- slippage
    - limit: place limit at open*(1-off) for buys and open*(1+off) for sells, cancel EOD if unfilled
    """
    df = df.copy()
    w = target_weight_close.reindex(df.index).fillna(0.0).clip(0.0, 1.0)
    desired_w_open = w.shift(1).fillna(0.0)

    fee = fee_bps / 10000.0
    slip = slippage_bps / 10000.0
    off = limit_offset_bps / 10000.0

    cash = float(initial_cash)
    units = 0.0

    equity = []
    weight_series = []

    in_pos = False
    entry_equity = np.nan
    trade_returns: list[float] = []

    open_order: Order | None = None
    orders_submitted = 0
    orders_filled = 0
    orders_canceled = 0

    for i in range(len(df)):
        o = float(df["Open"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])

        # 1) handle open limit order (1-day TIF)
        if execution == "limit" and open_order is not None and open_order.status == "OPEN":
            filled, fill_px = _maybe_fill_limit(open_order, o, h, l)
            if filled:
                px = float(fill_px)
                open_order.status = "FILLED"
                open_order.fill_price = px
                orders_filled += 1

                if open_order.side == "BUY":
                    spend = min(cash, open_order.qty * px)
                    got_units = (spend * (1 - fee)) / px
                    units += got_units
                    cash -= spend
                    if not in_pos and units * px > 1e-9:
                        in_pos = True
                        entry_equity = cash + units * px
                else:
                    sell_units = min(units, open_order.qty)
                    recv = sell_units * px * (1 - fee)
                    units -= sell_units
                    cash += recv
                    if in_pos and (units * px) <= 1e-9:
                        if entry_equity > 0:
                            trade_returns.append((cash / entry_equity) - 1.0)
                        in_pos = False
                        entry_equity = np.nan

                open_order = None
            else:
                open_order.status = "CANCELED"
                orders_canceled += 1
                open_order = None

        # 2) rebalance at OPEN
        desired_w = float(desired_w_open.iloc[i])
        eq_open = cash + units * o
        if eq_open <= 0:
            equity.append(0.0)
            weight_series.append(0.0)
            continue

        cur_w = (units * o) / eq_open
        desired_asset_value = desired_w * eq_open
        current_asset_value = units * o
        diff_value = desired_asset_value - current_asset_value

        if abs(diff_value) > 1e-6:
            if diff_value > 0:
                buy_value = min(diff_value, cash)
                if buy_value > 0:
                    if execution == "market":
                        px = o * (1 + slip)
                        got_units = (buy_value * (1 - fee)) / px
                        units += got_units
                        cash -= buy_value
                        if not in_pos and got_units * px > 1e-9:
                            in_pos = True
                            entry_equity = cash + units * px
                    else:
                        limit_px = o * (1 - off)
                        qty = buy_value / limit_px if limit_px > 0 else 0.0
                        open_order = Order("BUY", float(qty), float(limit_px), i)
                        orders_submitted += 1
            else:
                sell_value = -diff_value
                qty = min(units, sell_value / o) if o > 0 else 0.0
                if qty > 0:
                    if execution == "market":
                        px = o * (1 - slip)
                        recv = qty * px * (1 - fee)
                        units -= qty
                        cash += recv
                        if in_pos and (units * px) <= 1e-9:
                            if entry_equity > 0:
                                trade_returns.append((cash / entry_equity) - 1.0)
                            in_pos = False
                            entry_equity = np.nan
                    else:
                        limit_px = o * (1 + off)
                        open_order = Order("SELL", float(qty), float(limit_px), i)
                        orders_submitted += 1

        # 3) mark-to-market at CLOSE
        eq = cash + units * c
        equity.append(eq)
        w_close = (units * c) / eq if eq > 0 else 0.0
        weight_series.append(w_close)

    equity_s = pd.Series(equity, index=df.index, name="equity")
    w_s = pd.Series(weight_series, index=df.index, name="weight")

    tr = np.array(trade_returns, dtype=float)
    trades = int(len(tr))
    win_rate = float(np.mean(tr > 0)) if trades > 0 else 0.0
    avg_trade = float(np.mean(tr)) if trades > 0 else 0.0

    return BacktestResult(
        equity=equity_s,
        weight=w_s,
        trades=trades,
        win_rate=win_rate,
        avg_trade=avg_trade,
        trade_returns=tr,
        orders_submitted=orders_submitted,
        orders_filled=orders_filled,
        orders_canceled=orders_canceled,
    )


# =============================
# Strategies (target weight at CLOSE)
# =============================
def strat_buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def strat_ema_regime(df: pd.DataFrame, ema_len: int = 140) -> pd.Series:
    close = df["Close"]
    e = ema(close, ema_len)
    return (close > e).astype(float).fillna(0.0)


def strat_ema_regime_fast(df: pd.DataFrame, ema_len: int = 140, fast: int = 20, slow: int = 50) -> pd.Series:
    close = df["Close"]
    regime = close > ema(close, ema_len)
    trig = ema(close, fast) > ema(close, slow)
    return (regime & trig).astype(float).fillna(0.0)


def strat_ema_fast_rsi(df: pd.DataFrame, ema_len: int = 140, fast: int = 20, slow: int = 50, rsi_n: int = 14, rsi_th: float = 50.0) -> pd.Series:
    close = df["Close"]
    base = strat_ema_regime_fast(df, ema_len, fast, slow).astype(bool)
    r = rsi(close, rsi_n)
    return (base & (r > rsi_th)).astype(float).fillna(0.0)


def strat_ema_fast_macd_hist(df: pd.DataFrame, ema_len: int = 140, fast: int = 20, slow: int = 50,
                             macd_f: int = 12, macd_s: int = 26, macd_sig: int = 9) -> pd.Series:
    close = df["Close"]
    base = strat_ema_regime_fast(df, ema_len, fast, slow).astype(bool)
    _, _, hist = macd(close, macd_f, macd_s, macd_sig)
    return (base & (hist > 0)).astype(float).fillna(0.0)


def strat_ema_fast_rsi_macd(df: pd.DataFrame, ema_len: int = 140, fast: int = 20, slow: int = 50,
                            rsi_n: int = 14, rsi_th: float = 50.0,
                            macd_f: int = 12, macd_s: int = 26, macd_sig: int = 9) -> pd.Series:
    close = df["Close"]
    base = strat_ema_regime_fast(df, ema_len, fast, slow).astype(bool)
    r = rsi(close, rsi_n)
    _, _, hist = macd(close, macd_f, macd_s, macd_sig)
    return (base & (r > rsi_th) & (hist > 0)).astype(float).fillna(0.0)


def strat_ema_fast_adx(df: pd.DataFrame, ema_len: int = 140, fast: int = 20, slow: int = 50,
                       adx_n: int = 14, adx_th: float = 20.0) -> pd.Series:
    base = strat_ema_regime_fast(df, ema_len, fast, slow).astype(bool)
    a = adx(df, adx_n)
    return (base & (a > adx_th)).astype(float).fillna(0.0)


def strat_ema_fast_12m_mom(df: pd.DataFrame, ema_len: int = 140, fast: int = 20, slow: int = 50, lookback: int = 252) -> pd.Series:
    close = df["Close"]
    base = strat_ema_regime_fast(df, ema_len, fast, slow).astype(bool)
    mom = (close / close.shift(lookback) - 1.0)
    return (base & (mom > 0)).astype(float).fillna(0.0)


def strat_mtf_weekly_regime_daily_trigger(df: pd.DataFrame, weekly_ema_weeks: int = 26, fast: int = 20, slow: int = 50) -> pd.Series:
    # Weekly regime
    w = df.resample("W-SUN").last()
    w_close = w["Close"]
    w_reg = (w_close > ema(w_close, weekly_ema_weeks)).astype(float)
    w_reg_daily = w_reg.reindex(df.index, method="ffill").fillna(0.0).astype(bool)

    # Daily trigger
    close = df["Close"]
    trig = (ema(close, fast) > ema(close, slow)).astype(bool)

    return (w_reg_daily & trig).astype(float).fillna(0.0)


def strat_mtf_plus_rsi(df: pd.DataFrame, weekly_ema_weeks: int = 26, fast: int = 20, slow: int = 50, rsi_n: int = 14, rsi_th: float = 50.0) -> pd.Series:
    base = strat_mtf_weekly_regime_daily_trigger(df, weekly_ema_weeks, fast, slow).astype(bool)
    r = rsi(df["Close"], rsi_n)
    return (base & (r > rsi_th)).astype(float).fillna(0.0)


def strat_mtf_plus_macd(df: pd.DataFrame, weekly_ema_weeks: int = 26, fast: int = 20, slow: int = 50,
                        macd_f: int = 12, macd_s: int = 26, macd_sig: int = 9) -> pd.Series:
    base = strat_mtf_weekly_regime_daily_trigger(df, weekly_ema_weeks, fast, slow).astype(bool)
    _, _, hist = macd(df["Close"], macd_f, macd_s, macd_sig)
    return (base & (hist > 0)).astype(float).fillna(0.0)


# =============================
# Metrics
# =============================
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def perf_stats(equity: pd.Series, weight: pd.Series, trades: int, win_rate: float, avg_trade: float, annual_days: int = 365) -> dict:
    rets = equity.pct_change().fillna(0.0)
    days = max(1, (equity.index[-1] - equity.index[0]).days)

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (365.0 / days) - 1.0) if equity.iloc[0] > 0 else np.nan
    vol = float(rets.std(ddof=0) * math.sqrt(float(annual_days)))
    sharpe = float((rets.mean() / rets.std(ddof=0)) * math.sqrt(float(annual_days))) if rets.std(ddof=0) > 0 else np.nan


    mdd = max_drawdown(equity)
    exposure = float(weight.mean())

    return {
        "FinalEquity": float(equity.iloc[-1]),
        "TotalReturn": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Vol(ann.)": vol,
        "MaxDD": mdd,
        "Exposure": exposure,
        "Trades": trades,
        "WinRate": win_rate,
        "AvgTrade": avg_trade,
    }


# =============================
# Suite runner
# =============================
def run_suite(
    df: pd.DataFrame,
    *,
    fee_bps: float,
    slippage_bps: float,
    execution: ExecutionMode,
    limit_offset_bps: float,
    annual_days: int = 365,
) -> pd.DataFrame:
    strategies: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        # Baselines (keep)
        ("BuyHold", lambda d: strat_buy_hold(d)),
        ("EMA140", lambda d: strat_ema_regime(d, 140)),
        ("EMA140+EMA20/50", lambda d: strat_ema_regime_fast(d, 140, 20, 50)),

        # New confirmations off the winner
        ("EMA140+20/50+RSI>50", lambda d: strat_ema_fast_rsi(d, 140, 20, 50, 14, 50.0)),
        ("EMA140+20/50+RSI>55", lambda d: strat_ema_fast_rsi(d, 140, 20, 50, 14, 55.0)),
        ("EMA140+20/50+MACDhist>0", lambda d: strat_ema_fast_macd_hist(d, 140, 20, 50, 12, 26, 9)),
        ("EMA140+20/50+RSI>50+MACD>0", lambda d: strat_ema_fast_rsi_macd(d, 140, 20, 50, 14, 50.0, 12, 26, 9)),
        ("EMA140+20/50+ADX>20", lambda d: strat_ema_fast_adx(d, 140, 20, 50, 14, 20.0)),
        ("EMA140+20/50+12mMom>0", lambda d: strat_ema_fast_12m_mom(d, 140, 20, 50, 252)),

        # Keep your MTF keeper + confirmations
        ("MTF(WeeklyEMA26)+Daily20/50", lambda d: strat_mtf_weekly_regime_daily_trigger(d, 26, 20, 50)),
        ("MTF+RSI>50", lambda d: strat_mtf_plus_rsi(d, 26, 20, 50, 14, 50.0)),
        ("MTF+MACDhist>0", lambda d: strat_mtf_plus_macd(d, 26, 20, 50, 12, 26, 9)),
    ]

    rows = []
    for name, build_w in strategies:
        w_close = build_w(df).reindex(df.index).fillna(0.0).clip(0.0, 1.0)

        res = backtest_target_weight(
            df,
            w_close,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            execution=execution,
            limit_offset_bps=limit_offset_bps,
            initial_cash=10_000.0,
        )

        stats = perf_stats(
            res.equity,
            res.weight,
            trades=res.trades,
            win_rate=res.win_rate,
            avg_trade=res.avg_trade,
            annual_days=annual_days,
        )
        stats["Strategy"] = name
        stats["Orders_Submitted"] = res.orders_submitted
        stats["Orders_Filled"] = res.orders_filled
        stats["Orders_Canceled"] = res.orders_canceled
        stats["FillRate"] = (res.orders_filled / res.orders_submitted) if res.orders_submitted else np.nan
        rows.append(stats)

    out = pd.DataFrame(rows).set_index("Strategy")

    if "BuyHold" in out.index:
        bh = out.loc["BuyHold"]
        out["TotalReturn_vs_BH"] = out["TotalReturn"] - float(bh["TotalReturn"])
        out["CAGR_vs_BH"] = out["CAGR"] - float(bh["CAGR"])
        out["Sharpe_vs_BH"] = out["Sharpe"] - float(bh["Sharpe"])
        out["MaxDD_vs_BH"] = out["MaxDD"] - float(bh["MaxDD"])

    cols = [
        "FinalEquity","TotalReturn","CAGR","Sharpe","Vol(ann.)","MaxDD","Exposure","Trades","WinRate","AvgTrade",
        "Orders_Submitted","Orders_Filled","Orders_Canceled","FillRate",
        "TotalReturn_vs_BH","CAGR_vs_BH","Sharpe_vs_BH","MaxDD_vs_BH",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values("CAGR", ascending=False)


# =============================
# Main
# =============================
if __name__ == "__main__":
    TICKER = "BTC-USD"          # set "BTC-EUR" to match your live bot accounting
    horizons = [2, 3, 5, 6, 10]

    fee_bps = 20.0
    slippage_bps = 10.0
    annual_days = 365

    execution: ExecutionMode = "market"     # "market" or "limit"
    limit_offset_bps = 10.0                 # only used if execution="limit"

    data = download_crypto_daily(TICKER, years=max(horizons), auto_adjust=True)

    for y in horizons:
        dfx = slice_last_years(data, y)
        if len(dfx) < 200:
            print(f"\nSkipping horizon {y}y: not enough rows ({len(dfx)})")
            continue

        summary = run_suite(
            dfx,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            execution=execution,
            limit_offset_bps=limit_offset_bps,
            annual_days=annual_days,
        )

        print("\n" + "=" * 130)
        print(f"{TICKER} Daily Backtest — last {y} years — rows={len(dfx)} — fee={fee_bps}bps, slip={slippage_bps}bps — exec={execution}")
        if execution == "limit":
            print(f"Limit offset: {limit_offset_bps} bps (buy below open / sell above open)")
        print("=" * 130)

        with pd.option_context("display.max_rows", 300, "display.width", 220):
            print(summary.round({
                "FinalEquity": 2,
                "TotalReturn": 4,
                "CAGR": 4,
                "Sharpe": 3,
                "Vol(ann.)": 4,
                "MaxDD": 4,
                "Exposure": 3,
                "WinRate": 3,
                "AvgTrade": 4,
                "FillRate": 3,
                "TotalReturn_vs_BH": 4,
                "CAGR_vs_BH": 4,
                "Sharpe_vs_BH": 3,
                "MaxDD_vs_BH": 4,
            }))