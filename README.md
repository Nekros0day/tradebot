# BTC Coinbase Trade Bot (Advanced Trade API)

A simple, conservative **long/flat** BTC bot for Coinbase Advanced Trade.
It uses **daily candles** and can run in **dry-run** mode by default.

It’s designed like a small repo so you can add strategies cleanly.

---

## What this bot does

- Fetches the latest **closed daily** candle for your product (default `BTC-EUR` or `BTC-USD`)
- Computes a **target position**:  
  - `1` = LONG (hold BTC)
  - `0` = FLAT (hold quote currency, e.g. EUR/USD)
- Rebalances your account toward that target using **market IOC** orders (or logs the payload in dry-run)
- Stores state in `STATE_PATH` so it does not act twice on the same candle

> Important: Because this is **daily**, you’ll often see:  
> **"No new closed candle since last run; skipping."**  
> That’s expected until the next daily candle closes (typically **00:00 UTC**).

---

## Strategies available

Set `STRATEGY` in your `.env`:

### 1) `ema_filter` (classic)
**Name:** `EMAFilter(140)`  
**Rule:** LONG if `Close > EMA(140)` else FLAT  
Best as a simple baseline.

### 2) `ema140_ema20_50`
**Name:** `EMA140+EMA20/50`  
**Rule:** LONG if:
- `Close > EMA(140)` **and**
- `EMA(20) > EMA(50)`

This reduces chop vs EMA140 alone.

### 3) `mtf_weeklyema26_daily20_50`
**Name:** `MTF(WeeklyEMA26)+Daily20/50`  
**Rule:** LONG if:
- Weekly regime is bullish: `WeeklyClose > WeeklyEMA(26)` **and**
- Daily trigger confirms: `EMA(20) > EMA(50)`

Weekly regime is computed by resampling daily candles (`W-SUN`, last close of the week) and forward-filling to daily.

**History behavior:** the bot fetches the lookback window directly from Coinbase on each run (it does **not** rely on local accumulation). So you do **not** need to wait a week after starting the bot, as long as Coinbase has enough historical daily candles for that product.

---

## Coinbase API key setup (Advanced Trade)

You need a **Coinbase Advanced Trade API key** using **ECDSA (ES256)**.

1) In Coinbase (Advanced Trade / Developer / API):
   - Create a new API key (Advanced Trade)
   - Choose permissions:
     - ✅ `can_view`
     - ✅ `can_trade`
     - ❌ `can_transfer` (recommended OFF for safety)
2) Copy:
   - **API Key Name** (the “kid” / key id)
   - **Private Key** (PEM)

### Permissions check

Run:

```bash
python -m tradebot.scripts.check_permissions --raw