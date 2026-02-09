# BTC EMA(140) Coinbase Bot (Advanced Trade API)

This bot trades BTC-USD long/flat based on a daily EMA filter:
- If Close > EMA(140): target position = LONG
- Else: target position = FLAT

## Quick start
1) Create a Python venv
2) `pip install -r requirements.txt`
3) Copy `.env.example` to `.env` and fill in:
   - COINBASE_API_KEY_NAME
   - COINBASE_API_PRIVATE_KEY (ECDSA PEM)

## Run (dry-run by default)
- One-shot:
  `python -m tradebot run --once`

- Loop:
  `python -m tradebot run --loop`

## Safety
- Starts with DRY_RUN=true
- Has MAX_QUOTE_PER_TRADE to prevent spending your full balance
- Stores last processed candle in STATE_PATH to reduce repeat actions

## Add new strategies
Create a new file in `src/tradebot/strategies/`, subclass `Strategy`,
then register it in `strategies/__init__.py`.