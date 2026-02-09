from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


def _getenv(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v if v is None else str(v)


def _getenv_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _getenv_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None else int(v)


def _getenv_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)


@dataclass(frozen=True)
class Settings:
    # Coinbase auth
    api_key_name: str
    api_private_key: str
    base_url: str = "https://api.coinbase.com"
    jwt_host: str = "api.coinbase.com"  # used in the JWT "uri" claim

    # Trading
    product_id: str = "BTC-USD"
    granularity: str = "ONE_DAY"
    ema_length: int = 140

    base_currency: str = "BTC"
    quote_currency: str = "USD"

    # Execution controls
    dry_run: bool = True

    allocation_pct: float = 0.98
    max_quote_per_trade: float = 500.0
    min_quote_per_trade: float = 10.0
    min_base_sell: float = 0.00001

    # Looping
    mode: str = "once"  # once|loop
    poll_seconds: int = 300

    # State/logging
    state_path: Path = Path("./state.json")
    log_level: str = "INFO"

    target_long_pct: float = 0.98
    rebalance_tol_pct: float = 0.02
    min_position_notional: float = 25.0

    strategy_name: str = "ema_filter"

    @staticmethod
    def load() -> "Settings":
        # load .env if present
        load_dotenv(override=False)

        key_name = _getenv("COINBASE_API_KEY_NAME")
        priv_key = _getenv("COINBASE_API_PRIVATE_KEY")

        if not key_name or not priv_key:
            raise RuntimeError(
                "Missing COINBASE_API_KEY_NAME or COINBASE_API_PRIVATE_KEY in environment."
            )

        # allow \n escaped PEM in env var
        priv_key = priv_key.replace("\\n", "\n")

        return Settings(
            api_key_name=key_name,
            api_private_key=priv_key,
            base_url=_getenv("COINBASE_BASE_URL", "https://api.coinbase.com") or "https://api.coinbase.com",
            jwt_host=_getenv("COINBASE_JWT_HOST", "api.coinbase.com") or "api.coinbase.com",
            product_id=_getenv("PRODUCT_ID", "BTC-USD") or "BTC-USD",
            granularity=_getenv("GRANULARITY", "ONE_DAY") or "ONE_DAY",
            ema_length=_getenv_int("EMA_LENGTH", 140),
            base_currency=_getenv("BASE_CURRENCY", "BTC") or "BTC",
            quote_currency=_getenv("QUOTE_CURRENCY", "USD") or "USD",
            dry_run=_getenv_bool("DRY_RUN", True),
            allocation_pct=_getenv_float("ALLOCATION_PCT", 0.98),
            max_quote_per_trade=_getenv_float("MAX_QUOTE_PER_TRADE", 500.0),
            min_quote_per_trade=_getenv_float("MIN_QUOTE_PER_TRADE", 10.0),
            min_base_sell=_getenv_float("MIN_BASE_SELL", 0.00001),
            mode=_getenv("MODE", "once") or "once",
            poll_seconds=_getenv_int("POLL_SECONDS", 300),
            state_path=Path(_getenv("STATE_PATH", "./state.json") or "./state.json"),
            log_level=_getenv("LOG_LEVEL", "INFO") or "INFO",
            target_long_pct=_getenv_float("TARGET_LONG_PCT", 0.98),
            rebalance_tol_pct=_getenv_float("REBALANCE_TOL_PCT", 0.02),
            min_position_notional=_getenv_float("MIN_POSITION_NOTIONAL", 25.0),
            strategy_name=_getenv("STRATEGY", "ema_filter") or "ema_filter",
        )