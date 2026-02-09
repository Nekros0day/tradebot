from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from ..config import Settings
from ..data.candles import get_latest_closed_daily_candles
from ..exchange.coinbase_client import CoinbaseClient
from ..execution.trader import Trader
from ..strategies import STRATEGY_REGISTRY
from ..utils.decimal_utils import D, to_str

log = logging.getLogger("tradebot.runner")


@dataclass
class BotState:
    last_candle_start: int | None = None
    last_target_position: int | None = None

    @staticmethod
    def load(path: Path) -> "BotState":
        if not path.exists():
            return BotState()
        try:
            obj = json.loads(path.read_text())
            return BotState(
                last_candle_start=obj.get("last_candle_start"),
                last_target_position=obj.get("last_target_position"),
            )
        except Exception:
            return BotState()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(
            {
                "last_candle_start": self.last_candle_start,
                "last_target_position": self.last_target_position,
            },
            indent=2
        ))


class BotRunner:
    def __init__(self, settings: Settings, strategy_name: str | None = None):
        self.s = settings
        strategy_name = strategy_name or self.s.strategy_name
        self.client = CoinbaseClient(
            base_url=self.s.base_url,
            jwt_host=self.s.jwt_host,
            api_key_name=self.s.api_key_name,
            api_private_key=self.s.api_private_key,
        )

        StratCls = STRATEGY_REGISTRY.get(strategy_name)
        if StratCls is None:
            raise RuntimeError(f"Unknown strategy '{strategy_name}'. Known: {list(STRATEGY_REGISTRY)}")
        if strategy_name == "ema_filter":
            self.strategy = StratCls(ema_length=self.s.ema_length)
        else:
            self.strategy = StratCls()

        self.trader = Trader(
            self.client,
            product_id=self.s.product_id,
            base_currency=self.s.base_currency,
            quote_currency=self.s.quote_currency,
            dry_run=self.s.dry_run,
            allocation_pct=self.s.allocation_pct,
            max_quote_per_trade=self.s.max_quote_per_trade,
            min_quote_per_trade=self.s.min_quote_per_trade,
            min_base_sell=self.s.min_base_sell,
            # NEW:
            target_long_pct=self.s.target_long_pct,
            rebalance_tol_pct=self.s.rebalance_tol_pct,
            min_position_notional=self.s.min_position_notional,
        )

        self.state = BotState.load(self.s.state_path)
        log.info(f"Using strategy: {self.strategy.name} (key={strategy_name})")

    def run_once(self) -> None:
        lookback = max(self.strategy.lookback_bars, self.s.ema_length + 30)
        fetch = get_latest_closed_daily_candles(
            self.client,
            self.s.product_id,
            lookback_days=lookback,
            granularity=self.s.granularity,
        )

        if self.state.last_candle_start == fetch.latest_closed_start:
            log.info("No new closed candle since last run; skipping.")
            return

        candles = fetch.df  # must be a DataFrame
        sig = self.strategy.generate_signal(candles)

        log.info(f"candles type={type(fetch.df)} shape={fetch.df.shape} cols={list(fetch.df.columns)}")
        log.info(f"tail:\n{fetch.df.tail(3)}")

        last_close = D(sig.info["last_close"])
        last_ema = D(sig.info["last_ema"])

        log.info(
            f"Signal {self.strategy.name}: close={to_str(last_close)} ema={to_str(last_ema)} -> target={sig.target_position}"
        )

        # Execute (rebalance)
        self.trader.rebalance_to_target(target_position=sig.target_position, last_price=last_close)

        # Save state
        self.state.last_candle_start = fetch.latest_closed_start
        self.state.last_target_position = sig.target_position
        self.state.save(self.s.state_path)

    def run_loop(self) -> None:
        log.info(f"Looping every {self.s.poll_seconds}s...")
        while True:
            try:
                self.run_once()
            except Exception as e:
                log.exception(f"Run failed: {e}")
            time.sleep(self.s.poll_seconds)