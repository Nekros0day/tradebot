from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from uuid import uuid4

from ..exchange.coinbase_client import CoinbaseClient
from ..utils.decimal_utils import D, round_down_step, to_str

log = logging.getLogger("tradebot.trader")


@dataclass(frozen=True)
class Balances:
    base: Decimal
    quote: Decimal


@dataclass(frozen=True)
class ProductRules:
    base_increment: Decimal
    quote_increment: Decimal
    base_min_size: Decimal
    quote_min_size: Decimal


class Trader:
    def __init__(
        self,
        client: CoinbaseClient,
        *,
        product_id: str,
        base_currency: str,
        quote_currency: str,
        dry_run: bool,
        allocation_pct: float,
        max_quote_per_trade: float,
        min_quote_per_trade: float,
        min_base_sell: float,
        # NEW:
        target_long_pct: float,
        rebalance_tol_pct: float,
        min_position_notional: float,
    ):
        self.client = client
        self.product_id = product_id
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.dry_run = dry_run

        # spending controls
        self.allocation_pct = D(allocation_pct)
        self.max_quote_per_trade = D(max_quote_per_trade)
        self.min_quote_per_trade = D(min_quote_per_trade)
        self.min_base_sell = D(min_base_sell)

        # NEW: portfolio-style rebalance controls
        self.target_long_pct = D(target_long_pct)
        self.rebalance_tol_pct = D(rebalance_tol_pct)
        self.min_position_notional = D(min_position_notional)

    def get_balances(self) -> Balances:
        accounts = self.client.list_all_accounts()
        base = D("0")
        quote = D("0")

        for a in accounts:
            cur = a.get("currency")
            avail = a.get("available_balance", {}).get("value")
            if avail is None:
                continue
            if cur == self.base_currency:
                base = D(avail)
            elif cur == self.quote_currency:
                quote = D(avail)

        return Balances(base=base, quote=quote)

    def get_product_rules(self) -> ProductRules:
        p = self.client.get_product(self.product_id)
        return ProductRules(
            base_increment=D(p.get("base_increment", "0.00000001")),
            quote_increment=D(p.get("quote_increment", "0.01")),
            base_min_size=D(p.get("base_min_size", "0")),
            quote_min_size=D(p.get("quote_min_size", "0")),
        )

    def market_buy_quote(self, quote_size: Decimal) -> dict:
        payload = {
            "client_order_id": str(uuid4()),
            "product_id": self.product_id,
            "side": "BUY",
            "order_configuration": {
                "market_market_ioc": {
                    "quote_size": to_str(quote_size),
                }
            },
        }

        if self.dry_run:
            log.warning(f"[DRY_RUN] Would BUY {self.product_id} quote_size={to_str(quote_size)}")
            return {"dry_run": True, "payload": payload}

        return self.client.create_order(payload)

    def market_sell_base(self, base_size: Decimal) -> dict:
        payload = {
            "client_order_id": str(uuid4()),
            "product_id": self.product_id,
            "side": "SELL",
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": to_str(base_size),
                }
            },
        }

        if self.dry_run:
            log.warning(f"[DRY_RUN] Would SELL {self.product_id} base_size={to_str(base_size)}")
            return {"dry_run": True, "payload": payload}

        return self.client.create_order(payload)

    def rebalance_to_target(self, *, target_position: int, last_price: Decimal) -> None:
        """
        Fully functional EMA long/flat execution using allocation rebalancing.

        target_position:
          1 => target BTC allocation ~= target_long_pct of total value
          0 => target BTC allocation ~= 0
        """
        rules = self.get_product_rules()
        bal = self.get_balances()

        base_value = bal.base * last_price
        total_value = base_value + bal.quote

        if total_value <= D("0"):
            log.warning("Total value is 0; nothing to do.")
            return

        cur_base_pct = base_value / total_value
        tgt_base_pct = self.target_long_pct if target_position == 1 else D("0")
        tol = self.rebalance_tol_pct

        log.info(
            f"Balances: {self.base_currency}={to_str(bal.base)} (~{to_str(base_value)} {self.quote_currency}), "
            f"{self.quote_currency}={to_str(bal.quote)}, Total~{to_str(total_value)} | "
            f"cur_base_pct={to_str(cur_base_pct)} tgt_base_pct={to_str(tgt_base_pct)} tol={to_str(tol)}"
        )

        # Dust guard when flattening: ignore tiny BTC positions
        if target_position == 0 and base_value < self.min_position_notional:
            log.info(
                f"{self.base_currency} value {to_str(base_value)} < MIN_POSITION_NOTIONAL={to_str(self.min_position_notional)}; "
                f"treating as flat (no sell)."
            )
            return

        # --- If we have too much base (BTC), sell down toward target ---
        if cur_base_pct > tgt_base_pct + tol:
            desired_base_value = tgt_base_pct * total_value
            excess_value = base_value - desired_base_value

            # Convert quote-value excess into base qty to sell
            sell_qty = excess_value / last_price

            # never sell more than we have; leave a tiny dust buffer
            sell_qty = min(sell_qty, bal.base * D("0.999"))
            sell_qty = round_down_step(sell_qty, rules.base_increment)

            if sell_qty < max(self.min_base_sell, rules.base_min_size):
                log.info(f"SELL skipped: qty {to_str(sell_qty)} below min.")
                return

            resp = self.market_sell_base(sell_qty)
            log.info(f"SELL response: {resp}")
            return

        # --- If we have too little base (BTC), buy up toward target ---
        if cur_base_pct < tgt_base_pct - tol:
            desired_base_value = tgt_base_pct * total_value
            needed_value = desired_base_value - base_value
            if needed_value <= D("0"):
                log.info("BUY not needed (already at/above target).")
                return

            # Spend limited by:
            # - what we need
            # - allocation_pct of available quote (safety)
            # - max cap per trade
            spend = min(needed_value, bal.quote * self.allocation_pct, self.max_quote_per_trade)
            spend = round_down_step(spend, rules.quote_increment)

            if spend < max(self.min_quote_per_trade, rules.quote_min_size):
                log.info(f"BUY skipped: spend {to_str(spend)} below min.")
                return

            resp = self.market_buy_quote(spend)
            log.info(f"BUY response: {resp}")
            return

        log.info("No rebalance needed (within tolerance band).")