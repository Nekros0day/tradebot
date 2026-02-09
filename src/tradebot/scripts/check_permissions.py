from __future__ import annotations

import argparse
import json
import logging

from tradebot.config import Settings
from tradebot.exchange.coinbase_client import CoinbaseClient
from tradebot.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Check Coinbase API key permissions + portfolio scoping.")
    parser.add_argument("--raw", action="store_true", help="Print full raw JSON response")
    args = parser.parse_args()

    s = Settings.load()
    setup_logging(s.log_level)
    log = logging.getLogger("tradebot.check_permissions")

    client = CoinbaseClient(
        base_url=s.base_url,
        jwt_host=s.jwt_host,
        api_key_name=s.api_key_name,
        api_private_key=s.api_private_key,
    )

    # Endpoint: Get API key permissions (Advanced Trade)
    # Docs: https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/data-api/get-api-key-permissions
    resp = client._request("GET", "/api/v3/brokerage/key_permissions")

    perms = resp.get("permissions", resp)  # some responses nest under 'permissions'

    # Common fields seen: can_view, can_trade, can_transfer, portfolio_uuid, etc.
    can_view = perms.get("can_view")
    can_trade = perms.get("can_trade")
    can_transfer = perms.get("can_transfer")
    portfolio_uuid = perms.get("portfolio_uuid") or perms.get("portfolio_id") or perms.get("portfolio")

    log.info("=== API KEY PERMISSIONS ===")
    log.info(f"can_view     : {can_view}")
    log.info(f"can_trade    : {can_trade}")
    log.info(f"can_transfer : {can_transfer}")
    log.info(f"portfolio    : {portfolio_uuid}")

    if args.raw:
        print("\n--- RAW JSON ---")
        print(json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()