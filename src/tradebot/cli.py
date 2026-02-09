from __future__ import annotations

import argparse
import logging

from .config import Settings
from .logging_config import setup_logging
from .bot.runner import BotRunner


def main():
    parser = argparse.ArgumentParser(prog="tradebot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Run the bot")
    mode = runp.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one cycle and exit")
    mode.add_argument("--loop", action="store_true", help="Run continuously (polling)")

    runp.add_argument("--strategy", default="ema_filter", help="Strategy name (default: ema_filter)")

    args = parser.parse_args()

    s = Settings.load()
    setup_logging(s.log_level)
    log = logging.getLogger("tradebot")

    runner = BotRunner(s, strategy_name=args.strategy)

    if args.loop or (not args.once and s.mode.lower() == "loop"):
        log.info("Starting in LOOP mode")
        runner.run_loop()
    else:
        log.info("Starting in ONCE mode")
        runner.run_once()