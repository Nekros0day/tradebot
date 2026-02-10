import unittest

import pandas as pd

from tradebot.data.candles import _candles_any_to_df
from tradebot.strategies import STRATEGY_REGISTRY
from tradebot.strategies.mtf_weekly_ema26_daily_20_50 import MTFWeeklyEMA26Daily2050Strategy


class StrategyAndCandleTests(unittest.TestCase):
    def _build_daily_df(self, periods: int = 320) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
        prices = pd.Series(range(100, 100 + periods), index=idx, dtype="float64")
        return pd.DataFrame(
            {
                "start": (idx.view("int64") // 10**9),
                "open": prices.values,
                "high": prices.values + 1,
                "low": prices.values - 1,
                "close": prices.values,
                "volume": 1.0,
            },
            index=idx,
        )

    def test_all_registered_strategies_emit_valid_signal(self):
        candles = self._build_daily_df(320)

        for key, cls in STRATEGY_REGISTRY.items():
            strategy = cls() if key != "ema_filter" else cls(ema_length=140)
            sig = strategy.generate_signal(candles)

            self.assertIn(sig.target_position, (0, 1), msg=f"bad target for {key}")
            self.assertIn("last_close", sig.info, msg=f"missing last_close for {key}")

    def test_mtf_requires_weekly_history_before_long(self):
        strategy = MTFWeeklyEMA26Daily2050Strategy(weekly_ema_weeks=26)

        short = self._build_daily_df(25)  # fewer than 26 weekly closes
        sig_short = strategy.generate_signal(short)

        self.assertEqual(sig_short.target_position, 0)
        self.assertEqual(sig_short.info["enough_weekly_history"], "0")

    def test_candle_parser_supports_dict_and_array_shapes(self):
        candles_dict = [
            {"start": 1704067200, "low": "1", "high": "3", "open": "2", "close": "2.5", "volume": "10"},
            {"start": 1704153600, "low": "2", "high": "4", "open": "3", "close": "3.5", "volume": "11"},
        ]
        candles_arr = [
            [1704067200, "1", "3", "2", "2.5", "10"],
            [1704153600, "2", "4", "3", "3.5", "11"],
        ]

        df_dict = _candles_any_to_df(candles_dict)
        df_arr = _candles_any_to_df(candles_arr)

        self.assertListEqual(list(df_dict.columns), ["start", "open", "high", "low", "close", "volume"])
        self.assertListEqual(list(df_arr.columns), ["start", "open", "high", "low", "close", "volume"])
        self.assertEqual(len(df_dict), 2)
        self.assertEqual(len(df_arr), 2)
        self.assertTrue(str(df_dict.index.tz) in ("UTC", "UTC+00:00"))


if __name__ == "__main__":
    unittest.main()
