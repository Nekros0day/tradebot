import unittest
from pathlib import Path
from unittest.mock import patch

from tradebot.config import Settings


class DummyRunner:
    created_with = None

    def __init__(self, settings, strategy_name=None):
        DummyRunner.created_with = strategy_name
        self.settings = settings

    def run_once(self):
        return None

    def run_loop(self):
        return None


class CLIStrategySelectionTests(unittest.TestCase):
    def _settings(self) -> Settings:
        return Settings(
            api_key_name="k",
            api_private_key="p",
            strategy_name="mtf_weeklyema26_daily20_50",
            mode="once",
            state_path=Path("./state.json"),
        )

    @patch("tradebot.cli.setup_logging")
    @patch("tradebot.cli.BotRunner", DummyRunner)
    @patch("tradebot.cli.Settings.load")
    def test_cli_uses_env_strategy_when_flag_not_provided(self, mock_load, _mock_log):
        mock_load.return_value = self._settings()

        with patch("sys.argv", ["tradebot", "run", "--once"]):
            from tradebot.cli import main

            main()

        self.assertIsNone(DummyRunner.created_with)

    @patch("tradebot.cli.setup_logging")
    @patch("tradebot.cli.BotRunner", DummyRunner)
    @patch("tradebot.cli.Settings.load")
    def test_cli_strategy_flag_overrides_env(self, mock_load, _mock_log):
        mock_load.return_value = self._settings()

        with patch("sys.argv", ["tradebot", "run", "--once", "--strategy", "ema_filter"]):
            from tradebot.cli import main

            main()

        self.assertEqual(DummyRunner.created_with, "ema_filter")


if __name__ == "__main__":
    unittest.main()
