import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SPEC = importlib.util.spec_from_file_location("nadaraya", Path(__file__).parents[1] / "automation" / "nadaraya.py")
nadaraya = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nadaraya)


class NadarayaTests(unittest.TestCase):
    def test_endpoint_uses_full_500_bar_window(self):
        close = np.arange(1.0, 700.0)
        out = nadaraya.endpoint_nwe(close)
        self.assertTrue(np.isnan(out[498]))
        self.assertTrue(np.isfinite(out[499]))

    def test_mae_becomes_valid_after_original_window_requirements(self):
        dates = pd.bdate_range("2020-01-01", periods=1000)
        close = np.linspace(100, 200, len(dates))
        frame = pd.DataFrame({
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(len(dates), 1000),
        }, index=dates)
        data = nadaraya.with_nadaraya(frame)
        self.assertTrue(np.isnan(data["NWE_MAE"].iloc[996]))
        self.assertTrue(np.isfinite(data["NWE_MAE"].iloc[997]))

    def test_buy_requires_reentry_rsi_turn_and_adx_filter(self):
        previous = {"Close": 90, "High": 91, "Low": 89, "NWE_UPPER": 110, "NWE_LOWER": 90, "RSI14": 28, "ADX14": 20}
        current = {"Close": 91, "High": 92, "Low": 88, "NWE_UPPER": 110, "NWE_LOWER": 90, "RSI14": 32, "ADX14": 22}
        signal, _ = nadaraya.classify_signal(current, previous)
        self.assertEqual(signal, "BUY")

        current["ADX14"] = 35
        signal, _ = nadaraya.classify_signal(current, previous)
        self.assertEqual(signal, "BUY_CANDIDATE")

    def test_sell_thresholds_use_channel_position(self):
        previous = {"Close": 107, "High": 108, "Low": 106, "NWE_UPPER": 110, "NWE_LOWER": 90, "RSI14": 62, "ADX14": 20}
        current = {"Close": 109.2, "High": 109.5, "Low": 108.5, "NWE_UPPER": 110, "NWE_LOWER": 90, "RSI14": 66, "ADX14": 20}
        signal, _ = nadaraya.classify_signal(current, previous)
        self.assertEqual(signal, "SELL")

        current["Close"] = 108.2
        current["RSI14"] = 61
        signal, _ = nadaraya.classify_signal(current, previous)
        self.assertEqual(signal, "SELL_APPROACH")

    def test_attach_preserves_existing_scanner_fields(self):
        original = {"observations": [{"symbol": "THYAO"}], "tracking": [{"id": "x"}], "version": 2}
        updated = nadaraya.attach_nadaraya(
            original,
            [{"symbol": "ASELS", "signal": "WATCH", "channelPosition": 5}],
            available_symbols=500,
            failed_symbols=3,
            insufficient_symbols=2,
            duration_seconds=1.0,
        )
        self.assertEqual(updated["observations"], original["observations"])
        self.assertEqual(updated["tracking"], original["tracking"])
        self.assertEqual(updated["nadaraya"][0]["symbol"], "ASELS")


if __name__ == "__main__":
    unittest.main()
