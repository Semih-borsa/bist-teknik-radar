import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


SPEC = importlib.util.spec_from_file_location("scanner", Path(__file__).parents[1] / "automation" / "scanner.py")
scanner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scanner)


class FollowUpTests(unittest.TestCase):
    def point(self, close=110, sma5=108, sma8=106, sma13=103, volume=1.2):
        return {"close": close, "sma5": sma5, "sma8": sma8, "sma13": sma13, "volumeRatio": volume}

    def test_momentum_confirmation_uses_state_not_second_cross(self):
        status, reason = scanner.evaluate_status("momentum", self.point())
        self.assertEqual(status, "CONFIRMED")
        self.assertIn("üç ortalamanın üzerinde", reason)

    def test_low_volume_is_partial_confirmation(self):
        status, _ = scanner.evaluate_status("momentum", self.point(volume=0.75))
        self.assertEqual(status, "PARTIAL")

    def test_momentum_below_sma13_is_reversed(self):
        status, _ = scanner.evaluate_status("momentum", self.point(close=100))
        self.assertEqual(status, "REVERSED")

    def test_pullback_below_sma8_is_not_confirmed(self):
        status, _ = scanner.evaluate_status("pullback", self.point(close=104))
        self.assertEqual(status, "NOT_CONFIRMED")

    def test_tracking_creates_t1_t3_t5_once(self):
        dates = pd.bdate_range("2026-01-01", periods=32)
        close = [80 + index for index in range(32)]
        frame = pd.DataFrame({"Open": close, "High": [x + 1 for x in close], "Low": [x - 1 for x in close], "Close": close, "Volume": [1000] * 32}, index=dates)
        signal_date = dates[22].date().isoformat()
        item = {"symbol": "THYAO", "strategy": "momentum", "event": "ABOVE", "marketDate": signal_date,
                "close": close[22], "sma5": 100, "sma8": 98, "sma13": 95, "volumeRatio": 1.2, "reason": "Test"}
        tracking = [scanner.signal_track(item)]
        tracking, added = scanner.evaluate_tracking(tracking, {"THYAO": frame})
        self.assertEqual([check["offset"] for check in tracking[0]["checks"]], [1, 3, 5])
        self.assertEqual(len(added), 3)
        _, added_again = scanner.evaluate_tracking(tracking, {"THYAO": frame})
        self.assertEqual(added_again, [])

    def test_telegram_contains_follow_up_summary(self):
        message = scanner.telegram_message({"marketDate": "2026-01-02", "scannedSymbols": 540, "observations": [],
                                            "followUps": [{"status": "CONFIRMED", "symbol": "THYAO", "offset": 1, "label": "Teyit aldı"}]})
        self.assertIn("Takip özeti", message)
        self.assertIn("THYAO T+1 · Teyit aldı", message)
        self.assertIn("işlem emri değildir", message)

    def test_invalid_chat_id_does_not_fail_scan(self):
        fake_requests = SimpleNamespace(RequestException=Exception)
        with patch.dict(os.environ, {"TELEGRAM_TOKEN": "123:test", "TELEGRAM_CHAT_ID": "123:yanlis"}), \
             patch.dict(sys.modules, {"requests": fake_requests}):
            self.assertFalse(scanner.send_telegram("Test"))

    def test_telegram_http_error_is_nonfatal_and_description_is_safe(self):
        class Response:
            ok = False
            status_code = 400

            @staticmethod
            def json():
                return {"description": "Bad Request: chat not found"}

        fake_requests = SimpleNamespace(RequestException=Exception, post=lambda *args, **kwargs: Response())
        with patch.dict(os.environ, {"TELEGRAM_TOKEN": "123:test", "TELEGRAM_CHAT_ID": "123456789"}), \
             patch.dict(sys.modules, {"requests": fake_requests}):
            self.assertFalse(scanner.send_telegram("Test"))


if __name__ == "__main__":
    unittest.main()
