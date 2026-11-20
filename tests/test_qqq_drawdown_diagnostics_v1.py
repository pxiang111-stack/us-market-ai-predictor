import unittest
import pandas as pd
from qqq_drawdown_diagnostics_v1 import evaluate_drawdown_warnings


class QQQDrawdownDiagnosticsTests(unittest.TestCase):
    def test_detects_future_drawdown_and_scores_early_warning(self):
        dates = pd.date_range("2026-01-01", periods=8, freq="B")
        bars = pd.DataFrame({
            "Open": [100, 99, 97, 94, 96, 98, 99, 100],
            "High": [101, 100, 98, 96, 99, 100, 101, 101],
            "Low": [99, 97, 93, 90, 95, 97, 98, 99],
            "Close": [100, 98, 95, 95, 98, 99, 100, 100],
        }, index=dates)
        forecasts = pd.DataFrame([{
            "as_of": dates[0], "horizon_sessions": 3,
            "probability_up": 0.30, "expected_return": -0.04,
            "decision": "FLAT",
        }])
        detail, summary = evaluate_drawdown_warnings(forecasts, bars, thresholds=(0.05, 0.08))
        self.assertTrue(detail.iloc[0]["bearish_model_warning"])
        self.assertTrue(detail.iloc[0]["drawdown_5pct"])
        self.assertTrue(detail.iloc[0]["drawdown_8pct"])
        self.assertEqual(detail.iloc[0]["drawdown_5pct_first_hit_session"], 2)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary.iloc[0]["warning_count"], 1)

    def test_trade_short_signal_is_reported_separately(self):
        dates = pd.date_range("2026-02-01", periods=4, freq="B")
        bars = pd.DataFrame({
            "Open": [100, 99, 98, 97], "High": [101, 100, 99, 98],
            "Low": [99, 97, 96, 95], "Close": [100, 98, 97, 96],
        }, index=dates)
        forecasts = pd.DataFrame([{
            "as_of": dates[0], "horizon_sessions": 2,
            "probability_up": 0.45, "expected_return": -0.01,
            "decision": "SHORT",
        }])
        detail, summary = evaluate_drawdown_warnings(forecasts, bars, thresholds=(0.03,))
        self.assertFalse(detail.iloc[0]["bearish_model_warning"])
        self.assertTrue(detail.iloc[0]["short_trade_signal"])
        self.assertEqual(set(summary["warning_type"]), {"bearish_model_warning", "short_trade_signal"})

    def test_asof_bar_is_excluded_from_future_drawdown(self):
        dates = pd.date_range("2026-03-01", periods=3, freq="B")
        bars = pd.DataFrame({
            "Open": [100, 100, 100], "High": [101, 101, 101],
            "Low": [80, 99, 99], "Close": [100, 100, 100],
        }, index=dates)
        forecasts = pd.DataFrame([{
            "as_of": dates[0], "horizon_sessions": 2,
            "probability_up": 0.7, "expected_return": 0.02,
            "decision": "LONG",
        }])
        detail, _ = evaluate_drawdown_warnings(forecasts, bars, thresholds=(0.05,))
        self.assertFalse(detail.iloc[0]["drawdown_5pct"])


if __name__ == "__main__":
    unittest.main()
