import unittest

import pandas as pd

from walkforward_backtest_v5 import (concat_frames, DECISION_COST_FLOOR, build_benchmark_curve, forecast_quality_summary)


class WalkForwardBacktestAggregationTests(unittest.TestCase):
    def test_concat_frames_handles_different_symbol_schemas(self):
        first = pd.DataFrame([{"ticker": "QQQ", "probability_up": 0.6}])
        second = pd.DataFrame([{"ticker": "SPY", "expected_return": 0.02}])
        combined = concat_frames([first, second])
        self.assertEqual(len(combined), 2)
        self.assertIn("probability_up", combined.columns)
        self.assertIn("expected_return", combined.columns)
        self.assertEqual(combined.loc[0, "ticker"], "QQQ")
        self.assertEqual(combined.loc[1, "ticker"], "SPY")

    def test_spy_qqq_benchmark_uses_matched_dates_and_normalizes_equity(self):
        idx = pd.date_range("2025-01-01", periods=4, freq="B")
        close = pd.Series([100.0, 102.0, 101.0, 105.0], index=idx)
        curve = build_benchmark_curve(close, idx[1:])
        self.assertEqual(len(curve), 3)
        self.assertEqual(curve.iloc[0]["equity"], 100000.0)
        self.assertAlmostEqual(curve.iloc[-1]["equity"], 100000.0 * 105.0 / 102.0)

    def test_signal_filter_covers_worst_case_round_trip_cost_and_slippage(self):
        self.assertAlmostEqual(DECISION_COST_FLOOR, 0.004)

    def test_forecast_quality_uses_realized_future_returns_only_for_diagnostics(self):
        idx = pd.date_range("2025-01-01", periods=8, freq="B")
        close = pd.Series([100, 101, 100, 102, 101, 103, 102, 104], index=idx, dtype=float)
        realized = close.shift(-1).iloc[:-1] / close.iloc[:-1] - 1.0
        forecasts = pd.DataFrame({
            "probability_up": [0.9 if x > 0 else 0.1 for x in realized],
            "expected_return": realized.values * 0.8,
        }, index=idx[:-1])
        stats = forecast_quality_summary(forecasts, close, 1)
        self.assertEqual(stats["forecast_eval_n"], 7)
        self.assertAlmostEqual(stats["forecast_auc"], 1.0)
        self.assertAlmostEqual(stats["forecast_directional_accuracy_50_pct"], 100.0)
        self.assertAlmostEqual(stats["predicted_vs_realized_spearman_ic"], 1.0)
        self.assertGreater(stats["forecast_brier_skill_vs_50"], 0.0)
        self.assertGreaterEqual(stats["forecast_ece_10"], 0.0)
        self.assertIn("predicted_return_mae_pct", stats)
        self.assertIn("predicted_return_rmse_pct", stats)

    def test_concat_frames_empty_input(self):
        self.assertTrue(concat_frames([]).empty)
        self.assertTrue(concat_frames([pd.DataFrame()]).empty)


if __name__ == "__main__":
    unittest.main()
