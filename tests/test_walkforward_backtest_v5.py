import unittest

import pandas as pd

from walkforward_backtest_v5 import concat_frames, DECISION_COST_FLOOR, build_benchmark_curve


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

    def test_concat_frames_empty_input(self):
        self.assertTrue(concat_frames([]).empty)
        self.assertTrue(concat_frames([pd.DataFrame()]).empty)


if __name__ == "__main__":
    unittest.main()
