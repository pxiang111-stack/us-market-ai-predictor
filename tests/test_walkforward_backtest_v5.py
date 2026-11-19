import unittest

import pandas as pd

from walkforward_backtest_v5 import concat_frames


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

    def test_concat_frames_empty_input(self):
        self.assertTrue(concat_frames([]).empty)
        self.assertTrue(concat_frames([pd.DataFrame()]).empty)


if __name__ == "__main__":
    unittest.main()
