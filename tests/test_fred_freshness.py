import unittest

import pandas as pd

from run_regime_first_v2 import align_fred_to_nq_sessions


class FredFreshnessTests(unittest.TestCase):
    def test_fred_values_are_not_carried_beyond_three_nq_sessions(self):
        source = pd.Series([13.6], index=[pd.Timestamp("2026-11-13")])
        dates = pd.date_range("2026-11-13", periods=6, freq="B")

        aligned = align_fred_to_nq_sessions(source, dates, max_stale_sessions=3)

        self.assertEqual(aligned.iloc[0], 13.6)
        self.assertEqual(aligned.iloc[3], 13.6)
        self.assertTrue(pd.isna(aligned.iloc[4]))
        self.assertTrue(pd.isna(aligned.iloc[5]))


if __name__ == "__main__":
    unittest.main()
