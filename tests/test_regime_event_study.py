import unittest
import pandas as pd

from regime_event_study import build_event_study


class RegimeEventStudyTests(unittest.TestCase):
    def test_small_samples_are_record_only(self):
        dates = pd.bdate_range("2026-01-01", periods=25)
        frame = pd.DataFrame({
            "date": dates,
            "state": ["RISK_OFF"] * 25,
            "NDX_close": [100.0 + i for i in range(25)],
        })
        result = build_event_study(frame)
        row = result[(result.sample_type == "DAILY_RISK_OFF") &
                     (result.horizon_days == 5)].iloc[0]
        self.assertEqual(row.n, 20)
        self.assertEqual(row.status, "RECORD_ONLY")
        self.assertAlmostEqual(row.coverage_pct, 100.0)


    def test_first_observation_is_not_counted_as_state_entry(self):
        dates = pd.bdate_range("2026-01-01", periods=25)
        frame = pd.DataFrame({
            "date": dates,
            "state": ["RISK_OFF"] * 25,
            "NDX_close": [100.0 + i for i in range(25)],
        })
        result = build_event_study(frame)
        row = result[(result.sample_type == "ENTRY_RISK_OFF") &
                     (result.horizon_days == 5)].iloc[0]
        self.assertEqual(row.n, 0)

    def test_state_coverage_is_fraction_of_valid_market_days(self):
        dates = pd.bdate_range("2026-01-01", periods=40)
        frame = pd.DataFrame({
            "date": dates,
            "state": ["RISK_OFF"] * 20 + ["NEUTRAL"] * 20,
            "NDX_close": [100.0 + i for i in range(40)],
        })
        result = build_event_study(frame)
        row = result[(result.sample_type == "DAILY_RISK_OFF") &
                     (result.horizon_days == 5)].iloc[0]
        self.assertEqual(row.n, 20)
        self.assertAlmostEqual(row.coverage_pct, 20 / 35 * 100)


if __name__ == "__main__":
    unittest.main()
