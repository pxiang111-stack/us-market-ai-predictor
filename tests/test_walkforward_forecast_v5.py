import unittest
import numpy as np
import pandas as pd
from walkforward_forecast_v5 import build_execution_aligned_labels, generate_walkforward_forecasts


class WalkForwardForecastTests(unittest.TestCase):
    def test_no_forecast_before_minimum_purged_training_history(self):
        idx = pd.date_range("2020-01-01", periods=180, freq="B")
        close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, .01, len(idx)))), index=idx)
        x = pd.DataFrame({"ret": close.pct_change(), "mom": close.pct_change(5)}, index=idx)
        out = generate_walkforward_forecasts(x, close, 5, entry_open=close, min_train_rows=176, retrain_every=10)
        self.assertTrue(out.empty)

    def test_start_date_limits_scored_forecasts_without_losing_training_history(self):
        idx = pd.date_range("2015-01-01", periods=900, freq="B")
        rng = np.random.default_rng(11)
        rets = rng.normal(.0003, .01, len(idx))
        close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
        x = pd.DataFrame({"ret1": rets, "mom5": close.pct_change(5), "vol20": pd.Series(rets, index=idx).rolling(20).std()}, index=idx).fillna(0)
        start = idx[700]
        out = generate_walkforward_forecasts(x, close, 5, entry_open=close, min_train_rows=300, retrain_every=50, start_date=start)
        self.assertFalse(out.empty)
        self.assertGreaterEqual(out.index.min(), start)
        self.assertGreaterEqual(int(out.iloc[0]["n_train"]), 300)

    def test_forecast_metadata_marks_probability_uncalibrated(self):
        idx = pd.date_range("2015-01-01", periods=900, freq="B")
        rng = np.random.default_rng(9)
        rets = rng.normal(.0004, .012, len(idx))
        close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
        x = pd.DataFrame({
            "ret1": pd.Series(rets, index=idx),
            "mom5": close.pct_change(5),
            "vol20": pd.Series(rets, index=idx).rolling(20).std(),
        }, index=idx).fillna(0)
        out = generate_walkforward_forecasts(x, close, 5, entry_open=close, min_train_rows=300, retrain_every=50)
        self.assertFalse(out.empty)
        self.assertTrue((out["probability_calibrated"] == False).all())
        for date, row in out.iterrows():
            self.assertLessEqual(pd.Timestamp(row["train_end"]), date - pd.offsets.BDay(5))


    def test_execution_aligned_label_uses_next_open_to_horizon_close(self):
        idx = pd.date_range("2026-01-01", periods=8, freq="B")
        close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107], index=idx, dtype=float)
        open_price = pd.Series([99, 110, 111, 112, 113, 114, 115, 116], index=idx, dtype=float)
        labels = build_execution_aligned_labels(close, open_price, horizon_sessions=3)

        self.assertAlmostEqual(labels.loc[idx[0], "forward_return"], 103.0 / 110.0 - 1.0)
        self.assertEqual(labels.loc[idx[0], "up"], 0.0)
        self.assertTrue(labels.iloc[-3:]["forward_return"].isna().all())
        self.assertTrue(labels.iloc[-3:]["up"].isna().all())

    def test_execution_aligned_label_requires_next_open_and_exit_close(self):
        idx = pd.date_range("2026-01-01", periods=6, freq="B")
        close = pd.Series([100, 101, 102, 103, 104, 105], index=idx, dtype=float)
        open_price = pd.Series([99, 100, np.nan, 102, 103, 104], index=idx, dtype=float)
        labels = build_execution_aligned_labels(close, open_price, horizon_sessions=2)

        self.assertTrue(pd.isna(labels.loc[idx[1], "forward_return"]))


if __name__ == "__main__":
    unittest.main()
