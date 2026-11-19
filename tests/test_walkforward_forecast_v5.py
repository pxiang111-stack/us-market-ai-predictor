import unittest
import numpy as np
import pandas as pd
from walkforward_forecast_v5 import generate_walkforward_forecasts


class WalkForwardForecastTests(unittest.TestCase):
    def test_no_forecast_before_minimum_purged_training_history(self):
        idx = pd.date_range("2020-01-01", periods=180, freq="B")
        close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, .01, len(idx)))), index=idx)
        x = pd.DataFrame({"ret": close.pct_change(), "mom": close.pct_change(5)}, index=idx)
        out = generate_walkforward_forecasts(x, close, 5, min_train_rows=150, retrain_every=10)
        self.assertTrue(out.empty)

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
        out = generate_walkforward_forecasts(x, close, 5, min_train_rows=300, retrain_every=50)
        self.assertFalse(out.empty)
        self.assertTrue((out["probability_calibrated"] == False).all())
        for date, row in out.iterrows():
            self.assertLessEqual(pd.Timestamp(row["train_end"]), date - pd.offsets.BDay(5))


if __name__ == "__main__":
    unittest.main()
