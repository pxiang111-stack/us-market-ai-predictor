import unittest
import numpy as np
import pandas as pd
from market_risk_v4 import make_features, labeled, market_stress, merge_market_structure_features, run_forecast

class V4Tests(unittest.TestCase):
    def setUp(self):
        idx=pd.bdate_range("2020-01-01",periods=600); rng=np.random.default_rng(5)
        self.p={"^NDX":pd.Series(1000*np.exp(np.cumsum(rng.normal(.0005,.01,600))),index=idx),
          "SOXL":pd.Series(20*np.exp(np.cumsum(rng.normal(.0008,.03,600))),index=idx),
          "^VIX":pd.Series(20+np.abs(rng.normal(0,4,600)),index=idx),
          "HYG":pd.Series(80*np.exp(np.cumsum(rng.normal(0,.003,600))),index=idx),
          "LQD":pd.Series(100*np.exp(np.cumsum(rng.normal(0,.002,600))),index=idx)}
    def test_features_and_no_embedded_targets(self):
        x=make_features(self.p,"^NDX")
        self.assertIn("vix_level",x); self.assertIn("credit_mom20",x)
        self.assertNotIn("up",x); self.assertNotIn("forward_return",x)
    def test_market_structure_features_join_to_model_inputs(self):
        idx = self.p["SOXL"].index
        bars = pd.DataFrame({
            "Open": self.p["SOXL"].values * 0.99,
            "High": self.p["SOXL"].values * 1.02,
            "Low": self.p["SOXL"].values * 0.98,
            "Close": self.p["SOXL"].values,
            "Volume": np.arange(1000, 1600, dtype=float),
        }, index=idx)
        merged, meta = merge_market_structure_features(make_features(self.p, "SOXL"), bars, "SOXL")
        self.assertIn("ms_relative_volume", merged.columns)
        self.assertIn("ms_close_location", merged.columns)
        self.assertIn("ms_trend_above_sma20", merged.columns)
        self.assertIn("ms_breakout_up_20", merged.columns)
        self.assertIn("ms_failed_breakout_up_20", merged.columns)
        self.assertTrue(merged["ms_relative_volume"].notna().any())
        self.assertTrue(meta["volume_valid"])

    def test_forecast_excludes_categorical_market_regime_features(self):
        idx = self.p["SOXL"].index
        bars = pd.DataFrame({
            "Open": self.p["SOXL"].values * 0.99,
            "High": self.p["SOXL"].values * 1.02,
            "Low": self.p["SOXL"].values * 0.98,
            "Close": self.p["SOXL"].values,
            "Volume": np.arange(1000, 1600, dtype=float),
        }, index=idx)
        x, _ = merge_market_structure_features(make_features(self.p, "SOXL"), bars, "SOXL")
        self.assertIn("ms_regime_trend", x.columns)
        self.assertTrue(x["ms_regime_trend"].dropna().isin(["uptrend", "downtrend", "range_or_transition"]).all())
        row, diag = run_forecast("SOXL", "SOXL", self.p, x, "5d")
        self.assertEqual(diag["status"], "ok")
        self.assertIsNotNone(row)
        self.assertIn("probability_up", row)

    def test_cash_index_volume_not_used_as_model_feature(self):
        idx = self.p["^NDX"].index
        bars = pd.DataFrame({
            "Open": self.p["^NDX"].values * 0.99,
            "High": self.p["^NDX"].values * 1.02,
            "Low": self.p["^NDX"].values * 0.98,
            "Close": self.p["^NDX"].values,
            "Volume": np.arange(1000, 1600, dtype=float),
        }, index=idx)
        merged, meta = merge_market_structure_features(make_features(self.p, "^NDX"), bars, "^NDX")
        self.assertFalse(meta["volume_valid"])
        self.assertTrue(merged["ms_relative_volume"].isna().all())

    def test_forward_labels_tail_nan(self):
        x=make_features(self.p,"SOXL"); y=labeled(x,self.p["SOXL"],"5d")
        self.assertTrue(y.up.tail(5).isna().all())
        self.assertTrue(y.forward_return.tail(5).isna().all())
    def test_stress_score_bounded(self):
        rows,s=market_stress(self.p)
        self.assertGreaterEqual(s["risk_score"],0); self.assertLessEqual(s["risk_score"],100)
        self.assertGreaterEqual(s["coverage_items"],4)

if __name__=="__main__": unittest.main()
