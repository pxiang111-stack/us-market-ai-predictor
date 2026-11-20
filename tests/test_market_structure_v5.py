import unittest

import numpy as np
import pandas as pd

from market_structure_v5 import build_market_structure


class MarketStructureTests(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
        self.ohlcv = pd.DataFrame({
            "Open": np.arange(100, 130, dtype=float),
            "High": np.arange(102, 132, dtype=float),
            "Low": np.arange(99, 129, dtype=float),
            "Close": np.arange(101, 131, dtype=float),
            "Volume": np.arange(1000, 1030, dtype=float),
        }, index=idx)

    def test_relative_volume_uses_prior_bars_only(self):
        out = build_market_structure(self.ohlcv, volume_window=5)
        self.assertTrue(np.isnan(out.relative_volume.iloc[4]))
        expected = self.ohlcv["Volume"].iloc[5] / self.ohlcv["Volume"].iloc[:5].mean()
        self.assertAlmostEqual(out.relative_volume.iloc[5], expected)
        self.assertAlmostEqual(out.close_location.iloc[-1], 2 / 3)

    def test_breakout_levels_use_prior_bars_only(self):
        d = self.ohlcv.copy()
        d.loc[d.index[-1], ["Open", "High", "Low", "Close"]] = [140, 150, 139, 149]
        out = build_market_structure(d, volume_window=5)
        self.assertEqual(out.breakout_up_20.iloc[-1], 1.0)
        extended = pd.concat([d, d.iloc[[-1]].set_axis([d.index[-1] + pd.Timedelta(hours=1)])])
        out2 = build_market_structure(extended, volume_window=5)
        self.assertEqual(out2.breakout_up_20.iloc[-2], 1.0)

    def test_failed_breakout_is_flagged(self):
        d = self.ohlcv.copy()
        d.loc[d.index[-1], ["Open", "High", "Low", "Close"]] = [131, 140, 129, 130]
        out = build_market_structure(d, volume_window=5)
        self.assertEqual(out.failed_breakout_up_20.iloc[-1], 1.0)

    def test_cash_index_volume_is_excluded(self):
        out = build_market_structure(self.ohlcv, volume_window=5, volume_is_valid=False)
        self.assertTrue(out.relative_volume.isna().all())
        self.assertTrue(out.signed_volume_proxy_ratio.isna().all())
        self.assertTrue((out.volume_data_status == "invalid_for_cash_index").all())

    def test_wyckoff_effort_result_proxies_are_computed(self):
        d = self.ohlcv.copy()
        d.loc[d.index[-1], ["Open", "High", "Low", "Close", "Volume"]] = [100, 112, 98, 100.5, 5000]
        out = build_market_structure(d, volume_window=5)
        self.assertGreater(out.relative_volume.iloc[-1], 1.0)
        self.assertGreater(out.effort_result_absorption_proxy.iloc[-1], 1.0)
        self.assertTrue(np.isfinite(out.close_location_volume_pressure.iloc[-1]))
        self.assertTrue(np.isfinite(out.volume_weighted_close_location_20.iloc[-1]))

    def test_spring_and_upthrust_rejection_patterns(self):
        spring = self.ohlcv.copy()
        spring.loc[spring.index[-1], ["Open", "High", "Low", "Close"]] = [105, 115, 100, 112]
        spring_out = build_market_structure(spring, volume_window=5)
        self.assertEqual(spring_out.spring_reclaim_20.iloc[-1], 1.0)
        upthrust = self.ohlcv.copy()
        upthrust.loc[upthrust.index[-1], ["Open", "High", "Low", "Close"]] = [130, 140, 127, 128]
        upthrust_out = build_market_structure(upthrust, volume_window=5)
        self.assertEqual(upthrust_out.upthrust_reject_20.iloc[-1], 1.0)

    def test_new_volume_proxies_are_missing_for_invalid_cash_index_volume(self):
        out = build_market_structure(self.ohlcv, volume_window=5, volume_is_valid=False)
        for col in ("close_location_volume_pressure", "effort_result_absorption_proxy",
                    "volume_weighted_close_location_20"):
            self.assertTrue(out[col].isna().all())

    def test_price_up_oi_up_label(self):
        oi = pd.Series(np.arange(200, 230, dtype=float), index=self.ohlcv.index)
        out = build_market_structure(self.ohlcv, oi, volume_window=5)
        self.assertEqual(
            out.price_oi_regime.iloc[-1],
            "price_up_oi_up_building_exposure",
        )

    def test_missing_oi_is_explicit(self):
        out = build_market_structure(self.ohlcv, volume_window=5)
        self.assertTrue((out.price_oi_regime == "oi_unavailable").all())

    def test_causal_trend_and_volatility_regime_labels(self):
        idx = pd.date_range("2020-01-01", periods=330, freq="B")
        close = 100 * np.exp(np.linspace(0, 0.5, len(idx)))
        d = pd.DataFrame({
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(idx), 100000.0),
        }, index=idx)
        out = build_market_structure(d)
        self.assertEqual(out.regime_trend.iloc[-1], "uptrend")
        self.assertIn(out.regime_volatility.iloc[-1],
                      {"high_volatility", "normal_volatility", "low_volatility"})
        self.assertTrue(pd.notna(out.market_regime.iloc[-1]))
        self.assertTrue(out.market_regime.iloc[:260].isna().any())

    def test_regime_volatility_baseline_does_not_use_future_bars(self):
        idx = pd.date_range("2020-01-01", periods=330, freq="B")
        close = 100 * np.exp(np.linspace(0, 0.2, len(idx)))
        d = pd.DataFrame({
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.full(len(idx), 100000.0),
        }, index=idx)
        first = build_market_structure(d)
        cutoff = idx[-20]
        extended = d.copy()
        extended.loc[idx[-1], ["High", "Low"]] = [close[-1] * 1.5, close[-1] * 0.5]
        second = build_market_structure(extended)
        self.assertAlmostEqual(first.loc[cutoff, "atr_pct_baseline_252"],
                               second.loc[cutoff, "atr_pct_baseline_252"])

    def test_rejects_missing_ohlcv_columns(self):
        with self.assertRaises(ValueError):
            build_market_structure(self.ohlcv[["Open", "Close", "Volume"]])


if __name__ == "__main__":
    unittest.main()
