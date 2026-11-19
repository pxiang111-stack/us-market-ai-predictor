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

    def test_rejects_missing_ohlcv_columns(self):
        with self.assertRaises(ValueError):
            build_market_structure(self.ohlcv[["Open", "Close", "Volume"]])


if __name__ == "__main__":
    unittest.main()
