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

    def test_relative_volume_and_close_location(self):
        out = build_market_structure(self.ohlcv, volume_window=5)
        self.assertTrue(np.isnan(out.relative_volume.iloc[3]))
        self.assertAlmostEqual(out.close_location.iloc[-1], 2 / 3)
        self.assertTrue(np.isfinite(out.relative_volume.iloc[-1]))

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
