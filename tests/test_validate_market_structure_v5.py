import unittest

import numpy as np
import pandas as pd

from validate_market_structure_v5 import forward_returns, evaluate_symbol


class MarketStructureValidationTests(unittest.TestCase):
    def test_forward_returns_are_aligned_to_next_horizon_bars(self):
        r = pd.Series([0.10, 0.20, -0.10, 0.05, 0.01])
        out = forward_returns(r, 2)
        self.assertAlmostEqual(out.iloc[0], (1.20 * 0.90) - 1.0)
        self.assertAlmostEqual(out.iloc[1], (0.90 * 1.05) - 1.0)
        self.assertTrue(out.iloc[-2:].isna().all())

    def test_training_labels_are_purged_and_result_is_descriptive(self):
        n = 420
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        rng = np.random.default_rng(7)
        r = rng.normal(0.0004, 0.012, n)
        close = 100 * np.cumprod(1 + r)
        frame = pd.DataFrame({
            "bar_date": idx,
            "return_1bar": r,
            "trend_above_sma20": (close > pd.Series(close).rolling(20).mean()).astype(float),
            "sma20_above_sma50": 1.0,
            "sma50_above_sma200": 1.0,
            "atr_pct_14": 0.02,
            "range_position_20": rng.uniform(0, 1, n),
            "relative_volume": rng.uniform(0.5, 2.0, n),
            "close_location": rng.uniform(0, 1, n),
        })
        rows = evaluate_symbol(
            frame, "TEST", horizons=(5,), first_test_index=220,
            retrain_every=20, min_train_rows=150,
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["ticker"] == "TEST" for row in rows))
        self.assertTrue(all(row["status"] in {"descriptive_oos", "insufficient_oos"} for row in rows))


if __name__ == "__main__":
    unittest.main()
