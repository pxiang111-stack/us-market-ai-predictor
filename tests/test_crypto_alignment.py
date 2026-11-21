import unittest

import pandas as pd

from run_regime_first_v2 import align_crypto_to_nq_close
from regime_first_v2 import evaluate_history


class CryptoAlignmentTests(unittest.TestCase):
    def test_uses_last_completed_utc_bar_not_same_date_bar(self):
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        crypto = pd.DataFrame({"Close": range(100, 130)}, index=dates)
        target = pd.DatetimeIndex([pd.Timestamp("2026-01-25")])

        aligned = align_crypto_to_nq_close(crypto, target)

        # Jan 24 UTC bar is the latest fully completed bar by the Jan 25
        # NQ session close. Jan 25's UTC bar is not complete yet.
        self.assertEqual(aligned.iloc[0]["Close"], 123)
        # The breakout band uses the 20 completed UTC bars before Jan 24.
        self.assertEqual(aligned.iloc[0]["prior_20d_high"], 122)
        self.assertEqual(aligned.iloc[0]["prior_20d_low"], 103)

    def test_btc_band_and_close_share_same_availability_date(self):
        dates = pd.date_range("2026-01-01", periods=25, freq="D")
        crypto = pd.DataFrame({"Close": range(1, 26)}, index=dates)
        target = pd.date_range("2026-01-22", periods=2, freq="D")

        aligned = align_crypto_to_nq_close(crypto, target)

        self.assertEqual(aligned.iloc[0]["Close"], 21)
        self.assertEqual(aligned.iloc[0]["prior_20d_high"], 20)
        self.assertEqual(aligned.iloc[1]["Close"], 22)
        self.assertEqual(aligned.iloc[1]["prior_20d_high"], 21)


class VolatilityMultiplierTests(unittest.TestCase):
    def test_vix_above_45_uses_quarter_multiplier(self):
        row = {
            "date": "2026-11-20", "NQ_close": 25000, "NQ_high": 25100,
            "NQ_low": 24900, "NDX_close": 25000, "BTC_close": 60000,
            "ETH_close": 2000, "VIX_close": 46, "NQ_200ma": 24000,
            "NQ_20d_high": 26000, "NQ_20d_low": 24500,
            "NQ_60d_high": 27000, "NQ_60d_low": 23000,
            "BTC_20d_high": 65000, "BTC_20d_low": 58000,
            "ETH_20d_high": 2200, "ETH_20d_low": 1900,
            "CCC_spread": 8.0, "rank_model_score": 0.0,
            "BBB_spread": 2.0, "HY_spread": 4.0,
        }
        result = evaluate_history(pd.DataFrame([row])).iloc[-1]
        self.assertEqual(result.volatility_multiplier, 0.25)
        self.assertAlmostEqual(result.max_risk_pct, 0.5)  # NEUTRAL cap 2% x VIX multiplier 0.25


if __name__ == "__main__":
    unittest.main()
