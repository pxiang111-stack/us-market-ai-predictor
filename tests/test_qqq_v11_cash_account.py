import unittest

import pandas as pd

from qqq_v11_raw_price_dividends import simulate


class QQQV11CashAccountTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2026-01-05", periods=3, freq="B")
        self.bars = pd.DataFrame({
            "Close": [100.0, 100.0, 100.0],
            "Adj Close": [100.0, 100.0, 99.0],
            "Dividends": [0.0, 0.0, 1.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        }, index=self.dates)

    def test_close_signal_enters_next_session_and_ex_date_dividend_belongs_to_prior_close_shares(self):
        # Signal on day 1 enters day 2. The day-3 ex-date distribution is
        # earned by shares held at day-2 close, even though the strategy sells
        # those shares at day-3 close.
        target = pd.Series([1.0, 0.0, 0.0], index=self.dates)
        summary, curve, ledger = simulate("test", self.bars, target)

        self.assertEqual(ledger.iloc[0]["date"], str(self.dates[1].date()))
        self.assertEqual(ledger.iloc[0]["side"], "BUY")
        self.assertEqual(ledger.iloc[0]["shares"], 99)
        self.assertEqual(ledger.iloc[1]["date"], str(self.dates[2].date()))
        self.assertEqual(ledger.iloc[1]["side"], "SELL")
        self.assertEqual(summary["dividends_received_usd"], 99.0)
        self.assertEqual(summary["ending_shares"], 0)
        self.assertAlmostEqual(summary["ending_cash_usd"], 10079.2)
        self.assertGreaterEqual(curve["cash"].min(), 0.0)
        self.assertTrue((curve["shares"] % 1 == 0).all())

    def test_buying_on_ex_date_does_not_receive_distribution(self):
        target = pd.Series([0.0, 1.0, 0.0], index=self.dates)
        summary, _, ledger = simulate("test", self.bars, target)

        self.assertEqual(ledger.iloc[0]["date"], str(self.dates[2].date()))
        self.assertEqual(ledger.iloc[0]["side"], "BUY")
        self.assertEqual(summary["dividends_received_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
