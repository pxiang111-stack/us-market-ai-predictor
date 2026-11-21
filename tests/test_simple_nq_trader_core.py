import unittest

from simple_nq_trader_core import classify_regime, risk_sized_shares


class TraderCoreTests(unittest.TestCase):
    def test_uptrend(self):
        closes = [100 + i * 0.5 for i in range(220)]
        self.assertEqual(classify_regime(closes)["regime"], "UPTREND")

    def test_downtrend(self):
        closes = [220 - i * 0.5 for i in range(220)]
        self.assertEqual(classify_regime(closes)["regime"], "DOWNTREND")

    def test_mixed_market_is_transition(self):
        closes = [100 + i * 0.1 for i in range(200)] + [119, 118, 117, 116, 115]
        self.assertEqual(classify_regime(closes)["regime"], "RANGE_TRANSITION")

    def test_cash_only_risk_sizing(self):
        result = risk_sized_shares(10_000, entry=100, stop=95)
        self.assertEqual(result["shares"], 10)
        self.assertEqual(result["planned_dollar_risk"], 50)
        self.assertEqual(result["notional"], 1_000)
        self.assertEqual(result["cash_remaining"], 9_000)

    def test_cash_cap_limits_size(self):
        result = risk_sized_shares(10_000, entry=1_000, stop=999)
        self.assertEqual(result["shares"], 10)

    def test_rejects_invalid_long_stop(self):
        with self.assertRaises(ValueError):
            risk_sized_shares(10_000, entry=100, stop=101)


if __name__ == "__main__":
    unittest.main()
