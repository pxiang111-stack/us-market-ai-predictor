import unittest
from trade_decision_v5 import build_trade_decision, atr_value
import pandas as pd
import numpy as np


class TradeDecisionTests(unittest.TestCase):
    def features(self):
        return {
            "ms_trend_above_sma20": 1, "ms_sma20_above_sma50": 1,
            "ms_sma50_above_sma200": 1, "ms_failed_breakout_up_20": 0,
            "ms_breakout_up_volume_confirmed": 1, "ms_relative_volume": 2.0,
            "ms_close_location": 0.8,
        }

    def test_long_signal_has_defined_risk_plan(self):
        d = build_trade_decision(close=100, atr=2, probability_up=.70,
            expected_return=.04, features=self.features())
        self.assertEqual(d["decision"], "LONG")
        self.assertEqual(d["stop_loss"], 96)
        self.assertEqual(d["take_profit_2r"], 108)
        self.assertAlmostEqual(d["max_position_fraction"], .125)

    def test_no_trade_when_edge_does_not_cover_cost(self):
        d = build_trade_decision(close=100, atr=2, probability_up=.70,
            expected_return=.001, features=self.features())
        self.assertEqual(d["decision"], "FLAT")

    def test_failed_breakout_blocks_long(self):
        f = self.features(); f["ms_failed_breakout_up_20"] = 1
        d = build_trade_decision(close=100, atr=2, probability_up=.75,
            expected_return=.04, features=f)
        self.assertEqual(d["decision"], "FLAT")

    def test_short_requires_explicit_permission_and_bear_structure(self):
        f = self.features()
        f.update({"ms_trend_above_sma20":0,"ms_sma20_above_sma50":0,
                  "ms_sma50_above_sma200":0,"ms_failed_breakout_down_20":0,
                  "ms_breakout_down_volume_confirmed":1,"ms_close_location":.2})
        denied = build_trade_decision(close=100, atr=2, probability_up=.25,
            expected_return=-.05, features=f, allow_short=False)
        allowed = build_trade_decision(close=100, atr=2, probability_up=.25,
            expected_return=-.05, features=f, allow_short=True)
        self.assertEqual(denied["decision"], "FLAT")
        self.assertEqual(allowed["decision"], "SHORT")
        self.assertEqual(allowed["stop_loss"], 104)

    def test_atr(self):
        idx = pd.date_range("2026-01-01", periods=15)
        bars = pd.DataFrame({"High":np.arange(16,31),"Low":np.arange(14,29),
                             "Close":np.arange(15,30)},index=idx)
        self.assertTrue(np.isfinite(atr_value(bars)))


if __name__ == "__main__":
    unittest.main()
