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
            "ms_close_location": 0.8, "ms_volume_weighted_close_location_20": 0.20,
            "ms_spring_reclaim_20": 0, "ms_upthrust_reject_20": 0,
        }

    def test_long_signal_has_defined_risk_plan(self):
        d = build_trade_decision(close=100, atr=2, probability_up=.70,
            expected_return=.04, features=self.features())
        self.assertEqual(d["decision"], "LONG")
        self.assertTrue(d["gate_probability_long"])
        self.assertTrue(d["gate_edge_long"])
        self.assertTrue(d["gate_bull_trend"])
        self.assertTrue(d["gate_long_structure"])
        self.assertEqual(d["stop_loss"], 96)
        self.assertEqual(d["take_profit_2r"], 108)
        self.assertAlmostEqual(d["max_position_fraction"], .125)

    def test_lower_probability_gate_can_be_compared_without_changing_cost_floor(self):
        baseline = build_trade_decision(close=100, atr=2, probability_up=.60,
            expected_return=.04, features=self.features(), round_trip_cost=.004)
        active = build_trade_decision(close=100, atr=2, probability_up=.60,
            expected_return=.04, features=self.features(), round_trip_cost=.004,
            probability_threshold_long=.58, probability_threshold_short=.42)
        self.assertEqual(baseline["decision"], "FLAT")
        self.assertEqual(active["decision"], "LONG")
        self.assertEqual(active["round_trip_cost_assumption"], .004)

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
                  "ms_breakout_down_volume_confirmed":1,"ms_close_location":.2,
                  "ms_volume_weighted_close_location_20":-.20})
        denied = build_trade_decision(close=100, atr=2, probability_up=.25,
            expected_return=-.05, features=f, allow_short=False)
        allowed = build_trade_decision(close=100, atr=2, probability_up=.25,
            expected_return=-.05, features=f, allow_short=True)
        self.assertEqual(denied["decision"], "FLAT")
        self.assertEqual(allowed["decision"], "SHORT")
        self.assertEqual(allowed["stop_loss"], 104)

    def test_long_blocked_by_upthrust_even_with_bullish_trend(self):
        f = self.features(); f["ms_upthrust_reject_20"] = 1
        d = build_trade_decision(close=100, atr=2, probability_up=.75,
            expected_return=.04, features=f)
        self.assertEqual(d["decision"], "FLAT")

    def test_long_requires_positive_volume_weighted_effort_or_spring(self):
        f = self.features(); f["ms_volume_weighted_close_location_20"] = -.10
        d = build_trade_decision(close=100, atr=2, probability_up=.75,
            expected_return=.04, features=f)
        self.assertEqual(d["decision"], "FLAT")
        f["ms_spring_reclaim_20"] = 1
        d2 = build_trade_decision(close=100, atr=2, probability_up=.75,
            expected_return=.04, features=f)
        self.assertEqual(d2["decision"], "LONG")


    def test_trend_breakout_requires_confirmed_volume_breakout(self):
        f = self.features()
        f["ms_breakout_up_volume_confirmed"] = 0
        d = build_trade_decision(close=100, atr=2, probability_up=.70,
            expected_return=.04, features=f, strategy_mode="trend_breakout")
        self.assertEqual(d["decision"], "FLAT")
        f["ms_breakout_up_volume_confirmed"] = 1
        d2 = build_trade_decision(close=100, atr=2, probability_up=.70,
            expected_return=.04, features=f, strategy_mode="trend_breakout")
        self.assertEqual(d2["decision"], "LONG")

    def test_reversal_absorption_uses_spring_without_full_bull_trend(self):
        f = self.features()
        f.update({"ms_trend_above_sma20":0,"ms_sma20_above_sma50":0,
                  "ms_sma50_above_sma200":0,"ms_spring_reclaim_20":1})
        d = build_trade_decision(close=100, atr=2, probability_up=.60,
            expected_return=.02, features=f, strategy_mode="reversal_absorption")
        self.assertEqual(d["decision"], "LONG")

    def test_reversal_short_requires_upthrust_and_short_permission(self):
        f = self.features()
        f.update({"ms_upthrust_reject_20":1,"ms_trend_above_sma20":1,
                  "ms_sma20_above_sma50":1,"ms_sma50_above_sma200":1})
        denied = build_trade_decision(close=100, atr=2, probability_up=.40,
            expected_return=-.02, features=f, strategy_mode="reversal_absorption",
            allow_short=False)
        allowed = build_trade_decision(close=100, atr=2, probability_up=.40,
            expected_return=-.02, features=f, strategy_mode="reversal_absorption",
            allow_short=True)
        self.assertEqual(denied["decision"], "FLAT")
        self.assertEqual(allowed["decision"], "SHORT")

    def test_unknown_strategy_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            build_trade_decision(close=100, atr=2, probability_up=.7,
                expected_return=.04, features=self.features(),
                strategy_mode="unknown_mode")

    def test_atr(self):
        idx = pd.date_range("2026-01-01", periods=15)
        bars = pd.DataFrame({"High":np.arange(16,31),"Low":np.arange(14,29),
                             "Close":np.arange(15,30)},index=idx)
        self.assertTrue(np.isfinite(atr_value(bars)))


if __name__ == "__main__":
    unittest.main()
