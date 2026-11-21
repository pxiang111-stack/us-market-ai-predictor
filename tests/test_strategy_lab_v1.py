import unittest

import numpy as np
import pandas as pd

from strategy_lab_v1 import build_strategy_signals, simulate_cash_account


class CashAccountStrategyLabTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2025-01-01", periods=5, freq="B")
        self.bars = pd.DataFrame({
            "Open": [100.0, 100.0, 110.0, 105.0, 100.0],
            "High": [101.0, 111.0, 111.0, 106.0, 101.0],
            "Low": [99.0, 99.0, 109.0, 99.0, 99.0],
            "Close": [100.0, 110.0, 110.0, 100.0, 100.0],
            "Dividends": [0.0, 0.0, 1.0, 0.0, 0.0],
        }, index=self.dates)

    def test_next_open_execution_and_integer_cash_only_sizing(self):
        signals = pd.Series([True, False, False, False, False], index=self.dates)
        ledger, curve, stats = simulate_cash_account(
            self.bars, signals, initial_equity=1000.0, cost_bps_per_order=0.0
        )
        self.assertEqual(ledger.iloc[0]["date"], self.dates[1])
        self.assertEqual(ledger.iloc[0]["side"], "BUY")
        self.assertEqual(ledger.iloc[0]["shares"], 10)
        self.assertEqual(ledger.iloc[1]["date"], self.dates[2])
        self.assertEqual(ledger.iloc[1]["side"], "SELL")
        self.assertEqual(ledger.iloc[1]["shares"], 10)
        self.assertTrue(stats["whole_shares_only"])
        self.assertFalse(stats["shorts_enabled"])
        self.assertFalse(stats["leverage_enabled"])
        self.assertTrue((curve["cash"] >= -1e-8).all())
        self.assertTrue((curve["shares"] >= 0).all())
        self.assertTrue(np.all(curve["shares"].to_numpy() % 1 == 0))

    def test_dividend_cash_paid_to_previous_close_holder(self):
        signals = pd.Series([True, True, False, False, False], index=self.dates)
        ledger, curve, stats = simulate_cash_account(
            self.bars, signals, initial_equity=1000.0, cost_bps_per_order=0.0
        )
        # The portfolio bought at the second session's open, so it owns shares
        # at the prior close entering the third session's ex-date.
        self.assertEqual(stats["dividend_cash_received_usd"], 10.0)
        self.assertGreater(curve.iloc[2]["cash"], 0.0)
        self.assertEqual(ledger.iloc[-1]["side"], "SELL")

    def test_costs_reduce_ending_equity(self):
        signals = pd.Series([True, False, False, False, False], index=self.dates)
        _, _, free = simulate_cash_account(
            self.bars, signals, initial_equity=1000.0, cost_bps_per_order=0.0
        )
        _, _, charged = simulate_cash_account(
            self.bars, signals, initial_equity=1000.0, cost_bps_per_order=20.0
        )
        self.assertLess(charged["ending_equity"], free["ending_equity"])
        self.assertGreater(charged["transaction_costs_usd"], 0.0)


    def test_start_date_uses_prior_warmup_signal_but_scores_only_requested_window(self):
        signals = pd.Series([False, True, True, False, False], index=self.dates)
        ledger, curve, stats = simulate_cash_account(
            self.bars, signals, initial_equity=1000.0,
            cost_bps_per_order=0.0, start_date=self.dates[2]
        )
        self.assertEqual(curve.index[0], self.dates[2])
        self.assertEqual(ledger.iloc[0]["date"], self.dates[2])
        self.assertEqual(ledger.iloc[0]["side"], "BUY")
        self.assertEqual(ledger.iloc[0]["shares"], 9)
        self.assertEqual(stats["initial_equity_usd"], 1000.0)

    def test_signals_use_only_current_and_prior_completed_bars(self):
        close = np.linspace(100.0, 500.0, 300)
        bars = pd.DataFrame({
            "Open": close, "High": close + 1, "Low": close - 1,
            "Close": close, "Dividends": 0.0,
        }, index=pd.date_range("2020-01-01", periods=300, freq="B"))
        signals = build_strategy_signals(bars)
        self.assertEqual(len(signals), 300)
        self.assertTrue(signals.index[0] == bars.index[0])
        self.assertTrue(signals["QQQ_buy_hold"].all())
        for name in [
            "SMA100_trend", "SMA150_trend", "SMA200_trend", "SMA250_trend",
            "SMA150_entry_SMA200_exit", "MOM_12_1_plus_SMA200", "Donchian_20_10", "Donchian_40_20",
            "Donchian_55_20",
        ]:
            self.assertIn(name, signals.columns)
            self.assertEqual(signals[name].dtype, bool)


    def test_full_candidate_set_is_present_for_three_year_comparison(self):
        close = np.linspace(100.0, 500.0, 300)
        bars = pd.DataFrame({
            "Open": close, "High": close + 1, "Low": close - 1,
            "Close": close, "Dividends": 0.0,
        }, index=pd.date_range("2020-01-01", periods=300, freq="B"))
        signals = build_strategy_signals(bars)
        expected = {
            "QQQ_buy_hold", "SMA100_trend", "SMA150_trend",
            "SMA200_trend", "SMA250_trend", "SMA150_entry_SMA200_exit",
            "MOM_12_1_plus_SMA200", "Donchian_20_10",
            "Donchian_40_20", "Donchian_55_20",
        }
        self.assertEqual(set(signals.columns), expected)


if __name__ == "__main__":
    unittest.main()
