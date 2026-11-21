import unittest
import pandas as pd
from walkforward_execution_v5 import simulate_trades, performance_summary


class WalkForwardExecutionTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2025-01-01", periods=5, freq="B")
        self.signals = pd.DataFrame([{
            "decision": "LONG", "stop_loss": 95.0,
            "take_profit_1r": 105.0, "take_profit_2r": 110.0,
        }], index=[self.dates[0]])

    def test_signal_enters_next_open_and_target_exit(self):
        bars = pd.DataFrame({
            "Open": [100, 101, 103, 106, 106],
            "High": [101, 106, 104, 107, 107],
            "Low": [99, 102, 102, 105, 105],
            "Close": [100, 104, 103, 106, 106],
        }, index=self.dates)
        ledger, _ = simulate_trades(bars, self.signals, round_trip_cost_bps=0, slippage_bps_per_side=0)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger.iloc[0]["entry_date"], self.dates[1])
        self.assertEqual(ledger.iloc[0]["exit_reason"], "time_exit_after_1r")
        self.assertEqual(ledger.iloc[0]["partial_exit_price"], 107.0)
        self.assertEqual(ledger.iloc[0]["partial_units"], ledger.iloc[0]["units"] * 0.5)
        self.assertEqual(len(_), len(bars))
        stats = performance_summary(ledger, _)
        self.assertEqual(stats["trade_count"], 1)
        self.assertIn("max_drawdown_pct", stats)

    def test_stop_first_when_stop_and_target_both_touch(self):
        bars = pd.DataFrame({
            "Open": [100, 100, 100, 100, 100],
            "High": [101, 106, 101, 101, 101],
            "Low": [99, 94, 99, 99, 99],
            "Close": [100, 100, 100, 100, 100],
        }, index=self.dates)
        ledger, _ = simulate_trades(bars, self.signals, round_trip_cost_bps=0, slippage_bps_per_side=0)
        self.assertEqual(ledger.iloc[0]["exit_reason"], "stop")
        self.assertEqual(ledger.iloc[0]["exit_price"], 95.0)

    def test_gap_stop_fills_at_adverse_open(self):
        bars = pd.DataFrame({
            "Open": [100, 100, 90, 91, 92],
            "High": [101, 102, 92, 93, 94],
            "Low": [99, 99, 89, 90, 91],
            "Close": [100, 101, 91, 92, 93],
        }, index=self.dates)
        ledger, _ = simulate_trades(bars, self.signals, round_trip_cost_bps=0, slippage_bps_per_side=0)
        self.assertEqual(ledger.iloc[0]["exit_reason"], "gap_stop")
        self.assertEqual(ledger.iloc[0]["exit_price"], 90.0)


    def test_two_stage_targets_take_half_at_1r_and_half_at_2r(self):
        bars = pd.DataFrame({
            "Open": [100, 101, 103, 106, 106],
            "High": [101, 114, 104, 107, 107],
            "Low": [99, 102, 102, 105, 105],
            "Close": [100, 112, 103, 106, 106],
        }, index=self.dates)
        ledger, curve = simulate_trades(bars, self.signals, round_trip_cost_bps=0, slippage_bps_per_side=0)
        trade = ledger.iloc[0]
        self.assertEqual(trade["exit_reason"], "target_2r")
        self.assertEqual(trade["partial_exit_price"], 107.0)
        self.assertEqual(trade["exit_price"], 113.0)
        expected = trade["units"] * 0.5 * (107.0 - 101.0) + trade["units"] * 0.5 * (113.0 - 101.0)
        self.assertAlmostEqual(trade["gross_pnl"], expected)
        exit_equity = curve.loc[curve["date"] == trade["exit_date"], "equity"].iloc[0]
        self.assertAlmostEqual(exit_equity, trade["equity_after"])

    def test_stop_after_first_target_keeps_partial_profit(self):
        bars = pd.DataFrame({
            "Open": [100, 101, 102, 100, 100],
            "High": [101, 104, 103, 101, 101],
            "Low": [99, 102, 94, 99, 99],
            "Close": [100, 104, 100, 100, 100],
        }, index=self.dates)
        signals = self.signals.copy()
        signals["stop_loss"] = 99.0
        ledger, _ = simulate_trades(bars, signals, round_trip_cost_bps=0, slippage_bps_per_side=0)
        trade = ledger.iloc[0]
        self.assertEqual(trade["exit_reason"], "stop_after_1r")
        self.assertEqual(trade["partial_exit_price"], 103.0)
        self.assertEqual(trade["exit_price"], 101.0)
        self.assertGreater(trade["gross_pnl"], 0)


    def test_same_bar_1r_and_breakeven_touch_exits_remainder_conservatively_long(self):
        bars = pd.DataFrame({
            "Open": [100, 101, 100, 100, 100],
            "High": [101, 104, 101, 101, 101],
            "Low": [99, 100, 99, 99, 99],
            "Close": [100, 102, 100, 100, 100],
        }, index=self.dates)
        signals = self.signals.copy()
        signals["stop_loss"] = 99.0
        ledger, _ = simulate_trades(bars, signals, round_trip_cost_bps=0, slippage_bps_per_side=0)
        trade = ledger.iloc[0]
        self.assertEqual(trade["partial_exit_date"], self.dates[1])
        self.assertEqual(trade["exit_date"], self.dates[1])
        self.assertEqual(trade["exit_reason"], "stop_after_1r")
        self.assertEqual(trade["partial_exit_price"], 103.0)
        self.assertEqual(trade["exit_price"], 101.0)
        self.assertGreater(trade["gross_pnl"], 0)

    def test_same_bar_1r_and_breakeven_touch_exits_remainder_conservatively_short(self):
        bars = pd.DataFrame({
            "Open": [100, 101, 102, 100, 100],
            "High": [101, 102, 103, 101, 101],
            "Low": [99, 98, 101, 99, 99],
            "Close": [100, 99, 102, 100, 100],
        }, index=self.dates)
        signals = self.signals.copy()
        signals["decision"] = "SHORT"
        signals["stop_loss"] = 103.0
        ledger, _ = simulate_trades(bars, signals, round_trip_cost_bps=0, slippage_bps_per_side=0)
        trade = ledger.iloc[0]
        self.assertEqual(trade["partial_exit_date"], self.dates[1])
        self.assertEqual(trade["exit_date"], self.dates[1])
        self.assertEqual(trade["exit_reason"], "stop_after_1r")
        self.assertEqual(trade["partial_exit_price"], 99.0)
        self.assertEqual(trade["exit_price"], 101.0)
        self.assertGreater(trade["gross_pnl"], 0)

    def test_whole_share_cash_mode_uses_integer_shares_and_full_exit_for_single_share(self):
        bars = pd.DataFrame({
            "Open": [100, 101, 102, 102, 102],
            "High": [101, 108, 103, 103, 103],
            "Low": [99, 100, 101, 101, 101],
            "Close": [100, 107, 102, 102, 102],
        }, index=self.dates)
        ledger, curve = simulate_trades(
            bars, self.signals, initial_equity=1000.0,
            risk_budget_pct=0.01, max_gross_exposure=1.0,
            round_trip_cost_bps=0.0, slippage_bps_per_side=0.0,
            whole_shares_only=True,
        )
        self.assertEqual(len(ledger), 1)
        trade = ledger.iloc[0]
        self.assertEqual(trade["units"], 1)
        self.assertEqual(trade["partial_units"], 0)
        self.assertTrue(trade["whole_shares_only"])
        self.assertLessEqual(trade["units"] * trade["entry_price"], 1000.0)
        self.assertEqual(trade["exit_reason"], "target_1r_full")
        self.assertEqual(trade["exit_price"], 107.0)
        self.assertTrue(float(trade["units"]).is_integer())

    def test_flat_signal_does_not_trade(self):
        bars = pd.DataFrame({"Open":[100]*5,"High":[101]*5,"Low":[99]*5,"Close":[100]*5},index=self.dates)
        signals = self.signals.copy()
        signals.iloc[0, signals.columns.get_loc("decision")] = "FLAT"
        ledger, _ = simulate_trades(bars, signals)
        self.assertTrue(ledger.empty)


    def test_execution_diagnostics_count_skipped_position_signal(self):
        bars = pd.DataFrame({
            "Open": [100, 100, 100, 100, 100],
            "High": [101, 101, 101, 101, 101],
            "Low": [99, 99, 99, 94, 99],
            "Close": [100, 100, 100, 96, 100],
        }, index=self.dates)
        signals = pd.DataFrame([
            {"decision": "LONG", "stop_loss": 95.0, "take_profit_1r": 105.0, "take_profit_2r": 110.0},
            {"decision": "LONG", "stop_loss": 95.0, "take_profit_1r": 105.0, "take_profit_2r": 110.0},
            {"decision": "LONG", "stop_loss": 95.0, "take_profit_1r": 105.0, "take_profit_2r": 110.0},
        ], index=[self.dates[0], self.dates[1], self.dates[3]])
        ledger, curve, diag = simulate_trades(
            bars, signals, round_trip_cost_bps=0, slippage_bps_per_side=0,
            return_diagnostics=True,
        )
        self.assertEqual(diag["signals_received"], 3)
        self.assertEqual(diag["executed_trades"], 2)
        self.assertEqual(diag["skipped_existing_position"], 1)
        self.assertEqual(diag["skipped_total"], 1)
        self.assertEqual(len(ledger), 2)

if __name__ == "__main__":
    unittest.main()
