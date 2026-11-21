import csv
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from simple_nq_backtest import (
    _trade_exit,
    load_bars,
    run_strategy,
)


class BacktestExecutionTests(unittest.TestCase):
    def test_gap_through_stop_fills_at_open_not_stop(self):
        bars = [
            {"date": "2026-01-01", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1},
            {"date": "2026-01-02", "open": 94, "high": 96, "low": 90, "close": 92, "volume": 1},
        ]
        idx, price, reason = _trade_exit(bars, 1, stop=95, target=110, slippage_bps=0)
        self.assertEqual(idx, 1)
        self.assertEqual(price, 94)
        self.assertEqual(reason, "GAP_STOP")

    def test_same_day_stop_and_target_assumes_stop_first(self):
        bars = [
            {"date": "2026-01-01", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1},
            {"date": "2026-01-02", "open": 100, "high": 112, "low": 94, "close": 108, "volume": 1},
        ]
        _, price, reason = _trade_exit(bars, 1, stop=95, target=110, slippage_bps=0)
        self.assertEqual(price, 95)
        self.assertEqual(reason, "STOP_FIRST_AMBIGUOUS")

    def test_load_bars_requires_consistent_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("date,close\n2026-01-01,100\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_bars(path)

    def test_strategy_rejects_reward_risk_below_1_5(self):
        bars = []
        for i in range(207):
            bars.append({
                "date": f"2025-01-{(i % 28) + 1:02d}",
                "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000,
            })
        fake_levels = {"atr": 1.0, "resistances": [{"low": 101.5}]}
        fake_setup = {"action": "LONG_CANDIDATE", "zone": {"low": 99.0}}
        with patch("simple_nq_backtest.classify_regime", return_value={"regime": "UPTREND"}), \\
             patch("simple_nq_backtest.build_support_resistance", return_value=fake_levels), \\
             patch("simple_nq_backtest.evaluate_setups", return_value=fake_setup):
            metrics, trades, _ = run_strategy(bars, initial_equity=10_000)
        self.assertEqual(trades, [])
        self.assertEqual(metrics["final_equity"], 10_000)

    def test_strategy_stays_in_cash_when_no_setup(self):
        bars = []
        for i in range(270):
            close = 100 + i * 0.1
            bars.append({
                "date": f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                "open": close, "high": close + 1, "low": close - 1,
                "close": close, "volume": 1000,
            })
        metrics, trades, curve = run_strategy(bars, initial_equity=10_000)
        self.assertEqual(metrics["final_equity"], 10_000)
        self.assertEqual(trades, [])
        self.assertTrue(all(value == 10_000 for value in curve))


if __name__ == "__main__":
    unittest.main()
