import csv
from datetime import date, timedelta
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from simple_nq_backtest import (
    _trade_exit,
    _trade_exit_with_actions,
    run_buy_hold,
    load_bars,
    run_strategy,
    run_backtest,
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

    def test_split_adjusts_shares_and_orders_and_dividends_credit(self):
        bars = [
            {"date": "2026-01-01", "open": 50, "high": 52, "low": 49, "close": 51,
             "volume": 1, "dividends": 9.0, "stock_splits": 0.0},
            {"date": "2026-01-02", "open": 25, "high": 26, "low": 24, "close": 25,
             "volume": 1, "dividends": 1.0, "stock_splits": 2.0},
            {"date": "2026-01-05", "open": 25, "high": 31, "low": 24, "close": 30,
             "volume": 1, "dividends": 0.5, "stock_splits": 0.0},
        ]
        idx, price, reason, shares, dividends, _ = _trade_exit_with_actions(
            bars, 0, stop=45, target=60, slippage_bps=0, initial_shares=10
        )
        self.assertEqual(idx, 2)
        self.assertEqual(price, 30)
        self.assertEqual(reason, "TARGET")
        self.assertEqual(shares, 20)
        # Entry-day distribution is excluded; ex-date distributions use post-split shares.
        self.assertEqual(dividends, 30)

    def test_buy_and_hold_credits_dividends(self):
        bars = [
            {"date": "2026-01-01", "open": 100, "high": 101, "low": 99, "close": 100,
             "volume": 1, "dividends": 5.0, "stock_splits": 0.0},
            {"date": "2026-01-02", "open": 100, "high": 101, "low": 99, "close": 100,
             "volume": 1, "dividends": 1.0, "stock_splits": 0.0},
            {"date": "2026-01-05", "open": 100, "high": 101, "low": 99, "close": 100,
             "volume": 1, "dividends": 1.0, "stock_splits": 0.0},
        ]
        result = run_buy_hold(bars, 1000, commission_bps=0, slippage_bps=0)
        # 10 shares bought at first open; first-day dividend is not earned.
        self.assertEqual(result["final_equity"], 1020.0)

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
        with patch("simple_nq_backtest.classify_regime", return_value={"regime": "UPTREND"}):
            with patch("simple_nq_backtest.build_support_resistance", return_value=fake_levels):
                with patch("simple_nq_backtest.evaluate_setups", return_value=fake_setup):
                    metrics, trades, _ = run_strategy(bars, initial_equity=10_000)
        self.assertEqual(trades, [])
        self.assertEqual(metrics["final_equity"], 10_000)

    def test_backtest_writes_summary_and_trade_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "qqq.csv"
            output_dir = Path(tmp) / "out"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "open", "high", "low", "close", "volume", "dividends", "stock_splits"])
                for i in range(270):
                    close = 100 + i * 0.1
                    writer.writerow([(date(2025, 1, 1) + timedelta(days=i)).isoformat(),
                                     close, close + 1, close - 1, close, 1000, 0, 0])
            result = run_backtest(csv_path, output_dir=output_dir)
            self.assertEqual(result["bars"], 270)
            self.assertEqual(result["holdout"]["bars"], 54)
            self.assertEqual(result["holdout"]["start_date"],
                             (date(2025, 1, 1) + timedelta(days=216)).isoformat())
            self.assertEqual(result["holdout"]["fresh_starting_equity"], 10_000)
            self.assertEqual(len(result["holdout"]["strategies"]), 4)
            self.assertTrue((output_dir / "simple_nq_backtest_summary.json").exists())
            self.assertTrue((output_dir / "simple_nq_backtest_trades.csv").exists())

    def test_holdout_starts_with_fresh_equity_and_keeps_warmup_out_of_curve(self):
        bars = []
        for i in range(270):
            close = 100 + i * 0.01
            bars.append({
                "date": (date(2025, 1, 1) + timedelta(days=i)).isoformat(),
                "open": close, "high": close + 0.5, "low": close - 0.5,
                "close": close, "volume": 1000,
            })
        split = 216
        metrics, trades, curve = run_strategy(
            bars, initial_equity=10_000, start_index=split,
            include_warmup=False,
        )
        self.assertEqual(len(curve), len(bars) - split)
        self.assertEqual(curve[0], 10_000)
        self.assertEqual(metrics["initial_equity"], 10_000)
        self.assertTrue(all(t["signal_date"] >= bars[split]["date"] for t in trades))

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
