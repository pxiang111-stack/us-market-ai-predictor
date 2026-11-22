import csv
from datetime import date, timedelta
import tempfile
import unittest
from pathlib import Path

from simple_nq_walkforward import run_walkforward


class WalkForwardTests(unittest.TestCase):
    def test_walkforward_uses_prior_bars_as_warmup_and_resets_equity(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "qqq.csv"
            out = Path(tmp) / "out"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "date", "open", "high", "low", "close", "volume",
                    "dividends", "stock_splits",
                ])
                for i in range(270):
                    close = 100 + i * 0.1
                    writer.writerow([
                        (date(2010, 1, 1) + timedelta(days=i)).isoformat(),
                        close, close + 1, close - 1, close, 1000, 0, 0,
                    ])
            result = run_walkforward(csv_path, out)
            self.assertEqual(len(result["folds"]), 1)
            fold = result["folds"][0]
            self.assertEqual(fold["fold"], "2010-2014")
            self.assertEqual(fold["warmup_bars"], 205)
            self.assertEqual(fold["bars_in_fold"], 65)
            self.assertEqual(fold["strategies"][0]["initial_equity"], 10_000)
            self.assertEqual(fold["strategies"][0]["final_equity"], 10_000)
            self.assertTrue((out / "simple_nq_walkforward_summary.json").exists())
            self.assertTrue((out / "simple_nq_walkforward_trades.csv").exists())


if __name__ == "__main__":
    unittest.main()
