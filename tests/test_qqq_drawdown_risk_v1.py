import unittest

import numpy as np
import pandas as pd

from qqq_drawdown_risk_v1 import (
    evaluate_drawdown_forecasts,
    future_drawdown_labels,
)


class QQQDrawdownRiskTests(unittest.TestCase):
    def test_future_label_excludes_asof_bar_and_purges_tail(self):
        dates = pd.date_range("2025-01-01", periods=4, freq="B")
        bars = pd.DataFrame({
            "Low": [80.0, 99.0, 98.0, 90.0],
            "Close": [100.0, 100.0, 100.0, 100.0],
        }, index=dates)
        labels, drawdown = future_drawdown_labels(bars, 2, 0.03)
        # Day one's own crash low is not part of its future label.
        self.assertFalse(bool(labels.iloc[0]))
        self.assertTrue(bool(labels.iloc[1]))
        self.assertTrue(np.isnan(labels.iloc[-1]))
        self.assertTrue(np.isnan(labels.iloc[-2]))
        self.assertTrue(np.isnan(drawdown.iloc[-1]))

    def test_incomplete_latest_forecasts_are_excluded_from_oos_metrics(self):
        dates = pd.date_range("2025-01-01", periods=4, freq="B")
        bars = pd.DataFrame({
            "Low": [100.0, 95.0, 99.0, 100.0],
            "Close": [100.0, 100.0, 100.0, 100.0],
        }, index=dates)
        forecasts = pd.DataFrame([
            {"as_of": dates[0], "horizon_sessions": 1, "drawdown_threshold_pct": 3,
             "drawdown_risk_score": 0.8, "future_event_eval_only": True},
            {"as_of": dates[1], "horizon_sessions": 1, "drawdown_threshold_pct": 3,
             "drawdown_risk_score": 0.1, "future_event_eval_only": False},
            {"as_of": dates[2], "horizon_sessions": 1, "drawdown_threshold_pct": 3,
             "drawdown_risk_score": 0.9, "future_event_eval_only": np.nan},
        ])
        detail, summary = evaluate_drawdown_forecasts(forecasts, bars)
        self.assertEqual(len(detail), 3)
        self.assertFalse(summary.empty)
        self.assertTrue((summary["forecast_count"] == 2).all())


if __name__ == "__main__":
    unittest.main()
