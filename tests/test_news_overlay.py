import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import us_market_ai_v3_1_news as scanner


class NewsOverlayTests(unittest.TestCase):
    @patch.object(scanner, "fetch_gdelt_articles", return_value=[])
    @patch.object(scanner, "score_headlines", return_value=(np.nan, 0))
    def test_missing_news_does_not_change_opportunity_score(self, score_mock, fetch_mock):
        original = pd.DataFrame([
            {"ticker": "TEST", "horizon": "6m", "opportunity_score": 72.5},
            {"ticker": "TEST", "horizon": "1m", "opportunity_score": 61.0},
        ])

        result, news = scanner.add_news_overlay(original.copy())

        self.assertEqual(result.loc[result["horizon"].eq("6m"), "opportunity_score"].iloc[0], 72.5)
        self.assertEqual(result.loc[result["horizon"].eq("1m"), "opportunity_score"].iloc[0], 61.0)
        self.assertTrue(pd.isna(result.loc[result["horizon"].eq("6m"), "combined_news_score"].iloc[0]))
        self.assertEqual(result.loc[result["horizon"].eq("6m"), "news_confidence"].iloc[0], 0.0)
        self.assertEqual(len(news), 1)

    def test_regression_target_bounds_limit_extreme_outliers(self):
        values = pd.Series([-50.0] + [0.01 * i for i in range(200)] + [100.0])
        lower, upper = scanner.regression_target_bounds(values)
        self.assertGreater(lower, -50.0)
        self.assertLess(upper, 100.0)
        self.assertLess(lower, upper)


if __name__ == "__main__":
    unittest.main()
