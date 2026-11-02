import unittest
import numpy as np
import pandas as pd

from quant_research import (
    add_factor_scores,
    evaluate_factor_portfolio,
    factor_information_coefficient,
)


class FactorResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2020-01-01", periods=420)
        tickers = [f"T{i:03d}" for i in range(60)] + ["SPY"]
        rows = []
        for dt in dates:
            latent = rng.normal(size=len(tickers))
            for i, ticker in enumerate(tickers):
                mom21 = latent[i] + rng.normal(scale=.2)
                future = .025 * latent[i] + rng.normal(scale=.04)
                rows.append({
                    "date": dt, "ticker": ticker,
                    "mom_21": mom21, "mom_63": latent[i] + rng.normal(scale=.2),
                    "mom_126": latent[i] + rng.normal(scale=.2),
                    "mom_252": latent[i] + rng.normal(scale=.2),
                    "relative_mom_21": latent[i] + rng.normal(scale=.2),
                    "ma_gap_21": latent[i] + rng.normal(scale=.2),
                    "ma_gap_63": latent[i] + rng.normal(scale=.2),
                    "vol_21": abs(rng.normal(.25, .08)),
                    "vol_63": abs(rng.normal(.30, .08)),
                    "drawdown_126": -abs(rng.normal(.15, .1)),
                    "rsi_14": rng.uniform(20, 80),
                    "target_ret_1m": future,
                    "target_ret_3m": future * 2,
                    "target_ret_6m": future * 3,
                })
        cls.data = pd.DataFrame(rows).set_index("date").sort_index()
        cls.data = add_factor_scores(cls.data)

    def test_factor_scores_are_finite_and_bounded(self):
        score = self.data["factor_score"].dropna()
        self.assertGreater(len(score), 0)
        self.assertTrue(np.isfinite(score).all())
        self.assertGreaterEqual(float(score.min()), 0.0)
        self.assertLessEqual(float(score.max()), 1.0)

    def test_factor_backtest_returns_rebalance_rows(self):
        summary, detail = evaluate_factor_portfolio(
            self.data, "1m", etfs={"SPY"}, transaction_cost_bps=10
        )
        self.assertEqual(summary["status"], "ok")
        self.assertGreaterEqual(summary["n_rebalances"], 3)
        self.assertTrue((detail["net_return"] <= detail["gross_return"]).all())

    def test_factor_ic_report(self):
        result = factor_information_coefficient(self.data, "1m", etfs={"SPY"})
        self.assertEqual(result["ic_status"], "ok")
        self.assertGreaterEqual(result["ic_observations"], 3)


if __name__ == "__main__":
    unittest.main()
