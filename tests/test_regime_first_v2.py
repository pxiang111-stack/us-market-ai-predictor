import unittest
import pandas as pd

from regime_first_v2 import evaluate_history


def row(date, **overrides):
    base = {
        "date": date, "NQ_close": 25000, "NQ_high": 25100, "NQ_low": 24900,
        "NDX_close": 25000, "BTC_close": 60000, "ETH_close": 2000,
        "VIX_close": 25, "NQ_200ma": 24000,
        "NQ_20d_high": 26000, "NQ_20d_low": 24500,
        "NQ_60d_high": 27000, "NQ_60d_low": 23000,
        "BTC_20d_high": 65000, "BTC_20d_low": 58000,
        "ETH_20d_high": 2200, "ETH_20d_low": 1900,
        "CCC_spread": 8.0, "rank_model_score": 0.0,
        "BBB_spread": 2.0, "HY_spread": 4.0,
    }
    base.update(overrides)
    return base


class RegimeFirstV2Tests(unittest.TestCase):
    def test_crypto_break_triggers_risk_off(self):
        data = pd.DataFrame([row("2026-11-20", BTC_close=57000)])
        result = evaluate_history(data)
        self.assertEqual(result.iloc[-1].state, "RISK_OFF")
        self.assertEqual(result.iloc[-1].allowed_action, "只减仓/对冲")

    def test_risk_off_requires_two_recovery_closes(self):
        data = pd.DataFrame([
            row("2026-11-20", BTC_close=57000),
            row("2026-11-23", BTC_close=60000),
            row("2026-11-24", BTC_close=60000),
        ])
        result = evaluate_history(data)
        self.assertEqual(result.iloc[0].state, "RISK_OFF")
        self.assertEqual(result.iloc[1].state, "RISK_OFF")
        self.assertEqual(result.iloc[2].state, "NEUTRAL")

    def test_missing_critical_data_halves_budget_once(self):
        data = pd.DataFrame([row("2026-11-20", VIX_close=38, NQ_20d_low=float("nan"))])
        result = evaluate_history(data)
        self.assertAlmostEqual(result.iloc[-1].data_gap_multiplier, 0.5)
        self.assertAlmostEqual(result.iloc[-1].max_risk_pct, 0.5)

    def test_flip_gate_requires_all_four_conditions(self):
        rows = []
        for j, score in enumerate([-2.0, -1.0, 0.0]):
            rows.append(row(
                f"2026-11-{20+j:02d}", NQ_close=28000,
                NQ_200ma=24000, VIX_close=18,
                BTC_close=70000, BTC_20d_high=69000,
                ETH_close=2400, ETH_20d_high=2300,
                NQ_20d_high=27500, rank_model_score=score,
            ))
        result = evaluate_history(pd.DataFrame(rows))
        self.assertEqual(result.iloc[-1].flip_count, 4)
        self.assertEqual(result.iloc[-1].state, "RISK_ON")

    def test_btc_break_is_not_hidden_by_missing_eth_feed(self):
        data = pd.DataFrame([row("2026-11-20", BTC_close=57000)]).drop(
            columns=["ETH_close", "ETH_20d_low"]
        )
        result = evaluate_history(data)
        self.assertEqual(result.iloc[-1].state, "RISK_OFF")
        self.assertIn("ETH_close", result.iloc[-1].data_gaps)

    def test_monitor_hard_veto_zeroes_new_risk(self):
        data = pd.DataFrame([row("2026-11-20", monitor_hard_veto=True)])
        result = evaluate_history(data)
        self.assertEqual(result.iloc[-1].monitor_veto_multiplier, 0.0)
        self.assertEqual(result.iloc[-1].max_risk_pct, 0.0)
        self.assertEqual(result.iloc[-1].allowed_action, "只减仓/对冲")


if __name__ == "__main__":
    unittest.main()
