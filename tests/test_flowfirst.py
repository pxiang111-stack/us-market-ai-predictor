import unittest

from flowfirst_backtest import (crowding_state, flow_direction, holding_window_end,
                                 oi_expanding, trend)


def flow_row(ratio):
    return {
        "buyVol": str(1.0 + ratio),
        "sellVol": str(1.0 - ratio),
    }


class FlowFirstUnitTests(unittest.TestCase):
    def test_flow_direction_detects_abnormal_buy_imbalance(self):
        baseline = [flow_row(-0.04 if i % 2 else 0.04) for i in range(48)]
        recent = [flow_row(0.20) for _ in range(4)]
        result = flow_direction(baseline + recent)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 1)
        self.assertGreater(result[2], 0.75)

    def test_flow_direction_detects_abnormal_sell_imbalance(self):
        baseline = [flow_row(-0.04 if i % 2 else 0.04) for i in range(48)]
        recent = [flow_row(-0.20) for _ in range(4)]
        result = flow_direction(baseline + recent)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], -1)
        self.assertLess(result[2], -0.75)

    def test_flow_direction_requires_full_lookback(self):
        self.assertIsNone(flow_direction([flow_row(0.1) for _ in range(51)]))

    def test_max_holding_window_counts_entry_bar_as_first_bar(self):\n        # Signal at index 60 enters at 61; 16 held bars end at index 76.\n        self.assertEqual(holding_window_end(60, 200), 76)\n\n    def test_max_holding_window_respects_available_history(self):\n        self.assertEqual(holding_window_end(60, 70), 69)\n\n    def test_oi_expansion_is_participation_filter(self):
        rows = [{"sumOpenInterestValue": str(x)} for x in (100, 100.1, 100.2, 100.3, 100.5)]
        self.assertTrue(oi_expanding(rows))

    def test_oi_contraction_does_not_pass_expansion_filter(self):
        rows = [{"sumOpenInterestValue": str(x)} for x in (100, 100.1, 100.2, 100.3, 99.9)]
        self.assertFalse(oi_expanding(rows))

    def test_trend_needs_sufficient_history(self):
        rows = [{"close": str(100 + i)} for i in range(30)]
        self.assertEqual(trend(rows), "unknown")

    def test_crowding_filter_flags_abnormal_long_concentration(self):
        rows = []
        for i in range(48):
            long_share = 0.69 if i % 2 else 0.71
            rows.append({"timestamp": i, "longAccount": str(long_share),
                         "shortAccount": str(1-long_share)})
        rows.append({"timestamp": 48, "longAccount": "0.85", "shortAccount": "0.15"})
        result = crowding_state(rows)
        self.assertIsNotNone(result)
        self.assertTrue(result["crowded_long"])
        self.assertFalse(result["crowded_short"])

    def test_crowding_filter_does_not_flag_normal_high_baseline(self):
        rows = []
        for i in range(48):
            long_share = 0.84 if i % 2 else 0.86
            rows.append({"timestamp": i, "longAccount": str(long_share),
                         "shortAccount": str(1-long_share)})
        rows.append({"timestamp": 48, "longAccount": "0.85", "shortAccount": "0.15"})
        result = crowding_state(rows)
        self.assertIsNotNone(result)
        self.assertFalse(result["crowded_long"])
        self.assertFalse(result["crowded_short"])


if __name__ == "__main__":
    unittest.main()
