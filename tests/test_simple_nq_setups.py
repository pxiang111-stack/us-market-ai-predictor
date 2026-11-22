import unittest

from simple_nq_setups import evaluate_setups


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.levels = {
            "atr": 2.0,
            "supports": [{"low": 98, "high": 100, "mid": 99, "touches": 3}],
            "resistances": [{"low": 110, "high": 112, "mid": 111, "touches": 2}],
        }

    def test_support_bounce_candidate(self):
        bars = [{"high": 103, "low": 99, "close": 101}]
        result = evaluate_setups(bars, "UPTREND", self.levels)
        self.assertEqual(result["setup"], "SUPPORT_BOUNCE")
        self.assertEqual(result["action"], "LONG_CANDIDATE")
        self.assertEqual(result["execution"], "NEXT_SESSION_ONLY")

    def test_no_support_bounce_in_downtrend(self):
        bars = [{"high": 103, "low": 99, "close": 101}]
        result = evaluate_setups(bars, "DOWNTREND", self.levels)
        self.assertEqual(result["action"], "WAIT")

    def test_resistance_rejection_is_research_only_in_downtrend(self):
        bars = [{"high": 111, "low": 107, "close": 109}]
        result = evaluate_setups(bars, "DOWNTREND", self.levels)
        self.assertEqual(result["setup"], "RESISTANCE_REJECTION")
        self.assertEqual(result["action"], "SHORT_SETUP_RESEARCH_ONLY")
        self.assertEqual(result["execution"], "NO_V1_SHORT_ORDERS")

    def test_resistance_rejection_does_not_short_in_range(self):
        bars = [{"high": 111, "low": 107, "close": 109}]
        result = evaluate_setups(bars, "RANGE_TRANSITION", self.levels)
        self.assertEqual(result["action"], "WAIT_FOR_CONFIRMATION")

    def test_no_touch_means_wait(self):
        bars = [{"high": 106, "low": 103, "close": 104}]
        result = evaluate_setups(bars, "UPTREND", self.levels)
        self.assertEqual(result["setup"], "NONE")
        self.assertEqual(result["action"], "WAIT")


if __name__ == "__main__":
    unittest.main()
