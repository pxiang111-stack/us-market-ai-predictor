import unittest

from simple_nq_support_resistance import (
    build_support_resistance,
    calculate_atr,
    confirmed_pivots,
)


def make_bars(highs, lows, closes=None):
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    return [{"high": h, "low": l, "close": c}
            for h, l, c in zip(highs, lows, closes)]


class SupportResistanceTests(unittest.TestCase):
    def test_pivot_not_confirmed_until_right_bars_exist(self):
        bars = make_bars(
            [10, 11, 12, 20, 13, 12],
            [8, 9, 10, 11, 9, 8],
        )
        pivots = confirmed_pivots(bars, left=2, right=2)
        self.assertTrue(any(p["kind"] == "resistance" and
                            p["pivot_index"] == 3 and p["confirmed_index"] == 5
                            for p in pivots))
        # Removing one right-hand bar must remove that pivot.
        self.assertFalse(any(p["pivot_index"] == 3
                             for p in confirmed_pivots(bars[:-1], left=2, right=2)))

    def test_atr_uses_previous_close(self):
        bars = make_bars([10, 12, 13], [8, 9, 10], [9, 10, 12])
        self.assertAlmostEqual(calculate_atr(bars, period=2), 2.0)

    def test_builder_returns_zones_below_and_above_price(self):
        highs = [101 + (i % 3) for i in range(65)]
        lows = [99 + (i % 3) for i in range(65)]
        closes = [100 + (i % 3) for i in range(65)]
        highs[-1], lows[-1], closes[-1] = 111, 108, 110
        bars = make_bars(highs, lows, closes)
        result = build_support_resistance(bars)
        self.assertEqual(result["last_close"], 110)
        self.assertTrue(result["supports"])
        self.assertTrue(result["resistances"])
        self.assertTrue(all(z["mid"] <= 110 for z in result["supports"]))
        self.assertTrue(all(z["mid"] > 110 for z in result["resistances"]))


if __name__ == "__main__":
    unittest.main()
